
import optuna
import yaml
import torch
import torch.optim as optim
import subprocess
import copy
import sys
import numpy as np
import os
from tqdm import tqdm

# --- We are copying all imports here ---
from src.environment import WarehouseEnvironment
from src.utils.buffer import ReplayBuffer
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.core.models import EnsembleDynamicsModel # <-- 1. IMPORT THE NEW MODEL
from src.planning.fsm_planner import FSMAutomaton, FSM_STATE_GOAL, FSM_STATE_FAILED
from src.utils.logger import Logger
from src.utils.seeding import set_seed
from src.utils.visualize import plot_critic_landscapes, create_evaluation_animation


# --- 1. Define the "Objective" Function ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    One "trial" of our auto-trainer.
    It suggests parameters, runs training, runs evaluation,
    and returns a score.
    """
    
    set_seed(42)

    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # --- These are the "knobs" Optuna will turn ---
    config['train']['lambda_cbf'] = trial.suggest_loguniform('lambda_cbf', 1e2, 1e5)
    config['train']['exploration_noise'] = trial.suggest_uniform('exploration_noise', 0.05, 0.3)
    config['train']['lr_actor'] = trial.suggest_loguniform('lr_actor', 1e-5, 1e-3)
    config['train']['lr_critic'] = trial.suggest_loguniform('lr_critic', 1e-5, 1e-3)
    config['train']['lr_dynamics'] = trial.suggest_loguniform('lr_dynamics', 1e-5, 1e-3) # <-- 2. TUNE DYNAMICS LR
    config['train']['num_episodes'] = 3000 # <-- 3. LONGER TRAINING

    print(f"\n--- TRIAL {trial.number} STARTING ---")
    print(f"Params: {trial.params}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = config['train']
    nn_config = config['nn']

    # --- 3. RUN TRAINING LOOP ---
    try:
        env = WarehouseEnv(config)
        fsm = FSMAutomaton(
            start_pos=np.array(config['fsm']['start_state']),
            goal_pos=np.array(config['fsm']['goal_state']),
            config=config
        )
        buffer = ReplayBuffer(
            state_dim=nn_config['state_dim'], action_dim=nn_config['action_dim'],
            subgoal_dim=nn_config['subgoal_dim'], max_size=train_config['buffer_size']
        )
        
        policy_net = SubgoalConditionedPolicy(
            state_dim=nn_config['state_dim'], subgoal_dim=nn_config['subgoal_dim'],
            action_dim=nn_config['action_dim'], hidden_dims=nn_config['hidden_dims'],
            a_max=env.a_max
        ).to(device)
        
        cbf_net = CBFNetwork(
            state_dim=nn_config['state_dim'], hidden_dims=nn_config['hidden_dims']
        ).to(device)
        
        clf_net = CLFNetwork(
            state_dim=nn_config['state_dim'], subgoal_dim=nn_config['subgoal_dim'],
            hidden_dims=nn_config['hidden_dims']
        ).to(device)
        
        # --- 4. USE THE NEW ENSEMBLE MODEL ---
        dynamics_net = EnsembleDynamicsModel(
            state_dim=nn_config['state_dim'], 
            action_dim=nn_config['action_dim'],
            hidden_dims=nn_config['hidden_dims'],
            num_ensemble=nn_config['num_ensemble']
        ).to(device)
        
        if train_config['load_anchored_models']:
            cbf_net.load_state_dict(torch.load(train_config['anchored_cbf_path'], map_location=device))
            clf_net.load_state_dict(torch.load(train_config['anchored_clf_path'], map_location=device))
            print("Successfully loaded pre-trained CBF and CLF models.")
            
        policy_optim = optim.Adam(policy_net.parameters(), lr=train_config['lr_actor'])
        cbf_optim = optim.Adam(cbf_net.parameters(), lr=train_config['lr_critic'])
        clf_optim = optim.Adam(clf_net.parameters(), lr=train_config['lr_critic'])
        # --- 5. CREATE THE NEW OPTIMIZER ---
        dynamics_optim = optim.Adam(dynamics_net.parameters(), lr=train_config['lr_dynamics'])

        print(f"--- Starting Main Training (Algorithm 2) ---")
        global_step = 0
        best_ep_reward = -np.inf
        
        for episode in range(train_config['num_episodes']):
            fsm.reset()
            s = env.reset(start_pos=fsm.start_pos)
            ep_reward = 0
            
            for t in range(train_config['max_episode_length']):
                global_step += 1
                g = fsm.get_current_subgoal()
                
                with torch.no_grad():
                    s_torch = torch.from_numpy(s).float().to(device).unsqueeze(0)
                    g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
                    a_det = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0)
                
                exploration_noise = train_config['exploration_noise'] * env.a_max 
                a = a_det + np.random.normal(0, exploration_noise, size=nn_config['action_dim']) 
                a = np.clip(a, -env.a_max, env.a_max)
                
                s_next, _, done, info = env.step(a)
                v_star = env.get_ground_truth_feasibility(s_next, g)
                collision_penalty = 100.0 if info['is_collision'] else 0.0
                r = -np.sqrt(v_star) - collision_penalty 
                ep_reward += r
                
                fsm_state = fsm.transition(s_next)
                done = info['is_collision'] or (fsm_state == FSM_STATE_GOAL)
                
                buffer.add(s, a, s_next, g, r, done, info['h_star'], v_star)
                s = s_next
                
                if len(buffer) > train_config['batch_size']:
                    # --- 6. UPDATE DYNAMICS ---
                    if global_step % train_config['model_update_freq'] == 0:
                        batch = buffer.sample(train_config['batch_size'])
                        b_s, b_a, b_s_next = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'actions', 'next_states'])
                        # The new compute_loss handles its own optimization
                        _, _ = dynamics_net.compute_loss(b_s, b_a, b_s_next, dynamics_optim)
                    
                    if global_step % train_config['cbf_update_freq'] == 0:
                        batch = buffer.sample(train_config['batch_size'])
                        b_s, b_s_next, b_h_star_next = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'next_states', 'h_stars'])
                        cbf_optim.zero_grad()
                        loss, _ = cbf_net.compute_loss_constraint(b_s, b_s_next, b_h_star_next, config)
                        loss.backward()
                        cbf_optim.step()

                    if global_step % train_config['clf_update_freq'] == 0:
                        batch = buffer.sample(train_config['batch_size'])
                        b_s, b_s_next, b_g, b_v_star = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'next_states', 'subgoals', 'v_stars'])
                        clf_optim.zero_grad()
                        loss, _ = clf_net.compute_loss_constraint(b_s, b_s_next, b_g, b_v_star, config)
                        loss.backward()
                        clf_optim.step()

                    if global_step % train_config['policy_update_freq'] == 0:
                        batch = buffer.sample(train_config['batch_size'])
                        b_s, b_g = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'subgoals'])
                        b_a = policy_net(b_s, b_g)
                        b_s_next_pred = dynamics_net(b_s, b_a)
                        policy_optim.zero_grad()
                        for param in cbf_net.parameters(): param.requires_grad = False
                        for param in clf_net.parameters(): param.requires_grad = False
                        loss, _ = policy_net.compute_loss(b_s, b_g, b_s_next_pred, cbf_net, clf_net, config)
                        loss.backward()
                        policy_optim.step()
                        for param in cbf_net.parameters(): param.requires_grad = True
                        for param in clf_net.parameters(): param.requires_grad = True

                if done:
                    break
            
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                os.makedirs(train_config['model_save_path'], exist_ok=True)
                torch.save(policy_net.state_dict(), os.path.join(train_config['model_save_path'], f"policy_trial_{trial.number}.pth"))
                torch.save(cbf_net.state_dict(), os.path.join(train_config['model_save_path'], f"cbf_trial_{trial.number}.pth"))
                torch.save(clf_net.state_dict(), os.path.join(train_config['model_save_path'], f"clf_trial_{trial.number}.pth"))
                torch.save(dynamics_net.state_dict(), os.path.join(train_config['model_save_path'], f"dynamics_trial_{trial.number}.pth"))
        
        print(f"--- Main Training Complete (Trial {trial.number}) ---")

        # --- 4. RUN EVALUATION ---
        policy_net.eval()
        cbf_net.eval()
        clf_net.eval()
        dynamics_net.eval()

        eval_buffer = ReplayBuffer(
            state_dim=nn_config['state_dim'], action_dim=nn_config['action_dim'],
            subgoal_dim=nn_config['subgoal_dim'], max_size=config['data_gen']['num_transitions']
        )
        eval_buffer.load(config['data_gen']['data_path'])
        
        eval_fsm = FSMAutomaton(
            start_pos=np.array(config['fsm']['start_state']),
            goal_pos=np.array(config['fsm']['goal_state']),
            config=config
        )
        
        is_valid, safety_rate, feasibility_rate = eval_fsm.prune_fsm_with_certificates(
            replay_buffer=eval_buffer,
            policy_net=policy_net,
            dynamics_net=dynamics_net,
            cbf_net=cbf_net,
            clf_net=clf_net,
            device=device
        )
        
        score = safety_rate * feasibility_rate
        
        print(f"--- TRIAL {trial.number} COMPLETE ---")
        print(f"Reward: {best_ep_reward:.2f}, Safety: {safety_rate:.2f}, Feasibility: {feasibility_rate:.2f}")
        print(f"==> Score: {score:.4f}")

        current_best_score = trial.study.best_value if trial.study.best_value is not None else 0.0
        if score > current_best_score:
            print(f"*** New Best Score! Saving Champion Models. ***")
            torch.save(policy_net.state_dict(), os.path.join(train_config['model_save_path'], "policy_best.pth"))
            torch.save(cbf_net.state_dict(), os.path.join(train_config['model_save_path'], "cbf_best.pth"))
            torch.save(clf_net.state_dict(), os.path.join(train_config['model_save_path'], "clf_best.pth"))
            torch.save(dynamics_net.state_dict(), os.path.join(train_config['model_save_path'], "dynamics_best.pth"))

        return score

    except Exception as e:
        print(f"TRIAL {trial.number} FAILED with exception: {e}")
        return 0.0 # Return a 0 score if the trial crashes

# --- 2. Main "Auto-Trainer" Script ---
if __name__ == "__main__":
    print("="*40)
    print("🚀 STARTING AUTO-TRAINER 🚀")
    print("="*40)

    python_executable = sys.executable 

    print("Running generate_data.py...")
    subprocess.run([python_executable, "-m", "scripts.generate_data"], check=True)
    
    print("Running pretrain.py...")
    subprocess.run([python_executable, "-m", "scripts.pretrain"], check=True)

    print("\n--- STARTING OPTUNA OPTIMIZATION ---")
    study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=50)

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score (Safety * Feasibility): {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    print("\nTo use these parameters, update your config/warehouse_v1.yaml file.")
    print("You can now run 'python -m scripts.run_final_demo' to see the final result.")