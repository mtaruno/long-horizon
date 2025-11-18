# scripts/hpo_trainer.py

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

from src.environment.warehouse import WarehouseEnv
from src.utils.buffer import ReplayBuffer
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.core.models import EnsembleDynamicsModel
from src.planning.fsm_planner import FSMAutomaton, FSM_STATE_GOAL, FSM_STATE_WAYPOINT_1
from src.utils.seeding import set_seed
from src.utils.visualization import plot_critic_landscapes, create_evaluation_animation
from typing import Dict, Tuple, Any

def compute_actor_loss(
    s: torch.Tensor,
    g: torch.Tensor,
    s_next: torch.Tensor, # Can be real or predicted
    cbf_net: CBFNetwork,
    clf_net: CLFNetwork,
    config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to compute the core actor loss."""
    train_config = config['train']
    
    # --- 1. CLF Penalty (Feasibility) ---
    v_psi = clf_net(s, g).detach() 
    v_psi_next = clf_net(s_next, g)
    beta = train_config['clf_beta']
    delta = train_config['clf_delta']
    clf_violation = v_psi_next - (1 - beta) * v_psi - delta
    penalty_clf_constraint = torch.mean(torch.relu(clf_violation) ** 2)

    # --- 2. CBF Penalty (Safety) ---
    h_phi = cbf_net(s).detach() 
    h_phi_next = cbf_net(s_next)
    alpha = train_config['cbf_alpha']
    cbf_violation = h_phi_next - (1 - alpha) * h_phi
    penalty_cbf_constraint = torch.mean(torch.relu(-cbf_violation) ** 2)

    # --- 3. Total Loss ---
    loss = (penalty_clf_constraint + 
            train_config['lambda_cbf'] * penalty_cbf_constraint)
            
    return loss, penalty_clf_constraint, penalty_cbf_constraint


def objective(trial: optuna.trial.Trial) -> float:
    """
    One "trial" of our auto-trainer.
    It suggests parameters, runs training, runs evaluation,
    and returns a score.
    """
    
    set_seed(42)

    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 1. Suggest new parameters for this trial
    config['train']['lambda_cbf'] = trial.suggest_float('lambda_cbf', 1e2, 1e5, log=True)
    config['train']['lambda_mf'] = trial.suggest_float('lambda_mf', 0.1, 0.9)
    config['train']['exploration_noise'] = trial.suggest_float('exploration_noise', 0.05, 0.3)
    config['train']['lr_actor'] = trial.suggest_float('lr_actor', 1e-5, 1e-3, log=True)
    config['train']['lr_critic'] = trial.suggest_float('lr_critic', 1e-5, 1e-3, log=True)
    config['train']['lr_dynamics'] = trial.suggest_float('lr_dynamics', 1e-5, 1e-3, log=True)
    config['train']['num_episodes'] = 300 # Reduced for a faster trial

    print(f"\n--- TRIAL {trial.number} STARTING ---")
    print(f"Params: {trial.params}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = config['train']
    nn_config = config['nn']

    try:
        # 2. Initialize all components
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
        
        # --- THIS IS THE FIX for 'omega_max' error ---
        policy_net = SubgoalConditionedPolicy(
            state_dim=nn_config['state_dim'], subgoal_dim=nn_config['subgoal_dim'],
            action_dim=nn_config['action_dim'], hidden_dims=nn_config['hidden_dims'],
            a_max=env.a_max, omega_max=env.omega_max # Pass both maxes
        ).to(device)
        # --- END FIX ---
        
        cbf_net = CBFNetwork(
            state_dim=nn_config['state_dim'], hidden_dims=nn_config['hidden_dims']
        ).to(device)
        
        clf_net = CLFNetwork(
            state_dim=nn_config['state_dim'], subgoal_dim=nn_config['subgoal_dim'],
            hidden_dims=nn_config['hidden_dims']
        ).to(device)
        
        dynamics_net = EnsembleDynamicsModel(
            state_dim=nn_config['state_dim'], action_dim=nn_config['action_dim'],
            hidden_dims=nn_config['hidden_dims'], num_ensemble=nn_config['num_ensemble']
        ).to(device)
        
        if train_config['load_anchored_models']:
            cbf_net.load_state_dict(torch.load(train_config['anchored_cbf_path'], map_location=device))
            clf_net.load_state_dict(torch.load(train_config['anchored_clf_path'], map_location=device))
            print("Successfully loaded pre-trained CBF and CLF models.")
            
        policy_optim = optim.Adam(policy_net.parameters(), lr=train_config['lr_actor'])
        cbf_optim = optim.Adam(cbf_net.parameters(), lr=train_config['lr_critic'])
        clf_optim = optim.Adam(clf_net.parameters(), lr=train_config['lr_critic'])
        dynamics_optim = optim.Adam(dynamics_net.parameters(), lr=train_config['lr_dynamics'])

        print(f"--- Starting Main Training (Algorithm 2) ---")
        global_step = 0
        
        # 3. Run Training Loop
        episode_iterator = tqdm(range(train_config['num_episodes'])) # Re-enabled tqdm
        
        for episode in episode_iterator:
            fsm.reset()
            s_nn = env.reset(start_pos=fsm.start_pos)
            episode_ended = False
            
            # 3a. Run first sub-task: START -> WAYPOINT_1
            g = fsm.get_current_subgoal() # This is waypoint_1
            for t in range(train_config['max_episode_length']):
                global_step += 1
                with torch.no_grad():
                    s_torch = torch.from_numpy(s_nn).float().to(device).unsqueeze(0)
                    g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
                    a_det_unscaled = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0)
                
                # --- THIS IS THE FIX for action scaling ---
                exploration_noise = train_config['exploration_noise']
                # Add noise to the [-1, 1] action
                a_unscaled = a_det_unscaled + np.random.normal(0, exploration_noise, size=nn_config['action_dim'])
                a_unscaled = np.clip(a_unscaled, -1.0, 1.0)
                
                # Scale action for the environment
                a_scaled = a_unscaled * np.array([env.a_max, env.omega_max])
                # --- END FIX ---
                
                s_next_nn, _, done, info = env.step(a_scaled)
                v_star = env.get_ground_truth_feasibility(s_next_nn, g)
                collision_penalty = 100.0 if info['is_collision'] else 0.0
                r = -np.sqrt(v_star) - collision_penalty 
                fsm_state = fsm.transition(s_next_nn)
                done = info['is_collision']
                
                buffer.add(s_nn, a_scaled, s_next_nn, g, r, done, info['h_star'], v_star)
                s_nn = s_next_nn
                
                if len(buffer) > train_config['batch_size']:
                    if global_step % train_config['model_update_freq'] == 0:
                        batch = buffer.sample(train_config['batch_size'])
                        b_s, b_a, b_s_next = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'actions', 'next_states'])
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
                        b_s, b_g, b_s_next_real = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'subgoals', 'next_states'])
                        
                        b_a_mb_unscaled = policy_net(b_s, b_g) # Raw output [-1, 1]
                        b_a_mb_scaled = b_a_mb_unscaled * torch.tensor([env.a_max, env.omega_max], device=device, dtype=torch.float32)
                        
                        b_s_next_pred = dynamics_net(b_s, b_a_mb_scaled)
                        
                        loss_mb, _, _ = compute_actor_loss(b_s, b_g, b_s_next_pred, cbf_net, clf_net, config)
                        loss_mf, _, _ = compute_actor_loss(b_s, b_g, b_s_next_real, cbf_net, clf_net, config)
                        
                        lambda_mf = train_config['lambda_mf']
                        loss = (1.0 - lambda_mf) * loss_mb + lambda_mf * loss_mf
                        
                        policy_optim.zero_grad()
                        for param in cbf_net.parameters(): param.requires_grad = False
                        for param in clf_net.parameters(): param.requires_grad = False
                        loss.backward()
                        policy_optim.step()
                        for param in cbf_net.parameters(): param.requires_grad = True
                        for param in clf_net.parameters(): param.requires_grad = True
                if done or fsm_state == FSM_STATE_WAYPOINT_1:
                    break
            if done: episode_ended = True
            
            # 3b. Run second sub-task: WAYPOINT_1 -> GOAL
            if not episode_ended:
                g = fsm.get_current_subgoal() # This is the final goal
                for t in range(train_config['max_episode_length']):
                    global_step += 1
                    with torch.no_grad():
                        s_torch = torch.from_numpy(s_nn).float().to(device).unsqueeze(0)
                        g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
                        a_det_unscaled = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0)
                    
                    exploration_noise_lin = train_config['exploration_noise'] * env.a_max 
                    exploration_noise_ang = train_config['exploration_noise'] * env.omega_max
                    a_unscaled = a_det_unscaled + np.array([np.random.normal(0, exploration_noise_lin), np.random.normal(0, exploration_noise_ang)])
                    a_unscaled = np.clip(a_unscaled, -1.0, 1.0)
                    a_scaled = a_unscaled * np.array([env.a_max, env.omega_max])

                    s_next_nn, _, done, info = env.step(a_scaled)
                    v_star = env.get_ground_truth_feasibility(s_next_nn, g)
                    collision_penalty = 100.0 if info['is_collision'] else 0.0
                    r = -np.sqrt(v_star) - collision_penalty 
                    fsm_state = fsm.transition(s_next_nn)
                    done = info['is_collision'] or (fsm_state == FSM_STATE_GOAL)
                    
                    buffer.add(s_nn, a_scaled, s_next_nn, g, r, done, info['h_star'], v_star)
                    s_nn = s_next_nn
                    
                    if len(buffer) > train_config['batch_size']:
                        if global_step % train_config['model_update_freq'] == 0:
                            batch = buffer.sample(train_config['batch_size'])
                            b_s, b_a, b_s_next = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'actions', 'next_states'])
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
                            b_s, b_g, b_s_next_real = (torch.from_numpy(batch[k]).float().to(device) for k in ['states', 'subgoals', 'next_states'])
                            
                            b_a_mb_unscaled = policy_net(b_s, b_g) # Raw output [-1, 1]
                            b_a_mb_scaled = b_a_mb_unscaled * torch.tensor([env.a_max, env.omega_max], device=device, dtype=torch.float32)
                            
                            b_s_next_pred = dynamics_net(b_s, b_a_mb_scaled)
                            
                            loss_mb, _, _ = compute_actor_loss(b_s, b_g, b_s_next_pred, cbf_net, clf_net, config)
                            loss_mf, _, _ = compute_actor_loss(b_s, b_g, b_s_next_real, cbf_net, clf_net, config)
                            
                            lambda_mf = train_config['lambda_mf']
                            loss = (1.0 - lambda_mf) * loss_mb + lambda_mf * loss_mf
                            
                            policy_optim.zero_grad()
                            for param in cbf_net.parameters(): param.requires_grad = False
                            for param in clf_net.parameters(): param.requires_grad = False
                            loss.backward()
                            policy_optim.step()
                            for param in cbf_net.parameters(): param.requires_grad = True
                            for param in clf_net.parameters(): param.requires_grad = True
                    if done:
                        break
        
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
        
        # 4a. Run the FSM check
        is_valid, safety_rate, feasibility_rate = eval_fsm.prune_fsm_with_certificates(
            replay_buffer=eval_buffer,
            policy_net=policy_net,
            dynamics_net=dynamics_net,
            cbf_net=cbf_net,
            clf_net=clf_net,
            device=device
        )
        
        # 4b. Run the deterministic demo
        demo_env = WarehouseEnv(config)
        s_nn = demo_env.reset(start_pos=eval_fsm.start_pos)
        eval_fsm.reset() 
        
        demo_success = False
        episode_ended = False
        
        # Run first leg
        for t in range(train_config['max_episode_length']):
            g = eval_fsm.get_current_subgoal()
            with torch.no_grad():
                s_torch = torch.from_numpy(s_nn).float().to(device).unsqueeze(0)
                g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
                a_unscaled = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0)
            
            a_scaled = a_unscaled * np.array([demo_env.a_max, demo_env.omega_max])

            s_next_nn, _, _, info = demo_env.step(a_scaled)
            fsm_state = eval_fsm.transition(s_next_nn)
            if info['is_collision']: episode_ended = True; break
            if fsm_state == FSM_STATE_WAYPOINT_1: break
            s_nn = s_next_nn
        else: episode_ended = True # Timed out

        # Run second leg
        if not episode_ended:
            for t in range(train_config['max_episode_length']):
                g = eval_fsm.get_current_subgoal()
                with torch.no_grad():
                    s_torch = torch.from_numpy(s_nn).float().to(device).unsqueeze(0)
                    g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
                    a_unscaled = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0)
                
                a_scaled = a_unscaled * np.array([demo_env.a_max, demo_env.omega_max])
                
                s_next_nn, _, _, info = demo_env.step(a_scaled)
                fsm_state = eval_fsm.transition(s_next_nn)
                if info['is_collision']: episode_ended = True; break
                if fsm_state == FSM_STATE_GOAL: 
                    demo_success = True # Reached the final goal
                    break
                s_nn = s_next_nn
            else: episode_ended = True # Timed out
        
        # --- 5. CALCULATE FINAL SCORE ---
        score = 0.0
        if is_valid and demo_success:
            score = safety_rate * feasibility_rate
        
        print(f"--- TRIAL {trial.number} COMPLETE ---")
        print(f"FSM_Check: {is_valid}, Demo_Success: {demo_success}")
        print(f"Avg Safety: {safety_rate:.2f}, Avg Feasibility: {feasibility_rate:.2f}")
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
    print("="*40) # Fixed typo 4g -> 40

    python_executable = sys.executable 

    print("Running generate_data.py...")
    subprocess.run([python_executable, "-m", "scripts.generate_data"], check=True)
    
    print("Running pretrain.py...")
    subprocess.run([python_executable, "-m", "scripts.pretrain"], check=True)

    print("\n--- STARTING OPTUNA OPTIMIZATION ---")
    study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=25) # 25 trials for a reasonable search time

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score (Safety * Feasibility): {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    print("\nTo use these parameters, update your config/warehouse_v1.yaml file.")
    print("You can now run 'python -m scripts.evaluate' to see the final result.")