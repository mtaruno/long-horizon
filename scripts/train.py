import yaml
import numpy as np
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from src.environment.warehouse import WarehouseEnv
from src.utils.buffer import ReplayBuffer
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.core.models import DynamicsModel
from src.planning.fsm_planner import FSMAutomaton, FSM_STATE_GOAL, FSM_STATE_FAILED
from src.utils.logger import Logger
from src.utils.seeding import set_seed

def run_training(config: dict, device: torch.device, use_tqdm: bool = True) -> float:
    """
    Runs the main training loop.
    Returns the best average episode reward.
    """
    train_config = config['train']
    nn_config = config['nn']
    
    # Initialize Logger
    # Disable logging to TensorBoard during optimization to save disk space
    logger = Logger("logs/main_train")
    
    # 1. Initialize Environment, FSM, and Buffer
    env = WarehouseEnv(config)
    fsm = FSMAutomaton(
        start_pos=np.array(config['fsm']['start_state']),
        goal_pos=np.array(config['fsm']['goal_state']),
        config=config
    )
    buffer = ReplayBuffer(
        state_dim=nn_config['state_dim'],
        action_dim=nn_config['action_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        max_size=train_config['buffer_size']
    )
    
    # 2. Initialize All Networks
    policy_net = SubgoalConditionedPolicy(
        state_dim=nn_config['state_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        action_dim=nn_config['action_dim'],
        hidden_dims=nn_config['hidden_dims'],
        a_max=env.a_max
    ).to(device)
    
    cbf_net = CBFNetwork(
        state_dim=nn_config['state_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    clf_net = CLFNetwork(
        state_dim=nn_config['state_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    dynamics_net = DynamicsModel(
        state_dim=nn_config['state_dim'],
        action_dim=nn_config['action_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    # 3. Load Anchored Models (Phase 1 result)
    if train_config['load_anchored_models']:
        try:
            cbf_net.load_state_dict(torch.load(train_config['anchored_cbf_path'], map_location=device))
            clf_net.load_state_dict(torch.load(train_config['anchored_clf_path'], map_location=device))
            print("Successfully loaded pre-trained CBF and CLF models.")
        except FileNotFoundError:
            print("Warning: Could not find anchored models. Training from scratch.")
            
    # 4. Setup Optimizers
    policy_optim = optim.Adam(policy_net.parameters(), lr=train_config['lr_actor'])
    cbf_optim = optim.Adam(cbf_net.parameters(), lr=train_config['lr_critic'])
    clf_optim = optim.Adam(clf_net.parameters(), lr=train_config['lr_critic'])
    dynamics_optim = optim.Adam(dynamics_net.parameters(), lr=train_config['lr_dynamics'])

    print(f"--- Starting Main Training (Algorithm 2) ---")
    global_step = 0
    best_ep_reward = -np.inf
    
    # Use tqdm only if specified
    episode_iterator = range(train_config['num_episodes'])
    if use_tqdm:
        episode_iterator = tqdm(episode_iterator)
    
    for episode in episode_iterator:
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
                if global_step % train_config['model_update_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    b_s = torch.from_numpy(batch['states']).float().to(device)
                    b_a = torch.from_numpy(batch['actions']).float().to(device)
                    b_s_next = torch.from_numpy(batch['next_states']).float().to(device)
                    dynamics_optim.zero_grad()
                    loss, metrics = dynamics_net.compute_loss(b_s, b_a, b_s_next)
                    loss.backward()
                    dynamics_optim.step()
                    if use_tqdm: logger.log_metrics(metrics, global_step, prefix="dynamics/")
                
                if global_step % train_config['cbf_update_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    b_s = torch.from_numpy(batch['states']).float().to(device)
                    b_s_next = torch.from_numpy(batch['next_states']).float().to(device)
                    b_h_star_next = torch.from_numpy(batch['h_stars']).float().to(device)
                    cbf_optim.zero_grad()
                    loss, metrics = cbf_net.compute_loss_constraint(b_s, b_s_next, b_h_star_next, config)
                    loss.backward()
                    cbf_optim.step()
                    if use_tqdm: logger.log_metrics(metrics, global_step, prefix="cbf/")

                if global_step % train_config['clf_update_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    b_s = torch.from_numpy(batch['states']).float().to(device)
                    b_s_next = torch.from_numpy(batch['next_states']).float().to(device)
                    b_g = torch.from_numpy(batch['subgoals']).float().to(device)
                    b_v_star = torch.from_numpy(batch['v_stars']).float().to(device)
                    clf_optim.zero_grad()
                    loss, metrics = clf_net.compute_loss_constraint(b_s, b_s_next, b_g, b_v_star, config)
                    loss.backward()
                    clf_optim.step()
                    if use_tqdm: logger.log_metrics(metrics, global_step, prefix="clf/")

                if global_step % train_config['policy_update_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    b_s = torch.from_numpy(batch['states']).float().to(device)
                    b_g = torch.from_numpy(batch['subgoals']).float().to(device)
                    b_a = policy_net(b_s, b_g)
                    b_s_next_pred = dynamics_net(b_s, b_a)
                    policy_optim.zero_grad()
                    for param in cbf_net.parameters(): param.requires_grad = False
                    for param in clf_net.parameters(): param.requires_grad = False
                    loss, metrics = policy_net.compute_loss(b_s, b_g, b_s_next_pred, cbf_net, clf_net, config)
                    loss.backward()
                    policy_optim.step()
                    for param in cbf_net.parameters(): param.requires_grad = True
                    for param in clf_net.parameters(): param.requires_grad = True
                    if use_tqdm: logger.log_metrics(metrics, global_step, prefix="policy/")

            if done:
                break
        
        if use_tqdm:
            logger.log_scalar("episode/reward", ep_reward, episode)
            logger.log_scalar("episode/length", t+1, episode)
            logger.log_scalar("episode/success", 1.0 if fsm_state == FSM_STATE_GOAL else 0.0, episode)
        
        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            if use_tqdm: 
                print(f"\nEpisode {episode}: New Best Reward: {ep_reward:.2f}. Saving best models.")
            os.makedirs(train_config['model_save_path'], exist_ok=True)
            torch.save(policy_net.state_dict(), os.path.join(train_config['model_save_path'], "policy_best.pth"))
            torch.save(cbf_net.state_dict(), os.path.join(train_config['model_save_path'], "cbf_best.pth"))
            torch.save(clf_net.state_dict(), os.path.join(train_config['model_save_path'], "clf_best.pth"))
            torch.save(dynamics_net.state_dict(), os.path.join(train_config['model_save_path'], "dynamics_best.pth"))

    print("--- Main Training Complete ---")
    logger.close()
    return best_ep_reward # Return the best reward for Optuna

# This block now just loads the config and calls the run_training function
if __name__ == "__main__":
    set_seed(42)  # Set a fixed seed for reproducibility
    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_training(config, device, use_tqdm=True) # Run with progress bars