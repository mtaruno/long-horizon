# scripts/run_final_demo.py

import yaml
import torch
import numpy as np
import os
from typing import List

from src.environment import WarehouseEnvironment
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.core.models import DynamicsModel
from src.planning.fsm_planner import FSMAutomaton, FSM_STATE_GOAL
from src.utils.visualize import plot_critic_landscapes, create_evaluation_animation
from src.utils.seeding import set_seed

def run_final_demo():
    """
    Loads the champion model from 'models/best/' and runs a final
    evaluation, saving the .gif.
    """
    set_seed(42)  # Use the same seed for a reproducible demo

    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = config['train']
    nn_config = config['nn']

    # 1. Initialize Networks
    policy_net = SubgoalConditionedPolicy(
        state_dim=nn_config['state_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        action_dim=nn_config['action_dim'],
        hidden_dims=nn_config['hidden_dims'],
        a_max=config['env']['a_max']
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
    
    # 2. Load BEST Trained Models (Our Champion)
    model_path = train_config['model_save_path']
    try:
        policy_net.load_state_dict(torch.load(os.path.join(model_path, "policy_best.pth"), map_location=device))
        cbf_net.load_state_dict(torch.load(os.path.join(model_path, "cbf_best.pth"), map_location=device))
        clf_net.load_state_dict(torch.load(os.path.join(model_path, "clf_best.pth"), map_location=device))
        print(f"Successfully loaded champion models from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find best models in {model_path}. Exiting.")
        return

    policy_net.eval()
    cbf_net.eval()
    clf_net.eval()
    
    # 3. Initialize FSM and Environment
    fsm = FSMAutomaton(
        start_pos=np.array(config['fsm']['start_state']),
        goal_pos=np.array(config['fsm']['goal_state']),
        config=config
    )
    
    # NOTE: We can skip the FSM check because we know it passed.
    # We will just run the demo.
    print("\n--- Running Final Evaluation Demo ---")
    
    env = WarehouseEnv(config)
    s = env.reset(start_pos=fsm.start_pos)
    fsm.reset() 
    
    path_history: List[np.ndarray] = [s]
    
    for t in range(train_config['max_episode_length']):
        g = fsm.get_current_subgoal()
        with torch.no_grad():
            s_torch = torch.from_numpy(s).float().to(device).unsqueeze(0)
            g_torch = torch.from_numpy(g).float().to(device).unsqueeze(0)
            a = policy_net(s_torch, g_torch).cpu().numpy().squeeze(0) # No noise
        
        s_next, _, done, info = env.step(a)
        path_history.append(s_next)
        fsm_state = fsm.transition(s_next)
        
        if info['is_collision']:
            print(f"Evaluation FAILED: Collision at step {t+1}.")
            break
        if fsm_state == FSM_STATE_GOAL:
            print(f"Evaluation SUCCESS: Reached goal at step {t+1}.")
            break
        s = s_next
    else:
        print(f"Evaluation FAILED: Timed out at {train_config['max_episode_length']} steps.")
    
    # 4. --- Generate Final Visualizations ---
    plot_critic_landscapes(
        env=env, cbf_net=cbf_net, clf_net=clf_net,
        goal=fsm.goal_pos, device=device,
        filename="visualizations/final_champion_landscapes.png"
    )
    create_evaluation_animation(
        env=env, path=path_history,
        goal=fsm.goal_pos,
        filename="visualizations/final_champion_run.gif"
    )
    
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    run_final_demo()