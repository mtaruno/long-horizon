import yaml
import numpy as np
import os
import torch
from typing import List, Tuple

from src.environment.warehouse import WarehouseEnv
from src.utils.buffer import ReplayBuffer
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.core.models import DynamicsModel
from src.planning.fsm_planner import FSMAutomaton, FSM_STATE_GOAL, FSM_STATE_FAILED
from src.utils.visualization import plot_critic_landscapes, create_evaluation_animation
from src.utils.seeding import set_seed

def run_evaluation(config: dict, device: torch.device, visualize: bool = False) -> Tuple[float, float]:
    """
    Runs the FSM pruning and evaluation.
    Returns (safety_rate, feasibility_rate) for Optuna.
    """
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
    
    dynamics_net = DynamicsModel(
        state_dim=nn_config['state_dim'],
        action_dim=nn_config['action_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    # 2. Load BEST Trained Models
    model_path = train_config['model_save_path']
    try:
        policy_net.load_state_dict(torch.load(os.path.join(model_path, "policy_best.pth"), map_location=device))
        cbf_net.load_state_dict(torch.load(os.path.join(model_path, "cbf_best.pth"), map_location=device))
        clf_net.load_state_dict(torch.load(os.path.join(model_path, "clf_best.pth"), map_location=device))
        dynamics_net.load_state_dict(torch.load(os.path.join(model_path, "dynamics_best.pth"), map_location=device))
        print(f"Successfully loaded best models from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find best models in {model_path}. Evaluation failed.")
        return 0.0, 0.0 # Return 0 score

    policy_net.eval()
    cbf_net.eval()
    clf_net.eval()
    dynamics_net.eval()

    # 3. Load Replay Buffer (for sampling)
    print(f"Loading data from {config['data_gen']['data_path']} for FSM pruning...")
    buffer = ReplayBuffer(
        state_dim=nn_config['state_dim'],
        action_dim=nn_config['action_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        max_size=config['data_gen']['num_transitions']
    )
    buffer.load(config['data_gen']['data_path'])
    
    # 4. Initialize FSM
    fsm = FSMAutomaton(
        start_pos=np.array(config['fsm']['start_state']),
        goal_pos=np.array(config['fsm']['goal_state']),
        config=config
    )
    
    # 5. --- Run FSM Pruning (Algorithm 1) ---
    is_valid, safety_rate, feasibility_rate = fsm.prune_fsm_with_certificates(
        replay_buffer=buffer,
        policy_net=policy_net,
        dynamics_net=dynamics_net,
        cbf_net=cbf_net,
        clf_net=clf_net,
        device=device
    )
    
    if not is_valid:
        print("\nEvaluation Failed: FSM pruning removed all valid paths to goal.")
        # We still return the scores so Optuna knows what happened
        return safety_rate, feasibility_rate
        
    print("\n--- Running Final Evaluation Demo ---")
    
    # 6. Initialize Environment
    env = WarehouseEnv(config)
    s = env.reset(start_pos=fsm.start_pos)
    fsm.reset() # Reset FSM for execution
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

    
    # 7. --- Generate Final Visualizations (if requested) ---
    if visualize:
        plot_critic_landscapes(
            env=env, cbf_net=cbf_net, clf_net=clf_net,
            goal=fsm.goal_pos, device=device,
            filename="visualizations/03_final_landscapes.png"
        )
        create_evaluation_animation(
            env=env, path=path_history, goal=fsm.goal_pos,
            filename="visualizations/03_final_run.gif"
        )
    
    print("\n--- Evaluation Complete ---")
    return safety_rate, feasibility_rate


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run evaluation and create visualizations
    run_evaluation(config, device, visualize=True)