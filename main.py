import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os 

# Custom modules
from src.environment import WarehouseEnv, ReplayBuffer
from src.dataset import create_warehouse_dataset, Transition
from src.core.models import EnsembleDynamics
from src.core.cbf import CBFNetwork
from src.core.clf import CLFNetwork
from src.core.policy import SubgoalConditionedPolicy
from src.planning.fsm_planner import FSMState, FSMTransition, FSMAutomaton, create_waypoint_fsm

import matplotlib.pyplot as plt
from src.utils.visualization import EnvironmentVisualizer, FunctionVisualizer


# --- Hyperparameters ---
STATE_DIM = 4      # [x, y, vx, vy]
ACTION_DIM = 2     # [ax, ay]
GOAL_DIM = 2       # [x_goal, y_goal]
HIDDEN_DIMS = (128, 128)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training loop params (from Algorithm 2)
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 200
REPLAY_BUFFER_SIZE = 100000 
BATCH_SIZE = 128
MODEL_UPDATE_FREQ = 10
CERTIFICATE_UPDATE_FREQ = 10
POLICY_UPDATE_FREQ = 1

# Loss penalties (from Equation 3)
LAMBDA_CBF = 0.1
LAMBDA_CLF = 0.05
CLF_EPSILON = 0.1

# Certificate loss params (from Equations 7 & 11)
CBF_ALPHA = 0.1
CLF_BETA = 0.1
CLF_DELTA = 0.0

# Loss Term Weights
CBF_W_SAFE = 0.1
CBF_W_UNSAFE = 0.1
CBF_W_CONSTRAINT = 1.0

CLF_W_GOAL = 10.0  # Increased to strongly penalize large V at goal states
CLF_W_CONSTRAINT = 0.1  # Decreased to prevent constraint from dominating
CLF_W_POSITIVE = 0.1

def update_dynamics_model(dynamics_model, optimizer_dyn, replay_buffer):
    """Update P_theta using MSE loss (Eq. 1)."""
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0

    batch = replay_buffer.sample(BATCH_SIZE)
    states = torch.FloatTensor(np.array([t.state for t in batch])).to(DEVICE)
    actions = torch.FloatTensor(np.array([t.action for t in batch])).to(DEVICE)
    next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(DEVICE)

    pred_next_states = dynamics_model(states, actions)
    loss_dyn = F.mse_loss(pred_next_states, next_states)

    optimizer_dyn.zero_grad()
    loss_dyn.backward()
    optimizer_dyn.step()
    return loss_dyn.item()


def update_policy(actor, dynamics_model, cbf, clf, optimizer_actor, replay_buffer):
    """Update pi_theta using (Eq. 3)."""
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0
        
    batch = replay_buffer.sample(BATCH_SIZE)
    states = torch.FloatTensor(np.array([t.state for t in batch])).to(DEVICE)
    goals = torch.FloatTensor(np.array([t.goal for t in batch])).to(DEVICE) 
    goals_pos = goals[:, :2] 

    actions = actor(states, goals_pos)
    pred_next_states = dynamics_model(states, actions)

    pred_pos = pred_next_states[:, :2]
    loss_subgoal = F.mse_loss(pred_pos, goals_pos)

    h_pred = cbf(pred_next_states).squeeze()
    penalty_cbf = (F.relu(-h_pred) ** 2).mean()

    v_pred = clf(pred_next_states).squeeze()
    penalty_clf = (F.relu(v_pred - CLF_EPSILON) ** 2).mean()
    
    loss_actor = loss_subgoal + LAMBDA_CBF * penalty_cbf + LAMBDA_CLF * penalty_clf

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()
    return loss_actor.item()

def update_certificates(cbf, clf, optimizer_cbf, optimizer_clf, replay_buffer, env):
    """Update h_phi (Eq. 7) and V_psi (Eq. 11)."""
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0, 0.0
        
    batch = replay_buffer.sample(BATCH_SIZE)
    
    states_list = [t.state for t in batch]
    next_states_list = [t.next_state for t in batch]
    safe_states_list = [t.state for t in batch if t.is_safe]
    unsafe_states_list = [t.state for t in batch if not t.is_safe]
    goal_states_list = [t.state for t in batch if t.is_goal]
    non_goal_states_list = [t.state for t in batch if not t.is_goal]

    states = torch.FloatTensor(np.array(states_list)).to(DEVICE)
    next_states = torch.FloatTensor(np.array(next_states_list)).to(DEVICE)
    
    # Handle empty lists with proper tensor shapes
    if safe_states_list:
        safe_states = torch.FloatTensor(np.array(safe_states_list)).to(DEVICE)
    else:
        safe_states = torch.empty((0, STATE_DIM), device=DEVICE)
    
    if unsafe_states_list:
        unsafe_states = torch.FloatTensor(np.array(unsafe_states_list)).to(DEVICE)
    else:
        unsafe_states = torch.empty((0, STATE_DIM), device=DEVICE)
    
    if goal_states_list:
        goal_states = torch.FloatTensor(np.array(goal_states_list)).to(DEVICE)
    else:
        goal_states = torch.empty((0, STATE_DIM), device=DEVICE)
    
    if non_goal_states_list:
        non_goal_states = torch.FloatTensor(np.array(non_goal_states_list)).to(DEVICE)
    else:
        non_goal_states = torch.empty((0, STATE_DIM), device=DEVICE)
    
    # --- CBF Update (Eq. 7) ---
    # Initialize as tensors to ensure proper gradient flow
    loss_safe = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    loss_unsafe = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    loss_h_constraint = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    if len(safe_states) > 0:
        h_safe = cbf(safe_states).squeeze()
        loss_safe = (F.relu(-h_safe) ** 2).mean()
    else:
        loss_safe = torch.tensor(0.0, device=DEVICE)
    
    if len(unsafe_states) > 0:
        h_unsafe = cbf(unsafe_states).squeeze()
        loss_unsafe = (F.relu(h_unsafe) ** 2).mean()
    else:
        loss_unsafe = torch.tensor(0.0, device=DEVICE)

    # Invariance constraint: h(s_k+1) - h(s_k) >= -alpha * h(s_k)
    h_s = cbf(states).squeeze()
    h_s_next = cbf(next_states).squeeze().detach() 
    h_constraint = h_s_next - h_s + CBF_ALPHA * h_s
    loss_h_constraint = (F.relu(-h_constraint) ** 2).mean()

    loss_cbf = (CBF_W_SAFE * loss_safe + 
                CBF_W_UNSAFE * loss_unsafe + 
                CBF_W_CONSTRAINT * loss_h_constraint)
    
    optimizer_cbf.zero_grad()
    loss_cbf.backward()
    optimizer_cbf.step()

    # --- CLF Update (Eq. 11) ---
    # Initialize as tensors to ensure proper gradient flow
    loss_goal = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    loss_v_positive = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    loss_v_constraint = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    if len(goal_states) > 0:
        v_goal = clf(goal_states).squeeze()
        loss_goal = (v_goal ** 2).mean()
    else:
        loss_goal = torch.tensor(0.0, device=DEVICE)

    # Decrease condition: V(s_k+1) - V(s_k) <= -beta * V(s_k) + delta
    v_s = clf(states).squeeze()
    v_s_next = clf(next_states).squeeze().detach()
    v_constraint = (v_s_next - v_s) + CLF_BETA * v_s - CLF_DELTA
    loss_v_constraint = (F.relu(v_constraint) ** 2).mean()
    
    if len(non_goal_states) > 0:
        v_non_goal = clf(non_goal_states).squeeze()
        loss_v_positive = (F.relu(0.01 - v_non_goal) ** 2).mean()
    else:
        loss_v_positive = torch.tensor(0.0, device=DEVICE)

    loss_clf = (CLF_W_GOAL * loss_goal + 
                CLF_W_CONSTRAINT * loss_v_constraint + 
                CLF_W_POSITIVE * loss_v_positive)

    optimizer_clf.zero_grad()
    loss_clf.backward()
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
    optimizer_clf.step()
    return loss_cbf.item(), loss_clf.item()


# --- Main Function ---
def main():
    print(f"Starting training on {DEVICE}...")
    env = WarehouseEnv()

    # Use waypoint-based FSM for better navigation
    fsm, fsm_goal_pos, waypoints = create_waypoint_fsm(env)
    print(f"Created waypoint-based FSM with {len(waypoints)} waypoints")
    print(f"  Start: [0.5, 10.0]")
    for i, wp in enumerate(waypoints[:-1]):
        print(f"  Waypoint {i+1}: {wp}")
    print(f"  Goal: {waypoints[-1]}")
    
    # Create networks
    dynamics_model = EnsembleDynamics(3, STATE_DIM, ACTION_DIM, HIDDEN_DIMS, DEVICE)
    cbf = CBFNetwork(STATE_DIM, HIDDEN_DIMS, DEVICE)
    clf = CLFNetwork(STATE_DIM, HIDDEN_DIMS, DEVICE)
    actor = SubgoalConditionedPolicy(STATE_DIM, GOAL_DIM, ACTION_DIM, HIDDEN_DIMS, DEVICE)

    # Create optimizers
    optimizer_dyn = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    optimizer_cbf = optim.Adam(cbf.parameters(), lr=1e-3)
    optimizer_clf = optim.Adam(clf.parameters(), lr=1e-4)  # Lower LR to prevent explosion
    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)

    # Create replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    print("Pre-populating replay buffer with transitions for all waypoints...")
    # Generate transitions for each waypoint to help learning
    for i, waypoint in enumerate(waypoints):
        print(f"  Generating transitions for waypoint {i+1}/{len(waypoints)}...")
        initial_transitions = create_warehouse_dataset(
            1250  # 5000 / 4 waypoints
        )
        for trans in initial_transitions:
            replay_buffer.add(trans)
    print(f"Buffer populated with {len(replay_buffer)} transitions.")


    # --- 2. Training Loop (Algorithm 2) ---
    episode_rewards = deque(maxlen=100)
    print_freq = 50

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        fsm.reset() 
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            goal_pos = fsm.get_active_subgoal(state)
            if goal_pos is None:
                goal_pos = fsm_goal_pos
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            goal_tensor = torch.FloatTensor(goal_pos).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                action = actor(state_tensor, goal_tensor).squeeze(0).cpu().numpy()
            action += np.random.normal(0, 0.2, size=ACTION_DIM)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            trans = Transition(state, action, next_state, reward, 
                               terminated or truncated, info['is_safe'], 
                               info['is_goal'], goal_pos)
            replay_buffer.add(trans) 
            
            state = next_state
            fsm.update_state(state) 

            # --- Periodic Updates ---
            if len(replay_buffer) > BATCH_SIZE:
                if step % MODEL_UPDATE_FREQ == 0:
                    update_dynamics_model(dynamics_model, optimizer_dyn, replay_buffer)
                
                if step % CERTIFICATE_UPDATE_FREQ == 0:
                    update_certificates(cbf, clf, optimizer_cbf, optimizer_clf, replay_buffer, env)
                
                if step % POLICY_UPDATE_FREQ == 0:
                    update_policy(actor, dynamics_model, cbf, clf, optimizer_actor, replay_buffer)

            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % print_freq == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode} | Avg. Reward (last 100): {avg_reward:.2f} | Buffer: {len(replay_buffer)} | FSM State: {fsm.current_state.name}")

    print("--- Training Complete ---")

    # --- CHANGED: Run Pruning AFTER Training ---
    print("\n--- Running Final FSM Pruning (Algorithm 1) ---")
    cbf.eval()
    clf.eval()
    dynamics_model.eval()
    actor.eval()
    
    # Add diagnostic information before pruning
    print("\n--- Certificate Diagnostics ---")
    with torch.no_grad():
        sample_batch = replay_buffer.sample(min(100, len(replay_buffer)))
        sample_states = torch.FloatTensor(np.array([t.state for t in sample_batch])).to(DEVICE)
        h_values = cbf(sample_states).squeeze().cpu().numpy()
        v_values = clf(sample_states).squeeze().cpu().numpy()
        print(f"CBF Statistics: min={h_values.min():.4f}, max={h_values.max():.4f}, mean={h_values.mean():.4f}")
        print(f"CLF Statistics: min={v_values.min():.4f}, max={v_values.max():.4f}, mean={v_values.mean():.4f}")
        print(f"Safe states (h >= 0): {np.sum(h_values >= 0.0) / len(h_values) * 100:.1f}%")
        print(f"Feasible states (V <= {CLF_EPSILON}): {np.sum(v_values <= CLF_EPSILON) / len(v_values) * 100:.1f}%")
    
    fsm.prune_fsm_with_certificates(
        cbf, clf, dynamics_model, actor, replay_buffer, 
        safe_margin=0.0, feas_margin=CLF_EPSILON,
        pruning_threshold=0.5  # Lowered threshold: require 50% of samples to be valid
    )

    save_path = "checkpoints"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_save_path = os.path.join(save_path, "final_models.pt")
    torch.save({
        'dynamics_model_state_dict': dynamics_model.state_dict(),
        'cbf_state_dict': cbf.state_dict(),
        'clf_state_dict': clf.state_dict(),
        'actor_state_dict': actor.state_dict(),
    }, model_save_path)
    
    print(f"Models saved to {model_save_path}")
    
    # Test multiple episodes to find successful trajectories
    print("\n--- Testing Trained Agent ---")
    max_test_episodes = 10
    for test_ep in range(max_test_episodes):
        print(f"\nTest Episode {test_ep + 1}/{max_test_episodes}")
        success = test_episode(env, actor, cbf, clf, fsm, fsm_goal_pos,
                               visualize=(test_ep == 0), waypoints=waypoints)  # Only visualize first episode
        if success:
            print(f"\nSuccess! Agent reached goal in test episode {test_ep + 1}")
            if test_ep > 0:  # Re-run with visualization if it wasn't the first
                print("Running successful episode again with visualization...")
                test_episode(env, actor, cbf, clf, fsm, fsm_goal_pos, visualize=True, waypoints=waypoints)
            break
    else:
        print("\nNo successful episodes found in testing. Visualizing best attempt...")


def test_episode(env, actor, cbf, clf, fsm, goal_pos, visualize=True, waypoints=None):
    """Run a test episode and optionally visualize."""
    state, _ = env.reset()
    fsm.reset()
    trajectory = [state.copy()]

    for step in range(100):
        goal = fsm.get_active_subgoal(state)
        if goal is None:
            goal = goal_pos
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action = actor(state_tensor, goal_tensor).squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, info = env.step(action)
        trajectory.append(next_state.copy())
        state = next_state
        fsm.update_state(state)

        if terminated or truncated:
            break

    if visualize:
 

        # Create comprehensive visualization with CBF, CLF, and trajectory
        fig = plt.figure(figsize=(20, 6))

        # Plot 1: CBF (Safety) Heatmap with trajectory
        ax1 = plt.subplot(1, 3, 1)
        func_viz = FunctionVisualizer(env, resolution=100)
        func_viz.plot_cbf_heatmap(cbf, ax=ax1)

        # Overlay trajectory on CBF
        traj = np.array(trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.8,
                label='Trajectory', zorder=4)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color='cyan', markersize=12,
                label='Start', zorder=5, markeredgecolor='black', markeredgewidth=2)
        ax1.plot(traj[-1, 0], traj[-1, 1], 's', color='magenta', markersize=12,
                label='End', zorder=5, markeredgecolor='black', markeredgewidth=2)

        # Add waypoints
        if waypoints is not None:
            for i, wp in enumerate(waypoints[:-1]):  # Skip last one (goal)
                ax1.plot(wp[0], wp[1], '*', color='yellow', markersize=15,
                        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                ax1.text(wp[0], wp[1] + 0.3, f'WP{i+1}', ha='center',
                        fontsize=9, fontweight='bold', color='yellow',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_title('CBF (Safety) Heatmap + Trajectory', fontsize=14, fontweight='bold')

        # Plot 2: CLF (Goal Distance) Heatmap with trajectory
        ax2 = plt.subplot(1, 3, 2)
        func_viz.plot_clf_heatmap(clf, ax=ax2)

        # Overlay trajectory on CLF
        ax2.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.8,
                label='Trajectory', zorder=4)
        ax2.plot(traj[0, 0], traj[0, 1], 'o', color='cyan', markersize=12,
                label='Start', zorder=5, markeredgecolor='black', markeredgewidth=2)
        ax2.plot(traj[-1, 0], traj[-1, 1], 's', color='magenta', markersize=12,
                label='End', zorder=5, markeredgecolor='black', markeredgewidth=2)

        # Add waypoints
        if waypoints is not None:
            for i, wp in enumerate(waypoints[:-1]):  # Skip last one (goal)
                ax2.plot(wp[0], wp[1], '*', color='yellow', markersize=15,
                        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                ax2.text(wp[0], wp[1] + 0.3, f'WP{i+1}', ha='center',
                        fontsize=9, fontweight='bold', color='yellow',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_title('CLF (Goal Distance) Heatmap + Trajectory', fontsize=14, fontweight='bold')

        # Plot 3: Clean trajectory view
        plt.subplot(1, 3, 3)
        env_viz = EnvironmentVisualizer(env)
        env_viz.plot_trajectory_sequence(trajectory, "Final Trajectory")

        plt.tight_layout()
        plt.savefig('comprehensive_visualization.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to 'comprehensive_visualization.png'")
        plt.show()

    goal_dist = np.linalg.norm(state[:2] - env.target_goal['center'])
    success = goal_dist <= env.target_goal['radius']
    print(f"Test episode: {len(trajectory)} steps, goal distance: {goal_dist:.3f}, success: {success}")
    return success


if __name__ == "__main__":
    main()