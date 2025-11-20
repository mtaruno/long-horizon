"""
Enhanced example with live visualization of training process.
Shows trajectories, CBF/CLF values, and real-time metrics.
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from src.core.cbf import EnsembleCBF
from src.core.clf import EnsembleCLF
from src.core.models import EnsembleDynamics, ReplayBuffer
from src.core.policy import SubgoalConditionedPolicy
from src.planning.fsm_planner import FSMState, FSMTransition, FSMAutomaton
from src.training.integrated_trainer import FSMCBFCLFTrainer
from src.dataset import create_warehouse_dataset
from src.environment import WarehouseEnv


class TrainingVisualizer:
    """Real-time visualization of training progress."""

    def __init__(self, env: WarehouseEnv, output_dir: str = "visualizations"):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.episodes = []
        self.rewards = []
        self.success_rates = []
        self.collision_rates = []
        self.cbf_losses = []
        self.clf_losses = []
        self.dynamics_losses = []

    def plot_environment_with_trajectory(self, trajectory, episode_num, title_suffix=""):
        """Plot the warehouse environment with a trajectory overlay."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set limits
        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw obstacles
        for i, obs in enumerate(self.env.obstacles):
            circle = patches.Circle(
                obs['center'], obs['radius'],
                color='red', alpha=0.6,
                label='Obstacles' if i == 0 else ""
            )
            ax.add_patch(circle)

        # Draw goals
        for i, goal in enumerate(self.env.goals):
            circle = patches.Circle(
                goal['center'], goal['radius'],
                color='green', alpha=0.4,
                label='Goals' if i == 0 else ""
            )
            ax.add_patch(circle)

        # Plot trajectory
        if len(trajectory) > 0:
            traj_array = np.array(trajectory)
            positions = traj_array[:, :2]

            # Color trajectory by time (blue -> yellow)
            colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

            for i in range(len(positions) - 1):
                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                       color=colors[i], linewidth=2, alpha=0.7)

            # Mark start and end
            ax.plot(positions[0, 0], positions[0, 1], 'bo',
                   markersize=12, label='Start')
            ax.plot(positions[-1, 0], positions[-1, 1], 'r*',
                   markersize=15, label='End')

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Episode {episode_num}: Trajectory {title_suffix}', fontsize=14)
        ax.legend(loc='upper right')

        plt.tight_layout()
        output_path = self.output_dir / f"trajectory_ep{episode_num}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_cbf_clf_heatmap(self, cbf, clf, episode_num):
        """Plot CBF and CLF value heatmaps over the workspace."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Create grid
        x = np.linspace(0, 12, 60)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)

        # Evaluate CBF and CLF
        states = np.stack([X.flatten(), Y.flatten(),
                          np.zeros_like(X.flatten()),
                          np.zeros_like(X.flatten())], axis=1)
        states_tensor = torch.FloatTensor(states)

        with torch.no_grad():
            cbf_values = cbf(states_tensor).squeeze().numpy()
            clf_values = clf(states_tensor).squeeze().numpy()

        cbf_grid = cbf_values.reshape(X.shape)
        clf_grid = clf_values.reshape(X.shape)

        # Plot CBF (Safety)
        im1 = ax1.contourf(X, Y, cbf_grid, levels=20, cmap='RdYlGn', alpha=0.8)
        ax1.contour(X, Y, cbf_grid, levels=[0], colors='black', linewidths=2)

        # Draw obstacles on CBF plot
        for obs in self.env.obstacles:
            circle = patches.Circle(obs['center'], obs['radius'],
                                   color='red', alpha=0.3)
            ax1.add_patch(circle)

        ax1.set_title(f'CBF Values (Safety) - Episode {episode_num}\nPositive=Safe, Negative=Unsafe', fontsize=12)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        plt.colorbar(im1, ax=ax1, label='h(s)')

        # Plot CLF (Goal distance)
        im2 = ax2.contourf(X, Y, clf_grid, levels=20, cmap='viridis', alpha=0.8)

        # Draw goals on CLF plot
        for goal in self.env.goals:
            circle = patches.Circle(goal['center'], goal['radius'],
                                   color='yellow', alpha=0.5)
            ax2.add_patch(circle)

        ax2.set_title(f'CLF Values (Goal Distance) - Episode {episode_num}\nLower=Closer to Goal', fontsize=12)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        plt.colorbar(im2, ax=ax2, label='V(s)')

        plt.tight_layout()
        output_path = self.output_dir / f"heatmap_ep{episode_num}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_training_metrics(self):
        """Plot training metrics over episodes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Rewards
        if len(self.rewards) > 0:
            ax1.plot(self.episodes, self.rewards, 'b-', linewidth=2)
            ax1.set_title('Episode Rewards', fontsize=12)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.grid(True, alpha=0.3)

        # 2. Success/Collision Rates
        if len(self.success_rates) > 0:
            ax2.plot(self.episodes, np.array(self.success_rates) * 100,
                    'g-', linewidth=2, label='Success Rate')
            ax2.plot(self.episodes, np.array(self.collision_rates) * 100,
                    'r-', linewidth=2, label='Collision Rate')
            ax2.set_title('Performance Metrics', fontsize=12)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Rate (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. CBF Loss
        if len(self.cbf_losses) > 0:
            ax3.plot(self.episodes, self.cbf_losses, 'r-', linewidth=2)
            ax3.set_title('CBF Loss (Safety)', fontsize=12)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)

        # 4. CLF Loss
        if len(self.clf_losses) > 0:
            ax4.plot(self.episodes, self.clf_losses, 'b-', linewidth=2)
            ax4.set_title('CLF Loss (Feasibility)', fontsize=12)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "training_metrics.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def update_metrics(self, episode, reward, success_rate, collision_rate,
                      cbf_loss=None, clf_loss=None, dyn_loss=None):
        """Update metrics storage."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.success_rates.append(success_rate)
        self.collision_rates.append(collision_rate)
        if cbf_loss is not None:
            self.cbf_losses.append(cbf_loss)
        if clf_loss is not None:
            self.clf_losses.append(clf_loss)
        if dyn_loss is not None:
            self.dynamics_losses.append(dyn_loss)


def evaluate_policy_with_viz(policy, fsm, env, visualizer, episode_num,
                             num_episodes=5, max_steps=100, device="cpu"):
    """Evaluate policy and return trajectory for visualization."""
    successes = 0
    collisions = 0
    trajectories = []
    waypoints_reached = []

    for ep in range(num_episodes):
        state = env.reset()
        fsm.current_state_id = list(fsm.states.keys())[0]
        trajectory = [state.copy()]
        prev_fsm_state = fsm.current_state_id

        for step in range(max_steps):
            subgoal = fsm.get_current_subgoal()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            subgoal_tensor = torch.FloatTensor(subgoal).unsqueeze(0).to(device)

            with torch.no_grad():
                action = policy(state_tensor, subgoal_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action * 2.0, -2.0, 2.0)

            next_state, reward, done, info = env.step(action, state)
            trajectory.append(next_state.copy())

            if info['collision']:
                collisions += 1
                break

            # Check goal
            if info['success']:
                successes += 1
                break

            state = next_state
            transitioned = fsm.step(state)

            # Track waypoint progress
            if transitioned and fsm.current_state_id != prev_fsm_state:
                waypoints_reached.append(fsm.current_state_id)
                prev_fsm_state = fsm.current_state_id

        if ep == 0:  # Save first trajectory
            trajectories.append(trajectory)

    # Visualize first trajectory
    if len(trajectories) > 0:
        success_label = " (Success)" if successes > 0 else " (Failed)"
        visualizer.plot_environment_with_trajectory(
            trajectories[0], episode_num, success_label
        )

    return {
        'success_rate': successes / num_episodes,
        'collision_rate': collisions / num_episodes,
        'trajectories': trajectories,
        'waypoints_reached': waypoints_reached
    }


def main():
    print("="*70)
    print("CBF-CLF Training with Live Visualization")
    print("="*70)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    state_dim = 4
    action_dim = 2
    subgoal_dim = 2
    device = "cpu"

    env = WarehouseEnv()
    visualizer = TrainingVisualizer(env)

    # Create buffer and models
    buffer = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=action_dim)

    policy = SubgoalConditionedPolicy(state_dim, action_dim, subgoal_dim, device=device)
    cbf = EnsembleCBF(num_models=3, state_dim=state_dim, device=device)
    clf = EnsembleCLF(num_models=3, state_dim=state_dim, device=device)
    dynamics = EnsembleDynamics(num_models=3, state_dim=state_dim,
                                action_dim=action_dim, device=device)

    policy_opt = optim.Adam(policy.parameters(), lr=3e-3)
    cbf_opt = optim.Adam(cbf.parameters(), lr=1e-3)
    clf_opt = optim.Adam(clf.parameters(), lr=1e-3)
    dyn_opt = optim.Adam(dynamics.parameters(), lr=1e-3)

    # PHASE 1: Load dataset
    print("\n" + "="*70)
    print("PHASE 1: Loading Dataset")
    print("="*70)

    transitions = create_warehouse_dataset(num_transitions=1000)

    safe_states = []
    unsafe_states = []
    goal_states = []

    for t in transitions:
        buffer.push(t.state, t.action, t.next_state, t.reward, t.done)

        if t.is_safe:
            safe_states.append(t.state)
        else:
            unsafe_states.append(t.state)

        if t.is_goal:
            goal_states.append(t.next_state)

    print(f"✓ Loaded {len(buffer)} transitions")
    print(f"  - Safe states: {len(safe_states)}")
    print(f"  - Unsafe states: {len(unsafe_states)}")
    print(f"  - Goal states: {len(goal_states)}")

    # PHASE 2: Pretrain models
    print("\n" + "="*70)
    print("PHASE 2: Pretraining Models")
    print("="*70)

    print("\n[Dynamics Model]")
    best_dyn_loss = float('inf')
    for epoch in range(100):  # Increased from 50
        if len(buffer) < 64:
            break
        batch = buffer.sample(min(len(buffer), 128))  # Larger batch
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        next_states = batch['next_states'].to(device)

        dyn_opt.zero_grad()
        pred_next = dynamics(states, actions)
        loss = torch.mean((pred_next - next_states) ** 2)
        loss.backward()
        dyn_opt.step()

        best_dyn_loss = min(best_dyn_loss, loss.item())

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

    print(f"  Best Dynamics Loss: {best_dyn_loss:.4f}")

    print("\n[CBF - Safety Function]")
    if len(safe_states) > 10 and len(unsafe_states) > 10:
        safe = torch.FloatTensor(safe_states).to(device)
        unsafe = torch.FloatTensor(unsafe_states).to(device)

        for epoch in range(50):
            cbf_opt.zero_grad()

            h_safe = cbf(safe).squeeze(-1)
            h_unsafe = cbf(unsafe).squeeze(-1)

            loss_safe = torch.mean(torch.clamp(-h_safe, min=0.0) ** 2)
            loss_unsafe = torch.mean(torch.clamp(h_unsafe, min=0.0) ** 2)
            loss = loss_safe + loss_unsafe

            loss.backward()
            cbf_opt.step()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

    print("\n[CLF - Goal Function]")
    if len(goal_states) > 5:
        goals = torch.FloatTensor(goal_states).to(device)

        for epoch in range(50):
            clf_opt.zero_grad()

            V_goal = clf(goals).squeeze(-1)
            loss = torch.mean(V_goal ** 2)

            loss.backward()
            clf_opt.step()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

    print("\n✓ Pretraining Complete")

    # Visualize initial CBF/CLF
    print("\n📊 Creating initial CBF/CLF heatmaps...")
    visualizer.plot_cbf_clf_heatmap(cbf, clf, episode_num=0)

    # PHASE 3: Online training with visualization
    print("\n" + "="*70)
    print("PHASE 3: Online Training with Policy")
    print("="*70)

    # Create FSM for warehouse navigation with intermediate waypoints
    # Safe path: (0,0) -> bottom edge -> right corridor -> goal
    waypoint1 = np.array([3.5, 0.5])   # Along bottom, avoid (1,1) obstacle
    waypoint2 = np.array([6.5, 0.5])   # Continue bottom edge
    waypoint3 = np.array([9.5, 2.5])   # Enter right corridor, avoid (9,1) obstacle
    waypoint4 = np.array([9.5, 5.0])   # Move up right corridor
    final_goal = np.array([10.5, 8.5]) # Loading dock 1

    states = [
        FSMState(id="waypoint1", subgoal=waypoint1, is_goal=False),
        FSMState(id="waypoint2", subgoal=waypoint2, is_goal=False),
        FSMState(id="waypoint3", subgoal=waypoint3, is_goal=False),
        FSMState(id="waypoint4", subgoal=waypoint4, is_goal=False),
        FSMState(id="goal", subgoal=final_goal, is_goal=True)
    ]
    transitions_fsm = [
        FSMTransition("waypoint1", "waypoint2", "at_waypoint1"),
        FSMTransition("waypoint2", "waypoint3", "at_waypoint2"),
        FSMTransition("waypoint3", "waypoint4", "at_waypoint3"),
        FSMTransition("waypoint4", "goal", "at_waypoint4"),
    ]
    predicates = {
        "at_waypoint1": lambda s: np.linalg.norm(s[:2] - waypoint1) < 0.8,
        "at_waypoint2": lambda s: np.linalg.norm(s[:2] - waypoint2) < 0.8,
        "at_waypoint3": lambda s: np.linalg.norm(s[:2] - waypoint3) < 0.8,
        "at_waypoint4": lambda s: np.linalg.norm(s[:2] - waypoint4) < 0.8,
    }
    fsm = FSMAutomaton(states, transitions_fsm, "waypoint1", predicates)

    config = {
        "lambda_cbf": 10.0,  # Strong safety constraint (strict, no margin)
        "lambda_clf": 1.0,
        "epsilon": 0.1,
        "batch_size": 64,
        "model_update_freq": 5,
        "cbf_update_freq": 10,
        "clf_update_freq": 10
    }

    trainer = FSMCBFCLFTrainer(
        fsm=fsm, policy=policy, cbf=cbf, clf=clf, dynamics=dynamics,
        replay_buffer=buffer, policy_optimizer=policy_opt,
        cbf_optimizer=cbf_opt, clf_optimizer=clf_opt,
        dynamics_optimizer=dyn_opt, config=config, device=device
    )

    # Initial evaluation
    print("\n[Initial Policy Evaluation]")
    metrics = evaluate_policy_with_viz(policy, fsm, env, visualizer, 0,
                                       num_episodes=5, device=device)
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Collision Rate: {metrics['collision_rate']*100:.1f}%")

    visualizer.update_metrics(0, 0, metrics['success_rate'],
                            metrics['collision_rate'])

    # Training loop
    num_episodes = 200  # Increased from 50
    eval_interval = 20

    print(f"\n[Training for {num_episodes} episodes]")
    for episode in range(1, num_episodes + 1):
        # Custom training episode to track actual environment
        state = env.reset()
        fsm.current_state_id = list(fsm.states.keys())[0]
        prev_fsm_state = fsm.current_state_id

        episode_reward = 0
        waypoints_reached_this_ep = 0

        for step in range(100):
            subgoal = fsm.get_current_subgoal()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            subgoal_tensor = torch.FloatTensor(subgoal).unsqueeze(0).to(device)

            with torch.no_grad():
                action = policy(state_tensor, subgoal_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action * 2.0, -2.0, 2.0)  # Scale actions
                if np.random.rand() < 0.1:  # Exploration
                    action = np.random.uniform(-1.0, 1.0, len(action))

            next_state, env_reward, done, info = env.step(action, state)

            # REWARD SHAPING: Add bonus for progress toward current subgoal
            dist_to_subgoal = np.linalg.norm(next_state[:2] - subgoal[:2])
            prev_dist = np.linalg.norm(state[:2] - subgoal[:2])
            progress_reward = (prev_dist - dist_to_subgoal) * 10.0  # Scale up progress

            # Combine rewards
            reward = env_reward + progress_reward

            is_safe = not info['collision']
            is_goal = info['success']

            buffer.push(state, action, next_state, reward, done)

            if is_safe:
                trainer.safe_states.append(state)
            else:
                trainer.unsafe_states.append(state)

            if is_goal:
                trainer.goal_states.append(next_state)

            # Periodic updates
            if trainer.step_count % config["model_update_freq"] == 0:
                trainer._update_dynamics()

            if trainer.step_count % config["cbf_update_freq"] == 0:
                trainer._update_cbf()

            if trainer.step_count % config["clf_update_freq"] == 0:
                trainer._update_clf()

            trainer._update_policy()

            # Track FSM transitions and reward waypoint progress
            transitioned = fsm.step(next_state)
            if transitioned and fsm.current_state_id != prev_fsm_state:
                waypoints_reached_this_ep += 1
                # Bonus reward for reaching waypoint
                reward += 100.0
                prev_fsm_state = fsm.current_state_id

            if fsm.is_goal_reached():
                reward += 500.0  # Big bonus for reaching final goal
                break

            state = next_state
            episode_reward += reward
            trainer.step_count += 1

            if done:
                break

        # Periodic evaluation and visualization
        if episode % eval_interval == 0:
            print(f"\n--- Episode {episode} ---")
            metrics = evaluate_policy_with_viz(policy, fsm, env, visualizer,
                                              episode, num_episodes=5, device=device)
            print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
            print(f"  Collision Rate: {metrics['collision_rate']*100:.1f}%")
            print(f"  Avg Reward: {episode_reward:.1f}")
            print(f"  Waypoints Reached (train): {waypoints_reached_this_ep}")
            if metrics['waypoints_reached']:
                print(f"  Waypoints Reached (eval): {metrics['waypoints_reached']}")

            visualizer.update_metrics(episode, episode_reward,
                                     metrics['success_rate'],
                                     metrics['collision_rate'])

            # Update heatmaps
            visualizer.plot_cbf_clf_heatmap(cbf, clf, episode_num=episode)
            visualizer.plot_training_metrics()

            print(f"  📊 Visualizations updated in {visualizer.output_dir}/")

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    metrics = evaluate_policy_with_viz(policy, fsm, env, visualizer,
                                      num_episodes, num_episodes=20, device=device)
    print(f"\n✓ Final Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"✓ Final Collision Rate: {metrics['collision_rate']*100:.1f}%")

    # Create final visualizations
    visualizer.plot_training_metrics()
    visualizer.plot_cbf_clf_heatmap(cbf, clf, episode_num=num_episodes)

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\n📁 All visualizations saved to: {visualizer.output_dir}/")
    print("\nGenerated files:")
    print("  - trajectory_ep*.png : Agent trajectories at different episodes")
    print("  - heatmap_ep*.png : CBF (safety) and CLF (goal) value heatmaps")
    print("  - training_metrics.png : Reward, success rate, and loss curves")
    print("\n💡 Check these files to understand how the agent learns!")


if __name__ == "__main__":
    main()