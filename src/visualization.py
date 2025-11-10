"""
Visualization utilities for CBF-CLF framework.
Clean, modular visualization functions for notebooks and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple, Dict
import torch


class EnvironmentVisualizer:
    """Visualize warehouse environment and robot trajectories."""

    def __init__(self, env):
        """
        Initialize visualizer with environment.

        Args:
            env: WarehouseEnvironment instance
        """
        self.env = env

    def plot_environment(self, ax=None, figsize=(14, 12), show_labels=True):
        """
        Plot the warehouse environment layout.

        Args:
            ax: Matplotlib axis (creates new if None)
            figsize: Figure size if creating new figure
            show_labels: Whether to show obstacle/goal labels

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Set limits and style
        ax.set_xlim(-0.5, self.env.bounds[0] + 0.5)
        ax.set_ylim(-0.5, self.env.bounds[1] + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Draw obstacles
        for i, obs in enumerate(self.env.obstacles):
            circle = patches.Circle(
                obs['center'], obs['radius'],
                color='red', alpha=0.6, edgecolor='darkred', linewidth=2,
                label='Obstacles' if i == 0 else ""
            )
            ax.add_patch(circle)

            if show_labels:
                ax.text(obs['center'][0], obs['center'][1], f'O{i}',
                       ha='center', va='center', fontsize=9,
                       fontweight='bold', color='white')

        # Draw goal regions
        for i, goal in enumerate(self.env.goals):
            circle = patches.Circle(
                goal['center'], goal['radius'],
                color='green', alpha=0.5, edgecolor='darkgreen', linewidth=2,
                label='Goals' if i == 0 else ""
            )
            ax.add_patch(circle)

            if show_labels:
                ax.text(goal['center'][0], goal['center'][1], f'G{i}',
                       ha='center', va='center', fontsize=11,
                       fontweight='bold', color='white')

        # Labels
        ax.set_xlabel('X Position (meters)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position (meters)', fontsize=13, fontweight='bold')

        # Grid ticks
        ax.set_xticks(np.arange(0, self.env.bounds[0]+1, 2))
        ax.set_yticks(np.arange(0, self.env.bounds[1]+1, 2))

        return fig, ax
    
    def plot_dataset(self, transitions: List):
        env = self.env
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(-0.5, env.bounds[0] + 0.5)
        ax.set_ylim(-0.5, env.bounds[1] + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # Draw environment (should now match dataset!)
        for obs in env.obstacles:
            circle = patches.Circle(obs['center'], obs['radius'], color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax.add_patch(circle)

        for goal in env.goals:
            circle = patches.Circle(goal['center'], goal['radius'], color='green', alpha=0.2, edgecolor='darkgreen', linewidth=2)
            ax.add_patch(circle)

        # Plot states with correct labels
        safe_positions = [t.state[:2] for t in transitions if t.is_safe and not t.is_goal]
        unsafe_positions = [t.state[:2] for t in transitions if not t.is_safe]
        goal_positions = [t.state[:2] for t in transitions if t.is_goal]

        if safe_positions:
            safe_array = np.array(safe_positions)
            ax.scatter(safe_array[:, 0], safe_array[:, 1], c='blue', s=10, alpha=0.5, label=f'Safe ({len(safe_positions)})')

        if unsafe_positions:
            unsafe_array = np.array(unsafe_positions)
            ax.scatter(unsafe_array[:, 0], unsafe_array[:, 1], c='red', s=30, marker='x', alpha=0.8, label=f'Unsafe ({len(unsafe_positions)})')

        if goal_positions:
            goal_array = np.array(goal_positions)
            ax.scatter(goal_array[:, 0], goal_array[:, 1], c='gold', s=50, marker='*', alpha=0.9, label=f'Goal ({len(goal_positions)})')

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Dataset State Distribution\n(Now matches environment layout!)', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self, states: List[np.ndarray], ax=None,
                       color='blue', label='Trajectory', show_arrows=False,
                       start_marker=True, end_marker=True):
        """
        Plot a trajectory on the environment.

        Args:
            states: List of states [x, y, vx, vy]
            ax: Matplotlib axis (creates new with environment if None)
            color: Trajectory color
            label: Legend label
            show_arrows: Show velocity arrows
            start_marker: Show start position marker
            end_marker: Show end position marker

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = self.plot_environment()
        else:
            fig = ax.figure

        # Extract positions
        positions = np.array([s[:2] for s in states])

        # Plot trajectory line
        ax.plot(positions[:, 0], positions[:, 1],
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.plot(positions[:, 0], positions[:, 1],
               'o', color=color, markersize=6, alpha=0.7)

        # Start marker
        if start_marker and len(positions) > 0:
            ax.plot(positions[0, 0], positions[0, 1],
                   'go', markersize=15, label='Start', zorder=5)

        # End marker
        if end_marker and len(positions) > 0:
            ax.plot(positions[-1, 0], positions[-1, 1],
                   'r*', markersize=20, label='End', zorder=5)

        # Velocity arrows
        if show_arrows:
            for i, state in enumerate(states[::3]):  # Every 3rd for clarity
                if i >= len(positions) // 3:
                    break
                idx = i * 3
                pos = positions[idx]
                vel = state[2:] * 2  # Scale for visibility
                ax.arrow(pos[0], pos[1], vel[0], vel[1],
                        head_width=0.2, head_length=0.15,
                        fc=color, ec=color, alpha=0.5)

        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

        return fig, ax

    def plot_trajectory_sequence(self, states: List[np.ndarray],
                                title: str = "Robot Trajectory"):
        """
        Plot trajectory with step numbers and details.

        Args:
            states: List of states
            title: Plot title

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        fig, ax = self.plot_environment()

        positions = np.array([s[:2] for s in states])

        # Plot trajectory with gradient color
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

        for i in range(len(positions) - 1):
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                   color=colors[i], linewidth=3, alpha=0.8)

        # Add step numbers
        for i, pos in enumerate(positions):
            ax.text(pos[0]+0.3, pos[1]+0.3, f'{i}',
                   fontsize=9,
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

        # Start and end markers
        ax.plot(positions[0, 0], positions[0, 1],
               'go', markersize=15, label='Start', zorder=5)
        ax.plot(positions[-1, 0], positions[-1, 1],
               'r*', markersize=20, label='End', zorder=5)

        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.legend()

        return fig, ax


class FunctionVisualizer:
    """Visualize CBF, CLF, and dynamics functions."""

    def __init__(self, env):
        """
        Initialize with environment for bounds.

        Args:
            env: WarehouseEnvironment instance
        """
        self.env = env

    def plot_cbf_heatmap(self, cbf, ax=None, resolution=60,
                        title="CBF Safety Function"):
        """
        Plot CBF values as heatmap over workspace.

        Args:
            cbf: CBF network
            ax: Matplotlib axis
            resolution: Grid resolution
            title: Plot title

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure

        # Create grid
        x = np.linspace(0, self.env.bounds[0], resolution)
        y = np.linspace(0, self.env.bounds[1], int(resolution * self.env.bounds[1] / self.env.bounds[0]))
        X, Y = np.meshgrid(x, y)

        # Evaluate CBF
        states = np.stack([X.flatten(), Y.flatten(),
                          np.zeros_like(X.flatten()),
                          np.zeros_like(X.flatten())], axis=1)
        states_tensor = torch.FloatTensor(states)

        with torch.no_grad():
            h_values = cbf(states_tensor).squeeze().numpy()

        h_grid = h_values.reshape(X.shape)

        # Plot heatmap
        im = ax.contourf(X, Y, h_grid, levels=20, cmap='RdYlGn', alpha=0.8)
        ax.contour(X, Y, h_grid, levels=[0], colors='black', linewidths=3)

        # Overlay obstacles
        for obs in self.env.obstacles:
            circle = patches.Circle(obs['center'], obs['radius'],
                                   color='red', alpha=0.4, edgecolor='darkred', linewidth=2)
            ax.add_patch(circle)

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'{title}\nGreen=Safe (h≥0), Red=Unsafe (h<0)',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        plt.colorbar(im, ax=ax, label='h(s)')

        return fig, ax

    def plot_clf_heatmap(self, clf, ax=None, resolution=60,
                        title="CLF Goal Function"):
        """
        Plot CLF values as heatmap over workspace.

        Args:
            clf: CLF network
            ax: Matplotlib axis
            resolution: Grid resolution
            title: Plot title

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure

        # Create grid
        x = np.linspace(0, self.env.bounds[0], resolution)
        y = np.linspace(0, self.env.bounds[1], int(resolution * self.env.bounds[1] / self.env.bounds[0]))
        X, Y = np.meshgrid(x, y)

        # Evaluate CLF
        states = np.stack([X.flatten(), Y.flatten(),
                          np.zeros_like(X.flatten()),
                          np.zeros_like(X.flatten())], axis=1)
        states_tensor = torch.FloatTensor(states)

        with torch.no_grad():
            V_values = clf(states_tensor).squeeze().numpy()

        V_grid = V_values.reshape(X.shape)

        # Plot heatmap
        im = ax.contourf(X, Y, V_grid, levels=20, cmap='viridis', alpha=0.8)

        # Overlay goals
        for goal in self.env.goals:
            circle = patches.Circle(goal['center'], goal['radius'],
                                   facecolor='yellow', edgecolor='red',
                                   linewidth=2, alpha=0.6)
            ax.add_patch(circle)
            ax.text(goal['center'][0], goal['center'][1], '★',
                   fontsize=20, ha='center', va='center', color='red')

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'{title}\nDark=Close to Goal (V≈0), Bright=Far (V>0)',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        plt.colorbar(im, ax=ax, label='V(s)')

        return fig, ax

    def plot_training_curves(self, losses: List[float], title: str = "Training Loss",
                           ax=None, color='blue', log_scale=False):
        """
        Plot training loss curves.

        Args:
            losses: List of loss values
            title: Plot title
            ax: Matplotlib axis
            color: Line color
            log_scale: Use log scale for y-axis

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ax.plot(losses, color=color, linewidth=2, label='Loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)

        # Add statistics to title
        initial_loss = losses[0] if len(losses) > 0 else 0
        final_loss = losses[-1] if len(losses) > 0 else 0
        min_loss = min(losses) if len(losses) > 0 else 0
        max_loss = max(losses) if len(losses) > 0 else 0

        title_with_stats = (f'{title}\n'
                           f'Initial: {initial_loss:.4f} → Final: {final_loss:.4f} '
                           f'(min: {min_loss:.4f}, max: {max_loss:.4f})')
        ax.set_title(title_with_stats, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')

        # Add legend if there's meaningful variation
        if max_loss - min_loss > 1e-6:
            ax.legend()

        return fig, ax


def plot_side_by_side(env, cbf=None, clf=None, figsize=(16, 7)):
    """
    Plot CBF and CLF side by side.

    Args:
        env: WarehouseEnvironment
        cbf: CBF network (optional)
        clf: CLF network (optional)
        figsize: Figure size

    Returns:
        fig, (ax1, ax2): Figure and axes
    """
    viz = FunctionVisualizer(env)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if cbf is not None:
        viz.plot_cbf_heatmap(cbf, ax=ax1)
    else:
        ax1.text(0.5, 0.5, 'CBF not trained yet',
                ha='center', va='center', transform=ax1.transAxes)

    if clf is not None:
        viz.plot_clf_heatmap(clf, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'CLF not trained yet',
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_training_progress(metrics: Dict, episode_steps: List[int], subgoals_reached: List[str],
                           episode_collisions: List[bool], figsize=(18, 12)):
    """
    Create comprehensive training progress visualization.

    Args:
        metrics: Dictionary from compute_training_metrics()
        episode_steps: List of steps per episode
        subgoals_reached: List of FSM states reached per episode
        episode_collisions: List of collision flags
        figsize: Figure size

    Returns:
        Figure object
    """
    num_episodes = metrics['total_episodes']
    window = 5

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Episode rewards
    ax1 = fig.add_subplot(gs[0, 0])
    episode_rewards = [metrics['mean_reward']] * num_episodes  # Placeholder
    ax1.plot(range(1, num_episodes+1), metrics['rolling_rewards'],
             '-', linewidth=3, color='darkblue', label=f'{window}-Episode Avg')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Success rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, num_episodes+1), metrics['rolling_success'],
             '-', linewidth=3, color='gold')
    ax2.fill_between(range(1, num_episodes+1), metrics['rolling_success'], alpha=0.3, color='gold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title(f'Success Rate ({window}-Episode Rolling)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Episode length
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(range(1, num_episodes+1), episode_steps, 'o-', linewidth=2,
             markersize=6, color='green')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Steps', fontsize=12)
    ax3.set_title('Episode Length', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative metrics
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative_success = np.cumsum([1 if s == 'goal' else 0 for s in subgoals_reached])
    cumulative_collisions = np.cumsum(episode_collisions)
    ax4.plot(range(1, num_episodes+1), cumulative_success, '-', linewidth=3,
             color='green', label='Successes')
    ax4.plot(range(1, num_episodes+1), cumulative_collisions, '-', linewidth=3,
             color='red', label='Collisions')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Cumulative Count', fontsize=12)
    ax4.set_title('Cumulative Successes vs Collisions', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: FSM progression
    ax5 = fig.add_subplot(gs[2, :])
    colors_map = {'start': 'red', 'at_g3': 'orange', 'at_g1': 'yellow', 'goal': 'green'}
    for i, state in enumerate(subgoals_reached):
        color = colors_map.get(state, 'gray')
        ax5.scatter(i+1, 1, c=color, s=100, marker='o', edgecolor='black', linewidth=1)
    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_yticks([])
    ax5.set_title('FSM State Progression\n(Red=start, Orange=G3, Yellow=G1, Green=goal)',
                  fontsize=14, fontweight='bold')
    ax5.set_xlim(0, num_episodes+1)
    ax5.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_cbf_comparison(cbf_original, cbf_fixed, env, path_waypoints, figsize=(20, 6)):
    """
    Create before/after comparison of CBF safety maps.

    Args:
        cbf_original: Original CBF network
        cbf_fixed: Fixed CBF network
        env: Environment with obstacles and goals
        path_waypoints: List of [x, y] waypoints
        figsize: Figure size

    Returns:
        Figure object
    """
    import torch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Create grid
    x_range = np.linspace(0, env.bounds[0], 60)
    y_range = np.linspace(0, env.bounds[1], 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute CBF values
    def compute_cbf_grid(cbf):
        grid_values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = torch.FloatTensor([[X[i,j], Y[i,j], 0.0, 0.0]])
                with torch.no_grad():
                    grid_values[i,j] = cbf(state).item()
        return grid_values

    cbf_grid_original = compute_cbf_grid(cbf_original)
    cbf_grid_fixed = compute_cbf_grid(cbf_fixed)

    # Plot original
    _plot_cbf_heatmap(ax1, X, Y, cbf_grid_original, env, path_waypoints,
                      title='Original CBF\n(Bottom marked unsafe)')

    # Plot fixed
    _plot_cbf_heatmap(ax2, X, Y, cbf_grid_fixed, env, path_waypoints,
                      title='Fixed CBF\n(Path-aware training)')

    plt.tight_layout()
    return fig


def _plot_cbf_heatmap(ax, X, Y, cbf_values, env, path_waypoints, title):
    """Helper to plot CBF heatmap."""
    im = ax.contourf(X, Y, cbf_values, levels=20, cmap='RdYlGn', alpha=0.8)
    ax.contour(X, Y, cbf_values, levels=[0], colors='black', linewidths=3)

    # Draw environment
    for obs in env.obstacles:
        circle = patches.Circle(obs['center'], obs['radius'], color='none',
                               edgecolor='darkred', linewidth=2)
        ax.add_patch(circle)

    for goal in env.goals:
        circle = patches.Circle(goal['center'], goal['radius'], color='none',
                               edgecolor='darkgreen', linewidth=2)
        ax.add_patch(circle)

    # Draw path
    if path_waypoints:
        path_x = [p[0] for p in path_waypoints]
        path_y = [p[1] for p in path_waypoints]
        ax.plot(path_x, path_y, 'c-', linewidth=4, marker='o', markersize=10,
               label='Intended Path')
        ax.scatter(path_x[1:], path_y[1:], c='gold', s=300, marker='*',
                  edgecolor='black', linewidth=2, label='Waypoints', zorder=11)

    ax.set_xlim(0, env.bounds[0])
    ax.set_ylim(0, env.bounds[1])
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='h(s)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def print_training_summary(metrics: Dict, subgoals_reached: List[str], trainer=None):
    """
    Print comprehensive training summary.

    Args:
        metrics: Dictionary from compute_training_metrics()
        subgoals_reached: List of FSM states reached
        trainer: Optional IntegratedTrainer with statistics
    """
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)

    print(f"\n📈 Performance Metrics:")
    print(f"  Total episodes: {metrics['total_episodes']}")
    print(f"  Success rate: {metrics['total_successes']}/{metrics['total_episodes']} "
          f"({metrics['total_successes']/metrics['total_episodes']*100:.1f}%)")
    print(f"  Mean reward: {metrics['mean_reward']:.2f}")
    print(f"  Final reward (last 10): {metrics['final_reward']:.2f}")

    print(f"\n📊 Learning Progress:")
    print(f"  Early success rate (ep 1-10): {metrics['early_success_rate']*100:.1f}%")
    print(f"  Late success rate (ep {metrics['total_episodes']-9}-{metrics['total_episodes']}): "
          f"{metrics['late_success_rate']*100:.1f}%")
    print(f"  Improvement: {metrics['improvement']:+.1f}%")

    if metrics['improvement'] > 0:
        print(f"  ✅ Policy improved with training!")
    else:
        print(f"  ⚠️  No clear improvement yet")

    # FSM progression
    subgoal_counts = {'start': 0, 'at_g3': 0, 'at_g1': 0, 'goal': 0}
    for state in subgoals_reached:
        if state in subgoal_counts:
            subgoal_counts[state] += 1

    print(f"\n🎯 FSM Progression:")
    print(f"  Stuck at start: {subgoal_counts['start']}")
    print(f"  Reached G3: {subgoal_counts['at_g3']}")
    print(f"  Reached G1: {subgoal_counts['at_g1']}")
    print(f"  Reached goal: {subgoal_counts['goal']}")

    # Trainer statistics
    if trainer:
        print(f"\n🔄 Co-Evolution Statistics:")
        print(f"  Safe states collected: {len(trainer.safe_states)}")
        print(f"  Unsafe states collected: {len(trainer.unsafe_states)}")
        print(f"  Goal states collected: {len(trainer.goal_states)}")
        print(f"  Total training steps: {trainer.step_count}")

    print("=" * 70)
