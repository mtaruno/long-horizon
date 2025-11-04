"""
Visualization utilities for CBF-CLF framework.
Clean, modular visualization functions for notebooks and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple
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
