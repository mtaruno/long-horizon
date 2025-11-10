"""
Training utilities for CBF-CLF framework.
Provides helper functions for path sampling, online updates, and reward shaping.
"""

import torch
import numpy as np
from typing import List, Tuple


def generate_safe_path_samples(env, start_pos, waypoints, samples_per_segment=30, offset_samples=5) -> torch.FloatTensor:
    """
    Generate safe state samples along a navigation path.

    Args:
        env: Environment with _check_collision method
        start_pos: Starting position [x, y]
        waypoints: List of waypoint positions [(x1, y1), (x2, y2), ...]
        samples_per_segment: Number of samples along each segment
        offset_samples: Number of offset samples perpendicular to path

    Returns:
        Tensor of safe states [N, 4] where N is number of safe samples
    """
    safe_samples = []

    # Add start position to waypoints
    all_points = [start_pos] + waypoints

    # Generate samples along each segment
    for i in range(len(all_points) - 1):
        p1, p2 = all_points[i], all_points[i + 1]

        # Sample along the segment
        for x in np.linspace(p1[0], p2[0], samples_per_segment):
            for y in np.linspace(p1[1], p2[1], samples_per_segment):
                # Add perpendicular offsets
                for offset in np.linspace(-0.3, 0.3, offset_samples):
                    # Perpendicular direction
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    length = np.sqrt(dx**2 + dy**2) + 1e-6
                    perp_x, perp_y = -dy / length, dx / length

                    x_sample = x + perp_x * offset
                    y_sample = y + perp_y * offset

                    # Check bounds and collision
                    if 0 <= x_sample <= env.bounds[0] and 0 <= y_sample <= env.bounds[1]:
                        state = np.array([x_sample, y_sample, 0.0, 0.0])
                        if not env._check_collision(state[:2]):
                            safe_samples.append(state)

    return torch.FloatTensor(safe_samples)


def train_cbf_with_path_augmentation(cbf, cbf_optimizer, safe_states, unsafe_states,
                                      path_samples=None, num_epochs=150, alpha=0.1,
                                      margin_weight=0.1, verbose=True):
    """
    Train CBF with optional path augmentation.

    Args:
        cbf: CBF network
        cbf_optimizer: Optimizer for CBF
        safe_states: Tensor of safe states
        unsafe_states: Tensor of unsafe states
        path_samples: Optional additional safe path samples
        num_epochs: Number of training epochs
        alpha: Decay rate for invariance constraint
        margin_weight: Weight for margin loss
        verbose: Print training progress

    Returns:
        List of training losses
    """
    # Combine safe states with path samples if provided
    if path_samples is not None:
        all_safe = torch.cat([safe_states, path_samples], dim=0)
    else:
        all_safe = safe_states

    losses = []

    for epoch in range(num_epochs):
        cbf_optimizer.zero_grad()

        # Safe states: h(s) >= 0
        h_safe = cbf(all_safe).squeeze()
        loss_safe = torch.mean(torch.clamp(-h_safe, min=0.0) ** 2)

        # Unsafe states: h(s) < 0
        h_unsafe = cbf(unsafe_states).squeeze()
        loss_unsafe = torch.mean(torch.clamp(h_unsafe, min=0.0) ** 2)

        # Margin loss: push safe/unsafe further apart
        margin_safe = torch.mean(torch.clamp(0.1 - h_safe, min=0.0) ** 2)
        margin_unsafe = torch.mean(torch.clamp(h_unsafe + 0.1, min=0.0) ** 2)

        loss = loss_safe + loss_unsafe + margin_weight * (margin_safe + margin_unsafe)
        loss.backward()
        cbf_optimizer.step()

        losses.append(loss.item())

        if verbose and epoch % 30 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

    return losses


def verify_cbf_on_path(cbf, waypoints, waypoint_names=None) -> Tuple[bool, List[float]]:
    """
    Verify that CBF marks all waypoints as safe.

    Args:
        cbf: Trained CBF network
        waypoints: List of waypoint states [N, 4]
        waypoint_names: Optional names for each waypoint

    Returns:
        (all_safe, h_values) - whether all safe and list of CBF values
    """
    waypoints_tensor = torch.FloatTensor(waypoints)

    with torch.no_grad():
        h_values = cbf(waypoints_tensor).squeeze().numpy()

    all_safe = True
    for i, (point, h) in enumerate(zip(waypoints, h_values)):
        is_safe = h >= 0
        all_safe = all_safe and is_safe

        status = "✅" if is_safe else "❌"
        name = f" ({waypoint_names[i]})" if waypoint_names and i < len(waypoint_names) else ""
        print(f"  {status} Point {i+1}{name} {point[:2]}: h(s) = {h:+.3f}")

    return all_safe, h_values.tolist()


class SubgoalRewardWrapper:
    """
    Environment wrapper that adds reward shaping for subgoal progress.
    Gives intermediate rewards for approaching and reaching subgoals.
    """

    def __init__(self, env, fsm, progress_reward_scale=5.0, subgoal_bonus=30.0):
        """
        Args:
            env: Base environment
            fsm: Finite state machine with subgoals
            progress_reward_scale: Reward per meter of progress
            subgoal_bonus: Bonus for reaching a subgoal
        """
        self.env = env
        self.fsm = fsm
        self.progress_reward_scale = progress_reward_scale
        self.subgoal_bonus = subgoal_bonus
        self.prev_distance = None

    def reset(self):
        """Reset environment and FSM."""
        state = self.env.reset()
        self.fsm.current_state_id = list(self.fsm.states.keys())[0]

        # Calculate initial distance to first subgoal
        subgoal = self.fsm.get_current_subgoal()
        self.prev_distance = np.linalg.norm(state[:2] - subgoal[:2])

        return state

    def step(self, action):
        """Execute action with reward shaping."""
        next_state, base_reward, done, info = self.env.step(action)

        # Get current subgoal
        current_subgoal = self.fsm.get_current_subgoal()
        current_distance = np.linalg.norm(next_state[:2] - current_subgoal[:2])

        # Progress reward
        progress_reward = 0.0
        if self.prev_distance is not None:
            distance_change = self.prev_distance - current_distance
            progress_reward = distance_change * self.progress_reward_scale

        # Subgoal reached bonus
        subgoal_reached = current_distance < 0.5  # Within radius
        if subgoal_reached:
            progress_reward += self.subgoal_bonus

        # Update FSM
        old_state = self.fsm.current_state_id
        self.fsm.step(next_state)
        new_state = self.fsm.current_state_id

        # Reset distance tracking if FSM transitioned
        if new_state != old_state:
            new_subgoal = self.fsm.get_current_subgoal()
            self.prev_distance = np.linalg.norm(next_state[:2] - new_subgoal[:2])
        else:
            self.prev_distance = current_distance

        # Combined reward
        shaped_reward = base_reward + progress_reward

        return next_state, shaped_reward, done, info

    @property
    def bounds(self):
        return self.env.bounds

    @property
    def obstacles(self):
        return self.env.obstacles

    @property
    def goals(self):
        return self.env.goals


def compute_training_metrics(episode_rewards, episode_success, window=5):
    """
    Compute training metrics and statistics.

    Args:
        episode_rewards: List of episode rewards
        episode_success: List of success flags (bool)
        window: Window size for rolling averages

    Returns:
        Dictionary with metrics
    """
    rolling_rewards = [
        np.mean(episode_rewards[max(0, i-window+1):i+1])
        for i in range(len(episode_rewards))
    ]

    rolling_success = [
        np.mean(episode_success[max(0, i-window+1):i+1])
        for i in range(len(episode_success))
    ]

    early_success = np.mean(episode_success[:10]) if len(episode_success) >= 10 else 0
    late_success = np.mean(episode_success[-10:]) if len(episode_success) >= 10 else 0

    return {
        'rolling_rewards': rolling_rewards,
        'rolling_success': rolling_success,
        'mean_reward': np.mean(episode_rewards),
        'final_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0,
        'early_success_rate': early_success,
        'late_success_rate': late_success,
        'improvement': (late_success - early_success) * 100,
        'total_episodes': len(episode_rewards),
        'total_successes': sum(episode_success)
    }
