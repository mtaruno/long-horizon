

import numpy as np
from typing import List, Dict
from .types import Transition

class WarehouseEnvironment:
    """Warehouse environment matching the dataset."""
    
    def __init__(self):
        self.bounds = (12.0, 10.0)  # 12m x 10m warehouse
        
        # Obstacles matching the dataset
        self.obstacles = [
            {"center": np.array([2.0, 3.0]), "radius": 0.8},
            {"center": np.array([2.0, 7.0]), "radius": 0.8},
            {"center": np.array([5.0, 2.0]), "radius": 0.6},
            {"center": np.array([5.0, 5.0]), "radius": 0.6},
            {"center": np.array([5.0, 8.0]), "radius": 0.6},
            {"center": np.array([8.0, 3.5]), "radius": 0.7},
            {"center": np.array([8.0, 6.5]), "radius": 0.7},
            {"center": np.array([4.0, 9.0]), "radius": 0.3},
            {"center": np.array([9.0, 1.0]), "radius": 0.3},
            {"center": np.array([1.0, 1.0]), "radius": 0.4},
            {"center": np.array([11.0, 9.0]), "radius": 0.4},
        ]
        
        # Goal regions matching the dataset
        self.goals = [
            {"center": np.array([10.5, 8.5]), "radius": 0.4},  # Loading dock 1
            {"center": np.array([10.5, 1.5]), "radius": 0.4},  # Loading dock 2
            {"center": np.array([1.5, 9.0]), "radius": 0.3},   # Pickup station 1
            {"center": np.array([6.5, 0.5]), "radius": 0.3},   # Pickup station 2
        ]

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.array([0.5, 0.2, 0.0, 0.0])  # [x, y, vx, vy] - Start at a safe starting position
        return self.state

    def step(self, action: np.ndarray, state: np.ndarray = None) -> tuple:
        """Simulate one environment step. Time Step = 0.1s.

        Args:
            action (np.ndarray): Action [ax, ay].
            state (np.ndarray): Optional current state [x, y, vx, vy]. Uses internal state if None.
        Returns:
            next_state (np.ndarray): Next state after action.
            reward (float): Reward obtained.
            done (bool): Whether episode has ended.
            info (Dict): Info dict with 'collision' and 'success' keys.
        """
        # Use provided state or internal state
        if state is None:
            state = self.state

        # Physics integration
        new_vel = state[2:] + action * 0.1
        new_pos = state[:2] + new_vel * 0.1
        new_state = np.concatenate([new_pos, new_vel])

        # Update internal state
        self.state = new_state

        # Check collisions
        collision = self._check_collision(new_pos)

        # Check bounds
        out_of_bounds = (new_pos[0] < 0 or new_pos[0] > self.bounds[0] or
                        new_pos[1] < 0 or new_pos[1] > self.bounds[1])

        # Compute reward
        min_goal_distance = min(
            np.linalg.norm(new_pos - goal["center"])
            for goal in self.goals
        )
        reward = -min_goal_distance

        if collision or out_of_bounds:
            reward -= 10.0

        # Check goal
        goal_reached = any(
            np.linalg.norm(new_pos - goal["center"]) < goal["radius"]
            for goal in self.goals
        )
        if goal_reached:
            reward += 50.0

        # Generate sensor data
        sensors = self._get_sensor_data(new_pos)

        # Create info dict with standard keys
        info = {
            'collision': collision or out_of_bounds,
            'success': goal_reached,
            'obstacle_distance': sensors['obstacle_distance'],
            'obstacle_direction': sensors['obstacle_direction']
        }

        return new_state, reward, goal_reached or collision or out_of_bounds, info
    
    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check collision with obstacles."""
        for obstacle in self.obstacles:
            if np.linalg.norm(pos - obstacle["center"]) < obstacle["radius"]:
                return True
        return False
    
    def _get_sensor_data(self, pos: np.ndarray) -> Dict:
        """Generate sensor readings.
        """
        min_distance = float('inf')
        closest_obstacle_dir = np.array([1.0, 0.0])
        
        for obstacle in self.obstacles:
            distance = np.linalg.norm(pos - obstacle["center"]) - obstacle["radius"]
            if distance < min_distance:
                min_distance = distance
                closest_obstacle_dir = (obstacle["center"] - pos) / (np.linalg.norm(obstacle["center"] - pos) + 1e-6)
        
        collision = self._check_collision(pos)
        
        return {
            "obstacle_distance": min_distance,
            "obstacle_direction": closest_obstacle_dir,
            "collision": collision
        }
    
    
    def get_dataset_statistics(self, transitions: List[Transition]) -> Dict:
        """Get statistics about the dataset."""
        total = len(transitions)
        safe_count = sum(1 for t in transitions if t.is_safe)
        goal_count = sum(1 for t in transitions if t.is_goal)
        
        return {
            'total_transitions': total,
            'safe_transitions': safe_count,
            'unsafe_transitions': total - safe_count,
            'goal_transitions': goal_count,
            'safety_ratio': safe_count / total if total > 0 else 0,
            'goal_ratio': goal_count / total if total > 0 else 0,
            'avg_reward': np.mean([t.reward for t in transitions]) if total > 0 else 0
        }
    

