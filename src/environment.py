

import numpy as np
from typing import List, Dict
import gymnasium as gym
from gymnasium import spaces
from collections import deque, namedtuple
import random

# Define the structure for a single transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_done',
                         'is_safe', 'is_goal', 'goal'))

class WarehouseEnvironment(gym.Env):
    """Warehouse environment matching the dataset."""
    
    def __init__(self, goal_pos=np.array([10.5, 8.5]), goal_radius=0.4):
        self.bounds = (12.0, 10.0)  # 12m x 10m warehouse
        self.dt = 0.1
        self.max_vel = 1.0
        self.max_accel = 5.0 # scaled from action
        self.safety_margin = 0.1

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
            # {"center": np.array([9.0, 1.0]), "radius": 0.3},
            {"center": np.array([1.0, 1.0]), "radius": 0.4},
            {"center": np.array([11.0, 9.0]), "radius": 0.4},
        ]
        

        self.target_goal = {"center": goal_pos, "radius": goal_radius}
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [ax, ay]

        # State space: [x, y, vx, vy]
        low_obs = np.array([0.0, 0.0, -self.max_vel, -self.max_vel])
        high_obs = np.array([self.bounds[0], self.bounds[1], self.max_vel, self.max_vel])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.state = None


    def _is_safe(self, state):
        """Checks if the state is outside all obstacles + safety margin."""
        pos = state[:2]
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['center'])
            # [cite: 186]
            if dist < (obs['radius'] + self.safety_margin):
                return False
        return True

    def _is_goal(self, state):
        """Checks if the state is within the target goal region."""
        pos = state[:2]
        dist_to_goal = np.linalg.norm(pos - self.target_goal['center'])
        return dist_to_goal <= self.target_goal['radius']


    def reset(self, seed=None, options=None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            super().reset(seed=seed)
        
        # use fixed start position for consistent training with waypoints
        # start position: bottom left area, safe from obstacles
        fixed_start_pos = np.array([0.5, 10.0])
        # TODO: Implement random start sampling if needed

        self.state = np.concatenate([fixed_start_pos, [0.0, 0.0]])
        info = {'is_safe': True, 'is_goal': self._is_goal(self.state)}
        return self.state, info
    

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
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        pos = self.state[:2]
        vel = self.state[2:]
        # Apply physics (Forward Euler) [cite: 181-184]
        accel = np.clip(action, -1.0, 1.0) * self.max_accel
        
        new_vel = vel + accel * self.dt
        new_vel = np.clip(new_vel, -self.max_vel, self.max_vel)
        
        new_pos = pos + new_vel * self.dt
        new_pos = np.clip(new_pos, [0.0, 0.0], self.bounds)
        
        self.state = np.concatenate([new_pos, new_vel])

        # Check collisions
        is_safe = self._is_safe(self.state)
        is_goal = self._is_goal(self.state)

        terminated = False # Episode ends (collision or goal)
        truncated = False  # Episode cut short (time limit)

        if not is_safe:
            reward = -10.0 # Collision penalty
            terminated = True
        elif is_goal:
            reward = 50.0  # Goal bonus
            terminated = True
        else:
            dist_to_goal = np.linalg.norm(new_pos - self.target_goal['center'])
            reward = -dist_to_goal # Negative distance
        
        reward -= 0.01 * np.linalg.norm(action) # Small action penalty

        info = {
            'is_safe': is_safe,
            'is_goal': is_goal
        }
        
        return self.state, reward, terminated, truncated, info
    
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
    

class ReplayBuffer:
    """A simple Replay Buffer for storing and sampling transitions."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)