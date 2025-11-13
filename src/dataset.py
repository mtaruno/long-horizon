"""
Dataset creation and management for CBF-CLF training.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import random
from .environment import WarehouseEnvironment, Transition


# class DatasetGenerator(ABC):
#     """Abstract base class for dataset generators."""
    
#     @abstractmethod
#     def generate_transitions(self, num_transitions: int) -> List[Transition]:
#         """Generate labeled transitions."""
#         pass

# class SimulationDatasetGenerator(DatasetGenerator):
#     """Generate dataset from simulation environment."""
    
#     def __init__(
#         self,
#         environment,
#         policy: Optional[Callable] = None,
#         state_dim: int = 4,
#         action_dim: int = 2
#     ):
#         self.environment = environment
#         self.policy = policy or self._random_policy
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#     def _random_policy(self, state: np.ndarray) -> np.ndarray:
#         """Default random policy for exploration."""
#         return np.random.uniform(-0.5, 0.5, self.action_dim)
    
#     def generate_transitions(self, num_transitions: int) -> List[Transition]:
#         """Generate transitions from simulation."""
#         transitions = []
        
#         while len(transitions) < num_transitions:
#             # Reset environment
#             state = self.environment.reset()
            
#             for _ in range(50):  # Max episode length
#                 # Get action from policy
#                 action = self.policy(state)
                
#                 # Step environment
#                 next_state, reward, done, info = self.environment.step(action)
                
#                 # Create transition
#                 transition = Transition(
#                     state=state.copy(),
#                     action=action.copy(),
#                     next_state=next_state.copy(),
#                     reward=reward,
#                     done=done,
#                     is_safe=info.get('is_safe', True),
#                     is_goal=info.get('is_goal', False)
#                 )
                
#                 transitions.append(transition)
#                 state = next_state
                
#                 if done or len(transitions) >= num_transitions:
#                     break
                    
#         return transitions[:num_transitions]

# class RuleBasedDatasetGenerator(DatasetGenerator):
#     """Generate dataset using rule-based labeling."""
    
#     def __init__(
#         self,
#         workspace_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
#         obstacles: List[Dict] = None,
#         goal_regions: List[Dict] = None,
#         state_dim: int = 4,
#         action_dim: int = 2
#     ):
#         self.workspace_bounds = workspace_bounds
#         self.obstacles = obstacles or []
#         self.goal_regions = goal_regions or []
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#     def generate_transitions(self, num_transitions: int) -> List[Transition]:
#         """Generate transitions with rule-based labels."""
#         transitions = []
        
#         for _ in range(num_transitions):
#             # Sample random state
#             state = self._sample_random_state()
            
#             # Sample random action
#             action = np.random.uniform(-1.0, 1.0, self.action_dim)
            
#             # Simulate next state (simple physics)
#             next_state = self._simulate_dynamics(state, action)
            
#             # Apply labeling rules
#             is_safe = self._is_safe(state, next_state)
#             is_goal = self._is_goal(next_state)
            
#             # Compute reward
#             reward = self._compute_reward(state, action, next_state, is_safe, is_goal)
            
#             transition = Transition(
#                 state=state,
#                 action=action,
#                 next_state=next_state,
#                 reward=reward,
#                 done=is_goal or not is_safe,
#                 is_safe=is_safe,
#                 is_goal=is_goal
#             )
            
#             transitions.append(transition)
            
#         return transitions
    
#     def _sample_random_state(self) -> np.ndarray:
#         """Sample random state within workspace."""
#         x_min, x_max, y_min, y_max = self.workspace_bounds
        
#         pos = np.array([
#             np.random.uniform(x_min, x_max),
#             np.random.uniform(y_min, y_max)
#         ])
        
#         vel = np.random.uniform(-0.5, 0.5, 2)
        
#         return np.concatenate([pos, vel])
    
#     def _simulate_dynamics(self, state: np.ndarray, action: np.ndarray, dt: float = 0.1) -> np.ndarray:
#         """Simple physics simulation."""
#         pos = state[:2]
#         vel = state[2:]
        
#         # Update velocity and position
#         new_vel = vel + action * dt
#         new_pos = pos + new_vel * dt
        
#         return np.concatenate([new_pos, new_vel])
    
#     def _is_safe(self, state: np.ndarray, next_state: np.ndarray) -> bool:
#         """Check if transition is safe."""
#         pos = next_state[:2]
        
#         # Check workspace bounds
#         x_min, x_max, y_min, y_max = self.workspace_bounds
#         if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
#             return False
        
#         # Check obstacles
#         for obstacle in self.obstacles:
#             center = obstacle['center']
#             radius = obstacle['radius']
#             if np.linalg.norm(pos - center) < radius:
#                 return False
                
#         return True
    
#     def _is_goal(self, state: np.ndarray) -> bool:
#         """Check if state is in goal region."""
#         pos = state[:2]
        
#         for goal in self.goal_regions:
#             center = goal['center']
#             radius = goal['radius']
#             if np.linalg.norm(pos - center) < radius:
#                 return True
                
#         return False
    
#     def _compute_reward(self, state: np.ndarray, action: np.ndarray, 
#                        next_state: np.ndarray, is_safe: bool, is_goal: bool) -> float:
#         """Compute reward for transition."""
#         reward = 0.0
        
#         # Distance-based reward (closer to goal is better)
#         if self.goal_regions:
#             goal_center = self.goal_regions[0]['center']
#             distance = np.linalg.norm(next_state[:2] - goal_center)
#             reward -= distance
        
#         # Safety penalty
#         if not is_safe:
#             reward -= 10.0
            
#         # Goal bonus
#         if is_goal:
#             reward += 50.0
            
#         return reward


def create_warehouse_dataset(num_transitions, env=None, goal_pos=None):
    """
    Generates a dataset of transitions using a random policy.
    Based on Algorithm 3 from the paper.
    """
    if env is None:
        env = WarehouseEnvironment()
        
    if goal_pos is None:
        goal_pos = env.target_goal['center']

    transitions = []
    state, _ = env.reset()
    
    for _ in range(num_transitions):
        # Sample a random action
        action = env.action_space.sample() 
        
        next_state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        
        trans = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            is_done=is_done,
            is_safe=info['is_safe'],
            is_goal=info['is_goal'],
            goal=goal_pos # Store the goal context
        )
        transitions.append(trans)
        
        if is_done:
            state, _ = env.reset()
        else:
            state = next_state
            
    return transitions

def create_warehouse_dataset(num_transitions, env=None, goal_pos=None):
    """
    Generates a dataset of transitions using a random policy.
    Based on Algorithm 3 from the paper.
    """
    if env is None:
        env = WarehouseEnvironment()
        
    if goal_pos is None:
        goal_pos = env.target_goal['center']

    transitions = []
    state, _ = env.reset()
    
    for _ in range(num_transitions):
        # Sample a random action
        action = env.action_space.sample() 
        
        next_state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        
        trans = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            is_done=is_done,
            is_safe=info['is_safe'],
            is_goal=info['is_goal'],
            goal=goal_pos # Store the goal context
        )
        transitions.append(trans)
        
        if is_done:
            state, _ = env.reset()
        else:
            state = next_state
            
    return transitions


# NOTE: Using older interface
# def create_warehouse_dataset(num_transitions: int = 10000) -> List[Transition]:
#     """Create dataset for warehouse robot scenario.

#     IMPORTANT: This must match the WarehouseEnvironment configuration in order to get good training results for that specific environment. 
#     """

#     # Match WarehouseEnvironment bounds: 12m x 10m
#     workspace_bounds = (0.0, 12.0, 0.0, 10.0)

#     # Match WarehouseEnvironment obstacles exactly
#     obstacles = [
#         {'center': np.array([2.0, 3.0]), 'radius': 0.8},
#         {'center': np.array([2.0, 7.0]), 'radius': 0.8},
#         {'center': np.array([5.0, 2.0]), 'radius': 0.6},
#         {'center': np.array([5.0, 5.0]), 'radius': 0.6},
#         {'center': np.array([5.0, 8.0]), 'radius': 0.6},
#         {'center': np.array([8.0, 3.5]), 'radius': 0.7},
#         {'center': np.array([8.0, 6.5]), 'radius': 0.7},
#         {'center': np.array([4.0, 9.0]), 'radius': 0.3},
#         # {'center': np.array([9.0, 1.0]), 'radius': 0.3},
#         {'center': np.array([1.0, 1.0]), 'radius': 0.4},
#         {'center': np.array([11.0, 9.0]), 'radius': 0.4},
#     ]

#     # Match WarehouseEnvironment goals exactly
#     goal_regions = [
#         {'center': np.array([10.5, 8.5]), 'radius': 0.4},  # Loading dock 1
#         {'center': np.array([10.5, 1.5]), 'radius': 0.4},  # Loading dock 2
#         {'center': np.array([1.5, 9.0]), 'radius': 0.3},   # Pickup station 1
#         {'center': np.array([6.5, 0.5]), 'radius': 0.3},   # Pickup station 2
#     ]

#     generator = RuleBasedDatasetGenerator(
#         workspace_bounds=workspace_bounds,
#         obstacles=obstacles,
#         goal_regions=goal_regions
#     )

#     return generator.generate_transitions(num_transitions)

def create_navigation_dataset(num_transitions: int = 5000) -> List[Transition]:
    """Create dataset for simple navigation scenario."""
    
    workspace_bounds = (0.0, 5.0, 0.0, 4.0)  # 5m x 4m space
    
    obstacles = [
        {'center': np.array([2.5, 2.0]), 'radius': 0.4},  # Central obstacle
    ]
    
    goal_regions = [
        {'center': np.array([4.0, 3.0]), 'radius': 0.2},  # Target corner
    ]
    
    generator = RuleBasedDatasetGenerator(
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        goal_regions=goal_regions
    )
    
    return generator.generate_transitions(num_transitions)

