"""
Dataset creation and management for CBF-CLF training.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from .types import Transition
import random

class DatasetGenerator(ABC):
    """Abstract base class for dataset generators."""
    
    @abstractmethod
    def generate_transitions(self, num_transitions: int) -> List[Transition]:
        """Generate labeled transitions."""
        pass

class SimulationDatasetGenerator(DatasetGenerator):
    """Generate dataset from simulation environment."""
    
    def __init__(
        self,
        environment: 'Environment',
        policy: Optional[Callable] = None,
        state_dim: int = 4,
        action_dim: int = 2
    ):
        self.environment = environment
        self.policy = policy or self._random_policy
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def _random_policy(self, state: np.ndarray) -> np.ndarray:
        """Default random policy for exploration."""
        return np.random.uniform(-0.5, 0.5, self.action_dim)
    
    def generate_transitions(self, num_transitions: int) -> List[Transition]:
        """Generate transitions from simulation."""
        transitions = []
        
        while len(transitions) < num_transitions:
            # Reset environment
            state = self.environment.reset()
            
            for _ in range(50):  # Max episode length
                # Get action from policy
                action = self.policy(state)
                
                # Step environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Create transition
                transition = Transition(
                    state=state.copy(),
                    action=action.copy(),
                    next_state=next_state.copy(),
                    reward=reward,
                    done=done,
                    is_safe=info.get('is_safe', True),
                    is_goal=info.get('is_goal', False)
                )
                
                transitions.append(transition)
                state = next_state
                
                if done or len(transitions) >= num_transitions:
                    break
                    
        return transitions[:num_transitions]

class RuleBasedDatasetGenerator(DatasetGenerator):
    """Generate dataset using rule-based labeling."""
    
    def __init__(
        self,
        workspace_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
        obstacles: List[Dict] = None,
        goal_regions: List[Dict] = None,
        state_dim: int = 4,
        action_dim: int = 2
    ):
        self.workspace_bounds = workspace_bounds
        self.obstacles = obstacles or []
        self.goal_regions = goal_regions or []
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def generate_transitions(self, num_transitions: int) -> List[Transition]:
        """Generate transitions with rule-based labels."""
        transitions = []
        
        for _ in range(num_transitions):
            # Sample random state
            state = self._sample_random_state()
            
            # Sample random action
            action = np.random.uniform(-1.0, 1.0, self.action_dim)
            
            # Simulate next state (simple physics)
            next_state = self._simulate_dynamics(state, action)
            
            # Apply labeling rules
            is_safe = self._is_safe(state, next_state)
            is_goal = self._is_goal(next_state)
            
            # Compute reward
            reward = self._compute_reward(state, action, next_state, is_safe, is_goal)
            
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=is_goal or not is_safe,
                is_safe=is_safe,
                is_goal=is_goal
            )
            
            transitions.append(transition)
            
        return transitions
    
    def _sample_random_state(self) -> np.ndarray:
        """Sample random state within workspace."""
        x_min, x_max, y_min, y_max = self.workspace_bounds
        
        pos = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
        
        vel = np.random.uniform(-0.5, 0.5, 2)
        
        return np.concatenate([pos, vel])
    
    def _simulate_dynamics(self, state: np.ndarray, action: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Simple physics simulation."""
        pos = state[:2]
        vel = state[2:]
        
        # Update velocity and position
        new_vel = vel + action * dt
        new_pos = pos + new_vel * dt
        
        return np.concatenate([new_pos, new_vel])
    
    def _is_safe(self, state: np.ndarray, next_state: np.ndarray) -> bool:
        """Check if transition is safe."""
        pos = next_state[:2]
        
        # Check workspace bounds
        x_min, x_max, y_min, y_max = self.workspace_bounds
        if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            center = obstacle['center']
            radius = obstacle['radius']
            if np.linalg.norm(pos - center) < radius:
                return False
                
        return True
    
    def _is_goal(self, state: np.ndarray) -> bool:
        """Check if state is in goal region."""
        pos = state[:2]
        
        for goal in self.goal_regions:
            center = goal['center']
            radius = goal['radius']
            if np.linalg.norm(pos - center) < radius:
                return True
                
        return False
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray, 
                       next_state: np.ndarray, is_safe: bool, is_goal: bool) -> float:
        """Compute reward for transition."""
        reward = 0.0
        
        # Distance-based reward (closer to goal is better)
        if self.goal_regions:
            goal_center = self.goal_regions[0]['center']
            distance = np.linalg.norm(next_state[:2] - goal_center)
            reward -= distance
        
        # Safety penalty
        if not is_safe:
            reward -= 10.0
            
        # Goal bonus
        if is_goal:
            reward += 50.0
            
        return reward

class ExpertDatasetGenerator(DatasetGenerator):
    """Generate dataset from expert demonstrations."""
    
    def __init__(
        self,
        expert_trajectories: List[List[Tuple[np.ndarray, np.ndarray]]],
        labeling_function: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[bool, bool]]
    ):
        self.expert_trajectories = expert_trajectories
        self.labeling_function = labeling_function
        
    def generate_transitions(self, num_transitions: int) -> List[Transition]:
        """Generate transitions from expert demonstrations."""
        transitions = []
        
        while len(transitions) < num_transitions:
            # Sample random trajectory
            trajectory = random.choice(self.expert_trajectories)
            
            # Sample random transition from trajectory
            if len(trajectory) < 2:
                continue
                
            idx = random.randint(0, len(trajectory) - 2)
            state, action = trajectory[idx]
            next_state, _ = trajectory[idx + 1]
            
            # Apply labeling function
            is_safe, is_goal = self.labeling_function(state, action, next_state)
            
            transition = Transition(
                state=state.copy(),
                action=action.copy(),
                next_state=next_state.copy(),
                reward=0.0,  # Not used in expert demos
                done=is_goal,
                is_safe=is_safe,
                is_goal=is_goal
            )
            
            transitions.append(transition)
            
        return transitions[:num_transitions]

class DatasetManager:
    """Manage and combine multiple dataset sources."""
    
    def __init__(self):
        self.generators = []
        
    def add_generator(self, generator: DatasetGenerator, weight: float = 1.0):
        """Add a dataset generator with specified weight."""
        self.generators.append((generator, weight))
        
    def generate_balanced_dataset(
        self, 
        total_transitions: int,
        min_unsafe_ratio: float = 0.1,
        min_goal_ratio: float = 0.05
    ) -> List[Transition]:
        """Generate balanced dataset with specified ratios."""
        
        # Calculate transitions per generator
        total_weight = sum(weight for _, weight in self.generators)
        transitions_per_gen = [
            int(total_transitions * weight / total_weight) 
            for _, weight in self.generators
        ]
        
        # Generate transitions from each generator
        all_transitions = []
        for (generator, _), num_trans in zip(self.generators, transitions_per_gen):
            transitions = generator.generate_transitions(num_trans)
            all_transitions.extend(transitions)
            
        # Analyze current distribution
        safe_count = sum(1 for t in all_transitions if t.is_safe)
        unsafe_count = len(all_transitions) - safe_count
        goal_count = sum(1 for t in all_transitions if t.is_goal)
        
        # Add more unsafe transitions if needed
        target_unsafe = max(int(total_transitions * min_unsafe_ratio), unsafe_count)
        if unsafe_count < target_unsafe:
            needed_unsafe = target_unsafe - unsafe_count
            # Generate more transitions and filter for unsafe ones
            # (Implementation would depend on specific generators)
            
        # Add more goal transitions if needed
        target_goal = max(int(total_transitions * min_goal_ratio), goal_count)
        if goal_count < target_goal:
            needed_goal = target_goal - goal_count
            # Generate more transitions and filter for goal ones
            # (Implementation would depend on specific generators)
            
        # Shuffle and return
        random.shuffle(all_transitions)
        return all_transitions[:total_transitions]
    
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

# ===== PREDEFINED DATASET CONFIGURATIONS =====

def create_warehouse_dataset(num_transitions: int = 10000) -> List[Transition]:
    """Create dataset for warehouse robot scenario."""
    
    # Define warehouse environment
    workspace_bounds = (0.0, 10.0, 0.0, 8.0)  # 10m x 8m warehouse
    
    obstacles = [
        {'center': np.array([2.0, 2.0]), 'radius': 0.5},  # Shelf 1
        {'center': np.array([5.0, 3.0]), 'radius': 0.4},  # Shelf 2
        {'center': np.array([8.0, 1.5]), 'radius': 0.3},  # Shelf 3
    ]
    
    goal_regions = [
        {'center': np.array([9.0, 7.0]), 'radius': 0.3},  # Delivery point 1
        {'center': np.array([1.0, 7.0]), 'radius': 0.3},  # Delivery point 2
    ]
    
    generator = RuleBasedDatasetGenerator(
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        goal_regions=goal_regions
    )
    
    return generator.generate_transitions(num_transitions)

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

def create_mixed_dataset(
    num_transitions: int = 15000,
    simulation_ratio: float = 0.6,
    rule_ratio: float = 0.4
) -> List[Transition]:
    """Create mixed dataset from multiple sources."""
    
    manager = DatasetManager()
    
    # Add rule-based generator
    rule_generator = RuleBasedDatasetGenerator(
        workspace_bounds=(0.0, 5.0, 0.0, 4.0),
        obstacles=[{'center': np.array([2.5, 2.0]), 'radius': 0.4}],
        goal_regions=[{'center': np.array([4.0, 3.0]), 'radius': 0.2}]
    )
    manager.add_generator(rule_generator, weight=rule_ratio)
    
    # Could add simulation generator here if environment is available
    # sim_generator = SimulationDatasetGenerator(environment)
    # manager.add_generator(sim_generator, weight=simulation_ratio)
    
    return manager.generate_balanced_dataset(
        total_transitions=num_transitions,
        min_unsafe_ratio=0.15,
        min_goal_ratio=0.05
    )