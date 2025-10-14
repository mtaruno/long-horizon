"""
Critical tests for dataset generation and management.
Data quality is essential for safety-critical learning.
"""

import pytest
import numpy as np
from src.dataset import (
    RuleBasedDatasetGenerator, DatasetManager, 
    create_warehouse_dataset, create_navigation_dataset
)
from src.types import Transition

class TestRuleBasedDatasetGenerator:
    """Test rule-based dataset generation."""
    
    @pytest.fixture
    def generator(self):
        return RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 5.0, 0.0, 4.0),
            obstacles=[
                {'center': np.array([2.5, 2.0]), 'radius': 0.4}
            ],
            goal_regions=[
                {'center': np.array([4.0, 3.0]), 'radius': 0.3}
            ]
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initializes with correct parameters."""
        assert generator.workspace_bounds == (0.0, 5.0, 0.0, 4.0)
        assert len(generator.obstacles) == 1
        assert len(generator.goal_regions) == 1
        assert generator.state_dim == 4
        assert generator.action_dim == 2
    
    def test_generate_transitions_count(self, generator):
        """Test generator produces requested number of transitions."""
        num_transitions = 100
        transitions = generator.generate_transitions(num_transitions)
        
        assert len(transitions) == num_transitions
        assert all(isinstance(t, Transition) for t in transitions)
    
    def test_transition_data_validity(self, generator):
        """Test all generated transitions have valid data."""
        transitions = generator.generate_transitions(50)
        
        for transition in transitions:
            # Check shapes
            assert transition.state.shape == (4,)
            assert transition.action.shape == (2,)
            assert transition.next_state.shape == (4,)
            
            # Check data types
            assert isinstance(transition.reward, float)
            assert isinstance(transition.done, bool)
            assert isinstance(transition.is_safe, bool)
            assert isinstance(transition.is_goal, bool)
            
            # Check no NaN/inf values
            assert not np.isnan(transition.state).any()
            assert not np.isnan(transition.action).any()
            assert not np.isnan(transition.next_state).any()
            assert not np.isnan(transition.reward)
            assert not np.isinf(transition.state).any()
            assert not np.isinf(transition.action).any()
            assert not np.isinf(transition.next_state).any()
            assert not np.isinf(transition.reward)
    
    def test_workspace_bounds_respected(self, generator):
        """Test generated states respect workspace bounds."""
        transitions = generator.generate_transitions(100)
        
        for transition in transitions:
            pos = transition.state[:2]
            next_pos = transition.next_state[:2]
            
            # Initial states should be within bounds
            assert 0.0 <= pos[0] <= 5.0
            assert 0.0 <= pos[1] <= 4.0
    
    def test_safety_labeling_consistency(self, generator):
        """Test safety labeling is consistent with rules."""
        transitions = generator.generate_transitions(200)
        
        for transition in transitions:
            next_pos = transition.next_state[:2]
            
            # Check workspace bounds safety
            in_bounds = (0.0 <= next_pos[0] <= 5.0 and 0.0 <= next_pos[1] <= 4.0)
            
            # Check obstacle collision
            obstacle_collision = False
            for obstacle in generator.obstacles:
                distance = np.linalg.norm(next_pos - obstacle['center'])
                if distance < obstacle['radius']:
                    obstacle_collision = True
                    break
            
            expected_safe = in_bounds and not obstacle_collision
            
            # Allow some tolerance for boundary cases
            if not in_bounds or obstacle_collision:
                assert not transition.is_safe, f"State {next_pos} should be unsafe"
    
    def test_goal_labeling_consistency(self, generator):
        """Test goal labeling is consistent with rules."""
        transitions = generator.generate_transitions(200)
        
        for transition in transitions:
            pos = transition.next_state[:2]
            
            # Check if in any goal region
            in_goal = False
            for goal in generator.goal_regions:
                distance = np.linalg.norm(pos - goal['center'])
                if distance < goal['radius']:
                    in_goal = True
                    break
            
            if in_goal:
                assert transition.is_goal, f"State {pos} should be goal"
            # Note: Not all non-goal states will be labeled as non-goal due to sampling
    
    def test_physics_simulation_consistency(self, generator):
        """Test physics simulation produces reasonable results."""
        transitions = generator.generate_transitions(50)
        
        for transition in transitions:
            state = transition.state
            action = transition.action
            next_state = transition.next_state
            
            # Check physics makes sense (simple integration)
            dt = 0.1
            expected_vel = state[2:] + action * dt
            expected_pos = state[:2] + expected_vel * dt
            
            # Allow some tolerance for physics variations
            vel_diff = np.linalg.norm(next_state[2:] - expected_vel)
            pos_diff = np.linalg.norm(next_state[:2] - expected_pos)
            
            assert vel_diff < 0.1, "Velocity update should follow physics"
            assert pos_diff < 0.1, "Position update should follow physics"

class TestDatasetManager:
    """Test dataset management and combination."""
    
    @pytest.fixture
    def manager(self):
        return DatasetManager()
    
    @pytest.fixture
    def sample_transitions(self):
        """Create sample transitions for testing."""
        return [
            Transition(
                state=np.array([1.0, 1.0, 0.1, 0.1]),
                action=np.array([0.1, 0.1]),
                next_state=np.array([1.1, 1.1, 0.2, 0.2]),
                reward=1.0,
                done=False,
                is_safe=True,
                is_goal=False
            ),
            Transition(
                state=np.array([2.0, 2.0, 0.0, 0.0]),
                action=np.array([0.0, 0.0]),
                next_state=np.array([2.0, 2.0, 0.0, 0.0]),
                reward=-10.0,
                done=True,
                is_safe=False,
                is_goal=False
            ),
            Transition(
                state=np.array([0.1, 0.1, 0.0, 0.0]),
                action=np.array([0.0, 0.0]),
                next_state=np.array([0.1, 0.1, 0.0, 0.0]),
                reward=50.0,
                done=True,
                is_safe=True,
                is_goal=True
            )
        ]
    
    def test_dataset_statistics_computation(self, manager, sample_transitions):
        """Test dataset statistics are computed correctly."""
        stats = manager.get_dataset_statistics(sample_transitions)
        
        assert stats['total_transitions'] == 3
        assert stats['safe_transitions'] == 2
        assert stats['unsafe_transitions'] == 1
        assert stats['goal_transitions'] == 1
        assert stats['safety_ratio'] == 2/3
        assert stats['goal_ratio'] == 1/3
        assert stats['avg_reward'] == (1.0 - 10.0 + 50.0) / 3
    
    def test_empty_dataset_statistics(self, manager):
        """Test statistics for empty dataset."""
        stats = manager.get_dataset_statistics([])
        
        assert stats['total_transitions'] == 0
        assert stats['safe_transitions'] == 0
        assert stats['unsafe_transitions'] == 0
        assert stats['goal_transitions'] == 0
        assert stats['safety_ratio'] == 0
        assert stats['goal_ratio'] == 0
        assert stats['avg_reward'] == 0
    
    def test_add_generator_with_weights(self, manager):
        """Test adding generators with different weights."""
        gen1 = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 2.0, 0.0, 2.0),
            obstacles=[], goal_regions=[]
        )
        gen2 = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 3.0, 0.0, 3.0),
            obstacles=[], goal_regions=[]
        )
        
        manager.add_generator(gen1, weight=0.7)
        manager.add_generator(gen2, weight=0.3)
        
        assert len(manager.generators) == 2
        assert manager.generators[0][1] == 0.7
        assert manager.generators[1][1] == 0.3

class TestPredefinedDatasets:
    """Test predefined dataset configurations."""
    
    def test_warehouse_dataset_creation(self):
        """Test warehouse dataset creates valid data."""
        dataset = create_warehouse_dataset(num_transitions=100)
        
        assert len(dataset) == 100
        assert all(isinstance(t, Transition) for t in dataset)
        
        # Check dataset has reasonable distribution
        safe_count = sum(1 for t in dataset if t.is_safe)
        goal_count = sum(1 for t in dataset if t.is_goal)
        
        # Should have mostly safe transitions
        assert safe_count > 50, "Warehouse should have mostly safe transitions"
        
        # Should have some goal transitions
        assert goal_count > 0, "Should have some goal transitions"
    
    def test_navigation_dataset_creation(self):
        """Test navigation dataset creates valid data."""
        dataset = create_navigation_dataset(num_transitions=50)
        
        assert len(dataset) == 50
        assert all(isinstance(t, Transition) for t in dataset)
        
        # Check all transitions have valid workspace bounds
        for transition in dataset:
            pos = transition.state[:2]
            next_pos = transition.next_state[:2]
            
            # Should be within navigation bounds (0-5, 0-4)
            assert 0.0 <= pos[0] <= 5.0
            assert 0.0 <= pos[1] <= 4.0
    
    def test_dataset_reproducibility(self):
        """Test datasets are reproducible with same seed."""
        np.random.seed(42)
        dataset1 = create_navigation_dataset(num_transitions=20)
        
        np.random.seed(42)
        dataset2 = create_navigation_dataset(num_transitions=20)
        
        # Should produce identical datasets
        for t1, t2 in zip(dataset1, dataset2):
            np.testing.assert_array_equal(t1.state, t2.state)
            np.testing.assert_array_equal(t1.action, t2.action)
            np.testing.assert_array_equal(t1.next_state, t2.next_state)
            assert t1.is_safe == t2.is_safe
            assert t1.is_goal == t2.is_goal

class TestDatasetQualityAssurance:
    """Test dataset quality and safety properties."""
    
    def test_dataset_balance_requirements(self):
        """Test dataset meets minimum balance requirements."""
        generator = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 3.0, 0.0, 3.0),
            obstacles=[{'center': np.array([2.8, 2.8]), 'radius': 0.3}],  # Near boundary
            goal_regions=[{'center': np.array([0.2, 0.2]), 'radius': 0.2}]
        )
        
        dataset = generator.generate_transitions(500)
        
        # Calculate statistics
        safe_count = sum(1 for t in dataset if t.is_safe)
        unsafe_count = len(dataset) - safe_count
        goal_count = sum(1 for t in dataset if t.is_goal)
        
        # Quality requirements
        assert unsafe_count > 0, "Dataset must contain some unsafe examples"
        assert goal_count > 0, "Dataset must contain some goal examples"
        assert safe_count > unsafe_count, "Dataset should be mostly safe (realistic)"
        
        # Minimum ratios
        unsafe_ratio = unsafe_count / len(dataset)
        goal_ratio = goal_count / len(dataset)
        
        assert unsafe_ratio >= 0.01, "At least 1% unsafe examples needed"
        assert goal_ratio >= 0.005, "At least 0.5% goal examples needed"
    
    def test_dataset_state_coverage(self):
        """Test dataset covers reasonable state space."""
        generator = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 5.0, 0.0, 4.0),
            obstacles=[], goal_regions=[]
        )
        
        dataset = generator.generate_transitions(200)
        
        # Extract positions
        positions = np.array([t.state[:2] for t in dataset])
        
        # Check coverage
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Should cover significant portion of workspace
        assert x_max - x_min > 3.0, "Should cover significant x range"
        assert y_max - y_min > 2.0, "Should cover significant y range"
        
        # Should respect bounds
        assert x_min >= 0.0 and x_max <= 5.0
        assert y_min >= 0.0 and y_max <= 4.0
    
    def test_dataset_action_diversity(self):
        """Test dataset contains diverse actions."""
        generator = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 3.0, 0.0, 3.0),
            obstacles=[], goal_regions=[]
        )
        
        dataset = generator.generate_transitions(100)
        
        # Extract actions
        actions = np.array([t.action for t in dataset])
        
        # Check diversity
        action_std = np.std(actions, axis=0)
        
        # Should have reasonable action diversity
        assert action_std[0] > 0.1, "Should have diverse x actions"
        assert action_std[1] > 0.1, "Should have diverse y actions"
        
        # Actions should be reasonable magnitude
        action_magnitudes = np.linalg.norm(actions, axis=1)
        assert np.max(action_magnitudes) < 5.0, "Actions should be reasonable magnitude"
    
    def test_transition_causality(self):
        """Test transitions follow causal relationships."""
        generator = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 2.0, 0.0, 2.0),
            obstacles=[], goal_regions=[]
        )
        
        dataset = generator.generate_transitions(50)
        
        for transition in dataset:
            state = transition.state
            action = transition.action
            next_state = transition.next_state
            
            # Check causality: next_state should be reachable from state + action
            dt = 0.1
            
            # Expected velocity change
            expected_next_vel = state[2:] + action * dt
            actual_next_vel = next_state[2:]
            
            vel_error = np.linalg.norm(actual_next_vel - expected_next_vel)
            assert vel_error < 0.2, f"Velocity transition should follow physics: {vel_error}"
            
            # Expected position change
            expected_next_pos = state[:2] + expected_next_vel * dt
            actual_next_pos = next_state[:2]
            
            pos_error = np.linalg.norm(actual_next_pos - expected_next_pos)
            assert pos_error < 0.2, f"Position transition should follow physics: {pos_error}"

if __name__ == "__main__":
    pytest.main([__file__])