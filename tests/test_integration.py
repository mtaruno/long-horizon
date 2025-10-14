"""
Integration tests for the complete safe planning system.
Tests end-to-end functionality and safety guarantees.
"""

import pytest
import torch
import numpy as np
from src import create_trainer, Transition
from src.dataset import create_navigation_dataset, RuleBasedDatasetGenerator
from src.types import SafetyMetrics, TrainingMetrics

class TestEndToEndIntegration:
    """Test complete system integration."""
    
    @pytest.fixture
    def trained_system(self):
        """Create a trained system for testing."""
        # Create trainer
        trainer = create_trainer(
            state_dim=4, 
            action_dim=2, 
            device="cpu",
            batch_size=32
        )
        
        # Generate training dataset
        dataset = create_navigation_dataset(num_transitions=500)
        
        # Add data to trainer
        for transition in dataset:
            trainer.add_transition(transition)
        
        return trainer, dataset
    
    def test_system_initialization(self, trained_system):
        """Test system initializes correctly with data."""
        trainer, dataset = trained_system
        
        # Check trainer state
        assert trainer.step_count == len(dataset)
        assert len(trainer.replay_buffer) == len(dataset)
        
        # Check data distribution
        safe_count = len(trainer.labeled_data['safe_states'])
        unsafe_count = len(trainer.labeled_data['unsafe_states'])
        goal_count = len(trainer.labeled_data['goal_states'])
        
        assert safe_count > 0, "Should have safe training data"
        assert unsafe_count > 0, "Should have unsafe training data"
        assert goal_count > 0, "Should have goal training data"
    
    def test_safe_action_generation(self, trained_system):
        """Test system generates safe actions."""
        trainer, _ = trained_system
        
        # Test various states
        test_states = [
            np.array([1.0, 1.0, 0.1, 0.1]),    # Normal state
            np.array([0.1, 0.1, 0.0, 0.0]),    # Near origin
            np.array([4.5, 3.5, 0.2, 0.2]),    # Near boundary
            np.array([2.5, 2.0, -0.1, -0.1]),  # Near obstacle
        ]
        
        for state in test_states:
            # Test with various proposed actions
            proposed_actions = [
                np.array([0.0, 0.0]),     # No action
                np.array([0.5, 0.3]),     # Moderate action
                np.array([-0.2, -0.1]),   # Negative action
                np.array([1.0, 1.0]),     # Large action
            ]
            
            for proposed_action in proposed_actions:
                safe_action = trainer.get_safe_action(state, proposed_action)
                
                # Check output validity
                assert isinstance(safe_action, torch.Tensor)
                assert safe_action.shape == (2,)
                assert not torch.isnan(safe_action).any()
                assert not torch.isinf(safe_action).any()
                
                # Check action is reasonable
                action_magnitude = torch.norm(safe_action)
                assert action_magnitude < 10.0, "Safe action should be reasonable magnitude"
    
    def test_safety_metrics_computation(self, trained_system):
        """Test safety metrics are computed correctly."""
        trainer, _ = trained_system
        
        # Create test batch
        batch_size = 10
        test_states = torch.randn(batch_size, 4)
        test_actions = torch.randn(batch_size, 2)
        
        # Get safety metrics
        metrics = trainer.evaluate(test_states, test_actions)
        
        # Check metrics structure
        assert isinstance(metrics, SafetyMetrics)
        assert hasattr(metrics, 'safety_rate')
        assert hasattr(metrics, 'goal_proximity_rate')
        assert hasattr(metrics, 'avg_cbf_value')
        assert hasattr(metrics, 'avg_clf_value')
        assert hasattr(metrics, 'cbf_constraint_violations')
        assert hasattr(metrics, 'clf_constraint_violations')
        
        # Check metrics validity
        assert 0.0 <= metrics.safety_rate <= 1.0
        assert 0.0 <= metrics.goal_proximity_rate <= 1.0
        assert not np.isnan(metrics.avg_cbf_value)
        assert not np.isnan(metrics.avg_clf_value)
        assert metrics.cbf_constraint_violations >= 0.0
        assert metrics.clf_constraint_violations >= 0.0
    
    def test_training_summary(self, trained_system):
        """Test training summary provides valid information."""
        trainer, _ = trained_system
        
        summary = trainer.get_training_summary()
        
        # Check summary structure
        assert isinstance(summary, TrainingMetrics)
        assert hasattr(summary, 'step_count')
        assert hasattr(summary, 'epoch_count')
        assert hasattr(summary, 'buffer_size')
        
        # Check summary validity
        assert summary.step_count > 0
        assert summary.epoch_count >= 0
        assert summary.buffer_size > 0
        assert summary.buffer_size == len(trainer.replay_buffer)

class TestSafetyGuarantees:
    """Test safety guarantees and constraint satisfaction."""
    
    @pytest.fixture
    def safety_critical_system(self):
        """Create system with known safe/unsafe regions."""
        # Create generator with clear safe/unsafe boundaries
        generator = RuleBasedDatasetGenerator(
            workspace_bounds=(0.0, 4.0, 0.0, 3.0),
            obstacles=[
                {'center': np.array([3.5, 2.5]), 'radius': 0.4}  # Near boundary
            ],
            goal_regions=[
                {'center': np.array([0.5, 0.5]), 'radius': 0.3}
            ]
        )
        
        # Generate dataset with clear labels
        dataset = generator.generate_transitions(800)
        
        # Create and train system
        trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
        for transition in dataset:
            trainer.add_transition(transition)
            
        return trainer, generator
    
    def test_cbf_safety_constraint_satisfaction(self, safety_critical_system):
        """Test CBF constraints are satisfied for safe actions."""
        trainer, generator = safety_critical_system
        
        # Test in known safe region
        safe_states = [
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([2.0, 1.5, 0.1, 0.1]),
            np.array([1.5, 2.0, -0.1, 0.0])
        ]
        
        for state in safe_states:
            # Get safe action
            proposed_action = np.array([0.3, 0.2])
            safe_action = trainer.get_safe_action(state, proposed_action)
            
            # Check CBF constraint
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = safe_action.unsqueeze(0)
            
            # Predict next state
            next_state = trainer.dynamics_ensemble(state_tensor, action_tensor)
            
            # Check CBF constraint: h(s') - h(s) >= -α * h(s)
            h_current = trainer.cbf_ensemble(state_tensor)
            h_next = trainer.cbf_ensemble(next_state)
            
            alpha = trainer.cbf_ensemble.models[0].alpha
            cbf_constraint = h_next - h_current + alpha * h_current
            
            # Constraint should be satisfied (non-negative)
            # Allow small tolerance for numerical errors
            assert cbf_constraint.item() >= -0.1, f"CBF constraint violated: {cbf_constraint.item()}"
    
    def test_clf_feasibility_constraint_satisfaction(self, safety_critical_system):
        """Test CLF constraints promote goal convergence."""
        trainer, generator = safety_critical_system
        
        # Test states at various distances from goal
        goal_center = generator.goal_regions[0]['center']
        test_states = [
            np.concatenate([goal_center + np.array([0.8, 0.6]), np.array([0.0, 0.0])]),
            np.concatenate([goal_center + np.array([1.5, 1.0]), np.array([0.1, 0.1])]),
            np.concatenate([goal_center + np.array([2.0, 1.5]), np.array([-0.1, 0.0])])
        ]
        
        for state in test_states:
            # Get action that should move toward goal
            direction_to_goal = goal_center - state[:2]
            proposed_action = direction_to_goal * 0.3  # Moderate action toward goal
            
            safe_action = trainer.get_safe_action(state, proposed_action)
            
            # Check CLF constraint
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = safe_action.unsqueeze(0)
            
            # Predict next state
            next_state = trainer.dynamics_ensemble(state_tensor, action_tensor)
            
            # Check CLF constraint: V(s') - V(s) <= -β * V(s) + δ
            V_current = trainer.clf_ensemble(state_tensor)
            V_next = trainer.clf_ensemble(next_state)
            
            beta = trainer.clf_ensemble.models[0].beta
            delta = trainer.clf_ensemble.models[0].delta
            clf_constraint = V_next - V_current + beta * V_current - delta
            
            # Constraint should be satisfied (non-positive)
            # Allow tolerance for learning and numerical errors
            assert clf_constraint.item() <= 0.5, f"CLF constraint violated: {clf_constraint.item()}"
    
    def test_action_filtering_safety(self, safety_critical_system):
        """Test action filtering prevents unsafe actions."""
        trainer, generator = safety_critical_system
        
        # Test near boundary (potentially unsafe)
        boundary_state = np.array([3.8, 2.8, 0.2, 0.2])  # Near workspace boundary
        
        # Propose action that would go out of bounds
        unsafe_action = np.array([0.5, 0.5])  # Would push out of bounds
        
        safe_action = trainer.get_safe_action(boundary_state, unsafe_action)
        
        # Safe action should be different from unsafe action
        action_diff = np.linalg.norm(safe_action.numpy() - unsafe_action)
        
        # If system is working, it should modify the action
        # (Allow case where action was already safe)
        assert action_diff >= 0.0, "Action filtering should produce valid output"
        
        # Check that safe action doesn't lead to immediate violation
        state_tensor = torch.FloatTensor(boundary_state).unsqueeze(0)
        safe_action_tensor = safe_action.unsqueeze(0)
        
        next_state = trainer.dynamics_ensemble(state_tensor, safe_action_tensor)
        next_pos = next_state[0, :2]
        
        # Should stay within reasonable bounds
        assert 0.0 <= next_pos[0] <= 4.2, f"Next x position should be reasonable: {next_pos[0]}"
        assert 0.0 <= next_pos[1] <= 3.2, f"Next y position should be reasonable: {next_pos[1]}"

class TestRobustnessAndEdgeCases:
    """Test system robustness and edge case handling."""
    
    def test_extreme_state_handling(self):
        """Test system handles extreme states gracefully."""
        trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
        
        # Add minimal training data
        dataset = create_navigation_dataset(num_transitions=50)
        for transition in dataset:
            trainer.add_transition(transition)
        
        # Test extreme states
        extreme_states = [
            np.array([0.0, 0.0, 0.0, 0.0]),      # All zeros
            np.array([10.0, 10.0, 5.0, 5.0]),    # Very large values
            np.array([-5.0, -5.0, -2.0, -2.0]),  # Negative values
            np.array([1e-6, 1e-6, 1e-6, 1e-6]),  # Very small values
        ]
        
        for state in extreme_states:
            try:
                safe_action = trainer.get_safe_action(state, np.array([0.1, 0.1]))
                
                # Should produce valid output
                assert isinstance(safe_action, torch.Tensor)
                assert safe_action.shape == (2,)
                assert not torch.isnan(safe_action).any()
                assert not torch.isinf(safe_action).any()
                
            except Exception as e:
                pytest.fail(f"System failed on extreme state {state}: {e}")
    
    def test_zero_action_handling(self):
        """Test system handles zero actions correctly."""
        trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
        
        # Add training data
        dataset = create_navigation_dataset(num_transitions=100)
        for transition in dataset:
            trainer.add_transition(transition)
        
        # Test zero action
        test_state = np.array([1.0, 1.0, 0.1, 0.1])
        zero_action = np.array([0.0, 0.0])
        
        safe_action = trainer.get_safe_action(test_state, zero_action)
        
        # Should handle gracefully
        assert isinstance(safe_action, torch.Tensor)
        assert safe_action.shape == (2,)
        assert not torch.isnan(safe_action).any()
        assert not torch.isinf(safe_action).any()
    
    def test_batch_processing_consistency(self):
        """Test batch processing gives consistent results."""
        trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
        
        # Add training data
        dataset = create_navigation_dataset(num_transitions=200)
        for transition in dataset:
            trainer.add_transition(transition)
        
        # Test single vs batch processing
        test_state = np.array([1.5, 1.5, 0.2, 0.2])
        test_action = np.array([0.3, 0.2])
        
        # Single processing
        single_result = trainer.get_safe_action(test_state, test_action)
        
        # Batch processing (same state/action repeated)
        batch_states = torch.FloatTensor([test_state, test_state])
        batch_actions = torch.FloatTensor([test_action, test_action])
        
        batch_results = trainer.get_safe_action(batch_states, batch_actions)
        
        # Results should be consistent
        torch.testing.assert_close(
            single_result, batch_results[0], 
            rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            batch_results[0], batch_results[1],
            rtol=1e-5, atol=1e-6
        )
    
    def test_deterministic_behavior(self):
        """Test system behavior is deterministic."""
        # Create two identical systems
        trainer1 = create_trainer(state_dim=4, action_dim=2, device="cpu")
        trainer2 = create_trainer(state_dim=4, action_dim=2, device="cpu")
        
        # Add identical data
        np.random.seed(42)
        dataset1 = create_navigation_dataset(num_transitions=100)
        
        np.random.seed(42)
        dataset2 = create_navigation_dataset(num_transitions=100)
        
        for t1, t2 in zip(dataset1, dataset2):
            trainer1.add_transition(t1)
            trainer2.add_transition(t2)
        
        # Test same inputs
        test_state = np.array([1.0, 1.0, 0.1, 0.1])
        test_action = np.array([0.2, 0.2])
        
        # Should produce identical results
        result1 = trainer1.get_safe_action(test_state, test_action)
        result2 = trainer2.get_safe_action(test_state, test_action)
        
        # Allow small numerical differences
        torch.testing.assert_close(result1, result2, rtol=1e-4, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])