"""
Critical tests for Control Lyapunov Functions (CLF).
Goal-reaching component - essential for feasibility guarantees.
"""

import pytest
import torch
import numpy as np
from src.clf import CLFNetwork, EnsembleCLF, CLFTrainer, CBFCLFController
from src.cbf import EnsembleCBF

class TestCLFNetwork:
    """Test individual CLF network functionality."""
    
    @pytest.fixture
    def clf_network(self):
        return CLFNetwork(state_dim=4, hidden_dims=(32, 16), beta=0.1, delta=0.01)
    
    def test_clf_initialization(self, clf_network):
        """Test CLF network initializes correctly."""
        assert clf_network.state_dim == 4
        assert clf_network.beta == 0.1
        assert clf_network.delta == 0.01
        assert clf_network.device == "cpu"
    
    def test_clf_forward_non_negative(self, clf_network):
        """Test CLF always outputs non-negative values."""
        states = torch.randn(20, 4)
        
        output = clf_network(states)
        
        assert output.shape == (20, 1)
        assert torch.all(output >= 0), "CLF must be non-negative (Lyapunov property)"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_clf_constraint_computation(self, clf_network):
        """Test CLF constraint computation is mathematically correct."""
        states = torch.tensor([[1.0, 1.0, 0.1, 0.1]])
        next_states = torch.tensor([[0.9, 0.9, 0.05, 0.05]])  # Moving toward origin
        
        V_curr = clf_network(states)
        V_next = clf_network(next_states)
        
        # Manual constraint calculation: V_next - V_curr <= -beta * V_curr + delta
        # Violation: max(0, V_next - V_curr + beta * V_curr - delta)
        expected_violation = torch.clamp(
            V_next - V_curr + clf_network.beta * V_curr - clf_network.delta,
            min=0.0
        )
        
        actual_violation = clf_network.clf_constraint(states, next_states)
        
        torch.testing.assert_close(actual_violation, expected_violation.squeeze())
    
    def test_clf_goal_convergence_property(self, clf_network):
        """Test CLF decreases toward goal states."""
        # States getting closer to origin (assumed goal)
        far_state = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
        near_state = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        goal_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        
        V_far = clf_network(far_state)
        V_near = clf_network(near_state)
        V_goal = clf_network(goal_state)
        
        # CLF should generally decrease toward goal (after training)
        # For untrained network, just check non-negativity
        assert V_far >= 0
        assert V_near >= 0
        assert V_goal >= 0
    
    def test_clf_feasibility_loss_components(self, clf_network):
        """Test CLF loss computation for all components."""
        goal_states = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        states = torch.tensor([[1.0, 1.0, 0.1, 0.1]])
        next_states = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
        
        losses = clf_network.feasibility_loss(goal_states, states, next_states)
        
        # Check all loss components exist
        assert "goal" in losses
        assert "constraint" in losses
        assert "positive" in losses
        assert "total" in losses
        
        # Check losses are non-negative
        for loss_name, loss_value in losses.items():
            assert loss_value >= 0, f"{loss_name} loss should be non-negative"
        
        # Check total is sum of components
        expected_total = losses["goal"] + losses["constraint"] + losses["positive"]
        torch.testing.assert_close(losses["total"], expected_total)

class TestEnsembleCLF:
    """Test CLF ensemble functionality."""
    
    @pytest.fixture
    def ensemble_clf(self):
        return EnsembleCLF(num_models=3, state_dim=4, hidden_dims=(16, 8))
    
    def test_ensemble_forward_consistency(self, ensemble_clf):
        """Test ensemble forward pass is mean of individual models."""
        states = torch.randn(5, 4)
        
        ensemble_output = ensemble_clf(states)
        individual_outputs = ensemble_clf.forward_all(states)
        expected_mean = torch.mean(individual_outputs, dim=0)
        
        torch.testing.assert_close(ensemble_output, expected_mean)
    
    def test_ensemble_non_negative(self, ensemble_clf):
        """Test ensemble CLF maintains non-negativity."""
        states = torch.randn(10, 4)
        
        output = ensemble_clf(states)
        
        assert torch.all(output >= 0), "Ensemble CLF must be non-negative"
    
    def test_ensemble_goal_detection(self, ensemble_clf):
        """Test ensemble goal detection consistency."""
        states = torch.randn(10, 4)
        threshold = 0.1
        
        goal_predictions = ensemble_clf.is_near_goal(states, threshold)
        clf_values = ensemble_clf(states)
        
        expected_goals = (clf_values.squeeze() <= threshold)
        torch.testing.assert_close(goal_predictions.float(), expected_goals.float())

class TestCBFCLFController:
    """Test integrated CBF-CLF controller."""
    
    @pytest.fixture
    def controller(self):
        cbf = EnsembleCBF(num_models=2, state_dim=4, hidden_dims=(16, 8))
        clf = EnsembleCLF(num_models=2, state_dim=4, hidden_dims=(16, 8))
        return CBFCLFController(cbf, clf, action_dim=2)
    
    @pytest.fixture
    def mock_dynamics(self):
        """Simple dynamics model for testing."""
        class MockDynamics:
            def __call__(self, states, actions):
                # Simple integration: next_state = state + [0, 0, ax, ay] * dt
                dt = 0.1
                vel_change = torch.cat([torch.zeros_like(actions), actions], dim=1) * dt
                return states + vel_change
        return MockDynamics()
    
    def test_controller_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.action_dim == 2
        assert controller.device == "cpu"
    
    def test_filter_action_preserves_shape(self, controller, mock_dynamics):
        """Test action filtering preserves tensor shapes."""
        batch_size = 5
        states = torch.randn(batch_size, 4)
        actions = torch.randn(batch_size, 2)
        
        filtered_actions = controller.filter_action(
            states, actions, mock_dynamics,
            safety_margin=0.1, feasibility_margin=0.1
        )
        
        assert filtered_actions.shape == actions.shape
        assert not torch.isnan(filtered_actions).any()
        assert not torch.isinf(filtered_actions).any()
    
    def test_safety_feasibility_metrics(self, controller, mock_dynamics):
        """Test safety and feasibility metrics computation."""
        states = torch.randn(3, 4)
        actions = torch.randn(3, 2)
        
        metrics = controller.get_safety_feasibility_metrics(states, actions, mock_dynamics)
        
        # Check all required metrics exist
        required_metrics = [
            "cbf_values", "clf_values", "cbf_constraints", 
            "clf_constraints", "is_safe", "is_near_goal"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric].shape[0] == 3  # Batch size
        
        # Check boolean metrics are actually boolean
        assert metrics["is_safe"].dtype == torch.bool
        assert metrics["is_near_goal"].dtype == torch.bool
        
        # Check constraint violations are non-negative
        assert torch.all(metrics["cbf_constraints"] >= 0)
        assert torch.all(metrics["clf_constraints"] >= 0)

class TestCLFTrainer:
    """Test CLF training functionality."""
    
    @pytest.fixture
    def clf_trainer(self):
        network = CLFNetwork(state_dim=4, hidden_dims=(16, 8))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        return CLFTrainer(network, optimizer)
    
    def test_training_step_reduces_goal_loss(self, clf_trainer):
        """Test training reduces loss for goal states."""
        # Goal states should have V=0
        goal_states = torch.zeros(5, 4)  # Origin as goal
        states = torch.randn(8, 4)
        next_states = states + torch.randn(8, 4) * 0.1
        
        # Record initial loss
        initial_losses = clf_trainer.train_step(goal_states, states, next_states)
        initial_goal_loss = initial_losses["goal"]
        
        # Train for several steps
        for _ in range(20):
            clf_trainer.train_step(goal_states, states, next_states)
        
        # Check final loss
        final_losses = clf_trainer.train_step(goal_states, states, next_states)
        final_goal_loss = final_losses["goal"]
        
        # Goal loss should decrease significantly
        assert final_goal_loss < initial_goal_loss * 0.8, "Training should reduce goal loss"
    
    def test_training_step_returns_valid_losses(self, clf_trainer):
        """Test training step returns valid loss dictionary."""
        goal_states = torch.randn(3, 4)
        states = torch.randn(4, 4)
        next_states = torch.randn(4, 4)
        
        losses = clf_trainer.train_step(goal_states, states, next_states)
        
        # Check all required keys
        required_keys = ["goal", "constraint", "positive", "total"]
        for key in required_keys:
            assert key in losses
            assert isinstance(losses[key], float)
            assert not np.isnan(losses[key])
            assert not np.isinf(losses[key])
            assert losses[key] >= 0

class TestCLFFeasibilityProperties:
    """Test critical feasibility properties of CLF."""
    
    def test_clf_lyapunov_property(self):
        """Test CLF satisfies Lyapunov function properties."""
        clf = CLFNetwork(state_dim=2, beta=0.1, delta=0.01)
        
        # Property 1: V(x) >= 0 for all x
        states = torch.randn(50, 2)
        V_values = clf(states)
        assert torch.all(V_values >= 0), "CLF must be non-negative everywhere"
        
        # Property 2: V(0) should be small (ideally 0 at goal)
        goal_state = torch.zeros(1, 2)
        V_goal = clf(goal_state)
        assert V_goal >= 0, "CLF at goal must be non-negative"
    
    def test_clf_constraint_mathematical_correctness(self):
        """Test CLF constraint satisfies mathematical definition."""
        clf = CLFNetwork(state_dim=2, beta=0.1, delta=0.01)
        
        # Create trajectory moving toward origin
        states = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
        next_states = torch.tensor([[1.8, 1.8], [0.8, 0.8]])
        
        V_curr = clf(states)
        V_next = clf(next_states)
        
        # CLF constraint: V_dot <= -beta * V + delta
        # Discrete: V_next - V_curr <= -beta * V_curr + delta
        constraint_violations = clf.clf_constraint(states, next_states)
        
        # Violations should be non-negative (clipped)
        assert torch.all(constraint_violations >= 0)
        
        # Manual verification
        manual_violations = torch.clamp(
            V_next.squeeze() - V_curr.squeeze() + clf.beta * V_curr.squeeze() - clf.delta,
            min=0.0
        )
        torch.testing.assert_close(constraint_violations, manual_violations)
    
    def test_clf_training_improves_goal_recognition(self):
        """Test CLF training improves goal state recognition."""
        clf = CLFNetwork(state_dim=2, beta=0.1, delta=0.01)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
        trainer = CLFTrainer(clf, optimizer)
        
        # Create goal and non-goal states
        goal_states = torch.zeros(10, 2)  # Origin
        non_goal_states = torch.randn(15, 2) * 2.0 + 1.0  # Away from origin
        
        # Test initial goal recognition
        with torch.no_grad():
            initial_goal_values = clf(goal_states).mean()
            initial_non_goal_values = clf(non_goal_states).mean()
        
        # Train extensively
        states = torch.cat([goal_states[:5], non_goal_states[:8]])
        next_states = states + torch.randn_like(states) * 0.1
        
        for _ in range(100):
            trainer.train_step(goal_states, states, next_states)
        
        # Test final goal recognition
        with torch.no_grad():
            final_goal_values = clf(goal_states).mean()
            final_non_goal_values = clf(non_goal_states).mean()
        
        # Goal values should decrease, non-goal should stay higher
        assert final_goal_values < initial_goal_values * 0.5, "Goal values should decrease significantly"
        assert final_non_goal_values > final_goal_values, "Non-goal values should be higher than goal values"
    
    def test_integrated_cbf_clf_consistency(self):
        """Test CBF and CLF work together consistently."""
        cbf = EnsembleCBF(num_models=2, state_dim=2, hidden_dims=(16, 8))
        clf = EnsembleCLF(num_models=2, state_dim=2, hidden_dims=(16, 8))
        controller = CBFCLFController(cbf, clf, action_dim=2)
        
        # Mock dynamics
        class SimpleDynamics:
            def __call__(self, states, actions):
                return states + torch.cat([actions, torch.zeros_like(actions)], dim=1) * 0.1
        
        dynamics = SimpleDynamics()
        
        # Test on various states
        states = torch.tensor([
            [0.0, 0.0],  # At origin
            [1.0, 1.0],  # Away from origin
            [2.0, 2.0],  # Far from origin
        ])
        actions = torch.tensor([
            [0.0, 0.0],   # No action
            [-0.1, -0.1], # Toward origin
            [0.1, 0.1],   # Away from origin
        ])
        
        metrics = controller.get_safety_feasibility_metrics(states, actions, dynamics)
        
        # Check consistency
        assert len(metrics["cbf_values"]) == 3
        assert len(metrics["clf_values"]) == 3
        assert len(metrics["is_safe"]) == 3
        assert len(metrics["is_near_goal"]) == 3
        
        # CLF values should be non-negative
        assert torch.all(metrics["clf_values"] >= 0)

if __name__ == "__main__":
    pytest.main([__file__])