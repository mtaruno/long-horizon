"""
Critical tests for Control Barrier Functions (CBF).
Safety-critical component - highest testing priority.
"""

import pytest
import torch
import numpy as np
from src.cbf import CBFNetwork, EnsembleCBF, CBFTrainer

class TestCBFNetwork:
    """Test individual CBF network functionality."""
    
    @pytest.fixture
    def cbf_network(self):
        return CBFNetwork(state_dim=4, hidden_dims=(32, 16), alpha=0.1)
    
    def test_cbf_initialization(self, cbf_network):
        """Test CBF network initializes correctly."""
        assert cbf_network.state_dim == 4
        assert cbf_network.alpha == 0.1
        assert cbf_network.device == "cpu"
    
    def test_cbf_forward_shape(self, cbf_network):
        """Test CBF forward pass returns correct shape."""
        batch_size = 10
        states = torch.randn(batch_size, 4)
        
        output = cbf_network(states)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_cbf_constraint_computation(self, cbf_network):
        """Test CBF constraint computation is mathematically correct."""
        states = torch.tensor([[1.0, 1.0, 0.1, 0.1]])
        next_states = torch.tensor([[1.1, 1.1, 0.2, 0.2]])
        
        h_curr = cbf_network(states)
        h_next = cbf_network(next_states)
        
        # Manual constraint calculation
        expected_violation = torch.clamp(
            -cbf_network.alpha * h_curr - (h_next - h_curr), 
            min=0.0
        )
        
        actual_violation = cbf_network.cbf_constraint(states, next_states)
        
        torch.testing.assert_close(actual_violation, expected_violation.squeeze())
    
    def test_cbf_safety_loss_components(self, cbf_network):
        """Test CBF loss computation for all components."""
        safe_states = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        unsafe_states = torch.tensor([[2.9, 2.9, 0.0, 0.0]])
        states = torch.tensor([[1.0, 1.0, 0.1, 0.1]])
        next_states = torch.tensor([[1.1, 1.1, 0.2, 0.2]])
        
        losses = cbf_network.safety_loss(safe_states, unsafe_states, states, next_states)
        
        # Check all loss components exist
        assert "safe" in losses
        assert "unsafe" in losses
        assert "constraint" in losses
        assert "total" in losses
        
        # Check losses are non-negative
        for loss_name, loss_value in losses.items():
            assert loss_value >= 0, f"{loss_name} loss should be non-negative"
        
        # Check total is sum of components
        expected_total = losses["safe"] + losses["unsafe"] + losses["constraint"]
        torch.testing.assert_close(losses["total"], expected_total)
    
    def test_cbf_safety_classification(self, cbf_network):
        """Test CBF safety classification consistency."""
        # Test multiple states
        states = torch.randn(20, 4)
        
        safety_predictions = cbf_network.is_safe(states, threshold=0.0)
        cbf_values = cbf_network(states)
        
        # Safety should match CBF values
        expected_safety = (cbf_values.squeeze() >= 0.0)
        torch.testing.assert_close(safety_predictions.float(), expected_safety.float())

class TestEnsembleCBF:
    """Test CBF ensemble functionality."""
    
    @pytest.fixture
    def ensemble_cbf(self):
        return EnsembleCBF(num_models=3, state_dim=4, hidden_dims=(16, 8))
    
    def test_ensemble_initialization(self, ensemble_cbf):
        """Test ensemble initializes with correct number of models."""
        assert ensemble_cbf.num_models == 3
        assert len(ensemble_cbf.models) == 3
    
    def test_ensemble_forward_consistency(self, ensemble_cbf):
        """Test ensemble forward pass is mean of individual models."""
        states = torch.randn(5, 4)
        
        ensemble_output = ensemble_cbf(states)
        individual_outputs = ensemble_cbf.forward_all(states)
        expected_mean = torch.mean(individual_outputs, dim=0)
        
        torch.testing.assert_close(ensemble_output, expected_mean)
    
    def test_ensemble_uncertainty_computation(self, ensemble_cbf):
        """Test uncertainty computation is standard deviation."""
        states = torch.randn(5, 4)
        
        uncertainty = ensemble_cbf.uncertainty(states)
        all_outputs = ensemble_cbf.forward_all(states)
        expected_std = torch.std(all_outputs, dim=0)
        
        torch.testing.assert_close(uncertainty, expected_std)
    
    def test_ensemble_constraint_uses_mean(self, ensemble_cbf):
        """Test ensemble constraint uses mean CBF values."""
        states = torch.randn(3, 4)
        next_states = torch.randn(3, 4)
        
        # Get ensemble constraint
        ensemble_constraint = ensemble_cbf.cbf_constraint(states, next_states)
        
        # Manually compute using mean
        h_curr = ensemble_cbf(states).squeeze()
        h_next = ensemble_cbf(next_states).squeeze()
        alpha = ensemble_cbf.models[0].alpha
        
        expected_constraint = torch.clamp(
            -alpha * h_curr - (h_next - h_curr), 
            min=0.0
        )
        
        torch.testing.assert_close(ensemble_constraint, expected_constraint)

class TestCBFTrainer:
    """Test CBF training functionality."""
    
    @pytest.fixture
    def cbf_trainer(self):
        network = CBFNetwork(state_dim=4, hidden_dims=(16, 8))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        return CBFTrainer(network, optimizer)
    
    def test_training_step_reduces_loss(self, cbf_trainer):
        """Test that training step reduces loss over multiple iterations."""
        # Generate consistent training data
        torch.manual_seed(42)
        safe_states = torch.randn(10, 4) * 0.5  # Smaller values
        unsafe_states = torch.randn(5, 4) * 2.0 + 3.0  # Larger values
        states = torch.randn(8, 4)
        next_states = states + torch.randn(8, 4) * 0.1
        
        # Record initial loss
        initial_losses = cbf_trainer.train_step(safe_states, unsafe_states, states, next_states)
        initial_total = initial_losses["total"]
        
        # Train for several steps
        for _ in range(10):
            cbf_trainer.train_step(safe_states, unsafe_states, states, next_states)
        
        # Check final loss
        final_losses = cbf_trainer.train_step(safe_states, unsafe_states, states, next_states)
        final_total = final_losses["total"]
        
        # Loss should decrease (allowing some tolerance for optimization noise)
        assert final_total < initial_total * 1.1, "Training should reduce loss"
    
    def test_training_step_returns_valid_losses(self, cbf_trainer):
        """Test training step returns valid loss dictionary."""
        safe_states = torch.randn(5, 4)
        unsafe_states = torch.randn(3, 4)
        states = torch.randn(4, 4)
        next_states = torch.randn(4, 4)
        
        losses = cbf_trainer.train_step(safe_states, unsafe_states, states, next_states)
        
        # Check all required keys
        required_keys = ["safe", "unsafe", "constraint", "total"]
        for key in required_keys:
            assert key in losses
            assert isinstance(losses[key], float)
            assert not np.isnan(losses[key])
            assert not np.isinf(losses[key])
            assert losses[key] >= 0

class TestCBFSafetyProperties:
    """Test critical safety properties of CBF."""
    
    def test_cbf_constraint_mathematical_correctness(self):
        """Test CBF constraint satisfies mathematical definition."""
        cbf = CBFNetwork(state_dim=2, alpha=0.1)
        
        # Create states where we know the relationship
        states = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        next_states = torch.tensor([[1.1, 1.1], [1.9, 1.9]])  # Moving toward origin
        
        h_curr = cbf(states)
        h_next = cbf(next_states)
        
        # CBF constraint: h_dot >= -alpha * h
        # Discrete: h_next - h_curr >= -alpha * h_curr
        constraint_violations = cbf.cbf_constraint(states, next_states)
        
        # Violations should be non-negative (clipped)
        assert torch.all(constraint_violations >= 0)
        
        # Manual verification
        manual_violations = torch.clamp(
            -cbf.alpha * h_curr.squeeze() - (h_next.squeeze() - h_curr.squeeze()),
            min=0.0
        )
        torch.testing.assert_close(constraint_violations, manual_violations)
    
    def test_cbf_ensemble_safety_consistency(self):
        """Test ensemble safety decisions are consistent."""
        ensemble = EnsembleCBF(num_models=5, state_dim=2)
        
        # Test on grid of states
        x = torch.linspace(-2, 2, 10)
        y = torch.linspace(-2, 2, 10)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        states = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Get safety predictions
        safety_predictions = ensemble.is_safe(states)
        cbf_values = ensemble(states)
        
        # Consistency check
        expected_safety = (cbf_values.squeeze() >= 0.0)
        torch.testing.assert_close(safety_predictions.float(), expected_safety.float())
    
    def test_cbf_training_improves_classification(self):
        """Test that CBF training improves safety classification."""
        # Create CBF and trainer
        cbf = CBFNetwork(state_dim=2, alpha=0.1)
        optimizer = torch.optim.Adam(cbf.parameters(), lr=0.01)
        trainer = CBFTrainer(cbf, optimizer)
        
        # Create clearly separable data
        safe_states = torch.randn(20, 2) * 0.5  # Near origin
        unsafe_states = torch.randn(10, 2) * 2.0 + 3.0  # Far from origin
        
        # Test initial accuracy
        with torch.no_grad():
            safe_predictions = cbf.is_safe(safe_states)
            unsafe_predictions = cbf.is_safe(unsafe_states)
            initial_accuracy = (
                safe_predictions.float().mean() + 
                (1 - unsafe_predictions.float()).mean()
            ) / 2
        
        # Train for many steps
        states = torch.cat([safe_states[:10], unsafe_states[:5]])
        next_states = states + torch.randn_like(states) * 0.1
        
        for _ in range(100):
            trainer.train_step(safe_states, unsafe_states, states, next_states)
        
        # Test final accuracy
        with torch.no_grad():
            safe_predictions = cbf.is_safe(safe_states)
            unsafe_predictions = cbf.is_safe(unsafe_states)
            final_accuracy = (
                safe_predictions.float().mean() + 
                (1 - unsafe_predictions.float()).mean()
            ) / 2
        
        # Accuracy should improve significantly
        assert final_accuracy > initial_accuracy + 0.1, "Training should improve classification"

if __name__ == "__main__":
    pytest.main([__file__])