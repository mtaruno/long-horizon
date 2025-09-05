"""
Control Barrier Function (CBF) implementation for safety constraints.

This module implements neural CBFs that ensure safety by maintaining
the robot in safe states S^+ and avoiding unsafe states S^-.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class CBFNetwork(nn.Module):
    """
    Neural Control Barrier Function.
    
    Learns h_φ: S → R such that:
    (i)   h_φ(s) ≥ 0 if s ∈ S^+  (safe states)
    (ii)  h_φ(s) < 0 if s ∈ S^-   (unsafe states)  
    (iii) h_φ(s_{k+1}) - h_φ(s_k) ≥ -α h_φ(s_k), α ∈ (0,1]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        activation: str = "relu",
        alpha: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize CBF network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
            alpha: CBF decay rate parameter
            device: Device to run on
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.alpha = alpha
        self.device = device
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.to(device)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CBF network.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            CBF values [batch_size, 1]
        """
        return self.network(states)
    
    def cbf_constraint(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CBF constraint: h(s_{k+1}) - h(s_k) ≥ -α h(s_k)
        
        Args:
            states: Current states [batch_size, state_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            CBF constraint violations [batch_size]
        """
        h_curr = self.forward(states).squeeze(-1)
        h_next = self.forward(next_states).squeeze(-1)
        
        # CBF constraint: h_next - h_curr >= -alpha * h_curr
        # Violation: max(0, -alpha * h_curr - (h_next - h_curr))
        constraint_violation = torch.clamp(
            -self.alpha * h_curr - (h_next - h_curr), 
            min=0.0
        )
        
        return constraint_violation
    
    def safety_loss(
        self,
        safe_states: torch.Tensor,
        unsafe_states: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CBF training loss.
        
        Args:
            safe_states: States known to be safe [batch_size, state_dim]
            unsafe_states: States known to be unsafe [batch_size, state_dim]
            states: Current states for dynamics [batch_size, state_dim]
            next_states: Next states for dynamics [batch_size, state_dim]
            weights: Loss component weights
            
        Returns:
            Dictionary of loss components
        """
        if weights is None:
            weights = {"safe": 1.0, "unsafe": 1.0, "constraint": 1.0}
        
        losses = {}
        
        # Safe state loss: h(s) >= 0 for s in S^+
        if safe_states.size(0) > 0:
            h_safe = self.forward(safe_states).squeeze(-1)
            losses["safe"] = weights["safe"] * torch.mean(
                torch.clamp(-h_safe, min=0.0) ** 2
            )
        else:
            losses["safe"] = torch.tensor(0.0, device=self.device)
        
        # Unsafe state loss: h(s) < 0 for s in S^-
        if unsafe_states.size(0) > 0:
            h_unsafe = self.forward(unsafe_states).squeeze(-1)
            losses["unsafe"] = weights["unsafe"] * torch.mean(
                torch.clamp(h_unsafe, min=0.0) ** 2
            )
        else:
            losses["unsafe"] = torch.tensor(0.0, device=self.device)
        
        # CBF constraint loss
        constraint_violations = self.cbf_constraint(states, next_states)
        losses["constraint"] = weights["constraint"] * torch.mean(
            constraint_violations ** 2
        )
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def is_safe(self, states: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Check if states are safe according to CBF.
        
        Args:
            states: States to check [batch_size, state_dim]
            threshold: Safety threshold
            
        Returns:
            Boolean tensor indicating safety [batch_size]
        """
        with torch.no_grad():
            h_values = self.forward(states).squeeze(-1)
            return h_values >= threshold


class EnsembleCBF(nn.Module):
    """
    Ensemble of CBF networks for improved robustness.
    """
    
    def __init__(
        self,
        num_models: int,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        activation: str = "relu",
        alpha: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize ensemble CBF.
        
        Args:
            num_models: Number of CBF models in ensemble
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            alpha: CBF decay rate parameter
            device: Device to run on
        """
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            CBFNetwork(state_dim, hidden_dims, activation, alpha, device)
            for _ in range(num_models)
        ])
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (returns mean).
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Mean CBF values [batch_size, 1]
        """
        outputs = torch.stack([model(states) for model in self.models], dim=0)
        return torch.mean(outputs, dim=0)
    
    def forward_all(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            All CBF values [num_models, batch_size, 1]
        """
        return torch.stack([model(states) for model in self.models], dim=0)
    
    def uncertainty(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction uncertainty (standard deviation).
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Uncertainty values [batch_size, 1]
        """
        outputs = self.forward_all(states)  # [num_models, batch_size, 1]
        return torch.std(outputs, dim=0)
    
    def cbf_constraint(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CBF constraint using ensemble mean.
        
        Args:
            states: Current states [batch_size, state_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            CBF constraint violations [batch_size]
        """
        h_curr = self.forward(states).squeeze(-1)
        h_next = self.forward(next_states).squeeze(-1)
        
        # Use alpha from first model (assuming all models have same alpha)
        alpha = self.models[0].alpha
        
        # CBF constraint: h_next - h_curr >= -alpha * h_curr
        constraint_violation = torch.clamp(
            -alpha * h_curr - (h_next - h_curr), 
            min=0.0
        )
        
        return constraint_violation
    
    def is_safe(self, states: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Check if states are safe according to ensemble CBF.
        
        Args:
            states: States to check [batch_size, state_dim]
            threshold: Safety threshold
            
        Returns:
            Boolean tensor indicating safety [batch_size]
        """
        with torch.no_grad():
            h_values = self.forward(states).squeeze(-1)
            return h_values >= threshold
    
    def conservative_safety(
        self, 
        states: torch.Tensor, 
        confidence: float = 0.95
    ) -> torch.Tensor:
        """
        Conservative safety check using ensemble statistics.
        
        Args:
            states: States to check [batch_size, state_dim]
            confidence: Confidence level for safety
            
        Returns:
            Conservative safety indicators [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward_all(states).squeeze(-1)  # [num_models, batch_size]
            mean_h = torch.mean(outputs, dim=0)
            std_h = torch.std(outputs, dim=0)
            
            # Conservative estimate: mean - k * std >= 0
            # where k is chosen based on confidence level
            from scipy.stats import norm
            k = norm.ppf(confidence)
            conservative_h = mean_h - k * std_h
            
            return conservative_h >= 0.0


class CBFTrainer:
    """
    Trainer for CBF networks.
    """
    
    def __init__(
        self,
        cbf_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ):
        """
        Initialize CBF trainer.
        
        Args:
            cbf_network: CBF network to train
            optimizer: Optimizer for training
            device: Device to run on
        """
        self.cbf = cbf_network
        self.optimizer = optimizer
        self.device = device
        
    def train_step(
        self,
        safe_states: torch.Tensor,
        unsafe_states: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            safe_states: Safe states
            unsafe_states: Unsafe states  
            states: Current states
            next_states: Next states
            weights: Loss weights
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        losses = self.cbf.safety_loss(
            safe_states, unsafe_states, states, next_states, weights
        )
        
        losses["total"].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}
