"""
Control Lyapunov Function (CLF) implementation for feasibility constraints.

This module implements neural CLFs that ensure feasibility by guaranteeing
convergence to goal states G ⊆ S.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class CLFNetwork(nn.Module):
    """
    Neural Control Lyapunov Function V(s)
    
    Learns V_ψ: S → R≥0 such that:
    (i)  V_ψ(s) = 0 if s ∈ G  (goal states)
    (ii) V_ψ(s_{k+1}) - V_ψ(s_k) ≤ -β V_ψ(s_k) + δ, β > 0

    
    - V(s) ≈ 0 implies at goal
    - V(s) > 0 implies not at goal
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        activation: str = "relu",
        beta: float = 0.1,
        delta: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize CLF network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
            beta: CLF convergence rate parameter
            delta: CLF tolerance parameter
            device: Device to run on
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.beta = beta
        self.delta = delta
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
            
        # Output layer with softplus to ensure non-negativity
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Softplus()  # Ensures V(s) ≥ 0
        ])
        
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
        Forward pass through CLF network.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            CLF values [batch_size, 1] (non-negative)
        """
        return self.network(states)
    
    def clf_constraint(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLF constraint: V(s_{k+1}) - V(s_k) ≤ -β V(s_k) + δ
        
        Args:
            states: Current states [batch_size, state_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            CLF constraint violations [batch_size]
        """
        V_curr = self.forward(states).squeeze(-1)
        V_next = self.forward(next_states).squeeze(-1)
        
        # CLF constraint: V_next - V_curr <= -beta * V_curr + delta
        # Violation: max(0, V_next - V_curr + beta * V_curr - delta)
        constraint_violation = torch.clamp(
            V_next - V_curr + self.beta * V_curr - self.delta,
            min=0.0
        )
        
        return constraint_violation
    
    def feasibility_loss(
        self,
        goal_states: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLF training loss.
        
        Args:
            goal_states: States known to be goals [batch_size, state_dim]
            states: Current states for dynamics [batch_size, state_dim]
            next_states: Next states for dynamics [batch_size, state_dim]
            weights: Loss component weights
            
        Returns:
            Dictionary of loss components
        """
        if weights is None:
            weights = {"goal": 1.0, "constraint": 1.0, "positive": 0.1}
        
        losses = {}
        
        # Goal state loss: V(s) = 0 for s in G
        if goal_states.size(0) > 0:
            V_goal = self.forward(goal_states).squeeze(-1)
            losses["goal"] = weights["goal"] * torch.mean(V_goal ** 2)
        else:
            losses["goal"] = torch.tensor(0.0, device=self.device)
        
        # CLF constraint loss
        constraint_violations = self.clf_constraint(states, next_states)
        losses["constraint"] = weights["constraint"] * torch.mean(
            constraint_violations ** 2
        )
        
        # Positive definiteness regularization (optional)
        # Encourage V(s) > 0 for non-goal states
        V_curr = self.forward(states).squeeze(-1)
        losses["positive"] = weights["positive"] * torch.mean(
            torch.clamp(-V_curr + 0.01, min=0.0) ** 2
        )
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def distance_to_goal(self, states: torch.Tensor) -> torch.Tensor:
        """
        Estimate distance to goal using CLF values.
        
        Args:
            states: States to evaluate [batch_size, state_dim]
            
        Returns:
            Estimated distances [batch_size]
        """
        with torch.no_grad():
            V_values = self.forward(states).squeeze(-1)
            return V_values
    
    def is_near_goal(
        self, 
        states: torch.Tensor, 
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Check if states are near goal according to CLF.
        
        Args:
            states: States to check [batch_size, state_dim]
            threshold: Goal proximity threshold
            
        Returns:
            Boolean tensor indicating proximity to goal [batch_size]
        """
        with torch.no_grad():
            V_values = self.forward(states).squeeze(-1)
            return V_values <= threshold


class EnsembleCLF(nn.Module):
    """
    Ensemble of CLF networks for improved robustness.
    """
    
    def __init__(
        self,
        num_models: int,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        activation: str = "relu",
        beta: float = 0.1,
        delta: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize ensemble CLF.
        
        Args:
            num_models: Number of CLF models in ensemble
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            beta: CLF convergence rate parameter
            delta: CLF tolerance parameter
            device: Device to run on
        """
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            CLFNetwork(state_dim, hidden_dims, activation, beta, delta, device)
            for _ in range(num_models)
        ])
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (returns mean).
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Mean CLF values [batch_size, 1]
        """
        outputs = torch.stack([model(states) for model in self.models], dim=0)
        return torch.mean(outputs, dim=0)
    
    def forward_all(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            All CLF values [num_models, batch_size, 1]
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
    
    def clf_constraint(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLF constraint using ensemble mean.
        
        Args:
            states: Current states [batch_size, state_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            CLF constraint violations [batch_size]
        """
        V_curr = self.forward(states).squeeze(-1)
        V_next = self.forward(next_states).squeeze(-1)
        
        # Use beta and delta from first model (assuming all models have same parameters)
        beta = self.models[0].beta
        delta = self.models[0].delta
        
        # CLF constraint: V_next - V_curr <= -beta * V_curr + delta
        constraint_violation = torch.clamp(
            V_next - V_curr + beta * V_curr - delta,
            min=0.0
        )
        
        return constraint_violation
    
    def is_near_goal(
        self, 
        states: torch.Tensor, 
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Check if states are near goal according to ensemble CLF.
        
        Args:
            states: States to check [batch_size, state_dim]
            threshold: Goal proximity threshold
            
        Returns:
            Boolean tensor indicating proximity to goal [batch_size]
        """
        with torch.no_grad():
            V_values = self.forward(states).squeeze(-1)
            return V_values <= threshold
    
    def conservative_feasibility(
        self, 
        states: torch.Tensor, 
        confidence: float = 0.95,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Conservative feasibility check using ensemble statistics.
        
        Args:
            states: States to check [batch_size, state_dim]
            confidence: Confidence level for feasibility
            threshold: Goal proximity threshold
            
        Returns:
            Conservative feasibility indicators [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward_all(states).squeeze(-1)  # [num_models, batch_size]
            mean_V = torch.mean(outputs, dim=0)
            std_V = torch.std(outputs, dim=0)
            
            # Conservative estimate: mean + k * std <= threshold
            # where k is chosen based on confidence level
            from scipy.stats import norm
            k = norm.ppf(confidence)
            conservative_V = mean_V + k * std_V
            
            return conservative_V <= threshold


class CLFTrainer:
    """
    Trainer for CLF networks.
    """
    
    def __init__(
        self,
        clf_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ):
        """
        Initialize CLF trainer.
        
        Args:
            clf_network: CLF network to train
            optimizer: Optimizer for training
            device: Device to run on
        """
        self.clf = clf_network
        self.optimizer = optimizer
        self.device = device
        
    def train_step(
        self,
        goal_states: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            goal_states: Goal states
            states: Current states
            next_states: Next states
            weights: Loss weights
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        losses = self.clf.feasibility_loss(
            goal_states, states, next_states, weights
        )
        
        losses["total"].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}

   
    def get_safety_feasibility_metrics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        dynamics_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Get safety and feasibility metrics for state-action pairs.
        
        Args:
            states: States [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            dynamics_model: Dynamics model
            
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            next_states = dynamics_model(states, actions)
            
            metrics = {
                "cbf_values": self.cbf(states),
                "clf_values": self.clf(states),
                "cbf_constraints": self.cbf.cbf_constraint(states, next_states),
                "clf_constraints": self.clf.clf_constraint(states, next_states),
                "is_safe": self.cbf.is_safe(states),
                "is_near_goal": self.clf.is_near_goal(states)
            }
            
        return metrics

if __name__ == "__main__":
    # Example usage
    state_dim = 4
    action_dim = 2

    clf = CLFNetwork(state_dim)
    print(clf)
    print("CLF output shape:", clf(torch.ones(1, state_dim)).shape)