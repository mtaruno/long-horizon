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
    - h(s) >= 0 implies safe
    - h(s) < 0 implies unsafe
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
        self.alpha = alpha # TODO: check if this is needed
        self.device = device 
        # build network layers 
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()] 

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())    
            
        # Output layer (single value)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        self.to(device)

        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)
    
    # def cbf_constraint(
    #     self, 
    #     states: torch.Tensor, 
    #     next_states: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute CBF constraint: h(s_{k+1}) - h(s_k) ≥ -α h(s_k)
        
    #     Args:
    #         states: Current states [batch_size, state_dim]
    #         next_states: Next states [batch_size, state_dim]
            
    #     Returns:
    #         CBF constraint violations [batch_size]
    #     """
    #     h_curr = self.forward(states).squeeze(-1)
    #     h_next = self.forward(next_states).squeeze(-1)
        
    #     # CBF constraint: h_next - h_curr >= -alpha * h_curr
    #     # Violation: max(0, -alpha * h_curr - (h_next - h_curr))
    #     constraint_violation = torch.clamp(
    #         -self.alpha * h_curr - (h_next - h_curr), 
    #         min=0.0
    #     )
        
    #     return constraint_violation

    def hinge_loss(self, safe_states: torch.Tensor, unsafe_states: torch.Tensor) -> torch.Tensor:
        loss_safe = 0.0
        if len(safe_states) > 0:
            h_safe = self.forward(safe_states).squeeze()
            loss_safe = (F.relu(-h_safe) ** 2).mean()
        
        loss_unsafe = 0.0
        if len(unsafe_states) > 0:
            h_unsafe = self.forward(unsafe_states).squeeze()
            loss_unsafe = (F.relu(h_unsafe) ** 2).mean()    
        
        return loss_safe + loss_unsafe

    
    def safety_loss(
        self,
        safe_states: torch.Tensor,
        unsafe_states: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
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
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            CBFNetwork(state_dim, hidden_dims, activation, alpha, device)
            for _ in range(num_models)
        ])
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([model(states) for model in self.models], dim=0)
        return torch.mean(outputs, dim=0)
    

    def forward_all(self, states: torch.Tensor) -> torch.Tensor:

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


