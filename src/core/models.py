# src/core/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.core.critics import create_mlp

class DynamicsModel(nn.Module):
    """
    This is a single deterministic dynamics model.
    It's used as a component in the ensemble.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=state_dim, # Predicts s_k+1
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predicts s_k+1"""
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)

class EnsembleDynamicsModel(nn.Module):
    """
    A robust Ensemble of Dynamics Models.
    Trains multiple models and averages their predictions.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int], 
                 num_ensemble: int = 5):
        super().__init__()
        self.num_ensemble = num_ensemble
        
        # Create a list of individual models
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dims) 
            for _ in range(num_ensemble)
        ])

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next state by averaging the predictions
        of all models in the ensemble.
        """
        # s: (batch_size, state_dim), a: (batch_size, action_dim)
        
        # Get predictions from all models
        # This creates a list of (batch_size, state_dim) tensors
        predictions = [model(s, a) for model in self.models]
        
        # Stack them into (num_ensemble, batch_size, state_dim)
        stacked_predictions = torch.stack(predictions, dim=0)
        
        # Average across the ensemble dimension (dim=0)
        # Output shape is (batch_size, state_dim)
        return torch.mean(stacked_predictions, dim=0)
    
    def compute_loss(self, 
                     s: torch.Tensor, 
                     a: torch.Tensor, 
                     s_next: torch.Tensor,
                     optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Computes the loss for all models and updates them.
        This uses "bootstrapping" - each model trains on a
        different random 70% subset of the batch.
        """
        batch_size = s.shape[0]
        total_loss = 0.0
        
        for model in self.models:
            # Create a bootstrap sample (sample with replacement)
            # Or simpler: just use a random 70% of the data
            indices = torch.randint(0, batch_size, (int(batch_size * 0.7),), device=s.device)
            s_sample, a_sample, s_next_sample = s[indices], a[indices], s_next[indices]
            
            # Predict and calculate loss for this model
            s_next_pred = model(s_sample, a_sample)
            loss = F.mse_loss(s_next_pred, s_next_sample)
            
            # Update this model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        metrics = {
            'dynamics_loss_mean': total_loss / self.num_ensemble
        }
        
        return total_loss, metrics # We return total_loss for logging, but updates are done