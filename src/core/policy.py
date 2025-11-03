"""
Subgoal-conditioned policy π_θ(s, g) for FSM-guided control.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class SubgoalConditionedPolicy(nn.Module):
    """
    Actor π_θ(s, g) that outputs actions conditioned on state and subgoal.
    
    Used in: a_k = π_θ(s_k, g_k) where g_k is the subgoal from FSM transition.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        subgoal_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        device: str = "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        self.device = device
        
        input_dim = state_dim + subgoal_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions
        
        self.network = nn.Sequential(*layers)
        self.to(device)
        
    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: π_θ(s, g) → a
        
        Args:
            state: [batch_size, state_dim]
            subgoal: [batch_size, subgoal_dim]
            
        Returns:
            action: [batch_size, action_dim]
        """
        x = torch.cat([state, subgoal], dim=-1)
        return self.network(x)


class PolicyTrainer:
    """
    Train policy with certificate penalties:
    L_actor = ℓ(ŝ', g) + λ_cbf·[max(0, -h(ŝ'))]² + λ_clf·[max(0, V(ŝ') - ε)]²
    """
    
    def __init__(
        self,
        policy: SubgoalConditionedPolicy,
        dynamics: nn.Module,
        cbf: nn.Module,
        clf: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_cbf: float = 1.0,
        lambda_clf: float = 1.0,
        epsilon: float = 0.1,
        device: str = "cpu"
    ):
        self.policy = policy
        self.dynamics = dynamics
        self.cbf = cbf
        self.clf = clf
        self.optimizer = optimizer
        self.lambda_cbf = lambda_cbf
        self.lambda_clf = lambda_clf
        self.epsilon = epsilon
        self.device = device
        
    def train_step(
        self,
        states: torch.Tensor,
        subgoals: torch.Tensor
    ) -> dict:
        """
        Single policy update step.
        
        Args:
            states: [batch_size, state_dim]
            subgoals: [batch_size, subgoal_dim]
            
        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad()
        
        # Get policy action
        actions = self.policy(states, subgoals)
        
        # Predict next state
        next_states = self.dynamics(states, actions)
        
        # Subgoal distance loss
        subgoal_loss = torch.mean((next_states[:, :self.policy.subgoal_dim] - subgoals) ** 2)
        
        # CBF penalty: λ_cbf·[max(0, -h(ŝ'))]²
        h_values = self.cbf(next_states).squeeze(-1)
        cbf_penalty = self.lambda_cbf * torch.mean(torch.clamp(-h_values, min=0.0) ** 2)
        
        # CLF penalty: λ_clf·[max(0, V(ŝ') - ε)]²
        V_values = self.clf(next_states).squeeze(-1)
        clf_penalty = self.lambda_clf * torch.mean(torch.clamp(V_values - self.epsilon, min=0.0) ** 2)
        
        # Total loss
        total_loss = subgoal_loss + cbf_penalty + clf_penalty
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total": total_loss.item(),
            "subgoal": subgoal_loss.item(),
            "cbf_penalty": cbf_penalty.item(),
            "clf_penalty": clf_penalty.item()
        }
