"""
Subgoal-conditioned policy π_θ(s, g) for FSM-guided control.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Dict
from src.core.critics import create_mlp, CBFNetwork, CLFNetwork

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
        a_max: float, 
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        device: str = "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        self.device = device
        self.a_max = a_max
        
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
            action a_k: [batch_size, action_dim]
        """
        x = torch.cat([state, subgoal], dim=-1)
        return self.a_max * self.network(x)
    
    def compute_loss(self,
                     states: torch.Tensor,
                     subgoals: torch.Tensor,
                     cbf_net: CBFNetwork
                     clf_net: CLFNetwork,
                     config: Dict[str]
                     ) -> torch.Tensor:
        train_config = config['train']


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
        
    def compute_loss(
        self,
        s: torch.Tensor,
        g: torch.Tensor,
        s_next_pred: torch.Tensor,
        cbf_net: CBFNetwork,
        clf_net: CLFNetwork,
        config: Dict[str, Any],
    ) -> dict:
        """
        Single policy update step.

        Args:
            states: [batch_size, state_dim]
            subgoals: [batch_size, subgoal_dim]
            rewards: [batch_size] optional rewards for policy gradient
            next_states: [batch_size, state_dim] actual next states from environment
            use_model_free: If True, use actual next_states; if False, predict from dynamics

        Returns:
            Dictionary of losses
        """
        train_config = config['train']
        # 1. CLF Loss (Task Progress)
        # The actor's job is to minimize the Lyapunov function of the next state.
        v_psi_next = clf_net(s_next_pred, g)
        loss_clf_progress = torch.mean(v_psi_next) # Make V(s_next) as small as possible
        
        # --- THIS IS THE FINAL FIX ---
        # 2. CBF Penalty (Safety)
        # We penalize the *violation of the CBF constraint*, not just being unsafe.
        # This gives the agent a gradient to follow to "steer away" from danger.
        # Rule: h(s_next) - (1 - alpha) * h(s) >= 0
        # Violation: -( h(s_next) - (1 - alpha) * h(s) )
        
        h_phi = cbf_net(s).detach() # Get current safety, no gradient
        h_phi_next = cbf_net(s_next_pred)
        alpha = train_config['cbf_alpha']

        constraint_violation = h_phi_next - (1 - alpha) * h_phi
        penalty_cbf_constraint = torch.mean(torch.relu(-constraint_violation) ** 2)
        # --- END FIX ---

        # Total Loss
        loss = (loss_clf_progress + 
                train_config['lambda_cbf'] * penalty_cbf_constraint)
                
        metrics = {
            'actor_loss': loss.item(),
            'actor_loss_clf_progress': loss_clf_progress.item(),
            'actor_penalty_cbf_constraint': penalty_cbf_constraint.item(),
        }
        
        return loss, metrics