import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from src.core.critics import create_mlp, CBFNetwork, CLFNetwork

class SubgoalConditionedPolicy(nn.Module):
    """
    The Actor (policy) network pi_theta(s, g).
    Outputs a deterministic action.
    """
    def __init__(self, 
                 state_dim: int, 
                 subgoal_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int],
                 a_max: float):
        super().__init__()
        self.a_max = a_max
        
        # Input is concatenated state and subgoal
        self.net = create_mlp(
            input_dim=state_dim + subgoal_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.ReLU,
            output_activation=nn.Tanh  # Scale output to [-1, 1]
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns action a_k"""
        sg = torch.cat([s, g], dim=-1)
        # Scale Tanh output from [-1, 1] to [-a_max, a_max]
        return self.a_max * self.net(sg)
        
    def compute_loss(self,
                     s: torch.Tensor,
                     g: torch.Tensor,
                     s_next_pred: torch.Tensor,
                     cbf_net: CBFNetwork,
                     clf_net: CLFNetwork,
                     config: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the actor loss (Corrected Version).
        The actor's job is to minimize the CLF (make progress)
        subject to the CBF constraint (stay safe).
        """
        train_config = config['train']
        
        # 1. CLF Loss (Task Progress)
        # The actor's job is to minimize the Lyapunov function of the next state.
        v_psi_next = clf_net(s_next_pred, g)
        loss_clf_progress = torch.mean(v_psi_next) # Make V(s_next) as small as possible
        
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

        # Total Loss
        loss = (loss_clf_progress + 
                train_config['lambda_cbf'] * penalty_cbf_constraint)
                
        metrics = {
            'actor_loss': loss.item(),
            'actor_loss_clf_progress': loss_clf_progress.item(),
            'actor_penalty_cbf_constraint': penalty_cbf_constraint.item(),
        }
        
        return loss, metrics