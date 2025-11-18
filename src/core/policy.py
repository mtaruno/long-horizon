import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.core.critics import create_mlp, CBFNetwork, CLFNetwork

class SubgoalConditionedPolicy(nn.Module):
    """
    The Actor (policy) network pi_theta(s, g).
    Outputs a deterministic action [a, omega],
    each scaled to [-1, 1] by a Tanh.
    """
    def __init__(self, 
                 state_dim: int, 
                 subgoal_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int],
                 a_max: float,
                 omega_max: float): 
        super().__init__()
        
        self.a_max = a_max
        self.omega_max = omega_max
        
        self.net = create_mlp(
            input_dim=state_dim + subgoal_dim,
            output_dim=action_dim, # action_dim is 2 ([a, omega])
            hidden_dims=hidden_dims,
            activation=nn.ReLU,
            output_activation=nn.Tanh  # Scale output to [-1, 1]
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns action a_k, scaled to [-1, 1]"""
        sg = torch.cat([s, g], dim=-1)
        return self.net(sg) # Output is already [-1, 1]
        
    def compute_loss(self,
                     s: torch.Tensor,
                     g: torch.Tensor,
                     s_next: torch.Tensor, # Can be real or predicted
                     cbf_net: CBFNetwork,
                     clf_net: CLFNetwork,
                     config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the actor loss (Corrected Version).
        The actor's job is to minimize *both* the CLF violation
        and the CBF violation.
        """
        train_config = config['train']
        
        # --- 1. CLF Penalty (Feasibility) ---
        # We penalize violation of the CLF *decrease* constraint.
        # This FORCES the agent to make progress.
        # Rule: V(s_next) - (1-beta)V(s) - delta <= 0
        v_psi = clf_net(s, g).detach() # Get current value, no gradient
        v_psi_next = clf_net(s_next, g)
        
        beta = train_config['clf_beta']
        delta = train_config['clf_delta']
        
        clf_violation = v_psi_next - (1 - beta) * v_psi - delta
        penalty_clf_constraint = torch.mean(torch.relu(clf_violation) ** 2)

        # --- 2. CBF Penalty (Safety) ---
        # We penalize violation of the CBF *safety* constraint.
        # Rule: h(s_next) - (1 - alpha) * h(s) >= 0
        h_phi = cbf_net(s).detach() # Get current safety, no gradient
        h_phi_next = cbf_net(s_next)
        alpha = train_config['cbf_alpha']

        cbf_violation = h_phi_next - (1 - alpha) * h_phi
        penalty_cbf_constraint = torch.mean(torch.relu(-cbf_violation) ** 2)

        # --- 3. Total Loss ---
        # The actor's job is to minimize *both* violations.
        loss = (penalty_clf_constraint + 
                train_config['lambda_cbf'] * penalty_cbf_constraint)
                
        metrics = {
            'actor_loss': loss.item(),
            'actor_loss_clf_constraint': penalty_clf_constraint.item(),
            'actor_penalty_cbf_constraint': penalty_cbf_constraint.item(),
        }
        
        return loss, penalty_clf_constraint, penalty_cbf_constraint