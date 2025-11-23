import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

def create_mlp(
    input_dim: int, 
    output_dim: int, 
    hidden_dims: List[int], 
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module = nn.Identity
) -> nn.Module:
    """Helper function to create a Multi-Layer Perceptron."""
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim
    
    layers.append(nn.Linear(current_dim, output_dim))
    layers.append(output_activation())
    
    return nn.Sequential(*layers)


class CBFNetwork(nn.Module):
    """
    Neural Control Barrier Function (CBF) h_phi(s).
    Predicts the "safety value" of a state.
    Positive = safe, Negative = unsafe.
    """
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = create_mlp(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns h_phi(s)"""
        return self.net(s)

    def compute_loss_pretrain(self, s: torch.Tensor, h_star: torch.Tensor) -> torch.Tensor:
        """Computes the MSE anchoring loss."""
        h_phi = self.forward(s)
        loss = F.mse_loss(h_phi, h_star)
        return loss

    def compute_loss_constraint(self, 
                                s: torch.Tensor, 
                                s_next: torch.Tensor, 
                                h_star_next: torch.Tensor, 
                                config: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the constraint-based loss (Eq 5 from paper).
        We use h_star_next to identify safe/unsafe states.
        """
        cbf_config = config['train']
        alpha = cbf_config['cbf_alpha']
        
        # Identify safe and unsafe states based on ground truth
        # We use a small margin for robustness
        safe_mask = (h_star_next.squeeze() >= 0.0)
        unsafe_mask = (h_star_next.squeeze() < 0.0)
        
        h_phi = self.forward(s)
        h_phi_next = self.forward(s_next)
        
        # 1. Safe loss: L_safe = [max(0, -h(s))]^2 for s in S_safe
        # We apply this to s_next, as h_star_next is what we have
        h_phi_next_safe = h_phi_next[safe_mask]
        loss_safe = torch.mean(torch.relu(-h_phi_next_safe) ** 2) if h_phi_next_safe.numel() > 0 else 0.0
        
        # 2. Unsafe loss: L_unsafe = [max(0, h(s))]^2 for s in S_unsafe
        h_phi_next_unsafe = h_phi_next[unsafe_mask]
        loss_unsafe = torch.mean(torch.relu(h_phi_next_unsafe) ** 2) if h_phi_next_unsafe.numel() > 0 else 0.0
        # 3. Constraint loss: L_constraint
        # h(s_k+1) - h(s_k) >= -alpha * h(s_k)
        # -> h(s_k+1) - (1 - alpha) * h(s_k) >= 0
        # Violation is when: (1 - alpha) * h(s_k) - h(s_k+1) > 0
        constraint_violation = h_phi_next - (1 - alpha) * h_phi
        loss_constraint = torch.mean(torch.relu(-constraint_violation) ** 2) # Penalize when violation < 0

        # Weighted sum of losses
        loss = (cbf_config['cbf_w_safe'] * loss_safe +
                cbf_config['cbf_w_unsafe'] * loss_unsafe +
                cbf_config['cbf_w_constraint'] * loss_constraint)
        
        metrics = {
            'cbf_loss': loss.item(),
            'cbf_loss_safe': loss_safe if isinstance(loss_safe, float) else loss_safe.item(),
            'cbf_loss_unsafe': loss_unsafe if isinstance(loss_unsafe, float) else loss_unsafe.item(),
            'cbf_loss_constraint': loss_constraint.item(),
            'h_phi_mean': h_phi.mean().item()
        }
        return loss, metrics

class CLFNetwork(nn.Module):
    """
    Neural Control Lyapunov Function (CLF) V_psi(s, g).
    Predicts the "progress value" of a state relative to a goal.
    Near 0 = at goal, Positive = far from goal.
    """
    def __init__(self, state_dim: int, subgoal_dim: int, hidden_dims: List[int]):
        super().__init__()
        # Input is concatenated state and subgoal
        self.net = create_mlp(
            input_dim=state_dim + subgoal_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns V_psi(s, g)"""
        sg = torch.cat([s, g], dim=-1)
        # Output V(s) which should be >= 0
        # Using relu or softplus can enforce this
        return F.softplus(self.net(sg))

    def compute_loss_pretrain(self, s: torch.Tensor, g: torch.Tensor, v_star: torch.Tensor) -> torch.Tensor:
        """Computes the MSE anchoring loss."""
        v_psi = self.forward(s, g)
        loss = F.mse_loss(v_psi, v_star)
        return loss

    def compute_loss_constraint(self, 
                                s: torch.Tensor, 
                                s_next: torch.Tensor, 
                                g: torch.Tensor, 
                                v_star: torch.Tensor,
                                config: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the constraint-based loss (Eq 9 from paper).
        We use v_star to identify goal states.
        """
        clf_config = config['train']
        beta = clf_config['clf_beta']
        delta = clf_config['clf_delta']
        epsilon = clf_config['clf_epsilon']
        
        # Identify goal states based on ground truth
        goal_mask = (v_star.squeeze() <= epsilon)
        
        v_psi = self.forward(s, g)
        v_psi_next = self.forward(s_next, g)
        
        # 1. Goal loss: L_goal = V(g)^2 for g in G
        # We apply this to states that are in the goal region
        v_psi_at_goal = v_psi[goal_mask]
        loss_goal = torch.mean(v_psi_at_goal ** 2) if v_psi_at_goal.numel() > 0 else 0.0
        
        # 2. Constraint loss: L_constraint
        # V(s_k+1) - V(s_k) <= -beta * V(s_k) + delta
        # Violation is when: V(s_k+1) - (1 - beta) * V(s_k) - delta > 0
        # --- THIS IS THE FIX ---
        constraint_violation = v_psi_next - (1 - beta) * v_psi - delta  # <-- WAS + delta
        loss_constraint = torch.mean(torch.relu(constraint_violation) ** 2)

        # 3. Positivity loss: L_positive (optional, but good)
        # Enforce V(s) > 0 for non-goal states
        # We use a small margin 0.01 as in paper
        v_psi_non_goal = v_psi[~goal_mask]
        loss_positive = torch.mean(torch.relu(0.01 - v_psi_non_goal) ** 2) if v_psi_non_goal.numel() > 0 else 0.0
        
        # Weighted sum of losses
        loss = (clf_config['clf_w_goal'] * loss_goal +
                clf_config['clf_w_constraint'] * loss_constraint +
                clf_config['clf_w_positive'] * loss_positive)
        
        metrics = {
            'clf_loss': loss.item(),
            'clf_loss_goal': loss_goal if isinstance(loss_goal, float) else loss_goal.item(),
            'clf_loss_constraint': loss_constraint.item(),
            'clf_loss_positive': loss_positive if isinstance(loss_positive, float) else loss_positive.item(),
            'v_psi_mean': v_psi.mean().item()
        }
        return loss, metrics