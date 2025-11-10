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
        subgoals: torch.Tensor,
        rewards: torch.Tensor = None,
        next_states: torch.Tensor = None,
        use_model_free: bool = True
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
        self.optimizer.zero_grad()

        # Get policy action
        actions = self.policy(states, subgoals)

        # Get next states: use actual if available (model-free), else predict (model-based)
        if use_model_free and next_states is not None:
            # MODEL-FREE: Use actual transitions from environment
            predicted_next = next_states
        else:
            # MODEL-BASED: Predict next state (has model bias)
            predicted_next = self.dynamics(states, actions)

        # Subgoal distance loss (drive toward subgoal)
        # Use position only for subgoal (first 2 dims), ignore velocity
        subgoal_loss = torch.mean((predicted_next[:, :2] - subgoals[:, :2]) ** 2) * 5.0

        # CBF penalty: Strict constraint (penalize h < 0)
        h_values = self.cbf(predicted_next).squeeze(-1)
        cbf_violation = torch.clamp(-h_values, min=0.0)  # Penalty when unsafe
        cbf_penalty = self.lambda_cbf * torch.mean(cbf_violation ** 2)

        # CLF penalty: λ_clf·[max(0, V(ŝ') - ε)]²
        V_values = self.clf(predicted_next).squeeze(-1)
        clf_penalty = self.lambda_clf * torch.mean(torch.clamp(V_values - self.epsilon, min=0.0) ** 2)

        # Reward-based loss: MUCH STRONGER WEIGHT
        reward_loss = 0.0
        if rewards is not None:
            # Normalize rewards to prevent scale issues
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            # Negative reward as loss (maximize reward = minimize negative reward)
            reward_loss = -torch.mean(normalized_rewards) * 2.0  # Increased from 0.01!

        # Total loss
        total_loss = subgoal_loss + cbf_penalty + clf_penalty + reward_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "subgoal": subgoal_loss.item(),
            "cbf_penalty": cbf_penalty.item(),
            "clf_penalty": clf_penalty.item(),
            "reward_loss": reward_loss if isinstance(reward_loss, float) else reward_loss.item()
        }
