"""
Model learning for dynamics approximation.

This module implements neural networks to learn approximate dynamics models
P̂: S × A → S for use with CBF and CLF constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from .buffer import ReplayBuffer


class DynamicsNetwork(nn.Module):
    """
    Neural network for learning dynamics model P̂(s_{k+1} | s_k, a_k).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 256),
        activation: str = "relu",
        output_activation: Optional[str] = None,
        residual: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize dynamics network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            output_activation: Output activation (None for linear)
            residual: Whether to learn residual dynamics (s_{k+1} - s_k)
            device: Device to run on
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.residual = residual
        self.device = device
        
        # Input is concatenated state and action
        input_dim = state_dim + action_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, state_dim))
        
        if output_activation is not None:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        self.to(device)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
        
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through dynamics network.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Predicted next states [batch_size, state_dim]
        """
        # Concatenate state and action
        inputs = torch.cat([states, actions], dim=-1)
        
        # Forward pass
        output = self.network(inputs)
        
        # If residual, add to current state
        if self.residual:
            output = states + output
            
        return output
    
    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Predict trajectory given initial state and action sequence.
        
        Args:
            initial_state: Initial state [state_dim]
            actions: Action sequence [horizon, action_dim]
            horizon: Prediction horizon
            
        Returns:
            Predicted trajectory [horizon+1, state_dim]
        """
        trajectory = [initial_state.unsqueeze(0)]
        current_state = initial_state.unsqueeze(0)
        
        with torch.no_grad():
            for t in range(horizon):
                next_state = self.forward(current_state, actions[t:t+1])
                trajectory.append(next_state)
                current_state = next_state
                
        return torch.cat(trajectory, dim=0)


class EnsembleDynamics(nn.Module):
    """
    Ensemble of dynamics models for uncertainty quantification.
    """
    
    def __init__(
        self,
        num_models: int,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 256),
        activation: str = "relu",
        residual: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize ensemble dynamics.
        
        Args:
            num_models: Number of models in ensemble
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            residual: Whether to learn residual dynamics
            device: Device to run on
        """
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            DynamicsNetwork(
                state_dim, action_dim, hidden_dims, 
                activation, residual=residual, device=device
            )
            for _ in range(num_models)
        ])
        
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through ensemble (returns mean prediction).
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Mean predicted next states [batch_size, state_dim]
        """
        predictions = torch.stack([
            model(states, actions) for model in self.models
        ], dim=0)
        return torch.mean(predictions, dim=0)
    
    def forward_all(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through all models.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            All predictions [num_models, batch_size, state_dim]
        """
        return torch.stack([
            model(states, actions) for model in self.models
        ], dim=0)
    
    def uncertainty(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction uncertainty (standard deviation).
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Uncertainty [batch_size, state_dim]
        """
        predictions = self.forward_all(states, actions)
        return torch.std(predictions, dim=0)
    
    def sample_prediction(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample predictions from ensemble.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            num_samples: Number of samples per input
            
        Returns:
            Sampled predictions [num_samples, batch_size, state_dim]
        """
        batch_size = states.size(0)
        samples = []
        
        for _ in range(num_samples):
            # Randomly select model for each batch element
            model_indices = torch.randint(
                0, self.num_models, (batch_size,), device=states.device
            )
            
            sample = torch.zeros_like(states)
            for i, model_idx in enumerate(model_indices):
                sample[i] = self.models[model_idx](
                    states[i:i+1], actions[i:i+1]
                ).squeeze(0)
            
            samples.append(sample)
            
        return torch.stack(samples, dim=0)


class AdaptiveModelLearner:
    """
    Adaptive model learning with uncertainty quantification.
    """
    
    def __init__(
        self,
        dynamics_ensemble: EnsembleDynamics,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        update_freq: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize adaptive model learner.
        
        Args:
            dynamics_ensemble: Ensemble dynamics model
            replay_buffer: Replay buffer for data
            batch_size: Training batch size
            update_freq: Model update frequency
            device: Device to run on
        """
        self.dynamics = dynamics_ensemble
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.device = device
        
        # Optimizer for dynamics models
        self.optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=1e-3)
        
        self.step_count = 0
        
    def add_data(
        self,
        state: np.ndarray,
        action: np.ndarray, 
        next_state: np.ndarray,
        reward: float = 0.0,
        done: bool = False
    ):
        """
        Add data and update model if needed.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer.push(state, action, next_state, reward, done)
        self.step_count += 1
        
        if (self.step_count % self.update_freq == 0 and 
            len(self.buffer) >= self.batch_size):
            self.update_model()
            
    def update_model(self) -> Dict[str, float]:
        """
        Update dynamics model.
        
        Returns:
            Dictionary of training metrics
        """
        batch = self.buffer.sample(self.batch_size)
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass through all models
        predictions = self.dynamics.forward_all(states, actions)
        
        # Compute loss for each model
        losses = []
        for i in range(self.dynamics.num_models):
            loss = F.mse_loss(predictions[i], next_states)
            losses.append(loss)
            
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'model_loss': total_loss.item(),
            'individual_losses': [l.item() for l in losses]
        }
        
    def get_model_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get model uncertainty estimates.
        
        Args:
            states: States [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Dictionary of uncertainty measures
        """
        with torch.no_grad():
            predictions = self.dynamics.forward_all(states, actions)
            
            # Epistemic uncertainty (model disagreement)
            epistemic = torch.std(predictions, dim=0)
            
            # Mean prediction
            mean_pred = torch.mean(predictions, dim=0)
            
            return {
                'epistemic_uncertainty': epistemic,
                'mean_prediction': mean_pred,
                'all_predictions': predictions
            }