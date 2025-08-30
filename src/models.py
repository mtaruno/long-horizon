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
from collections import deque
import random


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


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transition data.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def push(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        reward: float = 0.0,
        done: bool = False
    ):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch of transitions.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary of batched tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([t[1] for t in batch])
        next_states = torch.FloatTensor([t[2] for t in batch])
        rewards = torch.FloatTensor([t[3] for t in batch])
        dones = torch.BoolTensor([t[4] for t in batch])
        
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "dones": dones
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class ModelTrainer:
    """
    Trainer for dynamics models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ):
        """
        Initialize model trainer.
        
        Args:
            model: Dynamics model to train
            optimizer: Optimizer for training
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        loss_fn: str = "mse"
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            next_states: True next states [batch_size, state_dim]
            loss_fn: Loss function ("mse", "huber")
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_next_states = self.model(states, actions)
        
        # Compute loss
        if loss_fn == "mse":
            loss = F.mse_loss(predicted_next_states, next_states)
        elif loss_fn == "huber":
            loss = F.huber_loss(predicted_next_states, next_states)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(predicted_next_states, next_states)
            max_error = torch.max(torch.abs(predicted_next_states - next_states))
            
        return {
            "loss": loss.item(),
            "mae": mae.item(),
            "max_error": max_error.item()
        }
    
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            next_states: True next states [batch_size, state_dim]
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            predicted_next_states = self.model(states, actions)
            
            mse = F.mse_loss(predicted_next_states, next_states)
            mae = F.l1_loss(predicted_next_states, next_states)
            max_error = torch.max(torch.abs(predicted_next_states - next_states))
            
            # R² score
            ss_res = torch.sum((next_states - predicted_next_states) ** 2)
            ss_tot = torch.sum((next_states - torch.mean(next_states, dim=0)) ** 2)
            r2_score = 1 - ss_res / ss_tot
            
        self.model.train()
        
        return {
            "mse": mse.item(),
            "mae": mae.item(),
            "max_error": max_error.item(),
            "r2_score": r2_score.item()
        }


class AdaptiveModelLearner:
    """
    Adaptive model learner with online updates and uncertainty estimation.
    """
    
    def __init__(
        self,
        ensemble_model: EnsembleDynamics,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        update_freq: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize adaptive model learner.
        
        Args:
            ensemble_model: Ensemble dynamics model
            replay_buffer: Replay buffer for data storage
            batch_size: Training batch size
            update_freq: Frequency of model updates
            device: Device to run on
        """
        self.ensemble = ensemble_model
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.device = device
        
        # Create optimizers for each model in ensemble
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3)
            for model in self.ensemble.models
        ]
        
        self.trainers = [
            ModelTrainer(model, optimizer, device)
            for model, optimizer in zip(self.ensemble.models, self.optimizers)
        ]
        
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
        Add new data point and potentially update models.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer.push(state, action, next_state, reward, done)
        self.step_count += 1
        
        # Update models periodically
        if (self.step_count % self.update_freq == 0 and 
            len(self.buffer) >= self.batch_size):
            self.update_models()
    
    def update_models(self, num_updates: int = 10) -> Dict[str, List[float]]:
        """
        Update ensemble models with recent data.
        
        Args:
            num_updates: Number of gradient updates per model
            
        Returns:
            Training metrics for each model
        """
        metrics = {f"model_{i}": [] for i in range(self.ensemble.num_models)}
        
        for _ in range(num_updates):
            # Sample batch
            batch = self.buffer.sample(self.batch_size)
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            next_states = batch["next_states"].to(self.device)
            
            # Train each model in ensemble
            for i, trainer in enumerate(self.trainers):
                model_metrics = trainer.train_step(states, actions, next_states)
                metrics[f"model_{i}"].append(model_metrics)
        
        # Average metrics across updates
        for key in metrics:
            if metrics[key]:
                avg_metrics = {}
                for metric_name in metrics[key][0].keys():
                    avg_metrics[metric_name] = np.mean([
                        m[metric_name] for m in metrics[key]
                    ])
                metrics[key] = avg_metrics
        
        return metrics
    
    def get_model_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get model uncertainty estimates.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Dictionary of uncertainty measures
        """
        with torch.no_grad():
            # Epistemic uncertainty (model disagreement)
            epistemic_uncertainty = self.ensemble.uncertainty(states, actions)
            
            # Mean prediction
            mean_prediction = self.ensemble(states, actions)
            
            # Individual model predictions
            all_predictions = self.ensemble.forward_all(states, actions)
            
        return {
            "epistemic_uncertainty": epistemic_uncertainty,
            "mean_prediction": mean_prediction,
            "all_predictions": all_predictions,
            "prediction_std": torch.std(all_predictions, dim=0)
        }
            done: Whether episode ended
        """
        self.buffer.append({
            'state': state.copy(),
            'action': action.copy(), 
            'next_state': next_state.copy(),
            'reward': reward,
            'done': done
        })
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary of batched tensors
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': torch.FloatTensor([t['state'] for t in batch]),
            'actions': torch.FloatTensor([t['action'] for t in batch]),
            'next_states': torch.FloatTensor([t['next_state'] for t in batch]),
            'rewards': torch.FloatTensor([t['reward'] for t in batch]),
            'dones': torch.BoolTensor([t['done'] for t in batch])
        }
        
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


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