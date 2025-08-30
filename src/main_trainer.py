"""
Main training framework integrating CBF, CLF, and dynamics learning.

This module implements the complete training loop for safe and feasible
long-horizon planning with neural CBFs and CLFs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass
import time
from pathlib import Path

from .cbf import CBFNetwork, EnsembleCBF, CBFTrainer
from .clf import CLFNetwork, EnsembleCLF, CLFTrainer, CBFCLFController
from .models import EnsembleDynamics, AdaptiveModelLearner, ReplayBuffer


@dataclass
class TrainingConfig:
    """Configuration for training framework."""
    
    # Model dimensions
    state_dim: int
    action_dim: int
    
    # Network architectures
    cbf_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    clf_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    dynamics_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    
    # Ensemble sizes
    num_cbf_models: int = 3
    num_clf_models: int = 3
    num_dynamics_models: int = 5
    
    # CBF/CLF parameters
    cbf_alpha: float = 0.1
    clf_beta: float = 0.1
    clf_delta: float = 0.01
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 1000
    
    # Update frequencies
    cbf_update_freq: int = 10
    clf_update_freq: int = 10
    model_update_freq: int = 5
    
    # Loss weights
    cbf_weights: Dict[str, float] = None
    clf_weights: Dict[str, float] = None
    
    # Buffer settings
    buffer_capacity: int = 100000
    
    # Device
    device: str = "cpu"
    
    # Logging
    log_freq: int = 100
    save_freq: int = 1000
    
    def __post_init__(self):
        if self.cbf_weights is None:
            self.cbf_weights = {"safe": 1.0, "unsafe": 1.0, "constraint": 1.0}
        if self.clf_weights is None:
            self.clf_weights = {"goal": 1.0, "constraint": 1.0, "positive": 0.1}


class SafeFeasibleTrainer:
    """
    Main trainer for safe and feasible long-horizon planning.
    
    Implements the training loop:
    1. Update model P̂
    2. Update CBF h_φ and CLF V_ψ  
    3. Perform RL based on results of 1 & 2
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize trainers
        self._init_trainers()
        
        # Initialize data storage
        self.replay_buffer = ReplayBuffer(
            config.buffer_capacity, config.state_dim, config.action_dim
        )
        
        # Initialize adaptive model learner
        self.model_learner = AdaptiveModelLearner(
            self.dynamics_ensemble,
            self.replay_buffer,
            config.batch_size,
            config.model_update_freq,
            config.device
        )
        
        # Initialize controller
        self.controller = CBFCLFController(
            self.cbf_ensemble,
            self.clf_ensemble,
            config.action_dim,
            config.device
        )
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.training_history = {
            "cbf_losses": [],
            "clf_losses": [],
            "model_losses": [],
            "safety_metrics": [],
            "feasibility_metrics": []
        }
        
        # Setup logging
        self._setup_logging()
        
    def _init_networks(self):
        """Initialize all neural networks."""
        config = self.config
        
        # CBF ensemble
        self.cbf_ensemble = EnsembleCBF(
            config.num_cbf_models,
            config.state_dim,
            config.cbf_hidden_dims,
            alpha=config.cbf_alpha,
            device=config.device
        )
        
        # CLF ensemble
        self.clf_ensemble = EnsembleCLF(
            config.num_clf_models,
            config.state_dim,
            config.clf_hidden_dims,
            beta=config.clf_beta,
            delta=config.clf_delta,
            device=config.device
        )
        
        # Dynamics ensemble
        self.dynamics_ensemble = EnsembleDynamics(
            config.num_dynamics_models,
            config.state_dim,
            config.action_dim,
            config.dynamics_hidden_dims,
            device=config.device
        )
        
    def _init_optimizers(self):
        """Initialize optimizers for all networks."""
        lr = self.config.learning_rate
        
        # CBF optimizers (one per model in ensemble)
        self.cbf_optimizers = [
            optim.Adam(model.parameters(), lr=lr)
            for model in self.cbf_ensemble.models
        ]
        
        # CLF optimizers (one per model in ensemble)
        self.clf_optimizers = [
            optim.Adam(model.parameters(), lr=lr)
            for model in self.clf_ensemble.models
        ]
        
    def _init_trainers(self):
        """Initialize trainers for CBF and CLF networks."""
        # CBF trainers
        self.cbf_trainers = [
            CBFTrainer(model, optimizer, self.device)
            for model, optimizer in zip(
                self.cbf_ensemble.models, self.cbf_optimizers
            )
        ]
        
        # CLF trainers
        self.clf_trainers = [
            CLFTrainer(model, optimizer, self.device)
            for model, optimizer in zip(
                self.clf_ensemble.models, self.clf_optimizers
            )
        ]
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        is_safe: bool = True,
        is_goal: bool = False
    ):
        """
        Add transition data for training.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
            is_safe: Whether state is safe
            is_goal: Whether next_state is a goal state
        """
        # Add to model learner (handles model updates)
        self.model_learner.add_data(state, action, next_state, reward, done)
        
        # Store additional labels for CBF/CLF training
        if not hasattr(self, 'labeled_data'):
            self.labeled_data = {
                'safe_states': [],
                'unsafe_states': [],
                'goal_states': [],
                'transitions': []
            }
        
        # Label states for CBF training
        if is_safe:
            self.labeled_data['safe_states'].append(state)
        else:
            self.labeled_data['unsafe_states'].append(state)
            
        # Label goal states for CLF training
        if is_goal:
            self.labeled_data['goal_states'].append(next_state)
            
        # Store transition for dynamics training
        self.labeled_data['transitions'].append((state, action, next_state))
        
        self.step_count += 1
        
        # Periodic updates
        if self.step_count % self.config.cbf_update_freq == 0:
            self._update_cbf()
            
        if self.step_count % self.config.clf_update_freq == 0:
            self._update_clf()
            
    def _update_cbf(self):
        """Update CBF networks."""
        if not hasattr(self, 'labeled_data') or len(self.labeled_data['transitions']) < self.config.batch_size:
            return
            
        # Prepare data
        safe_states = torch.FloatTensor(self.labeled_data['safe_states']).to(self.device)
        unsafe_states = torch.FloatTensor(self.labeled_data['unsafe_states']).to(self.device)
        
        # Sample transitions for dynamics constraint
        transitions = self.labeled_data['transitions']
        if len(transitions) >= self.config.batch_size:
            sampled_transitions = np.random.choice(
                len(transitions), self.config.batch_size, replace=False
            )
            
            states = torch.FloatTensor([
                transitions[i][0] for i in sampled_transitions
            ]).to(self.device)
            
            next_states = torch.FloatTensor([
                transitions[i][2] for i in sampled_transitions
            ]).to(self.device)
            
            # Update each CBF model
            cbf_losses = []
            for trainer in self.cbf_trainers:
                losses = trainer.train_step(
                    safe_states, unsafe_states, states, next_states,
                    self.config.cbf_weights
                )
                cbf_losses.append(losses)
                
            # Log average losses
            avg_losses = {}
            for key in cbf_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in cbf_losses])
                
            self.training_history["cbf_losses"].append(avg_losses)
            
            if self.step_count % self.config.log_freq == 0:
                self.logger.info(f"Step {self.step_count} - CBF Losses: {avg_losses}")
                
    def _update_clf(self):
        """Update CLF networks."""
        if not hasattr(self, 'labeled_data') or len(self.labeled_data['transitions']) < self.config.batch_size:
            return
            
        # Prepare data
        goal_states = torch.FloatTensor(self.labeled_data['goal_states']).to(self.device)
        
        # Sample transitions for dynamics constraint
        transitions = self.labeled_data['transitions']
        if len(transitions) >= self.config.batch_size:
            sampled_transitions = np.random.choice(
                len(transitions), self.config.batch_size, replace=False
            )
            
            states = torch.FloatTensor([
                transitions[i][0] for i in sampled_transitions
            ]).to(self.device)
            
            next_states = torch.FloatTensor([
                transitions[i][2] for i in sampled_transitions
            ]).to(self.device)
            
            # Update each CLF model
            clf_losses = []
            for trainer in self.clf_trainers:
                losses = trainer.train_step(
                    goal_states, states, next_states,
                    self.config.clf_weights
                )
                clf_losses.append(losses)
                
            # Log average losses
            avg_losses = {}
            for key in clf_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in clf_losses])
                
            self.training_history["clf_losses"].append(avg_losses)
            
            if self.step_count % self.config.log_freq == 0:
                self.logger.info(f"Step {self.step_count} - CLF Losses: {avg_losses}")
                
    def evaluate_safety_feasibility(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate safety and feasibility metrics.
        
        Args:
            states: States to evaluate [batch_size, state_dim]
            actions: Actions to evaluate [batch_size, action_dim]
            
        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Get metrics from controller
            metrics = self.controller.get_safety_feasibility_metrics(
                states, actions, self.dynamics_ensemble
            )
            
            # Compute summary statistics
            eval_metrics = {
                "safety_rate": torch.mean(metrics["is_safe"].float()).item(),
                "goal_proximity_rate": torch.mean(metrics["is_near_goal"].float()).item(),
                "avg_cbf_value": torch.mean(metrics["cbf_values"]).item(),
                "avg_clf_value": torch.mean(metrics["clf_values"]).item(),
                "cbf_constraint_violations": torch.mean(
                    torch.clamp(metrics["cbf_constraints"], min=0.0)
                ).item(),
                "clf_constraint_violations": torch.mean(
                    torch.clamp(metrics["clf_constraints"], min=0.0)
                ).item()
            }
            
        return eval_metrics
        
    def get_safe_action(
        self,
        state: torch.Tensor,
        proposed_action: torch.Tensor,
        safety_margin: float = 0.1,
        feasibility_margin: float = 0.1
    ) -> torch.Tensor:
        """
        Get safe and feasible action using CBF-CLF controller.
        
        Args:
            state: Current state [state_dim]
            proposed_action: Proposed action [action_dim]
            safety_margin: Safety margin for CBF
            feasibility_margin: Feasibility margin for CLF
            
        Returns:
            Safe and feasible action [action_dim]
        """
        state_batch = state.unsqueeze(0)
        action_batch = proposed_action.unsqueeze(0)
        
        filtered_action = self.controller.filter_action(
            state_batch, action_batch, self.dynamics_ensemble,
            safety_margin, feasibility_margin
        )
        
        return filtered_action.squeeze(0)
        
    def save_checkpoint(self, filepath: str):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'config': self.config,
            'cbf_ensemble_state': self.cbf_ensemble.state_dict(),
            'clf_ensemble_state': self.clf_ensemble.state_dict(),
            'dynamics_ensemble_state': self.dynamics_ensemble.state_dict(),
            'cbf_optimizers_state': [opt.state_dict() for opt in self.cbf_optimizers],
            'clf_optimizers_state': [opt.state_dict() for opt in self.clf_optimizers],
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step_count = checkpoint['step_count']
        self.epoch_count = checkpoint['epoch_count']
        
        self.cbf_ensemble.load_state_dict(checkpoint['cbf_ensemble_state'])
        self.clf_ensemble.load_state_dict(checkpoint['clf_ensemble_state'])
        self.dynamics_ensemble.load_state_dict(checkpoint['dynamics_ensemble_state'])
        
        for opt, state in zip(self.cbf_optimizers, checkpoint['cbf_optimizers_state']):
            opt.load_state_dict(state)
            
        for opt, state in zip(self.clf_optimizers, checkpoint['clf_optimizers_state']):
            opt.load_state_dict(state)
            
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        
        Returns:
            Dictionary containing training summary
        """
        summary = {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "buffer_size": len(self.replay_buffer),
            "recent_cbf_loss": self.training_history["cbf_losses"][-1] if self.training_history["cbf_losses"] else None,
            "recent_clf_loss": self.training_history["clf_losses"][-1] if self.training_history["clf_losses"] else None,
        }
        
        # Add model uncertainty if available
        if len(self.replay_buffer) > 0:
            batch = self.replay_buffer.sample(min(100, len(self.replay_buffer)))
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            uncertainty_info = self.model_learner.get_model_uncertainty(states, actions)
            summary["avg_model_uncertainty"] = torch.mean(
                uncertainty_info["epistemic_uncertainty"]
            ).item()
            
        return summary


def create_trainer(
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
    **kwargs
) -> SafeFeasibleTrainer:
    """
    Create trainer with default configuration.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        device: Device to run on
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured trainer instance
    """
    config = TrainingConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **kwargs
    )
    
    return SafeFeasibleTrainer(config)