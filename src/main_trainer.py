"""
Main training framework integrating CBF, CLF, and dynamics learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any
import numpy as np
import logging

from .config import SafePlanningConfig
from .types import Transition, SafetyMetrics, TrainingMetrics, StateArray, ActionArray
from .interfaces import TrainerInterface
from .cbf import EnsembleCBF, CBFTrainer
from .clf import EnsembleCLF, CLFTrainer, CBFCLFController
from .models import EnsembleDynamics, AdaptiveModelLearner, ReplayBuffer


class SafeFeasibleTrainer(TrainerInterface):
    """
    Main trainer for safe and feasible long-horizon planning.
    """
    
    def __init__(self, config: SafePlanningConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self._init_networks()
        self._init_optimizers()
        self._init_trainers()
        
        # Initialize data storage
        self.replay_buffer = ReplayBuffer(
            config.training.buffer_capacity, 
            config.state_dim, 
            config.action_dim
        )
        
        # Initialize adaptive model learner
        self.model_learner = AdaptiveModelLearner(
            self.dynamics_ensemble,
            self.replay_buffer,
            config.training.batch_size,
            config.training.model_update_freq,
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
        }
        
        # Labeled data storage
        self.labeled_data = {
            'safe_states': [],
            'unsafe_states': [],
            'goal_states': [],
            'transitions': []
        }
        
        # Setup logging
        self._setup_logging()
        
    def _init_networks(self):
        """Initialize all neural networks."""
        config = self.config
        
        self.cbf_ensemble = EnsembleCBF(
            config.cbf.num_models,
            config.state_dim,
            config.cbf.network.hidden_dims,
            alpha=config.cbf.alpha,
            device=config.device
        )
        
        self.clf_ensemble = EnsembleCLF(
            config.clf.num_models,
            config.state_dim,
            config.clf.network.hidden_dims,
            beta=config.clf.beta,
            delta=config.clf.delta,
            device=config.device
        )
        
        self.dynamics_ensemble = EnsembleDynamics(
            config.dynamics.num_models,
            config.state_dim,
            config.action_dim,
            config.dynamics.network.hidden_dims,
            device=config.device
        )
        
    def _init_optimizers(self):
        """Initialize optimizers for all networks."""
        lr = self.config.training.learning_rate
        
        self.cbf_optimizers = [
            optim.Adam(model.parameters(), lr=lr)
            for model in self.cbf_ensemble.models
        ]
        
        self.clf_optimizers = [
            optim.Adam(model.parameters(), lr=lr)
            for model in self.clf_ensemble.models
        ]
        
    def _init_trainers(self):
        """Initialize trainers for CBF and CLF networks."""
        self.cbf_trainers = [
            CBFTrainer(model, optimizer, self.device)
            for model, optimizer in zip(
                self.cbf_ensemble.models, self.cbf_optimizers
            )
        ]
        
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
        
    def add_transition(self, transition: Transition) -> None:
        """Add a single transition to the trainer."""
        # Add to model learner
        self.model_learner.add_data(
            transition.state, transition.action, transition.next_state,
            transition.reward, transition.done
        )
        
        # Store labeled data
        if transition.is_safe:
            self.labeled_data['safe_states'].append(transition.state)
        else:
            self.labeled_data['unsafe_states'].append(transition.state)
            
        if transition.is_goal:
            self.labeled_data['goal_states'].append(transition.next_state)
            
        self.labeled_data['transitions'].append(
            (transition.state, transition.action, transition.next_state)
        )
        
        self.step_count += 1
        
        # Periodic updates
        if self.step_count % self.config.training.cbf_update_freq == 0:
            self._update_cbf()
            
        if self.step_count % self.config.training.clf_update_freq == 0:
            self._update_clf()
    
    def add_transitions(self, transitions: List[Transition]) -> None:
        """Add multiple transitions to the trainer."""
        for transition in transitions:
            self.add_transition(transition)
    
    def get_safe_action(
        self,
        state: StateArray,
        proposed_action: ActionArray,
        **kwargs
    ) -> torch.Tensor:
        """Get safe action using learned constraints."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(proposed_action, np.ndarray):
            proposed_action = torch.FloatTensor(proposed_action).to(self.device)
            
        state_batch = state.unsqueeze(0) if state.dim() == 1 else state
        action_batch = proposed_action.unsqueeze(0) if proposed_action.dim() == 1 else proposed_action
        
        filtered_action = self.controller.filter_action(
            state_batch, action_batch, self.dynamics_ensemble,
            self.config.controller.safety_margin,
            self.config.controller.feasibility_margin
        )
        
        return filtered_action.squeeze(0) if state.dim() == 1 else filtered_action
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> SafetyMetrics:
        """Evaluate current performance."""
        with torch.no_grad():
            metrics = self.controller.get_safety_feasibility_metrics(
                states, actions, self.dynamics_ensemble
            )
            
            return SafetyMetrics(
                safety_rate=torch.mean(metrics["is_safe"].float()).item(),
                goal_proximity_rate=torch.mean(metrics["is_near_goal"].float()).item(),
                avg_cbf_value=torch.mean(metrics["cbf_values"]).item(),
                avg_clf_value=torch.mean(metrics["clf_values"]).item(),
                cbf_constraint_violations=torch.mean(
                    torch.clamp(metrics["cbf_constraints"], min=0.0)
                ).item(),
                clf_constraint_violations=torch.mean(
                    torch.clamp(metrics["clf_constraints"], min=0.0)
                ).item()
            )
    
    def get_training_summary(self) -> TrainingMetrics:
        """Get summary of training progress."""
        avg_uncertainty = None
        if len(self.replay_buffer) > 0:
            batch = self.replay_buffer.sample(min(100, len(self.replay_buffer)))
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            uncertainty_info = self.model_learner.get_model_uncertainty(states, actions)
            avg_uncertainty = torch.mean(
                uncertainty_info["epistemic_uncertainty"]
            ).item()
        
        return TrainingMetrics(
            step_count=self.step_count,
            epoch_count=self.epoch_count,
            buffer_size=len(self.replay_buffer),
            recent_cbf_loss=self.training_history["cbf_losses"][-1] if self.training_history["cbf_losses"] else None,
            recent_clf_loss=self.training_history["clf_losses"][-1] if self.training_history["clf_losses"] else None,
            avg_model_uncertainty=avg_uncertainty
        )
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'config': self.config,
            'cbf_ensemble_state': self.cbf_ensemble.state_dict(),
            'clf_ensemble_state': self.clf_ensemble.state_dict(),
            'dynamics_ensemble_state': self.dynamics_ensemble.state_dict(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step_count = checkpoint['step_count']
        self.epoch_count = checkpoint['epoch_count']
        
        self.cbf_ensemble.load_state_dict(checkpoint['cbf_ensemble_state'])
        self.clf_ensemble.load_state_dict(checkpoint['clf_ensemble_state'])
        self.dynamics_ensemble.load_state_dict(checkpoint['dynamics_ensemble_state'])
        
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def _update_cbf(self):
        """Update CBF networks."""
        if len(self.labeled_data['transitions']) < self.config.training.batch_size:
            return
            
        safe_states = torch.FloatTensor(self.labeled_data['safe_states']).to(self.device)
        unsafe_states = torch.FloatTensor(self.labeled_data['unsafe_states']).to(self.device)
        
        transitions = self.labeled_data['transitions']
        sampled_transitions = np.random.choice(
            len(transitions), self.config.training.batch_size, replace=False
        )
        
        states = torch.FloatTensor([
            transitions[i][0] for i in sampled_transitions
        ]).to(self.device)
        
        next_states = torch.FloatTensor([
            transitions[i][2] for i in sampled_transitions
        ]).to(self.device)
        
        cbf_losses = []
        for trainer in self.cbf_trainers:
            losses = trainer.train_step(
                safe_states, unsafe_states, states, next_states,
                self.config.cbf.loss_weights
            )
            cbf_losses.append(losses)
            
        avg_losses = {}
        for key in cbf_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in cbf_losses])
            
        self.training_history["cbf_losses"].append(avg_losses)
        
        if self.step_count % self.config.training.log_freq == 0:
            self.logger.info(f"Step {self.step_count} - CBF Losses: {avg_losses}")
    
    def _update_clf(self):
        """Update CLF networks."""
        if len(self.labeled_data['transitions']) < self.config.training.batch_size:
            return
            
        goal_states = torch.FloatTensor(self.labeled_data['goal_states']).to(self.device)
        
        transitions = self.labeled_data['transitions']
        sampled_transitions = np.random.choice(
            len(transitions), self.config.training.batch_size, replace=False
        )
        
        states = torch.FloatTensor([
            transitions[i][0] for i in sampled_transitions
        ]).to(self.device)
        
        next_states = torch.FloatTensor([
            transitions[i][2] for i in sampled_transitions
        ]).to(self.device)
        
        clf_losses = []
        for trainer in self.clf_trainers:
            losses = trainer.train_step(
                goal_states, states, next_states,
                self.config.clf.loss_weights
            )
            clf_losses.append(losses)
            
        avg_losses = {}
        for key in clf_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in clf_losses])
            
        self.training_history["clf_losses"].append(avg_losses)
        
        if self.step_count % self.config.training.log_freq == 0:
            self.logger.info(f"Step {self.step_count} - CLF Losses: {avg_losses}")