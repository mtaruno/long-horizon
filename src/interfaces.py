"""
Abstract interfaces for the safe planning framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from .types import (
    Transition, SafetyMetrics, TrainingMetrics, 
    StateArray, ActionArray, BatchStates, BatchActions,
    LossDict, MetricsDict
)


class SafetyConstraintInterface(ABC):
    """Interface for safety constraint implementations."""
    
    @abstractmethod
    def evaluate(self, states: BatchStates) -> torch.Tensor:
        """Evaluate constraint values for given states."""
        pass
    
    @abstractmethod
    def constraint_violation(
        self, 
        states: BatchStates, 
        next_states: BatchStates
    ) -> torch.Tensor:
        """Compute constraint violations for state transitions."""
        pass
    
    @abstractmethod
    def is_safe(self, states: BatchStates, threshold: float = 0.0) -> torch.Tensor:
        """Check if states satisfy safety constraint."""
        pass


class FeasibilityConstraintInterface(ABC):
    """Interface for feasibility constraint implementations."""
    
    @abstractmethod
    def evaluate(self, states: BatchStates) -> torch.Tensor:
        """Evaluate constraint values for given states."""
        pass
    
    @abstractmethod
    def constraint_violation(
        self, 
        states: BatchStates, 
        next_states: BatchStates
    ) -> torch.Tensor:
        """Compute constraint violations for state transitions."""
        pass
    
    @abstractmethod
    def is_feasible(self, states: BatchStates, threshold: float = 0.1) -> torch.Tensor:
        """Check if states are feasible (near goal)."""
        pass


class DynamicsModelInterface(ABC):
    """Interface for dynamics model implementations."""
    
    @abstractmethod
    def predict(
        self, 
        states: BatchStates, 
        actions: BatchActions
    ) -> torch.Tensor:
        """Predict next states given current states and actions."""
        pass
    
    @abstractmethod
    def uncertainty(
        self, 
        states: BatchStates, 
        actions: BatchActions
    ) -> torch.Tensor:
        """Compute prediction uncertainty."""
        pass
    
    @abstractmethod
    def update(self, transitions: List[Transition]) -> MetricsDict:
        """Update model with new transition data."""
        pass


class ControllerInterface(ABC):
    """Interface for controller implementations."""
    
    @abstractmethod
    def compute_safe_action(
        self,
        state: StateArray,
        proposed_action: ActionArray,
        **kwargs
    ) -> torch.Tensor:
        """Compute safe action given state and proposed action."""
        pass
    
    @abstractmethod
    def evaluate_safety_feasibility(
        self,
        states: BatchStates,
        actions: BatchActions
    ) -> SafetyMetrics:
        """Evaluate safety and feasibility metrics."""
        pass


class TrainerInterface(ABC):
    """Interface for trainer implementations."""
    
    @abstractmethod
    def add_transition(self, transition: Transition) -> None:
        """Add a single transition to the trainer."""
        pass
    
    @abstractmethod
    def add_transitions(self, transitions: List[Transition]) -> None:
        """Add multiple transitions to the trainer."""
        pass
    
    @abstractmethod
    def get_safe_action(
        self,
        state: StateArray,
        proposed_action: ActionArray,
        **kwargs
    ) -> torch.Tensor:
        """Get safe action using learned constraints."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        states: BatchStates,
        actions: BatchActions
    ) -> SafetyMetrics:
        """Evaluate current performance."""
        pass
    
    @abstractmethod
    def get_training_summary(self) -> TrainingMetrics:
        """Get summary of training progress."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        pass


class DataBufferInterface(ABC):
    """Interface for data buffer implementations."""
    
    @abstractmethod
    def add(self, transition: Transition) -> None:
        """Add transition to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch of transitions."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return buffer size."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear buffer."""
        pass


class NetworkTrainerInterface(ABC):
    """Interface for network trainer implementations."""
    
    @abstractmethod
    def train_step(self, batch_data: Dict) -> LossDict:
        """Perform one training step."""
        pass
    
    @abstractmethod
    def evaluate_step(self, batch_data: Dict) -> MetricsDict:
        """Perform one evaluation step."""
        pass
    
    @abstractmethod
    def get_network(self) -> torch.nn.Module:
        """Get the underlying network."""
        pass