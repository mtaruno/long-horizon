"""
Data models and type definitions for the safe planning framework.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Transition:
    """Single environment transition."""
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float = 0.0
    done: bool = False
    is_safe: bool = True
    is_goal: bool = False


@dataclass
class SafetyMetrics:
    """Safety and feasibility evaluation metrics."""
    safety_rate: float
    goal_proximity_rate: float
    avg_cbf_value: float
    avg_clf_value: float
    cbf_constraint_violations: float
    clf_constraint_violations: float


@dataclass
class TrainingMetrics:
    """Training progress metrics."""
    step_count: int
    epoch_count: int
    buffer_size: int
    recent_cbf_loss: Optional[Dict[str, float]]
    recent_clf_loss: Optional[Dict[str, float]]
    avg_model_uncertainty: Optional[float]


@dataclass
class LossComponents:
    """Loss components for training."""
    safe: torch.Tensor
    unsafe: torch.Tensor
    constraint: torch.Tensor
    total: torch.Tensor


@dataclass
class ModelUncertainty:
    """Model uncertainty estimates."""
    epistemic_uncertainty: torch.Tensor
    mean_prediction: torch.Tensor
    all_predictions: torch.Tensor


class SafetyConstraint(ABC):
    """Abstract base class for safety constraints."""
    
    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute constraint values."""
        pass
    
    @abstractmethod
    def constraint_violation(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute constraint violations."""
        pass
    
    @abstractmethod
    def is_satisfied(
        self, 
        states: torch.Tensor, 
        threshold: float = 0.0
    ) -> torch.Tensor:
        """Check if constraint is satisfied."""
        pass


class DynamicsModel(ABC):
    """Abstract base class for dynamics models."""
    
    @abstractmethod
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict next states."""
        pass
    
    @abstractmethod
    def uncertainty(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction uncertainty."""
        pass


class Controller(ABC):
    """Abstract base class for controllers."""
    
    @abstractmethod
    def filter_action(
        self,
        states: torch.Tensor,
        proposed_actions: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Filter actions to satisfy constraints."""
        pass


# Type aliases for clarity
StateArray = Union[np.ndarray, torch.Tensor]
ActionArray = Union[np.ndarray, torch.Tensor]
BatchStates = torch.Tensor  # [batch_size, state_dim]
BatchActions = torch.Tensor  # [batch_size, action_dim]
LossDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]