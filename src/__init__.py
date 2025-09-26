"""
Long-horizon safe and feasible planning with neural CBFs and CLFs.
"""

# Core interfaces and types
from .types import Transition, SafetyMetrics, TrainingMetrics
from .interfaces import (
    TrainerInterface, ControllerInterface, 
    SafetyConstraintInterface, FeasibilityConstraintInterface
)

# Configuration
from .config import SafePlanningConfig, get_default_config, get_isaac_gym_config

# Factory functions (main API)
from .factory import (
    SafePlanningFactory,
    create_trainer,
    create_trainer_from_config, 
    create_isaac_gym_trainer
)

# Implementation classes (for advanced users)
from .cbf import CBFNetwork, EnsembleCBF, CBFTrainer
from .clf import CLFNetwork, EnsembleCLF, CLFTrainer, CBFCLFController
from .models import DynamicsNetwork, EnsembleDynamics, ReplayBuffer, AdaptiveModelLearner
from .main_trainer import SafeFeasibleTrainer

__version__ = "0.1.0"

__all__ = [
    # Main API
    "create_trainer",
    "create_trainer_from_config",
    "create_isaac_gym_trainer",
    "SafePlanningFactory",
    
    # Data models
    "Transition",
    "SafetyMetrics", 
    "TrainingMetrics",
    
    # Configuration
    "SafePlanningConfig",
    "get_default_config",
    "get_isaac_gym_config",
    
    # Interfaces
    "TrainerInterface",
    "ControllerInterface",
    "SafetyConstraintInterface",
    "FeasibilityConstraintInterface",
    
    # Implementation classes
    "SafeFeasibleTrainer",
    "CBFNetwork",
    "EnsembleCBF", 
    "CBFTrainer",
    "CLFNetwork",
    "EnsembleCLF",
    "CLFTrainer",
    "CBFCLFController",
    "DynamicsNetwork",
    "EnsembleDynamics",
    "ReplayBuffer",
    "AdaptiveModelLearner",
]