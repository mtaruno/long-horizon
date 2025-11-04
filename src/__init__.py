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

# Implementation classes (for advanced users)
from .cbf import CBFNetwork, EnsembleCBF, CBFTrainer
from .clf import CLFNetwork, EnsembleCLF, CLFTrainer, CBFCLFController
from .models import DynamicsNetwork, EnsembleDynamics, AdaptiveModelLearner
from .buffer import ReplayBuffer
from .training.integrated_trainer import FSMCBFCLFTrainer

# Dataset generation
from .dataset import (
    DatasetGenerator, SimulationDatasetGenerator, RuleBasedDatasetGenerator,
    create_warehouse_dataset,
    create_navigation_dataset
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "create_trainer",
    "create_trainer_from_config",

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
    "FSMCBFCLFTrainer",
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
    
    # Dataset generation
    "DatasetGenerator",
    "RuleBasedDatasetGenerator",
    "create_warehouse_dataset",
    "create_navigation_dataset",
]