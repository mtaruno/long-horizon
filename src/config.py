"""
Configuration management for the safe planning framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = "relu"
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class CBFConfig:
    """Control Barrier Function configuration."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    alpha: float = 0.1  # CBF decay rate
    num_models: int = 3  # Ensemble size
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "safe": 1.0, "unsafe": 1.0, "constraint": 1.0
    })


@dataclass
class CLFConfig:
    """Control Lyapunov Function configuration."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    beta: float = 0.1   # CLF convergence rate
    delta: float = 0.01 # CLF tolerance
    num_models: int = 3 # Ensemble size
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "goal": 1.0, "constraint": 1.0, "positive": 0.1
    })


@dataclass
class DynamicsConfig:
    """Dynamics model configuration."""
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig(
        hidden_dims=(512, 512, 256)
    ))
    num_models: int = 5  # Ensemble size
    residual: bool = True
    output_activation: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 1000
    
    # Update frequencies
    cbf_update_freq: int = 10
    clf_update_freq: int = 10
    model_update_freq: int = 5
    
    # Buffer settings
    buffer_capacity: int = 100000
    
    # Logging
    log_freq: int = 100
    save_freq: int = 1000


@dataclass
class ControllerConfig:
    """Controller configuration."""
    safety_margin: float = 0.1
    feasibility_margin: float = 0.1
    use_qp_solver: bool = False  # Placeholder for future QP implementation


@dataclass
class SafePlanningConfig:
    """Complete framework configuration."""
    # Model dimensions
    state_dim: int
    action_dim: int
    
    # Component configs
    cbf: CBFConfig = field(default_factory=CBFConfig)
    clf: CLFConfig = field(default_factory=CLFConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    
    # Device
    device: str = "cpu"
    
    # Experiment
    experiment_name: str = "safe_planning"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SafePlanningConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dicts to dataclasses
        if 'cbf' in config_dict:
            if 'network' in config_dict['cbf']:
                config_dict['cbf']['network'] = NetworkConfig(**config_dict['cbf']['network'])
            config_dict['cbf'] = CBFConfig(**config_dict['cbf'])
        
        if 'clf' in config_dict:
            if 'network' in config_dict['clf']:
                config_dict['clf']['network'] = NetworkConfig(**config_dict['clf']['network'])
            config_dict['clf'] = CLFConfig(**config_dict['clf'])
        
        if 'dynamics' in config_dict:
            if 'network' in config_dict['dynamics']:
                config_dict['dynamics']['network'] = NetworkConfig(**config_dict['dynamics']['network'])
            config_dict['dynamics'] = DynamicsConfig(**config_dict['dynamics'])
        
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        if 'controller' in config_dict:
            config_dict['controller'] = ControllerConfig(**config_dict['controller'])
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        # Convert nested dataclasses to dicts
        for key, value in config_dict.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.cbf.alpha > 0, "CBF alpha must be positive"
        assert self.clf.beta > 0, "CLF beta must be positive"
        assert self.clf.delta >= 0, "CLF delta must be non-negative"
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"


# Default configurations for common scenarios
def get_default_config(state_dim: int, action_dim: int, **kwargs) -> SafePlanningConfig:
    """Get default configuration with specified dimensions."""
    config = SafePlanningConfig(state_dim=state_dim, action_dim=action_dim)
    
    # Override any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.controller, key):
            setattr(config.controller, key, value)
    
    config.validate()
    return config


def get_isaac_gym_config(state_dim: int, action_dim: int) -> SafePlanningConfig:
    """Get configuration optimized for Isaac Gym environments."""
    return SafePlanningConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        training=TrainingConfig(
            batch_size=512,  # Larger batches for parallel envs
            cbf_update_freq=5,  # More frequent updates
            clf_update_freq=5,
            model_update_freq=2,
            buffer_capacity=200000  # Larger buffer
        ),
        device="cuda",
        experiment_name="isaac_gym_safe_planning"
    )