"""
Factory functions for creating framework components with proper abstractions.
"""
from typing import Optional
from .config import SafePlanningConfig, get_default_config, get_isaac_gym_config
from .types import Transition
from .interfaces import TrainerInterface
from .main_trainer import SafeFeasibleTrainer


class SafePlanningFactory:
    """Factory for creating safe planning components."""
    @staticmethod
    def create_trainer(
        state_dim: int,
        action_dim: int,
        config: Optional[SafePlanningConfig] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> TrainerInterface:
        """
        Create a safe planning trainer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration object (optional)
            device: Device to run on (optional)
            **kwargs: Additional config overrides
            
        Returns:
            Configured trainer instance
        """
        if config is None:
            config = get_default_config(state_dim, action_dim, **kwargs) 

        if device is not None:
            config.device = device
            
        config.validate()
        return SafeFeasibleTrainer(config)
    
    @staticmethod
    def create_trainer_from_yaml(
        config_path: str,
        **kwargs
    ) -> TrainerInterface:
        """
        Create trainer from YAML configuration.
        
        Args:
            config_path: Path to YAML config file
            **kwargs: Config overrides
            
        Returns:
            Configured trainer instance
        """
        config = SafePlanningConfig.from_yaml(config_path)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.validate()
        return SafeFeasibleTrainer(config)
    
    @staticmethod
    def create_isaac_gym_trainer(
        state_dim: int,
        action_dim: int,
        **kwargs
    ) -> TrainerInterface:
        """
        Create trainer optimized for Isaac Gym.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            **kwargs: Config overrides
            
        Returns:
            Isaac Gym optimized trainer
        """
        config = get_isaac_gym_config(state_dim, action_dim)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.validate()
        return SafeFeasibleTrainer(config)


# Convenience functions for backward compatibility
def create_trainer(
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
    **kwargs
) -> TrainerInterface:
    """Create trainer with default configuration."""
    return SafePlanningFactory.create_trainer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **kwargs
    )


def create_trainer_from_config(config_path: str, **kwargs) -> TrainerInterface:
    """Create trainer from configuration file."""
    return SafePlanningFactory.create_trainer_from_yaml(config_path, **kwargs)


def create_isaac_gym_trainer(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> TrainerInterface:
    """Create Isaac Gym optimized trainer."""
    return SafePlanningFactory.create_isaac_gym_trainer(
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs
    )