"""
Example using configuration-driven approach.
"""

import torch
import numpy as np
from src import create_trainer_from_config, Transition

def main():
    """Example using YAML configuration."""
    
    # Create trainer from config file
    trainer = create_trainer_from_config(
        "configs/default.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Training with configuration-driven approach")
    
    # Simple training loop
    for episode in range(50):
        state = np.random.randn(4) * 0.5
        
        for step in range(25):
            action = np.random.randn(2) * 0.1
            action_padded = np.pad(action, (0, 2))
            next_state = state + action_padded + np.random.randn(4) * 0.01
            
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=-np.linalg.norm(next_state[:2]),
                done=np.linalg.norm(next_state) < 0.1,
                is_safe=np.linalg.norm(state) < 2.0,
                is_goal=np.linalg.norm(next_state) < 0.1
            )
            
            trainer.add_transition(transition)
            state = next_state
            
            if transition.done:
                break
        
        if episode % 10 == 0:
            summary = trainer.get_training_summary()
            print(f"Episode {episode}: {summary.step_count} steps")
    
    print("Configuration-driven training completed!")

if __name__ == "__main__":
    main()