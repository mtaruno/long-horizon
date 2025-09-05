"""
Example usage of the safe and feasible long-horizon planning framework.
"""

import torch
import numpy as np
from src.main_trainer import create_trainer, TrainingConfig

def main():
    """Example training loop."""
    
    # Configuration
    state_dim = 4  # Example: 2D position + 2D velocity
    action_dim = 2  # Example: 2D acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create trainer
    trainer = create_trainer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        batch_size=128,
        learning_rate=1e-3,
        cbf_update_freq=10,
        clf_update_freq=10
    )
    
    print(f"Training on device: {device}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Simulate training data
    num_episodes = 100
    episode_length = 50
    
    for episode in range(num_episodes):
        # Initialize random state
        state = np.random.randn(state_dim) * 0.5
        
        for step in range(episode_length):
            # Random action (in practice, use policy)
            action = np.random.randn(action_dim) * 0.1
            
            # Simple dynamics: next_state = state + [action, 0, 0] + noise
            action_padded = np.pad(action, (0, state_dim - action_dim))
            next_state = state + action_padded + np.random.randn(state_dim) * 0.01
            
            # Simple reward
            reward = -np.linalg.norm(next_state[:2])  # Negative distance to origin
            
            # Safety and goal labels (example)
            is_safe = np.linalg.norm(state) < 2.0  # Safe if within radius 2
            is_goal = np.linalg.norm(next_state) < 0.1  # Goal if near origin
            done = is_goal or step == episode_length - 1
            
            # Add transition to trainer
            trainer.add_transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                is_safe=is_safe,
                is_goal=is_goal
            )
            
            state = next_state
            
            if done:
                break
        
        # Log progress
        if episode % 10 == 0:
            summary = trainer.get_training_summary()
            print(f"Episode {episode}: {summary}")
            
            # Evaluate on random batch
            if trainer.step_count > 100:
                test_states = torch.randn(32, state_dim).to(device)
                test_actions = torch.randn(32, action_dim).to(device)
                
                metrics = trainer.evaluate_safety_feasibility(test_states, test_actions)
                print(f"Safety metrics: {metrics}")
    
    # Save final model
    trainer.save_checkpoint("checkpoint.pth")
    print("Training completed and model saved!")
    
    # Example of using trained model for control
    test_state = torch.randn(state_dim).to(device)
    proposed_action = torch.randn(action_dim).to(device)
    
    safe_action = trainer.get_safe_action(test_state, proposed_action)
    print(f"Proposed action: {proposed_action}")
    print(f"Safe action: {safe_action}")

if __name__ == "__main__":
    main()