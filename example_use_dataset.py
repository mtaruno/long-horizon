"""
Example of how to use the generated dataset for CBF-CLF training.
"""

import numpy as np
from src import create_trainer
from generate_dataset import load_dataset

def main():
    """Demonstrate using the generated dataset."""
    
    print("=== USING GENERATED DATASET FOR TRAINING ===\n")
    
    # Load the dataset
    print("Loading dataset...")
    dataset, stats, metadata = load_dataset("warehouse_robot_dataset")
    
    print(f"Loaded dataset:")
    print(f"  - {len(dataset)} transitions")
    print(f"  - {stats['safe_transitions']} safe ({stats['safety_ratio']:.1%})")
    print(f"  - {stats['unsafe_transitions']} unsafe")
    print(f"  - {stats['goal_transitions']} goal ({stats['goal_ratio']:.1%})")
    print(f"  - Description: {metadata['description']}")
    
    # Create trainer
    print(f"\nCreating trainer...")
    trainer = create_trainer(
        state_dim=metadata['state_dim'],
        action_dim=metadata['action_dim'],
        device="cpu",
        batch_size=64
    )
    
    # Add dataset to trainer
    print(f"Adding dataset to trainer...")
    for transition in dataset:
        trainer.add_transition(transition)
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  - Total steps: {summary.step_count}")
    print(f"  - Buffer size: {summary.buffer_size}")
    print(f"  - Model uncertainty: {summary.avg_model_uncertainty:.4f}")
    
    # Test safe action generation
    print(f"\nTesting safe action generation...")
    
    # Test scenarios from the warehouse environment
    test_scenarios = [
        {
            "name": "Open area navigation",
            "state": np.array([6.0, 4.0, 0.2, 0.1]),
            "action": np.array([0.5, 0.3])
        },
        {
            "name": "Near obstacle",
            "state": np.array([2.5, 3.0, 0.1, 0.0]),
            "action": np.array([0.3, 0.2])
        },
        {
            "name": "Near boundary",
            "state": np.array([11.5, 9.5, 0.3, 0.2]),
            "action": np.array([0.4, 0.3])
        },
        {
            "name": "Near goal",
            "state": np.array([10.2, 8.3, 0.1, 0.1]),
            "action": np.array([0.2, 0.1])
        }
    ]
    
    for scenario in test_scenarios:
        state = scenario["state"]
        proposed_action = scenario["action"]
        
        safe_action = trainer.get_safe_action(state, proposed_action)
        
        print(f"\n{scenario['name']}:")
        print(f"  State: {state}")
        print(f"  Proposed: {proposed_action}")
        print(f"  Safe: {safe_action.numpy()}")
        
        # Check if action was modified
        modification = np.linalg.norm(safe_action.numpy() - proposed_action)
        if modification > 0.01:
            print(f"  ⚠️  Action modified (Δ={modification:.3f})")
        else:
            print(f"  ✅ Action approved as-is")
    
    # Evaluate on test batch
    print(f"\nEvaluating on test batch...")
    
    # Create test batch from dataset
    test_indices = np.random.choice(len(dataset), size=50, replace=False)
    test_states = np.array([dataset[i].state for i in test_indices])
    test_actions = np.array([dataset[i].action for i in test_indices])
    
    import torch
    test_states_tensor = torch.FloatTensor(test_states)
    test_actions_tensor = torch.FloatTensor(test_actions)
    
    metrics = trainer.evaluate(test_states_tensor, test_actions_tensor)
    
    print(f"Evaluation Results:")
    print(f"  - Safety rate: {metrics.safety_rate:.1%}")
    print(f"  - Goal proximity rate: {metrics.goal_proximity_rate:.1%}")
    print(f"  - Avg CBF value: {metrics.avg_cbf_value:.4f}")
    print(f"  - Avg CLF value: {metrics.avg_clf_value:.4f}")
    print(f"  - CBF violations: {metrics.cbf_constraint_violations:.6f}")
    print(f"  - CLF violations: {metrics.clf_constraint_violations:.6f}")
    
    print(f"\n=== DATASET USAGE COMPLETE ===")
    print(f"The dataset is ready for production CBF-CLF training!")

if __name__ == "__main__":
    main()