"""
Example demonstrating modular dataset creation and usage.
"""

import numpy as np
from src import create_trainer
from src.dataset import (
    create_warehouse_dataset,
    create_navigation_dataset, 
    create_mixed_dataset,
    RuleBasedDatasetGenerator,
    DatasetManager
)

def main():
    """Demonstrate different ways to create datasets."""
    
    print("=== MODULAR DATASET CREATION EXAMPLES ===\n")
    
    # ===== EXAMPLE 1: Predefined Datasets =====
    print("EXAMPLE 1: Predefined Datasets")
    print("-" * 40)
    
    # Warehouse scenario
    warehouse_data = create_warehouse_dataset(num_transitions=1000)
    print(f"Warehouse dataset: {len(warehouse_data)} transitions")
    
    # Navigation scenario  
    nav_data = create_navigation_dataset(num_transitions=500)
    print(f"Navigation dataset: {len(nav_data)} transitions")
    
    # Mixed dataset
    mixed_data = create_mixed_dataset(num_transitions=800)
    print(f"Mixed dataset: {len(mixed_data)} transitions")
    
    # ===== EXAMPLE 2: Custom Dataset Generator =====
    print(f"\nEXAMPLE 2: Custom Dataset Generator")
    print("-" * 40)
    
    # Define custom environment
    custom_generator = RuleBasedDatasetGenerator(
        workspace_bounds=(0.0, 6.0, 0.0, 5.0),  # 6m x 5m space
        obstacles=[
            {'center': np.array([2.0, 2.0]), 'radius': 0.4},  # Obstacle 1
            {'center': np.array([4.0, 3.0]), 'radius': 0.3},  # Obstacle 2
        ],
        goal_regions=[
            {'center': np.array([5.5, 4.5]), 'radius': 0.25},  # Goal region
        ]
    )
    
    custom_data = custom_generator.generate_transitions(600)
    print(f"Custom dataset: {len(custom_data)} transitions")
    
    # ===== EXAMPLE 3: Dataset Manager =====
    print(f"\nEXAMPLE 3: Dataset Manager")
    print("-" * 40)
    
    manager = DatasetManager()
    
    # Add multiple generators with different weights
    manager.add_generator(custom_generator, weight=0.6)
    
    # Create another generator for variety
    simple_generator = RuleBasedDatasetGenerator(
        workspace_bounds=(0.0, 3.0, 0.0, 3.0),
        obstacles=[],  # No obstacles
        goal_regions=[{'center': np.array([2.5, 2.5]), 'radius': 0.2}]
    )
    manager.add_generator(simple_generator, weight=0.4)
    
    # Generate balanced dataset
    balanced_data = manager.generate_balanced_dataset(
        total_transitions=1000,
        min_unsafe_ratio=0.15,  # At least 15% unsafe
        min_goal_ratio=0.08     # At least 8% goal
    )
    
    print(f"Balanced dataset: {len(balanced_data)} transitions")
    
    # ===== EXAMPLE 4: Dataset Statistics =====
    print(f"\nEXAMPLE 4: Dataset Analysis")
    print("-" * 40)
    
    datasets = {
        "Warehouse": warehouse_data,
        "Navigation": nav_data,
        "Custom": custom_data,
        "Balanced": balanced_data
    }
    
    for name, dataset in datasets.items():
        stats = manager.get_dataset_statistics(dataset)
        print(f"\n{name} Dataset Statistics:")
        print(f"  Total: {stats['total_transitions']}")
        print(f"  Safe: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
        print(f"  Unsafe: {stats['unsafe_transitions']}")
        print(f"  Goal: {stats['goal_transitions']} ({stats['goal_ratio']:.1%})")
        print(f"  Avg reward: {stats['avg_reward']:.2f}")
    
    # ===== EXAMPLE 5: Training with Dataset =====
    print(f"\nEXAMPLE 5: Training with Generated Dataset")
    print("-" * 40)
    
    # Create trainer
    trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
    
    # Use the balanced dataset for training
    print("Adding transitions to trainer...")
    for transition in balanced_data:
        trainer.add_transition(transition)
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"Training summary:")
    print(f"  Steps: {summary.step_count}")
    print(f"  Buffer size: {summary.buffer_size}")
    
    # Test safe action generation
    test_state = np.array([1.0, 1.0, 0.1, 0.1])
    test_action = np.array([0.5, 0.3])
    
    safe_action = trainer.get_safe_action(test_state, test_action)
    print(f"\nSafe action test:")
    print(f"  Proposed: {test_action}")
    print(f"  Safe: {safe_action.numpy()}")
    
    print(f"\n=== DATASET EXAMPLES COMPLETE ===")

if __name__ == "__main__":
    main()