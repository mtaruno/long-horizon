"""
Example using hierarchical control with FSM/LLM planners + CBF-CLF safety.
Updated to use the pre-generated high-quality dataset.
"""

import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
from src import create_trainer, Transition
from src.dataset import (
    create_warehouse_dataset, 
    create_navigation_dataset,
    RuleBasedDatasetGenerator,
    DatasetManager
)
from src.fsm import RobotMode, FSMPlanner, HighLevelCommand 
from generate_dataset import load_dataset
from src.warehouse import WarehouseEnvironment



# ===== MAIN EXAMPLE =====

def main():
    """Complete hierarchical control example with high-quality dataset."""
    
    print("=== HIERARCHICAL ROBOT CONTROL WITH HIGH-QUALITY DATASET ===\n")
    
    # Initialize system
    controller = FSMRobotController()
    env = WarehouseEnvironment()
    
    # Training phase: Load high-quality pre-generated dataset
    print("PHASE 1: Loading High-Quality Dataset")
    print("-" * 40)
    
    try:
        # Load the pre-generated warehouse dataset
        print("Loading warehouse robot dataset...")
        dataset, stats, metadata = load_dataset("warehouse_robot_dataset")
        
        print(f"Loaded high-quality dataset:")
        print(f"  - Description: {metadata['description']}")
        print(f"  - Total transitions: {stats['total_transitions']}")
        print(f"  - Safe transitions: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
        print(f"  - Unsafe transitions: {stats['unsafe_transitions']} ({1-stats['safety_ratio']:.1%})")
        print(f"  - Goal transitions: {stats['goal_transitions']} ({stats['goal_ratio']:.1%})")
        print(f"  - Average reward: {stats['avg_reward']:.2f}")
        
        # Add to trainer
        print(f"Adding dataset to safety trainer...")
        for transition in dataset:
            controller.safety_trainer.add_transition(transition)
            
    except FileNotFoundError:
        print("⚠️  Pre-generated dataset not found!")
        print("Please run 'python generate_dataset.py' first to create the dataset.")
        
        # Fallback: create smaller dataset
        print("Creating fallback dataset...")
        fallback_dataset = create_navigation_dataset(num_transitions=800)
        
        for transition in fallback_dataset:
            controller.safety_trainer.add_transition(transition)
        
        manager = DatasetManager()
        stats = manager.get_dataset_statistics(fallback_dataset)
        
        print(f"Fallback Dataset Statistics:")
        print(f"  - Total transitions: {stats['total_transitions']}")
        print(f"  - Safe transitions: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
    
    # Testing phase: Use hierarchical control
    print(f"\nPHASE 2: Hierarchical Control Testing")
    print("-" * 40)
    
    # Test scenarios in warehouse environment
    test_scenarios = [
        {
            "name": "FSM Navigation to Loading Dock",
            "initial_state": np.array([3.0, 4.0, 0.0, 0.0]),
            "control_type": "fsm"
        },
        {
            "name": "LLM Command: 'go to loading dock'",
            "initial_state": np.array([6.0, 5.0, 0.1, 0.0]),
            "control_type": "llm",
            "command": "go to loading dock"
        },
        {
            "name": "LLM Command: 'explore warehouse safely'",
            "initial_state": np.array([1.5, 2.0, 0.0, 0.0]),
            "control_type": "llm", 
            "command": "explore warehouse safely"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nSCENARIO: {scenario['name']}")
        print("-" * 30)
        
        state = scenario["initial_state"]
        
        for step in range(10):  # Shorter demo
            # Get sensor data
            _, _, _, sensors = env.step(state, np.array([0.0, 0.0]))  # No-op to get sensors
            
            # Execute hierarchical control
            if scenario["control_type"] == "fsm":
                result = controller.execute_fsm_control(state, sensors)
                print(f"Step {step}: Mode={result['mode']}, Priority={result['priority']}")
            else:
                result = controller.execute_llm_control(state, scenario["command"])
                print(f"Step {step}: Command='{scenario['command']}', Mode={result['interpreted_mode']}")
            
            # Execute safe action
            safe_action = result["safe_action"]
            next_state, reward, done, _ = env.step(state, safe_action)
            
            print(f"  State: [{state[0]:.1f}, {state[1]:.1f}] → [{next_state[0]:.1f}, {next_state[1]:.1f}]")
            print(f"  Action: {result['proposed_action']} → {safe_action} (safe)")
            
            state = next_state
            
            if done:
                print(f"  Episode ended: {'Goal reached!' if reward > 0 else 'Collision/Boundary'}")
                break
    
    # Final statistics
    print(f"\nFINAL STATISTICS:")
    print("-" * 20)
    summary = controller.safety_trainer.get_training_summary()
    print(f"Total training steps: {summary.step_count}")
    print(f"Buffer size: {summary.buffer_size}")
    print(f"Model uncertainty: {summary.avg_model_uncertainty:.4f}")
    
    print(f"\n=== HIERARCHICAL CONTROL WITH DATASET COMPLETE ===")
    print(f"System successfully integrated high-quality dataset for safe warehouse navigation!")

if __name__ == "__main__":
    main()