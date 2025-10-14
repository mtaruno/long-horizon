"""
Generate high-quality dataset for CBF-CLF training and save to file.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from src.dataset import RuleBasedDatasetGenerator, DatasetManager
from src.types import Transition

def create_warehouse_robot_dataset():
    """Create comprehensive warehouse robot dataset."""
    
    print("Creating warehouse robot dataset...")
    
    # Define realistic warehouse environment
    # 12m x 10m warehouse with multiple zones
    workspace_bounds = (0.0, 12.0, 0.0, 10.0)
    
    # Realistic obstacles (shelves, pillars, equipment)
    obstacles = [
        # Storage shelves
        {'center': np.array([2.0, 3.0]), 'radius': 0.8},
        {'center': np.array([2.0, 7.0]), 'radius': 0.8},
        {'center': np.array([5.0, 2.0]), 'radius': 0.6},
        {'center': np.array([5.0, 5.0]), 'radius': 0.6},
        {'center': np.array([5.0, 8.0]), 'radius': 0.6},
        {'center': np.array([8.0, 3.5]), 'radius': 0.7},
        {'center': np.array([8.0, 6.5]), 'radius': 0.7},
        
        # Structural pillars
        {'center': np.array([4.0, 9.0]), 'radius': 0.3},
        {'center': np.array([9.0, 1.0]), 'radius': 0.3},
        
        # Equipment/charging stations
        {'center': np.array([1.0, 1.0]), 'radius': 0.4},
        {'center': np.array([11.0, 9.0]), 'radius': 0.4},
    ]
    
    # Multiple goal regions (delivery/pickup points)
    goal_regions = [
        # Main delivery zones
        {'center': np.array([10.5, 8.5]), 'radius': 0.4},  # Loading dock 1
        {'center': np.array([10.5, 1.5]), 'radius': 0.4},  # Loading dock 2
        {'center': np.array([1.5, 9.0]), 'radius': 0.3},   # Pickup station 1
        {'center': np.array([6.5, 0.5]), 'radius': 0.3},   # Pickup station 2
        
        # Secondary zones
        {'center': np.array([3.5, 5.0]), 'radius': 0.25},  # Inspection area
        {'center': np.array([9.5, 5.0]), 'radius': 0.25},  # Sorting area
    ]
    
    generator = RuleBasedDatasetGenerator(
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        goal_regions=goal_regions,
        state_dim=4,
        action_dim=2
    )
    
    return generator

def create_multi_scenario_dataset():
    """Create dataset combining multiple scenarios for robustness."""
    
    manager = DatasetManager()
    
    # Scenario 1: Warehouse (dense obstacles)
    warehouse_gen = create_warehouse_robot_dataset()
    manager.add_generator(warehouse_gen, weight=0.4)
    
    # Scenario 2: Open navigation (sparse obstacles)
    open_gen = RuleBasedDatasetGenerator(
        workspace_bounds=(0.0, 15.0, 0.0, 12.0),
        obstacles=[
            {'center': np.array([7.5, 6.0]), 'radius': 1.0},  # Large central obstacle
            {'center': np.array([3.0, 9.0]), 'radius': 0.5},
            {'center': np.array([12.0, 3.0]), 'radius': 0.5},
        ],
        goal_regions=[
            {'center': np.array([14.0, 11.0]), 'radius': 0.5},
            {'center': np.array([1.0, 1.0]), 'radius': 0.5},
        ]
    )
    manager.add_generator(open_gen, weight=0.3)
    
    # Scenario 3: Narrow corridors (challenging navigation)
    corridor_gen = RuleBasedDatasetGenerator(
        workspace_bounds=(0.0, 8.0, 0.0, 6.0),
        obstacles=[
            # Create corridor-like environment
            {'center': np.array([2.0, 2.0]), 'radius': 0.4},
            {'center': np.array([2.0, 4.0]), 'radius': 0.4},
            {'center': np.array([6.0, 2.0]), 'radius': 0.4},
            {'center': np.array([6.0, 4.0]), 'radius': 0.4},
            {'center': np.array([4.0, 1.0]), 'radius': 0.3},
            {'center': np.array([4.0, 5.0]), 'radius': 0.3},
        ],
        goal_regions=[
            {'center': np.array([7.5, 3.0]), 'radius': 0.2},
            {'center': np.array([0.5, 3.0]), 'radius': 0.2},
        ]
    )
    manager.add_generator(corridor_gen, weight=0.3)
    
    return manager

def generate_balanced_dataset(num_transitions=15000):
    """Generate balanced, high-quality dataset."""
    
    print(f"Generating {num_transitions} transitions...")
    
    # Create multi-scenario generator
    manager = create_multi_scenario_dataset()
    
    # Generate balanced dataset with good ratios
    dataset = manager.generate_balanced_dataset(
        total_transitions=num_transitions,
        min_unsafe_ratio=0.08,  # 8% unsafe (realistic for warehouse)
        min_goal_ratio=0.03     # 3% goal states (achievable targets)
    )
    
    # Get statistics
    stats = manager.get_dataset_statistics(dataset)
    
    print(f"Dataset Statistics:")
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  Safe transitions: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
    print(f"  Unsafe transitions: {stats['unsafe_transitions']} ({1-stats['safety_ratio']:.1%})")
    print(f"  Goal transitions: {stats['goal_transitions']} ({stats['goal_ratio']:.1%})")
    print(f"  Average reward: {stats['avg_reward']:.2f}")
    
    return dataset, stats

def save_dataset(dataset, stats, filename="warehouse_robot_dataset"):
    """Save dataset to multiple formats."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save as pickle (for Python loading)
    pickle_path = data_dir / f"{filename}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'transitions': dataset,
            'statistics': stats,
            'metadata': {
                'version': '1.0',
                'description': 'High-quality warehouse robot dataset for CBF-CLF training',
                'state_dim': 4,
                'action_dim': 2,
                'coordinate_system': 'x-right, y-up, velocities in m/s, actions in m/s²'
            }
        }, f)
    
    print(f"Saved dataset to {pickle_path}")
    
    # Save statistics as JSON
    json_path = data_dir / f"{filename}_stats.json"
    with open(json_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'metadata': {
                'total_size_mb': pickle_path.stat().st_size / (1024 * 1024),
                'transitions_per_mb': stats['total_transitions'] / (pickle_path.stat().st_size / (1024 * 1024)),
                'generation_method': 'RuleBasedDatasetGenerator with multi-scenario approach'
            }
        }, f, indent=2)
    
    print(f"Saved statistics to {json_path}")
    
    # Save sample transitions as human-readable text
    sample_path = data_dir / f"{filename}_sample.txt"
    with open(sample_path, 'w') as f:
        f.write("SAMPLE TRANSITIONS FROM DATASET\n")
        f.write("=" * 50 + "\n\n")
        
        # Show examples of each type
        safe_examples = [t for t in dataset if t.is_safe and not t.is_goal][:3]
        unsafe_examples = [t for t in dataset if not t.is_safe][:2]
        goal_examples = [t for t in dataset if t.is_goal][:2]
        
        f.write("SAFE TRANSITIONS:\n")
        for i, t in enumerate(safe_examples):
            f.write(f"  {i+1}. State: {t.state}\n")
            f.write(f"     Action: {t.action}\n")
            f.write(f"     Next: {t.next_state}\n")
            f.write(f"     Reward: {t.reward:.2f}\n\n")
        
        f.write("UNSAFE TRANSITIONS:\n")
        for i, t in enumerate(unsafe_examples):
            f.write(f"  {i+1}. State: {t.state}\n")
            f.write(f"     Action: {t.action}\n")
            f.write(f"     Next: {t.next_state}\n")
            f.write(f"     Reward: {t.reward:.2f} (COLLISION/BOUNDARY)\n\n")
        
        f.write("GOAL TRANSITIONS:\n")
        for i, t in enumerate(goal_examples):
            f.write(f"  {i+1}. State: {t.state}\n")
            f.write(f"     Action: {t.action}\n")
            f.write(f"     Next: {t.next_state}\n")
            f.write(f"     Reward: {t.reward:.2f} (GOAL REACHED)\n\n")
    
    print(f"Saved sample data to {sample_path}")
    
    return pickle_path, json_path, sample_path

def load_dataset(filename="warehouse_robot_dataset"):
    """Load dataset from file."""
    
    data_dir = Path("data")
    pickle_path = data_dir / f"{filename}.pkl"
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['transitions'], data['statistics'], data['metadata']

def validate_dataset_quality(dataset, stats):
    """Validate dataset meets quality requirements."""
    
    print("Validating dataset quality...")
    
    # Check basic requirements
    assert len(dataset) > 1000, "Dataset too small"
    assert stats['safety_ratio'] > 0.7, "Not enough safe examples"
    assert stats['safety_ratio'] < 0.98, "Not enough unsafe examples"
    assert stats['goal_ratio'] > 0.008, "Not enough goal examples"
    
    # Check data diversity
    positions = np.array([t.state[:2] for t in dataset])
    velocities = np.array([t.state[2:] for t in dataset])
    actions = np.array([t.action for t in dataset])
    
    # Position coverage
    pos_std = np.std(positions, axis=0)
    assert pos_std[0] > 2.0, "Insufficient x-position diversity"
    assert pos_std[1] > 2.0, "Insufficient y-position diversity"
    
    # Velocity diversity
    vel_std = np.std(velocities, axis=0)
    assert vel_std[0] > 0.1, "Insufficient x-velocity diversity"
    assert vel_std[1] > 0.1, "Insufficient y-velocity diversity"
    
    # Action diversity
    action_std = np.std(actions, axis=0)
    assert action_std[0] > 0.2, "Insufficient x-action diversity"
    assert action_std[1] > 0.2, "Insufficient y-action diversity"
    
    # Check physics consistency
    physics_errors = []
    for t in dataset[:100]:  # Sample check
        dt = 0.1
        expected_vel = t.state[2:] + t.action * dt
        expected_pos = t.state[:2] + expected_vel * dt
        
        vel_error = np.linalg.norm(t.next_state[2:] - expected_vel)
        pos_error = np.linalg.norm(t.next_state[:2] - expected_pos)
        
        physics_errors.append(vel_error + pos_error)
    
    avg_physics_error = np.mean(physics_errors)
    assert avg_physics_error < 0.5, f"Physics inconsistency too high: {avg_physics_error}"
    
    print("✓ Dataset quality validation passed!")
    
    return True

def main():
    """Main dataset generation script."""
    
    print("=== HIGH-QUALITY DATASET GENERATION ===\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    dataset, stats = generate_balanced_dataset(num_transitions=12000)
    
    # Validate quality
    validate_dataset_quality(dataset, stats)
    
    # Save to files
    pickle_path, json_path, sample_path = save_dataset(dataset, stats)
    
    print(f"\n=== DATASET GENERATION COMPLETE ===")
    print(f"Generated high-quality dataset with {len(dataset)} transitions")
    print(f"Files saved:")
    print(f"  - Dataset: {pickle_path}")
    print(f"  - Statistics: {json_path}")
    print(f"  - Samples: {sample_path}")
    
    # Demonstrate loading
    print(f"\nTesting dataset loading...")
    loaded_dataset, loaded_stats, metadata = load_dataset()
    print(f"✓ Successfully loaded {len(loaded_dataset)} transitions")
    print(f"✓ Metadata: {metadata['description']}")
    
    return dataset, stats

if __name__ == "__main__":
    main()