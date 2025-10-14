# Modular Dataset System

## Overview

The dataset module (`src/dataset.py`) provides a flexible, modular system for creating labeled training data for CBF-CLF learning. It separates dataset creation from the main training pipeline, making it easy to experiment with different data sources and labeling strategies.

## Key Components

### 1. Dataset Generators

**Abstract Base Class:**
```python
class DatasetGenerator(ABC):
    @abstractmethod
    def generate_transitions(self, num_transitions: int) -> List[Transition]:
        pass
```

**Concrete Implementations:**

#### SimulationDatasetGenerator
- Generates data from simulation environments
- Automatically labels transitions based on environment feedback
- Best for: Rapid prototyping, large-scale data collection

```python
generator = SimulationDatasetGenerator(
    environment=isaac_gym_env,
    policy=exploration_policy
)
transitions = generator.generate_transitions(10000)
```

#### RuleBasedDatasetGenerator
- Uses predefined rules to label transitions
- Fast and deterministic
- Best for: Well-defined environments, initial datasets

```python
generator = RuleBasedDatasetGenerator(
    workspace_bounds=(0.0, 5.0, 0.0, 4.0),
    obstacles=[{'center': [2.5, 2.0], 'radius': 0.4}],
    goal_regions=[{'center': [4.0, 3.0], 'radius': 0.2}]
)
```

#### ExpertDatasetGenerator
- Creates transitions from expert demonstrations
- High-quality but limited scale
- Best for: Critical applications, validation data

```python
generator = ExpertDatasetGenerator(
    expert_trajectories=demo_data,
    labeling_function=custom_labeler
)
```

### 2. Dataset Manager

Combines multiple generators and ensures balanced datasets:

```python
manager = DatasetManager()
manager.add_generator(rule_generator, weight=0.6)
manager.add_generator(sim_generator, weight=0.4)

dataset = manager.generate_balanced_dataset(
    total_transitions=5000,
    min_unsafe_ratio=0.15,  # At least 15% unsafe
    min_goal_ratio=0.05     # At least 5% goal
)
```

### 3. Predefined Configurations

Ready-to-use dataset creators:

```python
# Warehouse robot scenario
warehouse_data = create_warehouse_dataset(num_transitions=10000)

# Simple navigation scenario  
nav_data = create_navigation_dataset(num_transitions=5000)

# Mixed dataset from multiple sources
mixed_data = create_mixed_dataset(num_transitions=15000)
```

## Usage Examples

### Basic Usage

```python
from src.dataset import create_navigation_dataset
from src import create_trainer

# Generate dataset
dataset = create_navigation_dataset(1000)

# Create trainer
trainer = create_trainer(state_dim=4, action_dim=2)

# Add data to trainer
for transition in dataset:
    trainer.add_transition(transition)
```

### Custom Environment

```python
from src.dataset import RuleBasedDatasetGenerator

# Define custom environment
generator = RuleBasedDatasetGenerator(
    workspace_bounds=(0.0, 8.0, 0.0, 6.0),
    obstacles=[
        {'center': np.array([2.0, 3.0]), 'radius': 0.5},
        {'center': np.array([6.0, 2.0]), 'radius': 0.4},
    ],
    goal_regions=[
        {'center': np.array([7.0, 5.0]), 'radius': 0.3},
    ]
)

# Generate labeled data
transitions = generator.generate_transitions(2000)
```

### Multiple Data Sources

```python
from src.dataset import DatasetManager, RuleBasedDatasetGenerator

manager = DatasetManager()

# Add different generators
safe_generator = RuleBasedDatasetGenerator(...)  # Mostly safe data
risky_generator = RuleBasedDatasetGenerator(...)  # More unsafe scenarios

manager.add_generator(safe_generator, weight=0.7)
manager.add_generator(risky_generator, weight=0.3)

# Generate balanced dataset
dataset = manager.generate_balanced_dataset(
    total_transitions=5000,
    min_unsafe_ratio=0.2,
    min_goal_ratio=0.1
)
```

## Dataset Statistics

Analyze your dataset composition:

```python
from src.dataset import DatasetManager

manager = DatasetManager()
stats = manager.get_dataset_statistics(dataset)

print(f"Total: {stats['total_transitions']}")
print(f"Safe: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
print(f"Unsafe: {stats['unsafe_transitions']}")
print(f"Goal: {stats['goal_transitions']} ({stats['goal_ratio']:.1%})")
```

## Integration with Training

### Hierarchical Control Example

```python
from src.dataset import create_navigation_dataset
from example_hierarchical import HierarchicalRobotController

# Generate training data
dataset = create_navigation_dataset(1500)

# Create controller
controller = HierarchicalRobotController()

# Add dataset to safety trainer
for transition in dataset:
    controller.safety_trainer.add_transition(transition)

# Now ready for hierarchical control
```

### Custom Labeling Rules

```python
def custom_safety_check(state, next_state):
    """Custom safety labeling logic."""
    pos = next_state[:2]
    
    # Custom safety rules
    if pos[0] < 0.5 or pos[0] > 4.5:  # Near boundaries
        return False
    if np.linalg.norm(pos - np.array([2.5, 2.0])) < 0.6:  # Near obstacle
        return False
    
    return True

def custom_goal_check(state):
    """Custom goal labeling logic."""
    pos = state[:2]
    target = np.array([4.0, 3.5])
    
    return np.linalg.norm(pos - target) < 0.25

# Use in generator
generator = RuleBasedDatasetGenerator(
    workspace_bounds=(0.0, 5.0, 0.0, 4.0),
    # Override default labeling with custom functions
)
```

## Benefits

1. **Modularity**: Easy to swap different data sources
2. **Flexibility**: Support for simulation, rules, and expert data
3. **Scalability**: Generate datasets of any size
4. **Balance Control**: Ensure proper ratios of safe/unsafe/goal data
5. **Reusability**: Predefined configurations for common scenarios
6. **Analysis**: Built-in statistics and visualization tools

## Best Practices

1. **Start Simple**: Use predefined datasets first
2. **Balance Data**: Ensure sufficient unsafe and goal examples
3. **Validate Labels**: Check dataset statistics before training
4. **Mix Sources**: Combine multiple generators for robustness
5. **Domain-Specific**: Customize generators for your specific robot/environment
6. **Iterative**: Start small, analyze results, then scale up

This modular approach makes it easy to experiment with different training data while maintaining clean separation between data generation and model training.