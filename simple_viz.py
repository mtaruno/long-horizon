"""
Simple text-based visualization of the simulation.
"""

import numpy as np
from src.dataset import RuleBasedDatasetGenerator

def create_ascii_visualization(generator, num_samples=20):
    """Create ASCII visualization of the environment and transitions."""
    
    # Get workspace bounds
    x_min, x_max, y_min, y_max = generator.workspace_bounds
    
    # Create grid (50x40 characters)
    width, height = 60, 30
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Helper function to convert world coordinates to grid coordinates
    def world_to_grid(x, y):
        grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
        grid_y = height - 1 - grid_y  # Flip Y axis
        return max(0, min(width-1, grid_x)), max(0, min(height-1, grid_y))
    
    # Draw obstacles
    for obstacle in generator.obstacles:
        center = obstacle['center']
        radius = obstacle['radius']
        cx, cy = world_to_grid(center[0], center[1])
        
        # Draw obstacle as filled circle
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx*dx + dy*dy <= 9:  # Rough circle
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < width and 0 <= gy < height:
                        grid[gy][gx] = '█'
    
    # Draw goal regions
    for goal in generator.goal_regions:
        center = goal['center']
        radius = goal['radius']
        cx, cy = world_to_grid(center[0], center[1])
        
        # Draw goal as circle outline
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if 2 <= dx*dx + dy*dy <= 4:  # Circle outline
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < width and 0 <= gy < height and grid[gy][gx] == ' ':
                        grid[gy][gx] = '○'
    
    # Generate sample transitions
    print(f"Generating {num_samples} sample transitions...")
    transitions = generator.generate_transitions(num_samples)
    
    # Draw transitions
    safe_count = unsafe_count = goal_count = 0
    
    for transition in transitions:
        pos = transition.state[:2]
        gx, gy = world_to_grid(pos[0], pos[1])
        
        if transition.is_goal:
            if grid[gy][gx] == ' ':
                grid[gy][gx] = '★'
            goal_count += 1
        elif transition.is_safe:
            if grid[gy][gx] == ' ':
                grid[gy][gx] = '·'
            safe_count += 1
        else:
            if grid[gy][gx] == ' ':
                grid[gy][gx] = '✗'
            unsafe_count += 1
    
    # Print the visualization
    print("\n" + "="*70)
    print("WAREHOUSE ROBOT SIMULATION VISUALIZATION")
    print("="*70)
    print("Legend:")
    print("  █ = Obstacles (shelves, pillars)")
    print("  ○ = Goal regions (delivery points)")
    print("  · = Safe robot positions")
    print("  ✗ = Unsafe robot positions (collisions)")
    print("  ★ = Goal-reaching positions")
    print("-"*70)
    
    # Print grid
    for row in grid:
        print(''.join(row))
    
    print("-"*70)
    print(f"Statistics: Safe={safe_count}, Unsafe={unsafe_count}, Goal={goal_count}")
    print(f"Safety Rate: {safe_count/(safe_count+unsafe_count)*100:.1f}%")
    print("="*70)

def show_transition_details(generator, num_examples=5):
    """Show detailed transition information."""
    print(f"\nDETAILED TRANSITION EXAMPLES ({num_examples} samples):")
    print("="*70)
    
    transitions = generator.generate_transitions(num_examples)
    
    for i, t in enumerate(transitions, 1):
        print(f"\nTransition #{i}:")
        print(f"  Current State: pos=({t.state[0]:.2f}, {t.state[1]:.2f}), vel=({t.state[2]:.2f}, {t.state[3]:.2f})")
        print(f"  Action Taken:  acc=({t.action[0]:.2f}, {t.action[1]:.2f})")
        print(f"  Next State:    pos=({t.next_state[0]:.2f}, {t.next_state[1]:.2f}), vel=({t.next_state[2]:.2f}, {t.next_state[3]:.2f})")
        print(f"  Safety:        {'SAFE' if t.is_safe else 'UNSAFE'}")
        print(f"  Goal:          {'GOAL REACHED' if t.is_goal else 'Not at goal'}")
        print(f"  Reward:        {t.reward:.2f}")
        
        # Explain the transition
        if not t.is_safe:
            print(f"  → COLLISION: Robot hit obstacle or boundary")
        elif t.is_goal:
            print(f"  → SUCCESS: Robot reached delivery point")
        else:
            print(f"  → NORMAL: Robot moving safely in workspace")

def create_warehouse_generator():
    """Create the warehouse environment generator."""
    workspace_bounds = (0.0, 12.0, 0.0, 10.0)
    
    obstacles = [
        {'center': np.array([2.0, 3.0]), 'radius': 0.8},
        {'center': np.array([2.0, 7.0]), 'radius': 0.8},
        {'center': np.array([5.0, 2.0]), 'radius': 0.6},
        {'center': np.array([5.0, 5.0]), 'radius': 0.6},
        {'center': np.array([5.0, 8.0]), 'radius': 0.6},
        {'center': np.array([8.0, 3.5]), 'radius': 0.7},
        {'center': np.array([8.0, 6.5]), 'radius': 0.7},
        {'center': np.array([4.0, 9.0]), 'radius': 0.3},
        {'center': np.array([9.0, 1.0]), 'radius': 0.3},
    ]
    
    goal_regions = [
        {'center': np.array([10.5, 8.5]), 'radius': 0.4},
        {'center': np.array([10.5, 1.5]), 'radius': 0.4},
        {'center': np.array([1.5, 9.0]), 'radius': 0.3},
        {'center': np.array([6.5, 0.5]), 'radius': 0.3},
    ]
    
    return RuleBasedDatasetGenerator(
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        goal_regions=goal_regions
    )

if __name__ == "__main__":
    print("Creating simple visualization of robot simulation...")
    
    # Create generator
    generator = create_warehouse_generator()
    
    # Show ASCII visualization
    create_ascii_visualization(generator, num_samples=100)
    
    # Show detailed examples
    show_transition_details(generator, num_examples=3)