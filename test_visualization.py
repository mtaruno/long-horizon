"""
Quick test of the visualization system.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
from visualize_simulation import create_warehouse_visualizer

def test_visualization():
    """Test the visualization system."""
    print("Testing visualization system...")
    
    # Create visualizer
    viz = create_warehouse_visualizer()
    
    # Generate and save a static plot
    print("Generating sample dataset visualization...")
    viz.visualize_dataset_sample(30)
    
    # Save the plot
    plt.savefig('data/simulation_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to data/simulation_visualization.png")
    
    # Print some sample data for understanding
    print("\nSample transitions:")
    transitions = viz.generator.generate_transitions(5)
    
    for i, t in enumerate(transitions):
        print(f"\nTransition {i+1}:")
        print(f"  Position: ({t.state[0]:.2f}, {t.state[1]:.2f})")
        print(f"  Velocity: ({t.state[2]:.2f}, {t.state[3]:.2f})")
        print(f"  Action: ({t.action[0]:.2f}, {t.action[1]:.2f})")
        print(f"  Next Position: ({t.next_state[0]:.2f}, {t.next_state[1]:.2f})")
        print(f"  Safe: {t.is_safe}, Goal: {t.is_goal}, Reward: {t.reward:.2f}")

if __name__ == "__main__":
    test_visualization()