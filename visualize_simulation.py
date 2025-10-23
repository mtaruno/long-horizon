"""
Simple visualization of dataset generation and simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from src.dataset import RuleBasedDatasetGenerator
from src.types import Transition
import time

class SimulationVisualizer:
    """Visualize dataset generation and robot simulation."""
    
    def __init__(self, generator: RuleBasedDatasetGenerator):
        self.generator = generator
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_environment()
        
    def setup_environment(self):
        """Setup the visualization environment."""
        # Clear and setup axes
        self.ax.clear()
        self.ax.set_xlim(self.generator.workspace_bounds[0], self.generator.workspace_bounds[1])
        self.ax.set_ylim(self.generator.workspace_bounds[2], self.generator.workspace_bounds[3])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Robot Navigation Simulation')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        
        # Draw obstacles
        for obstacle in self.generator.obstacles:
            circle = patches.Circle(
                obstacle['center'], obstacle['radius'], 
                color='red', alpha=0.7, label='Obstacle'
            )
            self.ax.add_patch(circle)
            
        # Draw goal regions
        for i, goal in enumerate(self.generator.goal_regions):
            circle = patches.Circle(
                goal['center'], goal['radius'], 
                color='green', alpha=0.5, label='Goal' if i == 0 else ""
            )
            self.ax.add_patch(circle)
            
        # Add legend
        self.ax.legend()
        
    def visualize_single_transition(self, transition: Transition):
        """Visualize a single transition."""
        state = transition.state
        next_state = transition.next_state
        action = transition.action
        
        # Current position
        pos = state[:2]
        next_pos = next_state[:2]
        
        # Plot robot position
        color = 'blue' if transition.is_safe else 'red'
        marker = 's' if transition.is_goal else 'o'
        
        self.ax.plot(pos[0], pos[1], marker, color=color, markersize=8, alpha=0.7)
        
        # Plot velocity vector
        vel = state[2:]
        if np.linalg.norm(vel) > 0.01:
            self.ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
        
        # Plot action vector
        if np.linalg.norm(action) > 0.01:
            self.ax.arrow(pos[0], pos[1], action[0]*0.2, action[1]*0.2, 
                         head_width=0.08, head_length=0.08, fc='orange', ec='orange', alpha=0.7)
        
        # Plot trajectory
        self.ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], 
                    'k--', alpha=0.3, linewidth=1)
        
        return pos, next_pos, transition.is_safe, transition.is_goal
        
    def visualize_dataset_sample(self, num_samples: int = 50):
        """Visualize a sample of transitions from the dataset."""
        print(f"Generating {num_samples} sample transitions...")
        
        # Generate sample transitions
        transitions = self.generator.generate_transitions(num_samples)
        
        # Setup environment
        self.setup_environment()
        
        # Plot all transitions
        safe_count = 0
        unsafe_count = 0
        goal_count = 0
        
        for transition in transitions:
            pos, next_pos, is_safe, is_goal = self.visualize_single_transition(transition)
            
            if is_goal:
                goal_count += 1
            elif is_safe:
                safe_count += 1
            else:
                unsafe_count += 1
        
        # Add statistics
        stats_text = f"Safe: {safe_count}, Unsafe: {unsafe_count}, Goal: {goal_count}"
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend for colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Safe'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Unsafe'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Goal'),
            plt.Line2D([0], [0], color='blue', alpha=0.5, label='Velocity'),
            plt.Line2D([0], [0], color='orange', alpha=0.7, label='Action')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def animate_robot_trajectory(self, num_steps: int = 100):
        """Animate a robot following a trajectory."""
        print(f"Animating robot trajectory for {num_steps} steps...")
        
        # Generate trajectory
        trajectory = []
        current_state = self.generator._sample_random_state()
        
        for _ in range(num_steps):
            # Sample action (could be from a policy)
            action = np.random.uniform(-0.5, 0.5, 2)
            
            # Simulate dynamics
            next_state = self.generator._simulate_dynamics(current_state, action)
            
            # Check safety and goal
            is_safe = self.generator._is_safe(current_state, next_state)
            is_goal = self.generator._is_goal(next_state)
            
            trajectory.append({
                'state': current_state.copy(),
                'next_state': next_state.copy(),
                'action': action.copy(),
                'is_safe': is_safe,
                'is_goal': is_goal
            })
            
            current_state = next_state
            
            # Reset if unsafe or goal reached
            if not is_safe or is_goal:
                current_state = self.generator._sample_random_state()
        
        # Setup animation
        self.setup_environment()
        
        # Animation elements
        robot_dot, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        trail_line, = self.ax.plot([], [], 'b-', alpha=0.3, linewidth=2, label='Trail')
        velocity_arrow = patches.FancyArrowPatch((0, 0), (0, 0), 
                                               arrowstyle='->', mutation_scale=20, color='blue', alpha=0.7)
        action_arrow = patches.FancyArrowPatch((0, 0), (0, 0), 
                                             arrowstyle='->', mutation_scale=20, color='orange', alpha=0.7)
        
        self.ax.add_patch(velocity_arrow)
        self.ax.add_patch(action_arrow)
        
        # Trail data
        trail_x, trail_y = [], []
        
        def animate(frame):
            if frame >= len(trajectory):
                return robot_dot, trail_line, velocity_arrow, action_arrow
                
            step = trajectory[frame]
            pos = step['state'][:2]
            vel = step['state'][2:]
            action = step['action']
            
            # Update robot position
            robot_dot.set_data([pos[0]], [pos[1]])
            
            # Update robot color based on safety
            color = 'green' if step['is_goal'] else ('blue' if step['is_safe'] else 'red')
            robot_dot.set_color(color)
            
            # Update trail
            trail_x.append(pos[0])
            trail_y.append(pos[1])
            trail_line.set_data(trail_x, trail_y)
            
            # Update velocity arrow
            if np.linalg.norm(vel) > 0.01:
                velocity_arrow.set_positions(pos, pos + vel * 0.5)
            else:
                velocity_arrow.set_positions(pos, pos)
            
            # Update action arrow
            if np.linalg.norm(action) > 0.01:
                action_arrow.set_positions(pos, pos + action * 0.3)
            else:
                action_arrow.set_positions(pos, pos)
            
            # Update title with current status
            status = "GOAL!" if step['is_goal'] else ("SAFE" if step['is_safe'] else "UNSAFE!")
            self.ax.set_title(f'Robot Navigation - Step {frame}: {status}')
            
            return robot_dot, trail_line, velocity_arrow, action_arrow
        
        # Create animation
        anim = FuncAnimation(self.fig, animate, frames=len(trajectory), 
                           interval=100, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

def create_warehouse_visualizer():
    """Create visualizer for warehouse environment."""
    # Warehouse environment
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
    
    generator = RuleBasedDatasetGenerator(
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        goal_regions=goal_regions
    )
    
    return SimulationVisualizer(generator)

if __name__ == "__main__":
    print("Creating warehouse simulation visualizer...")
    
    # Create visualizer
    viz = create_warehouse_visualizer()
    
    # Show menu
    print("\nVisualization Options:")
    print("1. Static dataset sample (50 transitions)")
    print("2. Animated robot trajectory (100 steps)")
    print("3. Large dataset sample (200 transitions)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        viz.visualize_dataset_sample(50)
    elif choice == "2":
        anim = viz.animate_robot_trajectory(100)
        input("Press Enter to close animation...")
    elif choice == "3":
        viz.visualize_dataset_sample(200)
    else:
        print("Invalid choice, showing default dataset sample...")
        viz.visualize_dataset_sample(50)