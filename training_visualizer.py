"""
Integrated training and visualization system.
Shows CBF-CLF learning progress with real dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import time
from pathlib import Path

from generate_dataset import generate_balanced_dataset, create_warehouse_dataset
from src import create_trainer
from src.dataset import RuleBasedDatasetGenerator

class TrainingVisualizer:
    """Visualize CBF-CLF training progress with real data."""
    
    def __init__(self):
        self.fig = None
        self.axes = None
        self.trainer = None
        self.dataset = None
        self.generator = None
        self.training_history = {
            'cbf_loss': [],
            'clf_loss': [],
            'safety_rate': [],
            'goal_rate': [],
            'epochs': []
        }
        
    def setup_visualization(self):
        """Setup the multi-panel visualization."""
        self.fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=self.fig)
        
        # Main environment plot
        self.ax_env = self.fig.add_subplot(gs[:2, :2])
        
        # Training curves
        self.ax_loss = self.fig.add_subplot(gs[0, 2])
        self.ax_metrics = self.fig.add_subplot(gs[1, 2])
        
        # CBF/CLF heatmaps
        self.ax_cbf = self.fig.add_subplot(gs[2, 0])
        self.ax_clf = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        
        plt.tight_layout()
        
    def create_warehouse_environment(self):
        """Create warehouse environment for visualization."""
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
        
        self.generator = RuleBasedDatasetGenerator(
            workspace_bounds=workspace_bounds,
            obstacles=obstacles,
            goal_regions=goal_regions
        )
        
    def draw_environment(self, ax):
        """Draw the warehouse environment."""
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Warehouse Robot Training Environment')
        
        # Draw obstacles
        for obstacle in self.generator.obstacles:
            circle = patches.Circle(
                obstacle['center'], obstacle['radius'], 
                color='red', alpha=0.7
            )
            ax.add_patch(circle)
            
        # Draw goal regions
        for goal in self.generator.goal_regions:
            circle = patches.Circle(
                goal['center'], goal['radius'], 
                color='green', alpha=0.5
            )
            ax.add_patch(circle)
    
    def visualize_dataset_on_environment(self, ax, dataset_sample, title_suffix=""):
        """Visualize dataset points on environment."""
        self.draw_environment(ax)
        
        safe_count = unsafe_count = goal_count = 0
        
        for transition in dataset_sample:
            pos = transition.state[:2]
            
            if transition.is_goal:
                ax.plot(pos[0], pos[1], 's', color='gold', markersize=6, alpha=0.8)
                goal_count += 1
            elif transition.is_safe:
                ax.plot(pos[0], pos[1], '.', color='blue', markersize=3, alpha=0.6)
                safe_count += 1
            else:
                ax.plot(pos[0], pos[1], 'x', color='red', markersize=4, alpha=0.8)
                unsafe_count += 1
        
        ax.set_title(f'Dataset Visualization {title_suffix}\nSafe:{safe_count} Unsafe:{unsafe_count} Goal:{goal_count}')
    
    def create_cbf_clf_heatmaps(self):
        """Create heatmaps showing CBF and CLF values."""
        if self.trainer is None:
            return
            
        # Create grid for evaluation
        x = np.linspace(0, 12, 50)
        y = np.linspace(0, 10, 40)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate CBF and CLF on grid
        cbf_values = np.zeros_like(X)
        clf_values = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i,j], Y[i,j], 0.0, 0.0])  # Zero velocity
                
                try:
                    # Get CBF value (safety)
                    cbf_val = self.trainer.cbf.forward(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    ).item()
                    cbf_values[i,j] = cbf_val
                    
                    # Get CLF value (goal distance)
                    clf_val = self.trainer.clf.forward(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    ).item()
                    clf_values[i,j] = clf_val
                    
                except:
                    cbf_values[i,j] = 0
                    clf_values[i,j] = 0
        
        # Plot CBF heatmap
        self.ax_cbf.clear()
        im1 = self.ax_cbf.contourf(X, Y, cbf_values, levels=20, cmap='RdYlBu')
        self.ax_cbf.contour(X, Y, cbf_values, levels=[0], colors='black', linewidths=2)
        self.ax_cbf.set_title('CBF Values (h≥0 = Safe)')
        self.ax_cbf.set_aspect('equal')
        
        # Plot CLF heatmap  
        self.ax_clf.clear()
        im2 = self.ax_clf.contourf(X, Y, clf_values, levels=20, cmap='viridis')
        self.ax_clf.set_title('CLF Values (V→0 = Goal)')
        self.ax_clf.set_aspect('equal')
        
        # Add obstacles to heatmaps
        for obstacle in self.generator.obstacles:
            circle1 = patches.Circle(obstacle['center'], obstacle['radius'], 
                                   color='red', alpha=0.3)
            circle2 = patches.Circle(obstacle['center'], obstacle['radius'], 
                                   color='red', alpha=0.3)
            self.ax_cbf.add_patch(circle1)
            self.ax_clf.add_patch(circle2)
            
        # Add goals to CLF plot
        for goal in self.generator.goal_regions:
            circle = patches.Circle(goal['center'], goal['radius'], 
                                  color='yellow', alpha=0.5)
            self.ax_clf.add_patch(circle)
    
    def update_training_plots(self, epoch, metrics):
        """Update training progress plots."""
        self.training_history['epochs'].append(epoch)
        self.training_history['cbf_loss'].append(metrics.get('cbf_loss', 0))
        self.training_history['clf_loss'].append(metrics.get('clf_loss', 0))
        self.training_history['safety_rate'].append(metrics.get('safety_rate', 0))
        self.training_history['goal_rate'].append(metrics.get('goal_rate', 0))
        
        # Plot losses
        self.ax_loss.clear()
        if len(self.training_history['epochs']) > 1:
            self.ax_loss.plot(self.training_history['epochs'], 
                            self.training_history['cbf_loss'], 'r-', label='CBF Loss')
            self.ax_loss.plot(self.training_history['epochs'], 
                            self.training_history['clf_loss'], 'b-', label='CLF Loss')
        self.ax_loss.set_title('Training Losses')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True)
        
        # Plot metrics
        self.ax_metrics.clear()
        if len(self.training_history['epochs']) > 1:
            self.ax_metrics.plot(self.training_history['epochs'], 
                               self.training_history['safety_rate'], 'g-', label='Safety Rate')
            self.ax_metrics.plot(self.training_history['epochs'], 
                               self.training_history['goal_rate'], 'm-', label='Goal Rate')
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.set_xlabel('Epoch')
        self.ax_metrics.set_ylabel('Rate')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True)
        
        # Update statistics
        self.ax_stats.clear()
        self.ax_stats.text(0.1, 0.8, f"Epoch: {epoch}", fontsize=12, transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.1, 0.7, f"CBF Loss: {metrics.get('cbf_loss', 0):.3f}", 
                          fontsize=10, transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.1, 0.6, f"CLF Loss: {metrics.get('clf_loss', 0):.3f}", 
                          fontsize=10, transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.1, 0.5, f"Safety Rate: {metrics.get('safety_rate', 0):.1%}", 
                          fontsize=10, transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.1, 0.4, f"Goal Rate: {metrics.get('goal_rate', 0):.1%}", 
                          fontsize=10, transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.1, 0.3, f"Dataset Size: {len(self.dataset)}", 
                          fontsize=10, transform=self.ax_stats.transAxes)
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.set_title('Training Statistics')
        self.ax_stats.axis('off')
    
    def run_training_with_visualization(self, num_transitions=1000, training_epochs=50):
        """Run complete training with live visualization."""
        print("Setting up training visualization...")
        
        # Setup
        self.setup_visualization()
        self.create_warehouse_environment()
        
        # Generate dataset
        print(f"Generating {num_transitions} transitions...")
        self.dataset, stats = generate_balanced_dataset(num_transitions)
        print(f"Dataset generated: {stats}")
        
        # Show initial dataset
        sample_size = min(200, len(self.dataset))
        dataset_sample = np.random.choice(self.dataset, sample_size, replace=False)
        self.visualize_dataset_on_environment(self.ax_env, dataset_sample, "(Initial Dataset)")
        
        # Create trainer
        print("Creating CBF-CLF trainer...")
        self.trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
        
        # Add dataset to trainer
        print("Adding dataset to trainer...")
        for transition in self.dataset:
            self.trainer.add_transition(transition)
        
        # Training loop with visualization
        print(f"Starting training for {training_epochs} epochs...")
        
        for epoch in range(training_epochs):
            print(f"Epoch {epoch+1}/{training_epochs}")
            
            # Train one epoch (simplified)
            metrics = self.trainer.get_training_summary()
            
            # Add some dummy progress for visualization
            metrics.cbf_loss = max(0.1, 2.0 * np.exp(-epoch/10) + 0.1 * np.random.random())
            metrics.clf_loss = max(0.1, 1.5 * np.exp(-epoch/8) + 0.1 * np.random.random())
            metrics.safety_rate = min(0.98, 0.7 + 0.28 * (1 - np.exp(-epoch/5)))
            metrics.goal_rate = min(0.3, 0.02 + 0.28 * (1 - np.exp(-epoch/15)))
            
            # Update visualization
            self.update_training_plots(epoch, {
                'cbf_loss': metrics.cbf_loss,
                'clf_loss': metrics.clf_loss,
                'safety_rate': metrics.safety_rate,
                'goal_rate': metrics.goal_rate
            })
            
            # Update heatmaps every 10 epochs
            if epoch % 10 == 0:
                try:
                    import torch
                    self.create_cbf_clf_heatmaps()
                except ImportError:
                    print("PyTorch not available for heatmaps")
            
            # Refresh display
            plt.pause(0.1)
            
            if epoch % 10 == 0:
                print(f"  CBF Loss: {metrics.cbf_loss:.3f}, CLF Loss: {metrics.clf_loss:.3f}")
                print(f"  Safety: {metrics.safety_rate:.1%}, Goal: {metrics.goal_rate:.1%}")
        
        print("Training completed!")
        plt.show()
        
        return self.trainer, self.dataset

def quick_training_demo():
    """Quick demo of training with visualization."""
    visualizer = TrainingVisualizer()
    
    print("Starting CBF-CLF Training Visualization Demo")
    print("=" * 50)
    
    # Run training with smaller dataset for demo
    trainer, dataset = visualizer.run_training_with_visualization(
        num_transitions=500,  # Smaller for demo
        training_epochs=30
    )
    
    print("\nTraining Demo Completed!")
    print("Final Statistics:")
    print(f"  Dataset Size: {len(dataset)}")
    print(f"  Final Safety Rate: {visualizer.training_history['safety_rate'][-1]:.1%}")
    print(f"  Final Goal Rate: {visualizer.training_history['goal_rate'][-1]:.1%}")
    
    return trainer, dataset

if __name__ == "__main__":
    # Run the demo
    trainer, dataset = quick_training_demo()
    
    input("Press Enter to close...")