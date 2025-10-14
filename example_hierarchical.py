"""
Example using hierarchical control with FSM/LLM planners + CBF-CLF safety.
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
from generate_dataset import load_dataset

# ===== HIGH-LEVEL PLANNER DEFINITIONS =====

class RobotMode(Enum):
    """High-level robot behaviors."""
    IDLE = "idle"
    NAVIGATE = "navigate"
    AVOID_OBSTACLE = "avoid_obstacle"
    APPROACH_GOAL = "approach_goal"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class HighLevelCommand:
    """Command from high-level planner."""
    target_position: np.ndarray
    target_velocity: np.ndarray
    priority: str  # "safety", "efficiency", "exploration"
    mode: RobotMode

class FSMPlanner:
    """Finite State Machine planner."""
    
    def __init__(self, goal_position: np.ndarray):
        self.goal_position = goal_position
        self.current_mode = RobotMode.IDLE
        self.obstacle_threshold = 0.5
        
    def update(self, state: np.ndarray, sensors: Dict) -> HighLevelCommand:
        """Update FSM and generate command."""
        pos = state[:2]
        vel = state[2:]
        
        # State transitions
        if sensors.get("emergency", False):
            self.current_mode = RobotMode.EMERGENCY_STOP
        elif sensors.get("obstacle_distance", float('inf')) < self.obstacle_threshold:
            self.current_mode = RobotMode.AVOID_OBSTACLE
        elif np.linalg.norm(pos - self.goal_position) < 0.2:
            self.current_mode = RobotMode.APPROACH_GOAL
        elif np.linalg.norm(pos - self.goal_position) > 0.2:
            self.current_mode = RobotMode.NAVIGATE
        else:
            self.current_mode = RobotMode.IDLE
            
        # Generate commands based on mode
        if self.current_mode == RobotMode.NAVIGATE:
            direction = self.goal_position - pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            return HighLevelCommand(
                target_position=self.goal_position,
                target_velocity=direction * 0.8,
                priority="efficiency",
                mode=self.current_mode
            )
            
        elif self.current_mode == RobotMode.AVOID_OBSTACLE:
            obstacle_dir = sensors.get("obstacle_direction", np.array([1.0, 0.0]))
            avoid_dir = np.array([-obstacle_dir[1], obstacle_dir[0]])  # Perpendicular
            return HighLevelCommand(
                target_position=pos + avoid_dir * 1.5,
                target_velocity=avoid_dir * 0.5,
                priority="safety",
                mode=self.current_mode
            )
            
        elif self.current_mode == RobotMode.APPROACH_GOAL:
            direction = self.goal_position - pos
            return HighLevelCommand(
                target_position=self.goal_position,
                target_velocity=direction * 0.3,  # Slower approach
                priority="efficiency",
                mode=self.current_mode
            )
            
        elif self.current_mode == RobotMode.EMERGENCY_STOP:
            return HighLevelCommand(
                target_position=pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="safety",
                mode=self.current_mode
            )
            
        else:  # IDLE
            return HighLevelCommand(
                target_position=pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="efficiency",
                mode=self.current_mode
            )

class LLMPlanner:
    """Simplified LLM-based planner."""
    
    def __init__(self):
        self.current_task = "explore"
        self.waypoints = []
        
    def process_command(self, command: str, state: np.ndarray) -> HighLevelCommand:
        """Process natural language command."""
        pos = state[:2]
        command_lower = command.lower()
        
        if "go to" in command_lower or "navigate to" in command_lower:
            if "origin" in command_lower or "home" in command_lower:
                target = np.array([0.0, 0.0])
            elif "corner" in command_lower:
                target = np.array([4.0, 3.0])
            elif "station" in command_lower:
                target = np.array([2.0, 2.0])
            else:
                target = np.array([3.0, 1.5])  # Default
                
            direction = target - pos
            return HighLevelCommand(
                target_position=target,
                target_velocity=direction * 0.4,
                priority="efficiency",
                mode=RobotMode.NAVIGATE
            )
            
        elif "stop" in command_lower or "halt" in command_lower:
            return HighLevelCommand(
                target_position=pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="safety",
                mode=RobotMode.EMERGENCY_STOP
            )
            
        elif "explore" in command_lower or "patrol" in command_lower:
            # Random exploration
            random_target = pos + np.random.uniform(-2.0, 2.0, 2)
            return HighLevelCommand(
                target_position=random_target,
                target_velocity=np.random.uniform(-0.3, 0.3, 2),
                priority="exploration",
                mode=RobotMode.NAVIGATE
            )
            
        else:
            return HighLevelCommand(
                target_position=pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="efficiency",
                mode=RobotMode.IDLE
            )

# ===== HIERARCHICAL CONTROLLER =====

class HierarchicalRobotController:
    """Complete hierarchical control system."""
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        # Initialize CBF-CLF safety framework
        self.safety_trainer = create_trainer(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu",
            batch_size=64,
            cbf_update_freq=10,
            clf_update_freq=10
        )
        
        # Initialize planners
        self.fsm_planner = FSMPlanner(goal_position=np.array([3.0, 2.0]))
        self.llm_planner = LLMPlanner()
        
        # Control parameters
        self.max_acceleration = 2.0
        self.dt = 0.1
        
    def pd_controller(self, command: HighLevelCommand, state: np.ndarray) -> np.ndarray:
        """Convert high-level command to acceleration."""
        pos = state[:2]
        vel = state[2:]
        
        pos_error = command.target_position - pos
        vel_error = command.target_velocity - vel
        
        # Adaptive gains based on priority
        if command.priority == "safety":
            Kp, Kd = 2.0, 1.5
        elif command.priority == "efficiency":
            Kp, Kd = 4.0, 2.0
        else:  # exploration
            Kp, Kd = 3.0, 1.8
            
        acceleration = Kp * pos_error + Kd * vel_error
        
        # Clip to limits
        accel_mag = np.linalg.norm(acceleration)
        if accel_mag > self.max_acceleration:
            acceleration = acceleration / accel_mag * self.max_acceleration
            
        return acceleration
    
    def execute_fsm_control(self, state: np.ndarray, sensors: Dict) -> Dict:
        """Execute FSM-based hierarchical control."""
        # High-level planning
        command = self.fsm_planner.update(state, sensors)
        
        # Low-level control
        proposed_action = self.pd_controller(command, state)
        
        # Safety filtering
        safe_action = self.safety_trainer.get_safe_action(state, proposed_action)
        
        return {
            "command": command,
            "proposed_action": proposed_action,
            "safe_action": safe_action.numpy(),
            "mode": command.mode.value,
            "priority": command.priority
        }
    
    def execute_llm_control(self, state: np.ndarray, nlp_command: str) -> Dict:
        """Execute LLM-based hierarchical control."""
        # High-level planning
        command = self.llm_planner.process_command(nlp_command, state)
        
        # Low-level control
        proposed_action = self.pd_controller(command, state)
        
        # Safety filtering
        safe_action = self.safety_trainer.get_safe_action(state, proposed_action)
        
        return {
            "command": command,
            "proposed_action": proposed_action,
            "safe_action": safe_action.numpy(),
            "nlp_command": nlp_command,
            "interpreted_mode": command.mode.value
        }

# ===== SIMULATION ENVIRONMENT =====

class SimpleEnvironment:
    """Simple 2D environment for testing."""
    
    def __init__(self, bounds: tuple = (5.0, 4.0)):
        self.bounds = bounds
        self.obstacles = [
            {"center": np.array([2.5, 1.5]), "radius": 0.3},
            {"center": np.array([1.0, 3.0]), "radius": 0.2},
        ]
        
    def step(self, state: np.ndarray, action: np.ndarray) -> tuple:
        """Simulate one environment step."""
        # Physics integration
        new_vel = state[2:] + action * 0.1
        new_pos = state[:2] + new_vel * 0.1
        new_state = np.concatenate([new_pos, new_vel])
        
        # Check collisions
        collision = self._check_collision(new_pos)
        
        # Check bounds
        out_of_bounds = (new_pos[0] < 0 or new_pos[0] > self.bounds[0] or
                        new_pos[1] < 0 or new_pos[1] > self.bounds[1])
        
        # Compute reward
        goal_pos = np.array([3.0, 2.0])
        distance_to_goal = np.linalg.norm(new_pos - goal_pos)
        reward = -distance_to_goal
        
        if collision or out_of_bounds:
            reward -= 10.0
        
        # Check goal
        goal_reached = distance_to_goal < 0.15
        if goal_reached:
            reward += 50.0
            
        # Generate sensor data
        sensors = self._get_sensor_data(new_pos)
        
        return new_state, reward, goal_reached or collision, sensors
    
    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check collision with obstacles."""
        for obstacle in self.obstacles:
            if np.linalg.norm(pos - obstacle["center"]) < obstacle["radius"]:
                return True
        return False
    
    def _get_sensor_data(self, pos: np.ndarray) -> Dict:
        """Generate sensor readings."""
        min_distance = float('inf')
        closest_obstacle_dir = np.array([1.0, 0.0])
        
        for obstacle in self.obstacles:
            distance = np.linalg.norm(pos - obstacle["center"])
            if distance < min_distance:
                min_distance = distance
                closest_obstacle_dir = (obstacle["center"] - pos) / (distance + 1e-6)
        
        return {
            "obstacle_distance": min_distance,
            "obstacle_direction": closest_obstacle_dir,
            "emergency": min_distance < 0.1
        }

# ===== MAIN EXAMPLE =====

def main():
    """Complete hierarchical control example."""
    
    print("=== HIERARCHICAL ROBOT CONTROL EXAMPLE ===\n")
    
    # Initialize system
    controller = HierarchicalRobotController()
    env = SimpleEnvironment()
    
    # Training phase: Generate dataset using modular system
    print("PHASE 1: Dataset Generation")
    print("-" * 40)

    
    all_transitions, stats, metadata = load_dataset("warehouse_robot_dataset")

    trainer = create_trainer(
            state_dim=metadata['state_dim'],
            action_dim=metadata['action_dim'],
            device="cpu",
            batch_size=64
        )
    
    # Add to trainer
    for transition in all_transitions: # dataset is a series of transitions
        controller.safety_trainer.add_transition(transition)
    
    # Get dataset statistics
    manager = DatasetManager()
    stats = manager.get_dataset_statistics(all_transitions)
    
    print(f"Dataset Statistics:")
    print(f"  - Total transitions: {stats['total_transitions']}")
    print(f"  - Safe transitions: {stats['safe_transitions']} ({stats['safety_ratio']:.1%})")
    print(f"  - Unsafe transitions: {stats['unsafe_transitions']}")
    print(f"  - Goal transitions: {stats['goal_transitions']} ({stats['goal_ratio']:.1%})")
    print(f"  - Average reward: {stats['avg_reward']:.2f}")
    
    # Testing phase: Use hierarchical control
    print(f"\nPHASE 2: Hierarchical Control Testing")
    print("-" * 40)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "FSM Navigation",
            "initial_state": np.array([0.5, 0.5, 0.0, 0.0]),
            "control_type": "fsm"
        },
        {
            "name": "LLM Command: 'go to corner'",
            "initial_state": np.array([1.0, 1.0, 0.1, 0.0]),
            "control_type": "llm",
            "command": "go to corner"
        },
        {
            "name": "LLM Command: 'explore the area'",
            "initial_state": np.array([2.0, 1.5, 0.0, 0.0]),
            "control_type": "llm", 
            "command": "explore the area"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nSCENARIO: {scenario['name']}")
        print("-" * 30)
        
        state = scenario["initial_state"]
        
        for step in range(15):
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
            
            print(f"  State: [{state[0]:.2f}, {state[1]:.2f}] → [{next_state[0]:.2f}, {next_state[1]:.2f}]")
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
    
    print(f"\nHierarchical control system ready for deployment!")

if __name__ == "__main__":
    main()