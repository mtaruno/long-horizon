"""
Integration of CBF-CLF safety framework with high-level planners (FSM, LLM).
"""

import torch
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src import create_trainer

# ===== HIGH-LEVEL PLANNER INTERFACES =====

class RobotState(Enum):
    """FSM states for robot behavior."""
    IDLE = "idle"
    NAVIGATE_TO_GOAL = "navigate_to_goal"
    AVOID_OBSTACLE = "avoid_obstacle"
    PICKUP_OBJECT = "pickup_object"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class HighLevelCommand:
    """High-level command from planner."""
    target_position: np.ndarray  # [x, y]
    target_velocity: np.ndarray  # [vx, vy]
    priority: str  # "safety", "efficiency", "exploration"
    timeout: float = 5.0

class FSMPlanner:
    """Finite State Machine high-level planner."""
    
    def __init__(self):
        self.current_state = RobotState.IDLE
        self.goal_position = np.array([0.0, 0.0])
        self.obstacles = []
        
    def update(self, robot_pos: np.ndarray, sensor_data: Dict) -> HighLevelCommand:
        """Update FSM and return high-level command."""
        
        # State transitions based on sensor data
        if sensor_data.get("emergency", False):
            self.current_state = RobotState.EMERGENCY_STOP
        elif sensor_data.get("obstacle_detected", False):
            self.current_state = RobotState.AVOID_OBSTACLE
        elif np.linalg.norm(robot_pos - self.goal_position) > 0.1:
            self.current_state = RobotState.NAVIGATE_TO_GOAL
        else:
            self.current_state = RobotState.IDLE
            
        # Generate commands based on current state
        if self.current_state == RobotState.NAVIGATE_TO_GOAL:
            direction = self.goal_position - robot_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            return HighLevelCommand(
                target_position=self.goal_position,
                target_velocity=direction * 0.5,  # 0.5 m/s toward goal
                priority="efficiency"
            )
            
        elif self.current_state == RobotState.AVOID_OBSTACLE:
            # Simple obstacle avoidance - move perpendicular to obstacle
            obstacle_dir = sensor_data.get("obstacle_direction", np.array([1.0, 0.0]))
            avoid_dir = np.array([-obstacle_dir[1], obstacle_dir[0]])  # Perpendicular
            return HighLevelCommand(
                target_position=robot_pos + avoid_dir * 2.0,
                target_velocity=avoid_dir * 0.3,
                priority="safety"
            )
            
        elif self.current_state == RobotState.EMERGENCY_STOP:
            return HighLevelCommand(
                target_position=robot_pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="safety"
            )
            
        else:  # IDLE
            return HighLevelCommand(
                target_position=robot_pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="efficiency"
            )

class LLMPlanner:
    """LLM-based high-level planner (simplified)."""
    
    def __init__(self):
        self.conversation_history = []
        self.current_task = "explore"
        
    def process_natural_language(self, command: str, robot_state: np.ndarray) -> HighLevelCommand:
        """Process natural language command and return high-level action."""
        
        # Simplified LLM processing (in practice, use actual LLM API)
        command_lower = command.lower()
        robot_pos = robot_state[:2]
        
        if "go to" in command_lower or "move to" in command_lower:
            # Extract target coordinates (simplified parsing)
            if "origin" in command_lower:
                target = np.array([0.0, 0.0])
            elif "corner" in command_lower:
                target = np.array([5.0, 5.0])
            else:
                # Default target
                target = np.array([2.0, 2.0])
                
            return HighLevelCommand(
                target_position=target,
                target_velocity=(target - robot_pos) * 0.2,
                priority="efficiency"
            )
            
        elif "stop" in command_lower or "halt" in command_lower:
            return HighLevelCommand(
                target_position=robot_pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="safety"
            )
            
        elif "explore" in command_lower:
            # Random exploration
            random_target = robot_pos + np.random.randn(2) * 2.0
            return HighLevelCommand(
                target_position=random_target,
                target_velocity=np.random.randn(2) * 0.3,
                priority="exploration"
            )
            
        else:
            # Default: stay in place
            return HighLevelCommand(
                target_position=robot_pos,
                target_velocity=np.array([0.0, 0.0]),
                priority="efficiency"
            )

# ===== INTEGRATION LAYER =====

class HierarchicalController:
    """Integrates high-level planners with CBF-CLF safety framework."""
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        # Initialize CBF-CLF safety framework
        self.safety_trainer = create_trainer(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu",
            batch_size=32
        )
        
        # High-level planners
        self.fsm_planner = FSMPlanner()
        self.llm_planner = LLMPlanner()
        
        # Control parameters
        self.dt = 0.1  # Control timestep
        self.max_acceleration = 2.0  # m/s²
        
    def high_level_to_low_level(
        self, 
        command: HighLevelCommand, 
        current_state: np.ndarray
    ) -> np.ndarray:
        """Convert high-level command to low-level action."""
        
        current_pos = current_state[:2]  # [x, y]
        current_vel = current_state[2:]  # [vx, vy]
        
        # Compute desired acceleration using simple PD controller
        pos_error = command.target_position - current_pos
        vel_error = command.target_velocity - current_vel
        
        # PD gains based on priority
        if command.priority == "safety":
            kp, kd = 2.0, 1.0  # Conservative gains
        elif command.priority == "efficiency":
            kp, kd = 5.0, 2.0  # Aggressive gains
        else:  # exploration
            kp, kd = 3.0, 1.5  # Moderate gains
            
        desired_acceleration = kp * pos_error + kd * vel_error
        
        # Clip to maximum acceleration
        accel_magnitude = np.linalg.norm(desired_acceleration)
        if accel_magnitude > self.max_acceleration:
            desired_acceleration = desired_acceleration / accel_magnitude * self.max_acceleration
            
        return desired_acceleration
    
    def execute_fsm_control(
        self, 
        current_state: np.ndarray, 
        sensor_data: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Execute FSM-based control with safety filtering."""
        
        # 1. High-level planning
        high_level_command = self.fsm_planner.update(current_state[:2], sensor_data)
        
        # 2. Convert to low-level action
        proposed_action = self.high_level_to_low_level(high_level_command, current_state)
        
        # 3. Safety filtering through CBF-CLF
        safe_action = self.safety_trainer.get_safe_action(current_state, proposed_action)
        
        # 4. Return results
        control_info = {
            "fsm_state": self.fsm_planner.current_state.value,
            "high_level_command": high_level_command,
            "proposed_action": proposed_action,
            "safe_action": safe_action.numpy(),
            "action_modified": not np.allclose(proposed_action, safe_action.numpy(), atol=1e-6)
        }
        
        return safe_action.numpy(), control_info
    
    def execute_llm_control(
        self, 
        current_state: np.ndarray, 
        natural_language_command: str
    ) -> Tuple[np.ndarray, Dict]:
        """Execute LLM-based control with safety filtering."""
        
        # 1. High-level planning from natural language
        high_level_command = self.llm_planner.process_natural_language(
            natural_language_command, current_state
        )
        
        # 2. Convert to low-level action
        proposed_action = self.high_level_to_low_level(high_level_command, current_state)
        
        # 3. Safety filtering through CBF-CLF
        safe_action = self.safety_trainer.get_safe_action(current_state, proposed_action)
        
        # 4. Return results
        control_info = {
            "natural_language": natural_language_command,
            "high_level_command": high_level_command,
            "proposed_action": proposed_action,
            "safe_action": safe_action.numpy(),
            "action_modified": not np.allclose(proposed_action, safe_action.numpy(), atol=1e-6)
        }
        
        return safe_action.numpy(), control_info

# ===== DEMONSTRATION =====

def demonstrate_hierarchical_control():
    """Demonstrate integration of high-level planners with CBF-CLF safety."""
    
    print("=== HIERARCHICAL CONTROL INTEGRATION ===\n")
    
    # Initialize controller
    controller = HierarchicalController()
    
    # Set FSM goal
    controller.fsm_planner.goal_position = np.array([3.0, 2.0])
    
    # Simulate robot states
    scenarios = [
        {
            "name": "Normal Navigation",
            "state": np.array([1.0, 1.0, 0.1, 0.1]),
            "sensors": {"obstacle_detected": False, "emergency": False},
            "llm_command": "go to the corner"
        },
        {
            "name": "Obstacle Avoidance", 
            "state": np.array([2.0, 1.5, 0.2, 0.0]),
            "sensors": {
                "obstacle_detected": True, 
                "obstacle_direction": np.array([1.0, 0.0]),
                "emergency": False
            },
            "llm_command": "move carefully around the obstacle"
        },
        {
            "name": "Emergency Stop",
            "state": np.array([2.5, 2.0, 0.3, 0.2]),
            "sensors": {"obstacle_detected": False, "emergency": True},
            "llm_command": "stop immediately"
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print("-" * 50)
        
        state = scenario["state"]
        sensors = scenario["sensors"]
        llm_cmd = scenario["llm_command"]
        
        print(f"Robot state: [x={state[0]:.1f}, y={state[1]:.1f}, vx={state[2]:.1f}, vy={state[3]:.1f}]")
        print(f"Sensors: {sensors}")
        print(f"LLM command: '{llm_cmd}'")
        
        # FSM Control
        print(f"\n--- FSM CONTROL ---")
        fsm_action, fsm_info = controller.execute_fsm_control(state, sensors)
        
        print(f"FSM State: {fsm_info['fsm_state']}")
        print(f"Target pos: {fsm_info['high_level_command'].target_position}")
        print(f"Target vel: {fsm_info['high_level_command'].target_velocity}")
        print(f"Proposed action: {fsm_info['proposed_action']}")
        print(f"Safe action: {fsm_info['safe_action']}")
        print(f"Action modified: {fsm_info['action_modified']}")
        
        # LLM Control
        print(f"\n--- LLM CONTROL ---")
        llm_action, llm_info = controller.execute_llm_control(state, llm_cmd)
        
        print(f"Interpreted target: {llm_info['high_level_command'].target_position}")
        print(f"Proposed action: {llm_info['proposed_action']}")
        print(f"Safe action: {llm_info['safe_action']}")
        print(f"Action modified: {llm_info['action_modified']}")
        
        print(f"\n" + "="*60 + "\n")

def demonstrate_safety_override():
    """Demonstrate how safety framework overrides unsafe high-level commands."""
    
    print("=== SAFETY OVERRIDE DEMONSTRATION ===\n")
    
    controller = HierarchicalController()
    
    # Simulate unsafe scenario
    unsafe_state = np.array([2.8, 1.9, 0.5, 0.3])  # Near boundary, high velocity
    
    # High-level planner suggests aggressive action
    aggressive_command = HighLevelCommand(
        target_position=np.array([4.0, 3.0]),  # Far target
        target_velocity=np.array([1.0, 0.8]),  # High velocity
        priority="efficiency"
    )
    
    print(f"Unsafe scenario:")
    print(f"State: {unsafe_state}")
    print(f"High-level target: {aggressive_command.target_position}")
    print(f"High-level velocity: {aggressive_command.target_velocity}")
    
    # Convert to low-level action
    proposed_action = controller.high_level_to_low_level(aggressive_command, unsafe_state)
    print(f"Proposed acceleration: {proposed_action}")
    
    # Safety filtering
    safe_action = controller.safety_trainer.get_safe_action(unsafe_state, proposed_action)
    
    print(f"Safe acceleration: {safe_action.numpy()}")
    print(f"Safety modification: {proposed_action - safe_action.numpy()}")
    
    # Check safety metrics
    state_tensor = torch.FloatTensor(unsafe_state).unsqueeze(0)
    action_tensor = torch.FloatTensor(proposed_action).unsqueeze(0)
    
    metrics = controller.safety_trainer.controller.get_safety_feasibility_metrics(
        state_tensor, action_tensor, controller.safety_trainer.dynamics_ensemble
    )
    
    print(f"\nSafety Analysis:")
    print(f"CBF value: {metrics['cbf_values'].item():.4f}")
    print(f"CLF value: {metrics['clf_values'].item():.4f}")
    print(f"Is safe: {metrics['is_safe'].item()}")
    print(f"CBF constraint violation: {metrics['cbf_constraints'].item():.6f}")

if __name__ == "__main__":
    demonstrate_hierarchical_control()
    demonstrate_safety_override()