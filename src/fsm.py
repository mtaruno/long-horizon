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

class RobotMode(Enum):
    """FSM states for robot behavior."""
    IDLE = "idle"
    NAVIGATE_TO_GOAL = "navigate_to_goal"
    AVOID_OBSTACLE = "avoid_obstacle"
    PICKUP_OBJECT = "pickup_object"
    EMERGENCY_STOP = "emergency_stop"

# TODO: In our design later, these high level commands will be detemined by the RGB-D input
@dataclass
class HighLevelCommand:
    """High-level command from planner."""
    target_position: np.ndarray  # [x, y]
    target_velocity: np.ndarray  # [vx, vy]
    priority: str  # "safety", "efficiency", "exploration"
    timeout: float = 5.0

class FSMPlanner:
    """Finite State Machine planner.
    MDP Structure:
    - Each FSM node = set of MDP states with similar behavior
    - Each update() call = one MDP transition
    - HighLevelCommand = MDP action for current state set
    """
    
    def __init__(self, goal_position: np.ndarray):
        self.goal_position = goal_position
        self.current_mode = RobotMode.IDLE
        self.obstacle_threshold = 0.5
        
    def update(self, state: np.ndarray, sensors: Dict) -> HighLevelCommand:
        """FSM transition: determine which set of MDP states we're in."""
        pos = state[:2]
        vel = state[2:]
        
        # FSM node transitions (each represents different MDP state sets)
        if sensors.get("emergency", False):
            self.current_mode = RobotMode.EMERGENCY_STOP  # Emergency MDP states
        elif sensors.get("obstacle_distance", float('inf')) < self.obstacle_threshold:
            self.current_mode = RobotMode.AVOID_OBSTACLE  # Obstacle avoidance MDP states
        elif np.linalg.norm(pos - self.goal_position) < 0.2:
            self.current_mode = RobotMode.APPROACH_GOAL   # Goal approach MDP states
        elif np.linalg.norm(pos - self.goal_position) > 0.2:
            self.current_mode = RobotMode.NAVIGATE        # Navigation MDP states
        else:
            self.current_mode = RobotMode.IDLE            # Idle MDP states
            
        # Generate MDP action for current FSM node (set of states)
        if self.current_mode == RobotMode.NAVIGATE:
            # Action for navigation MDP states: move toward goal
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
            ))

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
                target = np.array([10.5, 8.5])  # Warehouse corner
            elif "dock" in command_lower or "loading" in command_lower:
                target = np.array([10.5, 1.5])  # Loading dock
            else:
                target = np.array([6.0, 5.0])  # Default warehouse center
                
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
            # Random exploration within warehouse bounds
            random_target = np.array([
                np.random.uniform(1.0, 11.0),
                np.random.uniform(1.0, 9.0)
            ])
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


# ===== HIERARCHICAL CONTROLLER =====

class FSMRobotController:
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
        
        # Initialize planners for warehouse environment
        self.fsm_planner = FSMPlanner(goal_position=np.array([10.5, 8.5]))  # Loading dock
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
a