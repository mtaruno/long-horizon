import numpy as np
import torch
from typing import Dict, Any, Tuple

from src.utils.buffer import ReplayBuffer
from src.core.policy import SubgoalConditionedPolicy
from src.core.critics import CBFNetwork, CLFNetwork
from src.core.models import EnsembleDynamicsModel




class FSMAutomaton:
    """
    Implements the Finite State Machine (FSM) planner.
    Handles state transitions and FSM pruning (Algorithm 1).
    
    Automatically builds a linear FSM chain from waypoints:
    START -> WAYPOINT_1 -> WAYPOINT_2 -> ... -> GOAL
    """
    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray, config: Dict[str, Any]):

        FSM_STATE_START = "START"
        FSM_STATE_GOAL = "GOAL"
        FSM_STATE_FAILED = "FAILED"

        self.fsm_config = config['fsm']
        self.clf_config = config['train']
        self.FSM_STATE_FAILED = FSM_STATE_FAILED
        self.start_node = FSM_STATE_START
        self.goal_node = FSM_STATE_GOAL
        
        # Get waypoints from config (list of [x, y] positions)
        waypoint_positions = config["fsm"].get("waypoints", [])
        waypoint_positions = [np.array(wp) for wp in waypoint_positions]
        
        # Build FSM states: START, WAYPOINT_1, WAYPOINT_2, ..., GOAL
        self.all_states = [FSM_STATE_START]
        self.waypoint_states = []
        for i in range(len(waypoint_positions)):
            state_name = f"WAYPOINT_{i+1}"
            self.waypoint_states.append(state_name)
            self.all_states.append(state_name)
        self.all_states.append(FSM_STATE_GOAL)
        
        # Build transitions: linear chain START -> WP1 -> WP2 -> ... -> GOAL
        self.transitions = {}
        self.subgoals = {}
        
        # START -> first waypoint (or goal if no waypoints)
        if len(waypoint_positions) > 0:
            self.transitions[FSM_STATE_START] = [self.waypoint_states[0]]
            self.subgoals[FSM_STATE_START] = waypoint_positions[0]
        else:
            # No waypoints: direct path to goal
            self.transitions[FSM_STATE_START] = [FSM_STATE_GOAL]
            self.subgoals[FSM_STATE_START] = goal_pos
        
        # Waypoint transitions: each waypoint -> next waypoint (or goal)
        for i, wp_state in enumerate(self.waypoint_states):
            if i < len(self.waypoint_states) - 1:
                # Waypoint -> next waypoint
                next_state = self.waypoint_states[i + 1]
                self.transitions[wp_state] = [next_state]
                self.subgoals[wp_state] = waypoint_positions[i + 1]
            else:
                # Last waypoint -> goal
                self.transitions[wp_state] = [FSM_STATE_GOAL]
                self.subgoals[wp_state] = goal_pos
        
        # Goal is terminal
        self.transitions[FSM_STATE_GOAL] = []
        self.subgoals[FSM_STATE_GOAL] = goal_pos
        
        # Store waypoint positions for reference
        self.waypoint_positions = waypoint_positions
        
        self.valid_transitions = self.transitions.copy()
        self.current_state = self.start_node
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def reset(self):
        """Resets the FSM to the start state."""
        self.current_state = self.start_node
        self.valid_transitions = self.transitions.copy()

    def get_current_subgoal(self) -> np.ndarray:
        """Returns the subgoal for the current FSM state."""
        return self.subgoals[self.current_state]

    def transition(self, nn_state: np.ndarray) -> str:
        """
        Checks if the robot's state triggers an FSM transition.
        Automatically handles transitions for any number of waypoints.
        """
        current_subgoal = self.get_current_subgoal()
        dist_to_subgoal_sq = np.sum((nn_state[:2] - current_subgoal[:2]) ** 2)

        # Check if we've reached the current subgoal
        if dist_to_subgoal_sq <= self.clf_config['clf_epsilon']:
            # Get valid next states
            next_states = self.valid_transitions.get(self.current_state, [])
            
            if next_states:
                next_state = next_states[0]  # Take first valid transition
                old_state = self.current_state
                self.current_state = next_state
                
                # Print transition message
                if self.current_state == self.goal_node:
                    print("FSM: Transitioned to GOAL")
                elif old_state == self.start_node:
                    wp_num = self.waypoint_states.index(next_state) + 1
                    print(f"FSM: Reached Waypoint {wp_num}. Transitioning to next state.")
                elif next_state in self.waypoint_states:
                    wp_num = self.waypoint_states.index(next_state) + 1
                    print(f"FSM: Reached Waypoint {wp_num}.")
                else:
                    print(f"FSM: Transitioned from {old_state} to {next_state}")
        
        return self.current_state

    def prune_fsm_with_certificates(self,
                                    replay_buffer: ReplayBuffer,
                                    policy_net: SubgoalConditionedPolicy,
                                    dynamics_net: EnsembleDynamicsModel,
                                    cbf_net: CBFNetwork,
                                    clf_net: CLFNetwork,
                                    device: torch.device) -> Tuple[bool, float, float]:
        """
        Implements FSM Pruning (Algorithm 1) for ALL transitions.
        """
        print("\n--- Starting FSM Pruning (Algorithm 1) ---")
        
        all_paths_valid = True
        total_safety = 0.0
        total_feasibility = 0.0
        num_transitions = 0

        # --- LOOP OVER ALL TRANSITIONS ---
        for from_state, to_states in self.transitions.items():
            if not to_states: # Skip terminal states
                continue
            
            to_state = to_states[0] # Assuming one transition for now
            num_transitions += 1
            
            g_transition = torch.from_numpy(self.subgoals[from_state]).float().to(device)
            
            # 1. Sample states from the buffer
            batch = replay_buffer.sample(self.fsm_config['pruning_samples'])
            s = torch.from_numpy(batch['states']).float().to(device)
            g_transition = g_transition.repeat(s.shape[0], 1)
            
            with torch.no_grad():
                a = policy_net(s, g_transition)
                s_prime = dynamics_net(s, a)
                
                h_prime = cbf_net(s_prime)
                safe_margin = self.fsm_config['safe_margin']
                is_safe = (h_prime.squeeze() >= safe_margin)
                
                v = clf_net(s, g_transition)
                v_prime = clf_net(s_prime, g_transition)
                
                beta = self.clf_config['clf_beta']
                delta = self.clf_config['clf_delta']
                
                clf_violation = v_prime - (1 - beta) * v - delta 
                is_feasible = (clf_violation.squeeze() <= 0.0)

            safety_rate = torch.mean(is_safe.float()).item()
            feasibility_rate = torch.mean(is_feasible.float()).item()
            total_safety += safety_rate
            total_feasibility += feasibility_rate
            
            print(f"Transition ({from_state} -> {to_state}):")
            print(f"  - Safety Check (CBF): {safety_rate * 100:.1f}% of states safe.")
            print(f"  - Feasibility Check (CLF): {feasibility_rate * 100:.1f}% of states show progress.")

            is_valid = (safety_rate > 0.75) and (feasibility_rate > 0.75) # Use 75% threshold
            
            if is_valid:
                print(f"  - RESULT: Transition VALID.")
                self.valid_transitions[from_state] = [to_state]
            else:
                print(f"  - RESULT: Transition PRUNED (unsafe or infeasible).")
                self.valid_transitions[from_state] = []
                all_paths_valid = False
            
        print("--- FSM Pruning Complete ---")
        
        avg_safety = total_safety / num_transitions
        avg_feasibility = total_feasibility / num_transitions
        
        return all_paths_valid, avg_safety, avg_feasibility