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
    FSM_STATE_START = "START"
    FSM_STATE_GOAL = "GOAL"
    FSM_STATE_FAILED = "FAILED"
    
    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray, config: Dict[str, Any]):
        self.fsm_config = config['fsm']
        self.clf_config = config['train']
        
        # Store action limits for scaling (needed for dynamics model)
        self.a_max = config['env']['a_max']
        self.omega_max = config['env']['omega_max']
        
        # Get waypoints from config (list of [x, y] positions)
        waypoint_positions = config["fsm"].get("waypoints", [])
        waypoint_positions = [np.array(wp) for wp in waypoint_positions]
        print(f"Waypoint positions: {waypoint_positions}")
        
        # Build FSM states: START, WAYPOINT_1, WAYPOINT_2, ..., GOAL
        self.all_states = [self.FSM_STATE_START]
        self.waypoint_states = []
        for i in range(len(waypoint_positions)):
            state_name = f"WAYPOINT_{i+1}"
            self.waypoint_states.append(state_name)
            self.all_states.append(state_name)
        self.all_states.append(self.FSM_STATE_GOAL)
        
        # Create mapping from state names to integer IDs (for fast filtering in buffer)
        # START=0, WAYPOINT_1=1, WAYPOINT_2=2, ..., GOAL=len(all_states)-1
        self.state_to_id = {state: idx for idx, state in enumerate(self.all_states)}
        self.id_to_state = {idx: state for state, idx in self.state_to_id.items()}
        
        # Build transitions: linear chain START -> WP1 -> WP2 -> ... -> GOAL
        self.transitions = {}
        self.subgoals = {}
        
        # START -> first waypoint (or goal if no waypoints)
        if len(waypoint_positions) > 0:
            self.transitions[self.FSM_STATE_START] = [self.waypoint_states[0]]
            self.subgoals[self.FSM_STATE_START] = waypoint_positions[0]
        else:
            # No waypoints: direct path to goal
            self.transitions[self.FSM_STATE_START] = [self.FSM_STATE_GOAL]
            self.subgoals[self.FSM_STATE_START] = goal_pos
        
        # Waypoint transitions: each waypoint -> next waypoint (or goal)
        for i, wp_state in enumerate(self.waypoint_states):
            if i < len(self.waypoint_states) - 1:
                # Waypoint -> next waypoint
                next_state = self.waypoint_states[i + 1]
                self.transitions[wp_state] = [next_state]
                self.subgoals[wp_state] = waypoint_positions[i + 1]
            else:
                # Last waypoint -> goal
                self.transitions[wp_state] = [self.FSM_STATE_GOAL]
                self.subgoals[wp_state] = goal_pos
        
        # Goal is terminal
        self.transitions[self.FSM_STATE_GOAL] = []
        self.subgoals[self.FSM_STATE_GOAL] = goal_pos
        
        # Store waypoint positions for reference
        self.waypoint_positions = waypoint_positions
        
        # Set start and goal node references
        self.start_node = self.FSM_STATE_START
        self.goal_node = self.FSM_STATE_GOAL
        
        self.valid_transitions = self.transitions.copy()
        self.current_state = self.start_node
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def update_waypoints(self, new_waypoint_positions: list):
        """
        Updates the FSM with new waypoints and rebuilds all transitions.
        This is used for replanning when pruning fails.
        
        Args:
            new_waypoint_positions: List of [x, y] waypoint positions
        """
        new_waypoint_positions = [np.array(wp) for wp in new_waypoint_positions]
        self.waypoint_positions = new_waypoint_positions
        
        # Rebuild FSM states
        self.all_states = [self.FSM_STATE_START]
        self.waypoint_states = []
        for i in range(len(new_waypoint_positions)):
            state_name = f"WAYPOINT_{i+1}"
            self.waypoint_states.append(state_name)
            self.all_states.append(state_name)
        self.all_states.append(self.FSM_STATE_GOAL)
        
        # Rebuild state-to-id mapping
        self.state_to_id = {state: idx for idx, state in enumerate(self.all_states)}
        self.id_to_state = {idx: state for state, idx in self.state_to_id.items()}
        
        # Rebuild transitions and subgoals
        self.transitions = {}
        self.subgoals = {}
        
        if len(new_waypoint_positions) > 0:
            self.transitions[self.FSM_STATE_START] = [self.waypoint_states[0]]
            self.subgoals[self.FSM_STATE_START] = new_waypoint_positions[0]
        else:
            self.transitions[self.FSM_STATE_START] = [self.FSM_STATE_GOAL]
            self.subgoals[self.FSM_STATE_START] = self.goal_pos
        
        for i, wp_state in enumerate(self.waypoint_states):
            if i < len(self.waypoint_states) - 1:
                next_state = self.waypoint_states[i + 1]
                self.transitions[wp_state] = [next_state]
                self.subgoals[wp_state] = new_waypoint_positions[i + 1]
            else:
                self.transitions[wp_state] = [self.FSM_STATE_GOAL]
                self.subgoals[wp_state] = self.goal_pos
        
        self.transitions[self.FSM_STATE_GOAL] = []
        self.subgoals[self.FSM_STATE_GOAL] = self.goal_pos
        
        # Reset valid_transitions to match new transitions
        self.valid_transitions = self.transitions.copy()
        self.current_state = self.start_node
        
        print(f"FSM updated with {len(new_waypoint_positions)} new waypoints")

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

    def prune_fsm_with_certificates_offline(self,
                                    replay_buffer: ReplayBuffer,
                                    policy_net: SubgoalConditionedPolicy,
                                    dynamics_net: EnsembleDynamicsModel,
                                    cbf_net: CBFNetwork,
                                    clf_net: CLFNetwork,
                                    device: torch.device,
                                    return_diagnostics: bool = False) -> Tuple[bool, float, float, Dict[str, Any]]:
        """
        Implements FSM Pruning (Algorithm 1) for ALL transitions.
        
        Args:
            return_diagnostics: If True, returns additional info about which transitions failed
                               for use in replanning.
        
        Returns:
            (all_paths_valid, avg_safety, avg_feasibility, diagnostics)
            - all_paths_valid: True if all transitions are valid
            - avg_safety: Average safety rate across all transitions
            - avg_feasibility: Average feasibility rate across all transitions
            - diagnostics: Dict with 'failed_transitions' list of (from_state, to_state, safety_rate, feasibility_rate)
        """
        print("\n--- Starting FSM Pruning (Algorithm 1) ---")
        
        all_paths_valid = True
        total_safety = 0.0
        total_feasibility = 0.0
        num_transitions = 0
        failed_transitions = []  # For diagnostics

        for from_state, to_states in self.transitions.items():
            if not to_states: # Skip terminal states
                continue
            
            to_state = to_states[0] # Assuming one transition for now
            num_transitions += 1
            
            g_transition = torch.from_numpy(self.subgoals[from_state]).float().to(device)
            
            # 1. Sample states from the buffer CONDITIONED on the from_state
            # This is critical: we only test transitions from states that were actually
            # collected when the FSM was in the 'from_state'. This ensures we're testing
            # the correct transition (e.g., WAYPOINT_1 -> WAYPOINT_2) from states near
            # WAYPOINT_1, not from random states anywhere in the workspace.
            
            # Get integer ID for fast filtering
            fsm_state_id = self.state_to_id.get(from_state, -1)
            
            # Get waypoint position for proximity filtering (if from_state is a waypoint)
            proximity_filter = None
            proximity_radius = None
            if from_state in self.waypoint_states:
                # Find the waypoint index
                wp_idx = self.waypoint_states.index(from_state)
                proximity_filter = self.waypoint_positions[wp_idx]  # [x, y] position
                # Use CLF epsilon as proximity radius (states within goal region)
                # Or use a slightly larger radius to include states approaching the waypoint
                proximity_radius = self.clf_config.get('clf_epsilon', 0.25) * 2.0  # 2x goal radius
            elif from_state == self.FSM_STATE_START:
                # For START state, filter by proximity to start position
                proximity_filter = self.start_pos[:2]  # [x, y] from start state
                proximity_radius = self.clf_config.get('clf_epsilon', 0.25) * 2.0
            
            batch = replay_buffer.sample(
                batch_size=self.fsm_config['pruning_samples'],
                fsm_state_id_filter=fsm_state_id,
                proximity_filter=proximity_filter,
                proximity_radius=proximity_radius
            )
            s = torch.from_numpy(batch['states']).float().to(device)
            g_transition = g_transition.repeat(s.shape[0], 1)
            
            with torch.no_grad():
                # Policy outputs actions in [-1, 1] range (tanh)
                a_unscaled = policy_net(s, g_transition)
                # Scale actions to actual range before passing to dynamics model
                # The dynamics model was trained on scaled actions
                a_scaled = a_unscaled.clone()
                a_scaled[:, 0] = a_unscaled[:, 0] * self.a_max  # Scale linear acceleration
                a_scaled[:, 1] = a_unscaled[:, 1] * self.omega_max  # Scale angular velocity
                s_prime = dynamics_net(s, a_scaled)
                
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

            is_valid = (safety_rate > 0.75) and (feasibility_rate > 0.6) # Use 75% threshold (trying 60%)
            
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

if __name__ == "__main__":
    config = yaml.load(open("config/warehouse_v1.yaml"), Loader=yaml.FullLoader)
    fsm = FSMAutomaton(
        start_pos=np.array(config['fsm']['start_state']),
        goal_pos=np.array(config['fsm']['goal_state']),
        config=config
    )
    print(fsm.transitions)
    print(fsm.subgoals)
    print(fsm.waypoint_positions)
    print(fsm.waypoint_states)
    print(fsm.all_states)
    print(fsm.start_node)
    print(fsm.goal_node)
    print(fsm.valid_transitions)
    print(fsm.current_state)
    print(fsm.start_pos)
    print(fsm.goal_pos)