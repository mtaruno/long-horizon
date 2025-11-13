"""
Integration of CBF-CLF safety framework with high-level planners (FSM, LLM).
"""

import torch
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

class FSMState:
    """Represents a single node (state) in the FSM, like q0, q1, etc."""
    def __init__(self, name, is_goal=False):
        self.name = name
        self.is_goal = is_goal
        self.out_transitions = []
    
    def add_transition(self, transition):
        self.out_transitions.append(transition)

    def __repr__(self):
        return f"<FSMState: {self.name}>"

class FSMTransition:
    """Represents an edge (skill) in the FSM."""
    def __init__(self, name, to_state, guard_fn, subgoal_fn):
        self.name = name
        self.to_state = to_state
        self.guard_fn = guard_fn
        self.subgoal_fn = subgoal_fn

    def check_guard(self, current_state):
        """Checks if this transition can be taken."""
        return self.guard_fn(current_state)
        
    def get_subgoal(self, current_state):
        """Returns the subgoal for this transition."""
        return self.subgoal_fn(current_state)

class FSMAutomaton:
    """
    Manages the FSM, including the CBF-CLF pruning logic
    from Algorithm 1.
    """
    def __init__(self, start_state: FSMState):
        self.start_state = start_state
        self.current_state = start_state
        self.all_states = {start_state.name: start_state}
        self.all_transitions = [] # We need a flat list for pruning

    def add_state(self, state: FSMState):
        if state.name not in self.all_states:
            self.all_states[state.name] = state

    def add_transition(self, from_state_name, transition: FSMTransition):
        if from_state_name in self.all_states:
            self.all_states[from_state_name].add_transition(transition)
            self.all_transitions.append((self.all_states[from_state_name], transition))
        else:
            raise ValueError(f"State '{from_state_name}' not in FSM.")

    def reset(self):
        """Resets the FSM to the start state."""
        self.current_state = self.start_state

    def get_active_subgoal(self, physical_state):
        """
        Checks guards and returns the subgoal for the first valid transition.
        """
        for transition in self.current_state.out_transitions:
            if transition.check_guard(physical_state):
                return transition.get_subgoal(physical_state)
        
        if self.current_state.out_transitions:
            return self.current_state.out_transitions[0].get_subgoal(physical_state)
        return None 

    def update_state(self, physical_state):
        """
        Checks if any transition guards are met and updates the FSM's 
        current_state.
        """
        for transition in self.current_state.out_transitions:
            if transition.check_guard(physical_state):
                self.current_state = transition.to_state
                return True 
        return False 
        
    def is_at_goal(self):
        return self.current_state.is_goal

    def prune_fsm_with_certificates(self, cbf, clf, dynamics, actor, replay_buffer, 
                                    safe_margin=0.0, feas_margin=0.1, n_samples=100,
                                    pruning_threshold=0.95): # <-- CHANGED: New threshold
        """
        Implements Algorithm 1: FSM Pruning.
        This version only prunes transitions, not states, and uses a
        percentage threshold instead of torch.all().
        """
        print("[FSM Pruning] Starting FSM prune...")
        
        Q_prime = set(self.all_states.values())
        print(f"[FSM Pruning] Skipping state pruning. All {len(Q_prime)} states kept.")


        # --- Algorithm 1: Prune invalid transitions ---
        delta_prime = []
        
        if len(replay_buffer) < n_samples:
             print("[FSM Pruning] Not enough samples in buffer to prune transitions yet.")
             delta_prime = [(q_from, t) for q_from, t in self.all_transitions if q_from in Q_prime and t.to_state in Q_prime]
        else:
            for q_from, transition in self.all_transitions:
                q_to = transition.to_state
                
                if q_from not in Q_prime or q_to not in Q_prime:
                    continue 
                
                samples = replay_buffer.sample(n_samples)
                states_s_list = [t.state for t in samples]
                states_s = torch.FloatTensor(np.array(states_s_list)).to(cbf.device)

                goals_g_np = [transition.get_subgoal(s)[:2] for s in states_s_list]
                goals_g = torch.FloatTensor(np.array(goals_g_np)).to(cbf.device)
                
                with torch.no_grad():
                    actions_a = actor(states_s, goals_g)
                    s_prime = dynamics(states_s, actions_a)
                    h_prime = cbf(s_prime).squeeze()
                    v_prime = clf(s_prime).squeeze()
                
                # --- PRUNING LOGIC CHANGED ---
                # Check if the *percentage* of valid samples is above the threshold
                safe_samples = torch.sum(h_prime >= safe_margin).item() / n_samples
                feasible_samples = torch.sum(v_prime <= feas_margin).item() / n_samples
                
                is_safe = safe_samples >= pruning_threshold
                is_feasible = feasible_samples >= pruning_threshold

                if is_safe and is_feasible:
                    delta_prime.append((q_from, transition))
                else:
                    print(f"[FSM Pruning] Pruning transition: {q_from.name} -> {q_to.name} "
                          f"(Safe: {safe_samples*100:.1f}%, Feasible: {feasible_samples*100:.1f}%)")

        # --- Reconstruct FSM ---
        self.all_states = {q.name: q for q in Q_prime}
        self.all_transitions = delta_prime
        
        for q in self.all_states.values():
            q.out_transitions = []
            
        for q_from, transition in self.all_transitions:
            if q_from.name in self.all_states:
                q_from.add_transition(transition)
            
        print(f"[FSM Pruning] Pruning complete. {len(self.all_states)} states, {len(self.all_transitions)} transitions remain.")



def create_simple_fsm(env):
    """Creates a basic FSM for our environment - DEPRECATED, use create_waypoint_fsm instead."""
    goal_center = env.target_goal['center']
    goal_radius = env.target_goal['radius']

    q0 = FSMState("q0_Seeking")
    q_goal = FSMState("q_Goal", is_goal=True)

    def guard_is_at_goal(state):
        dist = np.linalg.norm(state[:2] - goal_center)
        return dist <= goal_radius

    def guard_is_not_at_goal(state):
        dist = np.linalg.norm(state[:2] - goal_center)
        return dist > goal_radius

    def subgoal_seek_goal(state):
        return goal_center

    t1 = FSMTransition("Transition_to_Goal", q_goal, guard_is_at_goal, subgoal_seek_goal)
    t_loop = FSMTransition("Loop_Seeking", q0, guard_is_not_at_goal, subgoal_seek_goal)

    fsm = FSMAutomaton(start_state=q0)
    fsm.add_state(q_goal)
    fsm.add_transition("q0_Seeking", t1)
    fsm.add_transition("q0_Seeking", t_loop)

    return fsm, goal_center



def create_waypoint_fsm(env):
    """
    Creates an FSM with hardcoded waypoints to navigate around obstacles.

    The warehouse layout has obstacles, so we'll create a path:
    Start (0.5, 10.0) -> Waypoint1 (3.5, 5.0) -> Waypoint2 (6.0, 8.0) -> Goal (10.5, 8.5)

    This path navigates through the corridors between obstacles.
    """
    goal_center = env.target_goal['center']
    goal_radius = env.target_goal['radius']

    # Define waypoints based on the obstacle layout
    # Looking at the environment, obstacles are at:
    # [2,3], [2,7], [5,2], [5,5], [5,8], [7,3.5], [7,6.5], [9,1], [9,5], [9,9], [11,3]
    # Goal is at [10.5, 8.5]

    waypoint1 = np.array([3.5, 5.0])  # Between left obstacles
    waypoint2 = np.array([6.0, 8.0])   # Navigate around middle obstacles
    waypoint3 = np.array([8.0, 7.5])   # Get closer to goal region

    waypoint_radius = 0.8  # Radius to consider "reached" a waypoint

    # Create FSM states
    q0 = FSMState("q0_ToWaypoint1")
    q1 = FSMState("q1_ToWaypoint2")
    q2 = FSMState("q2_ToWaypoint3")
    q3 = FSMState("q3_ToGoal")
    q_goal = FSMState("q_Goal", is_goal=True)

    # Guards for waypoint 1
    def guard_at_waypoint1(state):
        dist = np.linalg.norm(state[:2] - waypoint1)
        return dist <= waypoint_radius

    def guard_not_at_waypoint1(state):
        return not guard_at_waypoint1(state)

    # Guards for waypoint 2
    def guard_at_waypoint2(state):
        dist = np.linalg.norm(state[:2] - waypoint2)
        return dist <= waypoint_radius

    def guard_not_at_waypoint2(state):
        return not guard_at_waypoint2(state)

    # Guards for waypoint 3
    def guard_at_waypoint3(state):
        dist = np.linalg.norm(state[:2] - waypoint3)
        return dist <= waypoint_radius

    def guard_not_at_waypoint3(state):
        return not guard_at_waypoint3(state)

    # Guards for goal
    def guard_at_goal(state):
        dist = np.linalg.norm(state[:2] - goal_center)
        return dist <= goal_radius

    def guard_not_at_goal(state):
        return not guard_at_goal(state)

    # Subgoal functions
    def subgoal_waypoint1(state):
        return waypoint1

    def subgoal_waypoint2(state):
        return waypoint2

    def subgoal_waypoint3(state):
        return waypoint3

    def subgoal_goal(state):
        return goal_center

    # Create transitions
    # From q0 (seeking waypoint1)
    t0_to_q1 = FSMTransition("Reached_WP1", q1, guard_at_waypoint1, subgoal_waypoint2)
    t0_loop = FSMTransition("Seeking_WP1", q0, guard_not_at_waypoint1, subgoal_waypoint1)

    # From q1 (seeking waypoint2)
    t1_to_q2 = FSMTransition("Reached_WP2", q2, guard_at_waypoint2, subgoal_waypoint3)
    t1_loop = FSMTransition("Seeking_WP2", q1, guard_not_at_waypoint2, subgoal_waypoint2)

    # From q2 (seeking waypoint3)
    t2_to_q3 = FSMTransition("Reached_WP3", q3, guard_at_waypoint3, subgoal_goal)
    t2_loop = FSMTransition("Seeking_WP3", q2, guard_not_at_waypoint3, subgoal_waypoint3)

    # From q3 (seeking goal)
    t3_to_goal = FSMTransition("Reached_Goal", q_goal, guard_at_goal, subgoal_goal)
    t3_loop = FSMTransition("Seeking_Goal", q3, guard_not_at_goal, subgoal_goal)

    # Build FSM
    fsm = FSMAutomaton(start_state=q0)
    fsm.add_state(q1)
    fsm.add_state(q2)
    fsm.add_state(q3)
    fsm.add_state(q_goal)

    fsm.add_transition("q0_ToWaypoint1", t0_to_q1)
    fsm.add_transition("q0_ToWaypoint1", t0_loop)

    fsm.add_transition("q1_ToWaypoint2", t1_to_q2)
    fsm.add_transition("q1_ToWaypoint2", t1_loop)

    fsm.add_transition("q2_ToWaypoint3", t2_to_q3)
    fsm.add_transition("q2_ToWaypoint3", t2_loop)

    fsm.add_transition("q3_ToGoal", t3_to_goal)
    fsm.add_transition("q3_ToGoal", t3_loop)

    # Return FSM and waypoints for visualization
    waypoints = [waypoint1, waypoint2, waypoint3, goal_center]
    return fsm, goal_center, waypoints