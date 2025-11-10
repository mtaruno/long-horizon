"""
FSM planner with CBF-CLF pruning (Algorithm 1).
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from enum import Enum


@dataclass
class FSMState:
    """Single FSM node q ∈ Q"""
    id: str
    subgoal: np.ndarray  # Target state for this FSM node
    is_goal: bool = False


@dataclass
class FSMTransition:
    """Edge (q, σ, q') with predicate guard σ"""
    from_state: str
    to_state: str
    predicate: str  # Name of predicate that triggers transition
    

class FSMAutomaton:
    """
    Finite State Machine A_φ = (Q, Σ, δ, q_0, q_f)
    
    Q: states
    Σ: alphabet (predicate evaluations)
    δ: transition function
    q_0: initial state
    q_f: goal state
    """
    
    def __init__(
        self,
        states: List[FSMState],
        transitions: List[FSMTransition],
        initial_state_id: str,
        predicates: Dict[str, Callable]
    ):
        self.states = {s.id: s for s in states}
        self.transitions = transitions
        self.current_state_id = initial_state_id
        self.predicates = predicates
        
        # Build transition map: state_id -> [(predicate, next_state_id)]
        self.transition_map = {}
        for t in transitions:
            if t.from_state not in self.transition_map:
                self.transition_map[t.from_state] = []
            self.transition_map[t.from_state].append((t.predicate, t.to_state))
    
    def get_current_state(self) -> FSMState:
        return self.states[self.current_state_id]
    
    def get_current_subgoal(self) -> np.ndarray:
        return self.get_current_state().subgoal
    
    def evaluate_predicates(self, state: np.ndarray) -> Dict[str, bool]:
        """Evaluate all predicates on current state"""
        return {name: pred(state) for name, pred in self.predicates.items()}
    
    def step(self, state: np.ndarray) -> bool:
        """
        Execute one FSM transition based on predicate evaluation.
        
        Returns:
            True if transitioned to new state, False otherwise
        """
        if self.current_state_id not in self.transition_map:
            return False
        
        pred_values = self.evaluate_predicates(state)
        
        for pred_name, next_state_id in self.transition_map[self.current_state_id]:
            if pred_values.get(pred_name, False):
                self.current_state_id = next_state_id
                return True
        
        return False
    
    def is_goal_reached(self) -> bool:
        return self.get_current_state().is_goal


class FSMPruner:
    """
    Algorithm 1: FSM Pruning with CBF-CLF Certificates
    
    Removes unsafe states and infeasible transitions.
    """
    
    def __init__(
        self,
        cbf: torch.nn.Module,
        clf: torch.nn.Module,
        dynamics: torch.nn.Module,
        epsilon_safe: float = 0.0,
        epsilon_feas: float = 0.1,
        num_samples: int = 100,
        device: str = "cpu"
    ):
        self.cbf = cbf
        self.clf = clf
        self.dynamics = dynamics
        self.epsilon_safe = epsilon_safe
        self.epsilon_feas = epsilon_feas
        self.num_samples = num_samples
        self.device = device
    
    def prune(self, fsm: FSMAutomaton) -> FSMAutomaton:
        """
        Prune FSM by removing unsafe states and infeasible transitions.
        
        Returns:
            Pruned FSM A'_φ = (Q', Σ, δ', q_0, q_f)
        """
        # Step 1: Prune unsafe states
        safe_states = []
        for state in fsm.states.values():
            if self._is_state_safe(state):
                safe_states.append(state)
        
        # Step 2: Prune infeasible transitions
        valid_transitions = []
        for trans in fsm.transitions:
            if trans.from_state in [s.id for s in safe_states] and \
               trans.to_state in [s.id for s in safe_states]:
                if self._is_transition_valid(trans, fsm):
                    valid_transitions.append(trans)
        
        return FSMAutomaton(
            states=safe_states,
            transitions=valid_transitions,
            initial_state_id=fsm.current_state_id,
            predicates=fsm.predicates
        )
    
    def _is_state_safe(self, state: FSMState) -> bool:
        """Check if FSM state's associated region is safe"""
        # Sample states around subgoal
        samples = self._sample_state_distribution(state.subgoal)
        
        with torch.no_grad():
            h_values = self.cbf(samples).squeeze(-1)
            return torch.all(h_values >= self.epsilon_safe).item()
    
    def _is_transition_valid(self, trans: FSMTransition, fsm: FSMAutomaton) -> bool:
        """Check if transition maintains safety and feasibility"""
        from_state = fsm.states[trans.from_state]
        to_state = fsm.states[trans.to_state]
        
        # Sample (s, a) pairs executing this transition
        states, actions = self._sample_transition_data(from_state, to_state)
        
        with torch.no_grad():
            next_states = self.dynamics(states, actions)
            
            # Check safety: h(s') >= ε_safe
            h_values = self.cbf(next_states).squeeze(-1)
            safe = torch.all(h_values >= self.epsilon_safe)
            
            # Check feasibility: V(s') <= ε_feas
            V_values = self.clf(next_states).squeeze(-1)
            feasible = torch.all(V_values <= self.epsilon_feas)
            
            return (safe and feasible).item()
    
    def _sample_state_distribution(self, center: np.ndarray) -> torch.Tensor:
        """Sample states around a center point"""
        noise = np.random.randn(self.num_samples, len(center)) * 0.1
        samples = center + noise
        return torch.FloatTensor(samples).to(self.device)
    
    def _sample_transition_data(
        self, 
        from_state: FSMState, 
        to_state: FSMState
    ) -> tuple:
        """Sample (s, a) pairs for transition"""
        # Sample states near from_state
        states = self._sample_state_distribution(from_state.subgoal)
        
        # Sample actions toward to_state
        direction = to_state.subgoal[:states.shape[1]] - from_state.subgoal[:states.shape[1]]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        actions = torch.FloatTensor(
            np.tile(direction, (self.num_samples, 1))
        ).to(self.device)
        
        return states, actions


def create_simple_navigation_fsm(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    state_dim: int = 4
) -> FSMAutomaton:
    """
    Example for warehouse robot navigation. 
    Create simple 3-state FSM: NAVIGATE → APPROACH → GOAL
    """
    # Define states
    states = [
        FSMState(id="navigate", subgoal=goal_pos, is_goal=False),
        FSMState(id="approach", subgoal=goal_pos, is_goal=False),
        FSMState(id="goal", subgoal=goal_pos, is_goal=True)
    ]
    
    # Define transitions
    transitions = [
        FSMTransition("navigate", "approach", "near_goal"),
        FSMTransition("approach", "goal", "at_goal")
    ]
    
    # Define predicates
    def near_goal(state: np.ndarray) -> bool:
        pos = state[:2]
        return np.linalg.norm(pos - goal_pos) < 1.0
    
    def at_goal(state: np.ndarray) -> bool:
        pos = state[:2]
        return np.linalg.norm(pos - goal_pos) < 0.2
    
    predicates = {
        "near_goal": near_goal,
        "at_goal": at_goal
    }
    
    return FSMAutomaton(states, transitions, "navigate", predicates)
