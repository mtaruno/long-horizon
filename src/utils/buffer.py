import numpy as np
import pickle
from typing import Dict

class ReplayBuffer:
    """
    A simple FIFO replay buffer for storing transitions.
    Stores all components needed for all training loops.
    """
    def __init__(self, state_dim: int, action_dim: int, subgoal_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.subgoals = np.zeros((max_size, subgoal_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        self.h_stars = np.zeros((max_size, 1), dtype=np.float32)
        self.v_stars = np.zeros((max_size, 1), dtype=np.float32)
        
        # Store FSM state as string (for logging/debugging) and integer ID (for fast filtering)
        self.fsm_states = np.empty((max_size,), dtype=object)
        self.fsm_state_ids = np.full((max_size,), -1, dtype=np.int32)

        # Create a dedicated, seeded random number generator for the buffer
        self.rng = np.random.default_rng(seed=42)

    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            next_state: np.ndarray, 
            subgoal: np.ndarray, 
            reward: float, 
            done: bool, 
            h_star: float, 
            v_star: float,
            fsm_state: str = None,
            fsm_state_id: int = -1):
        """
        Add a transition to the buffer.
        
        Args:
            fsm_state: The FSM state (string) that was active when this transition was collected.
                      Used for logging/debugging.
            fsm_state_id: The FSM state (integer ID) that was active when this transition was collected.
                         Used for fast filtering during FSM pruning.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.subgoals[self.ptr] = subgoal
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.h_stars[self.ptr] = h_star
        self.v_stars[self.ptr] = v_star
        self.fsm_states[self.ptr] = fsm_state
        self.fsm_state_ids[self.ptr] = fsm_state_id

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, 
               batch_size: int, 
               fsm_state_id_filter: int = -1,
               proximity_filter: np.ndarray = None,
               proximity_radius: float = None) -> Dict[str, np.ndarray]:
        """
        Samples a random minibatch of transitions.
        
        Args:
            batch_size: Number of samples to return
            fsm_state_id_filter: If >= 0, only sample transitions collected when FSM was in this state ID.
                                Use integer IDs for fast filtering (much faster than string comparison).
            proximity_filter: If provided (2D position [x, y]), only sample states within proximity_radius.
                            Used to filter by proximity to waypoint for more accurate FSM pruning.
            proximity_radius: Radius for proximity filtering (in meters). Only used if proximity_filter is provided.
        """
        valid_idxs = None
        
        # Filter by FSM state ID (fast integer comparison)
        if fsm_state_id_filter >= 0:
            valid_mask = (self.fsm_state_ids[:self.size] == fsm_state_id_filter)
            valid_idxs = np.where(valid_mask)[0]
            
            if len(valid_idxs) == 0:
                # Fallback: if no samples for this FSM state, use all samples
                print(f"Warning: No samples found for FSM state ID {fsm_state_id_filter}. Using all samples.")
                valid_idxs = np.arange(self.size)
        
        # Further filter by proximity to waypoint (if specified)
        if proximity_filter is not None and proximity_radius is not None:
            if valid_idxs is None:
                valid_idxs = np.arange(self.size)
            
            # Compute distances from states to proximity_filter point
            # States are [x, y, cos(theta), sin(theta), v], so first 2 dims are position
            state_positions = self.states[valid_idxs, :2]  # Extract [x, y]
            distances = np.linalg.norm(state_positions - proximity_filter, axis=1)
            proximity_mask = distances <= proximity_radius
            valid_idxs = valid_idxs[proximity_mask]
            
            if len(valid_idxs) == 0:
                print(f"Warning: No samples found within {proximity_radius}m of {proximity_filter}. Using all samples.")
                valid_idxs = np.arange(self.size)
        
        # Sample from valid indices
        if valid_idxs is None:
            # No filtering: sample from all transitions
            idxs = self.rng.integers(0, self.size, size=batch_size)
        elif len(valid_idxs) < batch_size:
            # If we don't have enough samples, sample with replacement
            idxs = self.rng.choice(valid_idxs, size=batch_size, replace=True)
        else:
            # Sample without replacement
            idxs = self.rng.choice(valid_idxs, size=batch_size, replace=False)

        return {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "next_states": self.next_states[idxs],
            "subgoals": self.subgoals[idxs],
            "rewards": self.rewards[idxs],
            "dones": self.dones[idxs],
            "h_stars": self.h_stars[idxs],
            "v_stars": self.v_stars[idxs],
            "fsm_states": self.fsm_states[idxs]  # Include for debugging
        }

    def save(self, file_path: str):
        """Saves the buffer to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                "states": self.states[:self.size],
                "actions": self.actions[:self.size],
                "next_states": self.next_states[:self.size],
                "subgoals": self.subgoals[:self.size],
                "rewards": self.rewards[:self.size],
                "dones": self.dones[:self.size],
                "h_stars": self.h_stars[:self.size],
                "v_stars": self.v_stars[:self.size],
                "fsm_states": self.fsm_states[:self.size],
                "fsm_state_ids": self.fsm_state_ids[:self.size]
            }, f)

    def load(self, file_path: str):
        """Loads the buffer from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        num_loaded = len(data["states"])
        self.states[:num_loaded] = data["states"]
        self.actions[:num_loaded] = data["actions"]
        self.next_states[:num_loaded] = data["next_states"]
        self.subgoals[:num_loaded] = data["subgoals"]
        self.rewards[:num_loaded] = data["rewards"]
        self.dones[:num_loaded] = data["dones"]
        self.h_stars[:num_loaded] = data["h_stars"]
        self.v_stars[:num_loaded] = data["v_stars"]
        
        # Handle optional fsm_states and fsm_state_ids (for backward compatibility)
        if "fsm_states" in data:
            self.fsm_states[:num_loaded] = data["fsm_states"]
        else:
            # If loading old buffer without fsm_states, set to None
            self.fsm_states[:num_loaded] = None
        
        if "fsm_state_ids" in data:
            self.fsm_state_ids[:num_loaded] = data["fsm_state_ids"]
        else:
            # If loading old buffer without fsm_state_ids, set to -1
            self.fsm_state_ids[:num_loaded] = -1
        
        self.size = num_loaded
        self.ptr = num_loaded % self.max_size
        print(f"Loaded {num_loaded} transitions into replay buffer.")

    def __len__(self) -> int:
        return self.size