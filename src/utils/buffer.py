import numpy as np
import pickle
from typing import Tuple, Dict, Any

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

        # --- THIS IS THE FIX ---
        # Create a dedicated, seeded random number generator for the buffer
        self.rng = np.random.default_rng(seed=42)
        # --- END FIX ---

    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            next_state: np.ndarray, 
            subgoal: np.ndarray, 
            reward: float, 
            done: bool, 
            h_star: float, 
            v_star: float):
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.subgoals[self.ptr] = subgoal
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.h_stars[self.ptr] = h_star
        self.v_stars[self.ptr] = v_star

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Samples a random minibatch of transitions."""
        # --- THIS IS THE FIX ---
        # Use our seeded generator, not the global one
        idxs = self.rng.integers(0, self.size, size=batch_size)
        # --- END FIX ---

        return {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "next_states": self.next_states[idxs],
            "subgoals": self.subgoals[idxs],
            "rewards": self.rewards[idxs],
            "dones": self.dones[idxs],
            "h_stars": self.h_stars[idxs],
            "v_stars": self.v_stars[idxs]
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
                "v_stars": self.v_stars[:self.size]
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
        
        self.size = num_loaded
        self.ptr = num_loaded % self.max_size
        print(f"Loaded {num_loaded} transitions into replay buffer.")

    def __len__(self) -> int:
        return self.size