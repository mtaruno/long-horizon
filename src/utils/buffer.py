"""
Replay buffer for storing transition data.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict
import random


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transition data.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def push(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        reward: float = 0.0,
        done: bool = False
    ):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer.append({
            'state': state.copy(),
            'action': action.copy(), 
            'next_state': next_state.copy(),
            'reward': reward,
            'done': done
        })
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary of batched tensors
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': torch.FloatTensor([t['state'] for t in batch]),
            'actions': torch.FloatTensor([t['action'] for t in batch]),
            'next_states': torch.FloatTensor([t['next_state'] for t in batch]),
            'rewards': torch.FloatTensor([t['reward'] for t in batch]),
            'dones': torch.tensor([bool(t['done']) for t in batch], dtype=torch.bool)
        }
        
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
