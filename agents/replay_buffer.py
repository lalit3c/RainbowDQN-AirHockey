"""
Replay Buffer implementations for DQN and Rainbow algorithms.
Includes standard replay buffer and prioritized experience replay.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, Dict, Any, Optional


class ReplayBuffer:
    """Standard Experience Replay Buffer for DQN."""
    
    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of the state space
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.rng = np.random.default_rng(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions."""
        indices = self.rng.integers(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        return self.size


class SumTree:
    """Sum Tree data structure for efficient priority-based sampling.
    NOTE: This implementation is based on:
    https://github.com/rlcode/per/blob/master/SumTree.py
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        
    def _propagate(self, idx: int, change: float) -> None:
        """Update tree nodes up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority."""
        return self.tree[0]
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority of a leaf node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float]:
        """Get leaf index and priority for a given value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Rainbow DQN.
    Uses Sum Tree for O(log n) sampling based on priorities.
    """
    
    def __init__(self, capacity: int, state_dim: int, 
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100000, seed: int = 42):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of the state space
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which beta increases to 1
            seed: Random seed
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.tree = SumTree(capacity)
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.max_priority = 1.0
        self.min_priority = 1e-6
        
        self.rng = np.random.default_rng(seed)
    
    @property
    def beta(self) -> float:
        """Annealed beta value for importance sampling."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add transition with maximum priority."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # New transitions get max priority
        priority = self.max_priority ** self.alpha
        tree_idx = self.position + self.capacity - 1
        self.tree.update(tree_idx, priority)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with prioritized experience replay.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        total = self.tree.total()
        
        # Handle edge case where total priority is 0 or very small
        if total < 1e-8:
            # Fall back to uniform sampling
            sample_indices = self.rng.choice(self.size, size=batch_size, replace=False if self.size >= batch_size else True)
            return (
                self.states[sample_indices],
                self.actions[sample_indices],
                self.rewards[sample_indices],
                self.next_states[sample_indices],
                self.dones[sample_indices],
                sample_indices,
                np.ones(batch_size, dtype=np.float32)
            )
        
        segment = total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            # Ensure valid range for uniform sampling
            if b <= a:
                b = a + 1e-8
            s = self.rng.uniform(a, min(b, total))
            idx, priority = self.tree.get(s)
            indices[i] = idx
            priorities[i] = max(priority, 1e-8)  # Prevent zero priorities
        
        # Calculate importance sampling weights
        probs = priorities / total
        probs = np.clip(probs, 1e-8, 1.0)  # Prevent numerical issues
        weights = (self.size * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)  # Normalize with small epsilon
        
        self.frame += 1
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())
        
        for idx, priority in zip(indices, priorities):
            tree_idx = idx + self.capacity - 1
            self.tree.update(tree_idx, priority)
    
    def __len__(self) -> int:
        return self.size

