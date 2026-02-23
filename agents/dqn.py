"""
DQN (Deep Q-Network) Agent implementation for Hockey Environment.
Includes vanilla DQN with target network and experience replay.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Deep Q-Network with configurable architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions."""
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture that separates state value and advantage functions.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (subtract mean advantage for uniqueness)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DQNAgent:
    """
    DQN Agent with optional Double DQN and Dueling architecture.
    
    Features:
    - Experience replay
    - Target network with soft updates
    - Epsilon-greedy exploration with decay
    - Optional Double DQN (DDQN)
    - Optional Dueling architecture
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        update_freq: int = 4,
        double_dqn: bool = True,
        dueling: bool = False,
        device: str = "auto",
        seed: int = 42
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target network
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            update_freq: Steps between network updates
            double_dqn: Use Double DQN
            dueling: Use Dueling architecture
            device: Computation device
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.double_dqn = double_dqn
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize networks
        NetworkClass = DuelingQNetwork if dueling else QNetwork
        self.q_network = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, seed)
        
        # Training stats
        self.step_count = 0
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'epsilon_values': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Store stats
        loss_value = loss.item()
        self.training_stats['losses'].append(loss_value)
        self.training_stats['q_values'].append(current_q_values.mean().item())
        
        return loss_value
    
    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_values'].append(self.epsilon)
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_stats': self.training_stats,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'double_dqn': self.double_dqn
            }
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.training_stats = checkpoint['training_stats']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0,
            'avg_q_value': np.mean(self.training_stats['q_values'][-100:]) if self.training_stats['q_values'] else 0
        }
