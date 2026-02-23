"""
Rainbow DQN Agent implementation for Hockey Environment.

Rainbow combines six key improvements to DQN:
1. Double Q-learning
2. Prioritized Experience Replay
3. Dueling Networks
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Networks

Reference: Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning", 2018
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import math

from .replay_buffer import PrioritizedReplayBuffer


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration in Rainbow DQN.
    Replaces epsilon-greedy exploration with parametric noise.
    
    NOTE: the implementation of this class is based on
    https://github.com/Kaixhin/Rainbow/blob/1745b184c3dfc03d4ffa3ce2342ced9996b39a60/model.py#L10
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network combining:
    - Dueling architecture
    - Noisy networks
    - Distributional RL (C51)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True
    ):
        super(RainbowNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.noisy = noisy
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # Noisy or standard linear layers
        LinearClass = NoisyLinear if noisy else nn.Linear
        
        # Dueling streams
        # Value stream
        self.value_hidden = LinearClass(hidden_dims[0], hidden_dims[1])
        self.value_out = LinearClass(hidden_dims[1], num_atoms)
        
        # Advantage stream
        self.advantage_hidden = LinearClass(hidden_dims[0], hidden_dims[1])
        self.advantage_out = LinearClass(hidden_dims[1], action_dim * num_atoms)
        
        if not noisy:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-noisy layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-value distributions.
        
        Returns:
            Tensor of shape (batch_size, action_dim, num_atoms)
        """
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature_layer(x)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(batch_size, 1, self.num_atoms)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(batch_size, self.action_dim, self.num_atoms)
        
        # Combine streams
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Softmax over atoms dimension to get probability distribution
        q_dist = F.softmax(q_dist, dim=2)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values from distribution."""
        dist = self.forward(x)
        return (dist * self.support).sum(dim=2)
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class RainbowAgent:
    """
    Rainbow DQN Agent combining all improvements.
    
    Features:
    1. Double Q-learning - uses online network to select actions
    2. Prioritized Experience Replay - samples important transitions more
    3. Dueling Networks - separates value and advantage
    4. Multi-step Learning - uses n-step returns
    5. Distributional RL (C51) - learns value distribution
    6. Noisy Networks - parameter noise for exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        buffer_size: int = 100000,
        batch_size: int = 32,
        update_freq: int = 4,
        target_update_freq: int = 8000,
        alpha: float = 0.5,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        noisy: bool = True,
        device: str = "auto",
        seed: int = 42
    ):
        """
        Initialize Rainbow Agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            n_step: N-step returns
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value support
            v_max: Maximum value support
            buffer_size: Replay buffer size
            batch_size: Training batch size
            update_freq: Steps between network updates
            target_update_freq: Steps between hard target updates
            alpha: PER prioritization exponent
            beta_start: Initial PER importance sampling weight
            beta_frames: Frames to anneal beta to 1
            noisy: Use noisy networks
            device: Computation device
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gamma_n = gamma ** n_step  # Discount for n-step
        self.tau = tau
        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.noisy = noisy
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set seeds
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize networks
        self.online_network = RainbowNetwork(
            state_dim, action_dim, hidden_dims, num_atoms, v_min, v_max, noisy
        ).to(self.device)
        
        self.target_network = RainbowNetwork(
            state_dim, action_dim, hidden_dims, num_atoms, v_min, v_max, noisy
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr, eps=1.5e-4)
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size, state_dim, alpha, beta_start, beta_frames, seed
        )
        
        # N-step buffer for computing n-step returns
        self.n_step_buffer: List[Tuple] = []
        
        # Training stats
        self.step_count = 0
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'td_errors': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action (noisy network provides exploration).
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network.get_q_values(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def _compute_n_step_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Compute n-step return from buffer."""
        # First transition
        state, action = self.n_step_buffer[0][:2]
        
        # Compute n-step reward
        n_step_reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * transition[2]
        
        # Last transition's next state and done
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        return state, action, n_step_reward, next_state, done
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Store transition with n-step processing."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Store n-step transition when buffer is full
        if len(self.n_step_buffer) >= self.n_step:
            n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = \
                self._compute_n_step_return()
            self.replay_buffer.push(n_step_state, n_step_action, n_step_reward, 
                                    n_step_next_state, n_step_done)
            self.n_step_buffer.pop(0)
        
        # Handle episode end
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = \
                    self._compute_n_step_return()
                self.replay_buffer.push(n_step_state, n_step_action, n_step_reward,
                                        n_step_next_state, True)
                self.n_step_buffer.pop(0)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using distributional RL.
        
        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            return None
        
        # Reset noise for exploration
        self.online_network.reset_noise()
        self.target_network.reset_noise()
        
        # Sample from prioritized buffer
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current distribution
        current_dist = self.online_network(states)  # (batch, action, atoms)
        current_dist = current_dist[range(self.batch_size), actions]  # (batch, atoms)
        
        # Compute target distribution using Double DQN
        with torch.no_grad():
            # Select actions with online network
            next_q_values = self.online_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Get distribution from target network
            next_dist = self.target_network(next_states)  # (batch, action, atoms)
            next_dist = next_dist[range(self.batch_size), next_actions]  # (batch, atoms)
            
            # Compute projected distribution
            target_dist = self._project_distribution(rewards, dones, next_dist)
        
        # Compute cross-entropy loss
        loss = -(target_dist * current_dist.clamp(min=1e-5).log()).sum(dim=1)
        
        # Weight by importance sampling
        weighted_loss = (loss * weights).mean()
        
        # Compute TD errors for priority update
        td_errors = loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        else:
            self._soft_update()
        
        # Store stats
        loss_value = weighted_loss.item()
        self.training_stats['losses'].append(loss_value)
        
        with torch.no_grad():
            q_values = self.online_network.get_q_values(states)
            self.training_stats['q_values'].append(q_values.mean().item())
        
        self.training_stats['td_errors'].append(np.mean(td_errors))
        
        return loss_value
    
    def _project_distribution(self, rewards: torch.Tensor, dones: torch.Tensor,
                              next_dist: torch.Tensor) -> torch.Tensor:
        """
        Project target distribution onto support using Bellman update.
        
        This is the key distributional RL computation.
        """
        batch_size = rewards.size(0)
        
        # Compute projected support
        # Tz = r + gamma * z (clipped to [v_min, v_max])
        Tz = rewards.unsqueeze(1) + self.gamma_n * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        
        # Compute which bins Tz falls into
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle edge case where b is exactly an integer
        l[(u > 0) & (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) & (l == u)] += 1
        
        # Distribute probability
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size,
                                device=self.device, dtype=torch.long).unsqueeze(1)
        
        proj_dist = torch.zeros((batch_size, self.num_atoms), device=self.device)
        
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        
        return proj_dist
    
    def _soft_update(self) -> None:
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.online_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'training_stats': self.training_stats,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'n_step': self.n_step,
                'num_atoms': self.num_atoms,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'noisy': self.noisy
            }
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.training_stats = checkpoint['training_stats']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'step_count': self.step_count,
            'buffer_size': len(self.replay_buffer),
            'beta': self.replay_buffer.beta,
            'avg_loss': np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0,
            'avg_q_value': np.mean(self.training_stats['q_values'][-100:]) if self.training_stats['q_values'] else 0,
            'avg_td_error': np.mean(self.training_stats['td_errors'][-100:]) if self.training_stats['td_errors'] else 0
        }
