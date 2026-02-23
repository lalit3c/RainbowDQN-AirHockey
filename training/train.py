"""
Training utilities for DQN and Rainbow agents in Hockey Environment.
Includes training loop, evaluation, and logging functionality.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent, BasicOpponent, Mode
from agents.dqn import DQNAgent
from agents.rainbow import RainbowAgent


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    mode: str = "NORMAL"  # NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE
    weak_opponent: bool = True
    keep_mode: bool = True
    mixed_opponents: bool = False  # Train against both weak and strong
    opponent_mix_strategy: str = "alternate"  # "alternate", "random", "curriculum"
    
    # Training
    num_episodes: int = 5000
    max_steps_per_episode: int = 250
    eval_frequency: int = 100
    eval_episodes: int = 20
    save_frequency: int = 5000
    
    # Agent
    agent_type: str = "dqn"  # "dqn" or "rainbow"
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    
    # DQN specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    double_dqn: bool = True
    dueling: bool = False
    
    # Rainbow specific
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    alpha: float = 0.5
    beta_start: float = 0.4
    noisy: bool = True
    
    # Misc
    seed: int = 42
    device: str = "auto"
    log_dir: str = "logs"
    experiment_name: str = "experiment"


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.exp_dir = self.log_dir / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[int] = []  # 1=win, 0=draw, -1=loss
        self.eval_rewards: List[float] = []
        self.eval_win_rates: List[float] = []
        self.losses: List[float] = []
        self.q_values: List[float] = []
        self.epsilons: List[float] = []
        
        # Timing
        self.episode_times: List[float] = []
        self.start_time = time.time()
    
    def log_episode(self, episode: int, reward: float, length: int, 
                    winner: int, loss: Optional[float] = None,
                    epsilon: Optional[float] = None, q_value: Optional[float] = None) -> None:
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_wins.append(winner)
        
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        if q_value is not None:
            self.q_values.append(q_value)
    
    def log_evaluation(self, avg_reward: float, win_rate: float) -> None:
        """Log evaluation metrics."""
        self.eval_rewards.append(avg_reward)
        self.eval_win_rates.append(win_rate)
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get statistics from recent episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_wins = self.episode_wins[-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'win_rate': np.mean([1 if w == 1 else 0 for w in recent_wins]),
            'loss_rate': np.mean([1 if w == -1 else 0 for w in recent_wins]),
            'draw_rate': np.mean([1 if w == 0 else 0 for w in recent_wins]),
            'avg_loss': np.mean(self.losses[-window:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-window:]) if self.q_values else 0
        }
    
    def save_metrics(self) -> None:
        """Save all metrics to files."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_wins': self.episode_wins,
            'eval_rewards': self.eval_rewards,
            'eval_win_rates': self.eval_win_rates,
            'losses': self.losses,
            'q_values': self.q_values,
            'epsilons': self.epsilons,
            'total_time': time.time() - self.start_time
        }
        
        with open(self.exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save numpy arrays for plotting
        np.save(self.exp_dir / 'episode_rewards.npy', np.array(self.episode_rewards))
        np.save(self.exp_dir / 'episode_wins.npy', np.array(self.episode_wins))
        np.save(self.exp_dir / 'eval_rewards.npy', np.array(self.eval_rewards))
        np.save(self.exp_dir / 'eval_win_rates.npy', np.array(self.eval_win_rates))
        np.save(self.exp_dir / 'losses.npy', np.array(self.losses) if self.losses else np.array([]))


def create_agent(config: TrainingConfig, state_dim: int, action_dim: int) -> Union[DQNAgent, RainbowAgent]:
    """Create agent based on configuration."""
    if config.agent_type.lower() == "dqn":
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.lr,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            double_dqn=config.double_dqn,
            dueling=config.dueling,
            device=config.device,
            seed=config.seed
        )
    elif config.agent_type.lower() == "rainbow":
        return RainbowAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.lr,
            gamma=config.gamma,
            n_step=config.n_step,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            alpha=config.alpha,
            beta_start=config.beta_start,
            noisy=config.noisy,
            device=config.device,
            seed=config.seed
        )
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def evaluate_agent(
    agent: Union[DQNAgent, RainbowAgent],
    env: HockeyEnv_BasicOpponent,
    num_episodes: int = 20
) -> Tuple[float, float, List[float]]:
    """
    Evaluate agent performance.
    
    Returns:
        Tuple of (average_reward, win_rate, episode_rewards)
    """
    rewards = []
    wins = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action 
            action = agent.select_action(obs, training=False)
            continuous_action = env.discrete_to_continous_action(action)
            
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        wins.append(info['winner'])
    
    avg_reward = np.mean(rewards)
    win_rate = np.mean([1 if w == 1 else 0 for w in wins])
    
    return avg_reward, win_rate, rewards


def train(config: TrainingConfig, resume_from: str = None, start_episode: int = None) -> Tuple[Union[DQNAgent, RainbowAgent], TrainingLogger]:
    """
    Main training loop.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint file to resume from (optional)
        start_episode: Episode number to start from (optional, auto-detected from checkpoint if not provided)
        
    Returns:
        Trained agent and logger with metrics
    """
    # Create environment(s)
    print(f"config: {config}")
    mode = Mode[config.mode] if isinstance(config.mode, str) else config.mode
    
    if config.mixed_opponents:
        # Create both weak and strong opponent environments
        env_weak = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=True)
        env_strong = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=False)
        env = env_weak  # Start with weak
        print(f"Mixed opponent training: strategy={config.opponent_mix_strategy}")
    else:
        env = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=config.weak_opponent)
        env_weak = env_strong = None
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 8 if config.keep_mode else 7  # Discrete actions including shoot
    
    # Create agent
    agent = create_agent(config, state_dim, action_dim)
    
    # Resume from checkpoint if provided
    resume_episode = 0
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=agent.device, weights_only=False)
        
        # Load network weights
        if 'online_network_state_dict' in checkpoint:
            # Rainbow format
            agent.online_network.load_state_dict(checkpoint['online_network_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        elif 'network_state_dict' in checkpoint:
            # DQN format
            agent.network.load_state_dict(checkpoint['network_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore step count
        if 'step_count' in checkpoint:
            agent.step_count = checkpoint['step_count']
            print(f"  Restored step count: {agent.step_count}")
        
        # Determine starting episode
        if start_episode is not None:
            resume_episode = start_episode
        elif 'training_stats' in checkpoint and 'episode' in checkpoint['training_stats']:
            resume_episode = checkpoint['training_stats']['episode']
        
        print(f"  Starting from episode: {resume_episode + 1}")
    
    # Create logger
    logger = TrainingLogger(config.log_dir, config.experiment_name)
    
    # Save config
    config_dict = asdict(config)
    config_dict['hidden_dims'] = list(config.hidden_dims)  # Convert tuple for JSON
    config_dict['resumed_from'] = resume_from
    config_dict['resume_episode'] = resume_episode
    with open(logger.exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training {config.agent_type.upper()} agent")
    print(f"Device: {agent.device}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Logs will be saved to: {logger.exp_dir}")
    if resume_from:
        print(f"Resuming from episode {resume_episode}")
    
    # Training loop
    best_eval_reward = -float('inf')
    total_episodes = config.num_episodes + resume_episode
    pbar = tqdm(range(resume_episode + 1, total_episodes + 1), desc="Training")
    
    for episode in pbar:
        # Select opponent for this episode (if mixed training)
        if config.mixed_opponents:
            if config.opponent_mix_strategy == "alternate":
                # Alternate between weak and strong each episode
                use_weak = (episode % 2 == 1)
            elif config.opponent_mix_strategy == "random":
                # Random 50/50 selection
                use_weak = np.random.random() < 0.5
            elif config.opponent_mix_strategy == "curriculum":
                # Start mostly weak, gradually increase strong
                progress = episode / total_episodes
                weak_prob = max(0.2, 1.0 - progress)  # Goes from 100% weak to 20% weak
                use_weak = np.random.random() < weak_prob
            else:
                use_weak = True
            
            env = env_weak if use_weak else env_strong
        
        obs, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(config.max_steps_per_episode):
            # Select action
            action = agent.select_action(obs, training=True)
            continuous_action = env.discrete_to_continous_action(action)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Decay epsilon (DQN only)
        if isinstance(agent, DQNAgent):
            agent.decay_epsilon()
        
        # Log episode
        avg_loss = episode_loss / max(loss_count, 1)
        stats = agent.get_stats()
        
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            length=step + 1,
            winner=info['winner'],
            loss=avg_loss,
            epsilon=stats.get('epsilon'),
            q_value=stats.get('avg_q_value')
        )
        
        # Update progress bar
        recent_stats = logger.get_recent_stats(100)
        pbar.set_postfix({
            'reward': f"{recent_stats.get('avg_reward', 0):.2f}",
            'win%': f"{recent_stats.get('win_rate', 0)*100:.1f}",
            'loss': f"{recent_stats.get('avg_loss', 0):.4f}"
        })
        
        # Evaluation
        if episode % config.eval_frequency == 0:
            if config.mixed_opponents:
                # Evaluate against both opponents
                eval_reward_weak, eval_win_rate_weak, _ = evaluate_agent(agent, env_weak, config.eval_episodes // 2)
                eval_reward_strong, eval_win_rate_strong, _ = evaluate_agent(agent, env_strong, config.eval_episodes // 2)
                eval_reward = (eval_reward_weak + eval_reward_strong) / 2
                eval_win_rate = (eval_win_rate_weak + eval_win_rate_strong) / 2
                logger.log_evaluation(eval_reward, eval_win_rate)
                
                print(f"\nEvaluation @ Episode {episode}:")
                print(f"  vs Weak:   Reward={eval_reward_weak:.2f}, Win Rate={eval_win_rate_weak*100:.1f}%")
                print(f"  vs Strong: Reward={eval_reward_strong:.2f}, Win Rate={eval_win_rate_strong*100:.1f}%")
                print(f"  Combined:  Reward={eval_reward:.2f}, Win Rate={eval_win_rate*100:.1f}%")
            else:
                eval_reward, eval_win_rate, _ = evaluate_agent(agent, env, config.eval_episodes)
                logger.log_evaluation(eval_reward, eval_win_rate)
                
                print(f"\nEvaluation @ Episode {episode}:")
                print(f"  Avg Reward: {eval_reward:.2f}, Win Rate: {eval_win_rate*100:.1f}%")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(logger.exp_dir / 'best_model.pt')
                print(f"  New best model saved!")
        
        # Save checkpoint
        if episode % config.save_frequency == 0:
            agent.save(logger.exp_dir / f'checkpoint_{episode}.pt')
            logger.save_metrics()
    
    # Final save
    agent.save(logger.exp_dir / 'final_model.pt')
    logger.save_metrics()
    
    print("\nTraining completed!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Total time: {time.time() - logger.start_time:.2f}s")
    
    # Close environments
    if config.mixed_opponents:
        env_weak.close()
        env_strong.close()
    else:
        env.close()
    return agent, logger


def train_self_play(config: TrainingConfig) -> Tuple[Union[DQNAgent, RainbowAgent], TrainingLogger]:
    """
    Train agent using self-play (both players controlled by learning agents).
    
    """
    # Create full 2-player environment
    mode = Mode[config.mode] if isinstance(config.mode, str) else config.mode
    env = HockeyEnv(mode=mode, keep_mode=config.keep_mode)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 8 if config.keep_mode else 7
    
    # Create two agents (or use single agent for both)
    agent1 = create_agent(config, state_dim, action_dim)
    agent2 = create_agent(config, state_dim, action_dim)  
    
    # Create logger
    logger = TrainingLogger(config.log_dir, f"{config.experiment_name}_selfplay")
    
    print(f"Training with self-play")
    print(f"Device: {agent1.device}")

    
    pbar = tqdm(range(1, config.num_episodes + 1), desc="Self-play Training")
    
    for episode in pbar:
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(config.max_steps_per_episode):
            # Agent 1 action
            action1 = agent1.select_action(obs, training=True)
            continuous_action1 = env.discrete_to_continous_action(action1)
            
            # Agent 2 action (using mirrored observation)
            obs2 = env.obs_agent_two()
            action2 = agent2.select_action(obs2, training=True)
            continuous_action2 = env.discrete_to_continous_action(action2)
            
            # Combined action
            full_action = np.hstack([continuous_action1, continuous_action2])
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(full_action)
            done = terminated or truncated
            
            # Store transitions for both agents
            next_obs2 = env.obs_agent_two()
            reward2 = env.get_reward_agent_two(env.get_info_agent_two())
            
            agent1.store_transition(obs, action1, reward, next_obs, done)
            agent2.store_transition(obs2, action2, reward2, next_obs2, done)
            
            # Train both agents
            agent1.train_step()
            agent2.train_step()
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Decay epsilon for both
        if isinstance(agent1, DQNAgent):
            agent1.decay_epsilon()
            agent2.decay_epsilon()
        
        # Log episode (from agent 1's perspective)
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            length=step + 1,
            winner=info['winner'],
            loss=agent1.get_stats().get('avg_loss'),
            epsilon=agent1.get_stats().get('epsilon')
        )
        
        # Periodically sync agent 2 to agent 1 (curriculum learning)
        if episode % 500 == 0:
            # Copy weights to make opponents stronger
            agent2.q_network.load_state_dict(agent1.q_network.state_dict()) if isinstance(agent1, DQNAgent) else \
            agent2.online_network.load_state_dict(agent1.online_network.state_dict())
        
        recent_stats = logger.get_recent_stats(100)
        pbar.set_postfix({
            'reward': f"{recent_stats.get('avg_reward', 0):.2f}",
            'win%': f"{recent_stats.get('win_rate', 0)*100:.1f}"
        })
    
    agent1.save(logger.exp_dir / 'final_model.pt')
    logger.save_metrics()
    
    env.close()
    return agent1, logger


if __name__ == "__main__":
    config = TrainingConfig(
        agent_type="dqn",
        num_episodes=1000,
        experiment_name="dqn_test",
        weak_opponent=True
    )
    
    agent, logger = train(config)
