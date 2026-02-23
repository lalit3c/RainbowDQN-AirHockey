"""
Visualization and plotting utilities for training analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import seaborn as sns
from scipy.ndimage import uniform_filter1d


# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

COLORS = {
    'dqn': '#2196F3',      # Blue
    'rainbow': '#FF5722',   # Orange/Red
    'ddqn': '#4CAF50',      # Green
    'dueling': '#9C27B0',   # Purple
}


def smooth_data(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to data."""
    if len(data) < window:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode='nearest')


def load_experiment_data(exp_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load experiment data from directory."""
    exp_dir = Path(exp_dir)
    
    data = {}
    
    # Load metrics JSON
    metrics_path = exp_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    # Load numpy arrays
    for name in ['episode_rewards', 'episode_wins', 'eval_rewards', 'eval_win_rates', 'losses']:
        npy_path = exp_dir / f'{name}.npy'
        if npy_path.exists():
            data[name] = np.load(npy_path)
    
    data['name'] = exp_dir.name
    return data


def plot_training_rewards(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Training Rewards Comparison",
    smooth_window: int = 50,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot training rewards for multiple experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        save_path: Path to save figure
        title: Plot title
        smooth_window: Smoothing window size
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp in experiments:
        if 'episode_rewards' not in exp:
            continue
            
        rewards = exp['episode_rewards']
        episodes = np.arange(1, len(rewards) + 1)
        
        # Determine agent type from config or name
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', exp.get('name', agent_type))
        
        # Plot raw data with transparency
        ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.5)
        
        # Plot smoothed data
        smoothed = smooth_data(rewards, smooth_window)
        ax.plot(episodes, smoothed, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_win_rates(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Win Rate Comparison",
    window: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot win rate over training.
    
    Args:
        experiments: List of experiment data
        save_path: Path to save figure
        title: Plot title
        window: Window for computing rolling win rate
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp in experiments:
        if 'episode_wins' not in exp:
            continue
            
        wins = exp['episode_wins']
        # Convert to binary wins
        binary_wins = np.array([1 if w == 1 else 0 for w in wins])
        
        # Compute rolling win rate
        if len(binary_wins) >= window:
            win_rate = np.convolve(binary_wins, np.ones(window)/window, mode='valid')
            episodes = np.arange(window, len(binary_wins) + 1)
        else:
            win_rate = binary_wins
            episodes = np.arange(1, len(binary_wins) + 1)
        
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', exp.get('name', agent_type))
        
        ax.plot(episodes, win_rate * 100, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_evaluation_metrics(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot evaluation metrics (reward and win rate) side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for exp in experiments:
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', exp.get('name', agent_type))
        
        # Evaluation rewards
        if 'eval_rewards' in exp and len(exp['eval_rewards']) > 0:
            eval_freq = exp.get('config', {}).get('eval_frequency', 100)
            episodes = np.arange(eval_freq, eval_freq * (len(exp['eval_rewards']) + 1), eval_freq)
            axes[0].plot(episodes, exp['eval_rewards'], color=color, label=label, 
                        linewidth=2, marker='o', markersize=3)
        
        # Evaluation win rates
        if 'eval_win_rates' in exp and len(exp['eval_win_rates']) > 0:
            eval_freq = exp.get('config', {}).get('eval_frequency', 100)
            episodes = np.arange(eval_freq, eval_freq * (len(exp['eval_win_rates']) + 1), eval_freq)
            axes[1].plot(episodes, np.array(exp['eval_win_rates']) * 100, color=color, 
                        label=label, linewidth=2, marker='o', markersize=3)
    
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Evaluation Reward', fontsize=12)
    axes[0].set_title('Evaluation Rewards', fontsize=14)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Win Rate (%)', fontsize=12)
    axes[1].set_title('Evaluation Win Rate', fontsize=14)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_curves(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Training Loss Comparison",
    smooth_window: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot training loss curves."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp in experiments:
        if 'losses' not in exp or len(exp['losses']) == 0:
            continue
            
        losses = np.array(exp['losses'])
        steps = np.arange(1, len(losses) + 1)
        
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', exp.get('name', agent_type))
        
        # Plot smoothed loss
        smoothed = smooth_data(losses, smooth_window)
        ax.plot(steps, smoothed, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_epsilon_decay(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """Plot epsilon decay (for DQN agents)."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp in experiments:
        metrics = exp.get('metrics', {})
        if 'epsilons' not in metrics or len(metrics['epsilons']) == 0:
            continue
            
        epsilons = np.array(metrics['epsilons'])
        episodes = np.arange(1, len(epsilons) + 1)
        
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', exp.get('name', agent_type))
        
        ax.plot(episodes, epsilons, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax.set_title('Epsilon Decay During Training', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comprehensive_comparison(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive comparison plot with multiple subplots.
    
    Includes:
    - Training rewards
    - Win rates
    - Evaluation metrics
    - Loss curves
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Training rewards
    ax1 = fig.add_subplot(gs[0, 0])
    for exp in experiments:
        if 'episode_rewards' not in exp:
            continue
        rewards = exp['episode_rewards']
        episodes = np.arange(1, len(rewards) + 1)
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', agent_type.upper())
        
        smoothed = smooth_data(rewards, 50)
        ax1.plot(episodes, smoothed, color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('(a) Training Rewards')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Win rates
    ax2 = fig.add_subplot(gs[0, 1])
    for exp in experiments:
        if 'episode_wins' not in exp:
            continue
        wins = exp['episode_wins']
        binary_wins = np.array([1 if w == 1 else 0 for w in wins])
        window = 100
        if len(binary_wins) >= window:
            win_rate = np.convolve(binary_wins, np.ones(window)/window, mode='valid')
            episodes = np.arange(window, len(binary_wins) + 1)
        else:
            continue
            
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', agent_type.upper())
        
        ax2.plot(episodes, win_rate * 100, color=color, label=label, linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('(b) Rolling Win Rate (100 episodes)')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Evaluation rewards
    ax3 = fig.add_subplot(gs[1, 0])
    for exp in experiments:
        if 'eval_rewards' not in exp or len(exp['eval_rewards']) == 0:
            continue
        eval_freq = exp.get('config', {}).get('eval_frequency', 100)
        episodes = np.arange(eval_freq, eval_freq * (len(exp['eval_rewards']) + 1), eval_freq)
        
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', agent_type.upper())
        
        ax3.plot(episodes, exp['eval_rewards'], color=color, label=label, 
                linewidth=2, marker='o', markersize=4)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('(c) Evaluation Rewards')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Loss curves
    ax4 = fig.add_subplot(gs[1, 1])
    for exp in experiments:
        if 'losses' not in exp or len(exp['losses']) == 0:
            continue
        losses = np.array(exp['losses'])
        steps = np.arange(1, len(losses) + 1)
        
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        color = COLORS.get(agent_type.lower(), '#666666')
        label = exp.get('config', {}).get('experiment_name', agent_type.upper())
        
        smoothed = smooth_data(losses, 100)
        ax4.plot(steps, smoothed, color=color, label=label, linewidth=2)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('(d) Training Loss')
    ax4.set_yscale('log')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DQN vs Rainbow Performance Comparison', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_final_statistics(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create bar plots comparing final statistics.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    names = []
    final_rewards = []
    final_win_rates = []
    final_loss_rates = []
    colors = []
    
    for exp in experiments:
        agent_type = exp.get('config', {}).get('agent_type', 'dqn')
        name = exp.get('config', {}).get('experiment_name', agent_type.upper())
        names.append(name)
        colors.append(COLORS.get(agent_type.lower(), '#666666'))
        
        # Get last 100 episodes stats
        if 'episode_rewards' in exp:
            final_rewards.append(np.mean(exp['episode_rewards'][-100:]))
        else:
            final_rewards.append(0)
            
        if 'episode_wins' in exp:
            wins = exp['episode_wins'][-100:]
            final_win_rates.append(np.mean([1 if w == 1 else 0 for w in wins]) * 100)
            final_loss_rates.append(np.mean([1 if w == -1 else 0 for w in wins]) * 100)
        else:
            final_win_rates.append(0)
            final_loss_rates.append(0)
    
    x = np.arange(len(names))
    width = 0.6
    
    # Final rewards
    bars1 = axes[0].bar(x, final_rewards, width, color=colors)
    axes[0].set_xlabel('Agent')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Final 100 Episodes - Avg Reward')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Win rates
    bars2 = axes[1].bar(x, final_win_rates, width, color=colors)
    axes[1].set_xlabel('Agent')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_title('Final 100 Episodes - Win Rate')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Win/Draw/Loss distribution
    draw_rates = [100 - w - l for w, l in zip(final_win_rates, final_loss_rates)]
    
    axes[2].bar(x, final_win_rates, width, label='Win', color='#4CAF50')
    axes[2].bar(x, draw_rates, width, bottom=final_win_rates, label='Draw', color='#FFC107')
    axes[2].bar(x, final_loss_rates, width, 
                bottom=[w + d for w, d in zip(final_win_rates, draw_rates)], 
                label='Loss', color='#F44336')
    
    axes[2].set_xlabel('Agent')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title('Final 100 Episodes - Outcome Distribution')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_report_plots(
    experiment_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path],
    prefix: str = "report"
) -> None:
    """
    Generate all plots for a training report.
    
    Args:
        experiment_dirs: List of experiment directory paths
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiment data
    experiments = [load_experiment_data(d) for d in experiment_dirs]
    
    print(f"Generating report plots for {len(experiments)} experiments...")
    
    # Generate all plots
    plot_training_rewards(experiments, output_dir / f"{prefix}_training_rewards.png")
    print("  - Training rewards plot saved")
    
    plot_win_rates(experiments, output_dir / f"{prefix}_win_rates.png")
    print("  - Win rates plot saved")
    
    plot_evaluation_metrics(experiments, output_dir / f"{prefix}_evaluation_metrics.png")
    print("  - Evaluation metrics plot saved")
    
    plot_loss_curves(experiments, output_dir / f"{prefix}_loss_curves.png")
    print("  - Loss curves plot saved")
    
    plot_comprehensive_comparison(experiments, output_dir / f"{prefix}_comprehensive.png")
    print("  - Comprehensive comparison plot saved")
    
    plot_final_statistics(experiments, output_dir / f"{prefix}_final_stats.png")
    print("  - Final statistics plot saved")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    # Example: Generate plots from experiment directories
    import sys
    
    if len(sys.argv) > 1:
        exp_dirs = sys.argv[1:]
        generate_report_plots(exp_dirs, "plots")
    else:
        print("Usage: python visualization.py <exp_dir1> [exp_dir2] ...")
