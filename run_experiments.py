"""
Main experiment runner for comparing DQN and Rainbow agents on Hockey Environment.

This script runs comprehensive experiments and generates analysis plots.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.train import train, TrainingConfig, evaluate_agent
from utils.visualization import generate_report_plots, load_experiment_data
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode


def run_dqn_experiments(base_config: dict) -> List[Path]:
    """Run DQN experiments with different configurations."""
    experiment_dirs = []
    
    # Standard DQN
    print("Running Standard DQN")
    
    config = TrainingConfig(
        agent_type="dqn",
        experiment_name="DQN_standard",
        double_dqn=False,
        dueling=False,
        **base_config
    )
    _, logger = train(config)
    experiment_dirs.append(logger.exp_dir)
    
    # Double DQN
    print("Running Double DQN")
    
    config = TrainingConfig(
        agent_type="dqn",
        experiment_name="DDQN",
        double_dqn=True,
        dueling=False,
        **base_config
    )
    _, logger = train(config)
    experiment_dirs.append(logger.exp_dir)
    
    # Dueling DQN
    print("Running Dueling DQN")
    
    config = TrainingConfig(
        agent_type="dqn",
        experiment_name="Dueling_DQN",
        double_dqn=True,
        dueling=True,
        **base_config
    )
    _, logger = train(config)
    experiment_dirs.append(logger.exp_dir)
    
    return experiment_dirs


def run_rainbow_experiment(base_config: dict) -> Path:
    """Run Rainbow DQN experiment."""
    print("Running Rainbow DQN")
    
    config = TrainingConfig(
        agent_type="rainbow",
        experiment_name="Rainbow",
        n_step=3,
        num_atoms=51,
        noisy=True,
        **base_config
    )
    _, logger = train(config)
    return logger.exp_dir


def run_ablation_study(base_config: dict) -> List[Path]:
    """Run Rainbow ablation study to understand component contributions."""
    experiment_dirs = []
    
    # Rainbow without noisy networks
    print("Running Rainbow (no noisy nets)")
    
    config = TrainingConfig(
        agent_type="rainbow",
        experiment_name="Rainbow_no_noisy",
        noisy=False,
        **base_config
    )
    _, logger = train(config)
    experiment_dirs.append(logger.exp_dir)
    
    # Rainbow with 1-step (no multi-step)
    print("Running Rainbow (1-step)")
    
    config = TrainingConfig(
        agent_type="rainbow",
        experiment_name="Rainbow_1step",
        n_step=1,
        **base_config
    )
    _, logger = train(config)
    experiment_dirs.append(logger.exp_dir)
    
    return experiment_dirs


def run_hyperparameter_study(base_config: dict) -> List[Path]:
    """Study effect of key hyperparameters."""
    experiment_dirs = []
    
    # Different learning rates
    for lr in [1e-3, 1e-4, 1e-5]:
        print(f"Running DQN with lr={lr}")
        
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name=f"DQN_lr_{lr}",
            lr=lr,
            **base_config
        )
        _, logger = train(config)
        experiment_dirs.append(logger.exp_dir)
    
    return experiment_dirs


def generate_analysis_report(experiment_dirs: List[Path], output_dir: Path) -> None:
    """Generate comprehensive analysis report with plots."""
    print("Generating Analysis Report")
    
    # Load all experiment data
    experiments = [load_experiment_data(d) for d in experiment_dirs]
    
    # Generate plots
    generate_report_plots(experiment_dirs, output_dir, prefix="comparison")
    
    # Generate summary statistics
    summary = []
    for exp in experiments:
        name = exp.get('config', {}).get('experiment_name', exp.get('name', 'Unknown'))
        
        if 'episode_rewards' in exp:
            final_rewards = exp['episode_rewards'][-100:]
            avg_reward = np.mean(final_rewards)
            std_reward = np.std(final_rewards)
        else:
            avg_reward, std_reward = 0, 0
        
        if 'episode_wins' in exp:
            wins = exp['episode_wins'][-100:]
            win_rate = np.mean([1 if w == 1 else 0 for w in wins]) * 100
            loss_rate = np.mean([1 if w == -1 else 0 for w in wins]) * 100
        else:
            win_rate, loss_rate = 0, 0
        
        if 'eval_win_rates' in exp and len(exp['eval_win_rates']) > 0:
            best_eval_win_rate = max(exp['eval_win_rates']) * 100
        else:
            best_eval_win_rate = 0
        
        summary.append({
            'name': name,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'best_eval_win_rate': best_eval_win_rate
        })
    
    # Print summary table
    print("Final Performance Summary (Last 100 Episodes)")
    print(f"{'Agent':<25} {'Avg Reward':>12} {'Win Rate':>12} {'Loss Rate':>12} {'Best Eval':>12}")
    
    for s in summary:
        print(f"{s['name']:<25} {s['avg_reward']:>10.2f}±{s['std_reward']:.2f} "
              f"{s['win_rate']:>11.1f}% {s['loss_rate']:>11.1f}% {s['best_eval_win_rate']:>11.1f}%")
    
    
    # Save summary to file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DQN vs Rainbow Performance Comparison\n")
        
        for s in summary:
            f.write(f"Agent: {s['name']}\n")
            f.write(f"  Average Reward: {s['avg_reward']:.2f} ± {s['std_reward']:.2f}\n")
            f.write(f"  Win Rate: {s['win_rate']:.1f}%\n")
            f.write(f"  Loss Rate: {s['loss_rate']:.1f}%\n")
            f.write(f"  Best Evaluation Win Rate: {s['best_eval_win_rate']:.1f}%\n\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run DQN/Rainbow experiments on Hockey Environment")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="comparison",
                        choices=["comparison", "dqn_only", "rainbow_only", "ablation", "hyperparams", "quick_test", "rainbow_not_noisy"],
                        help="Type of experiment to run")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes")
    parser.add_argument("--eval-freq", type=int, default=500,
                        help="Evaluation frequency (episodes)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    
    # Environment parameters
    parser.add_argument("--mode", type=str, default="NORMAL",
                        choices=["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"],
                        help="Environment mode")
    parser.add_argument("--weak-opponent", action="store_true", default=True,
                        help="Use weak opponent")
    parser.add_argument("--strong-opponent", action="store_true",
                        help="Use strong opponent")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        'num_episodes': args.episodes,
        'eval_frequency': args.eval_freq,
        'eval_episodes': args.eval_episodes,
        'mode': args.mode,
        'weak_opponent': not args.strong_opponent,
        'seed': args.seed,
        'device': args.device,
        'log_dir': args.log_dir,
        'mixed_opponents': True,
        'opponent_mix_strategy': 'alternate'  # Alternate between weak and strong opponents
    }
    
    print("Hockey Environment DQN/Rainbow Experiments")
    print(f"Experiment type: {args.experiment}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Opponent: {'Weak' if not args.strong_opponent else 'Strong'}")
    
    experiment_dirs = []
    
    if args.experiment == "quick_test":
        # Quick test with fewer episodes
        base_config['num_episodes'] = 100
        base_config['eval_frequency'] = 50
        
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name="quick_test",
            **base_config
        )
        _, logger = train(config)
        experiment_dirs.append(logger.exp_dir)
        
    elif args.experiment == "comparison":
        # Full DQN vs Rainbow comparison
        experiment_dirs.extend(run_dqn_experiments(base_config))
        experiment_dirs.append(run_rainbow_experiment(base_config))
        
    elif args.experiment == "dqn_only":
        experiment_dirs.extend(run_dqn_experiments(base_config))
        
    elif args.experiment == "rainbow_only":
        experiment_dirs.append(run_rainbow_experiment(base_config))
        
    elif args.experiment == "ablation":
        experiment_dirs.append(run_rainbow_experiment(base_config))
        experiment_dirs.extend(run_ablation_study(base_config))
        
    elif args.experiment == "hyperparams":
        experiment_dirs.extend(run_hyperparameter_study(base_config))
    elif args.experiment == "rainbow_not_noisy":
        # Run Rainbow without noisy networks
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_no_noisy",
            noisy=False,
            **base_config
        )
        _, logger = train(config)
        experiment_dirs.append(logger.exp_dir)
    
    # Generate analysis report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_analysis_report(experiment_dirs, output_dir)
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()
