"""
Comprehensive Ablation Study for DQN and Rainbow agents in Hockey Environment.

This script runs ablations on:
1. Agent architectures: DQN (standard, double, dueling) vs Rainbow (full, variants)
2. Rainbow component ablations: no noisy, no n-step
3. Opponent training strategies: weak-only, strong-only, alternate, random, curriculum

All results save win rates against BOTH weak AND strong opponents for fair comparison.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from training.train import TrainingConfig, train, TrainingLogger, evaluate_agent
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode
from agents.dqn import DQNAgent
from agents.rainbow import RainbowAgent


@dataclass
class AblationResult:
    """Result from a single ablation experiment."""
    name: str
    variant: str
    category: str  # "agent", "rainbow_component", "opponent_strategy"
    
    # Training metrics
    final_train_win_rate: float
    final_train_reward: float
    
    # Evaluation against weak opponent
    eval_win_rate_weak: float
    eval_reward_weak: float
    
    # Evaluation against strong opponent
    eval_win_rate_strong: float
    eval_reward_strong: float
    
    # Best evaluation (during training)
    best_eval_win_rate: float
    
    # Training config used
    config: Dict[str, Any]
    
    # Log directory
    log_dir: str


class AblationStudy:
    """Manager for comprehensive ablation study."""
    
    def __init__(
        self,
        num_episodes: int = 3000,
        eval_episodes: int = 50,
        eval_frequency: int = 100,
        device: str = "auto",
        seed: int = 42,
        log_dir: str = "logs/ablation",
        results_dir: str = "results/ablation"
    ):
        self.num_episodes = num_episodes
        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency
        self.device = device
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.results_dir / f"ablation_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[AblationResult] = []
        
        # Create evaluation environments (for final evaluation)
        self.eval_env_weak = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
        self.eval_env_strong = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
    
    def _base_config(self) -> dict:
        """Get base configuration for all experiments."""
        return {
            'num_episodes': self.num_episodes,
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'mode': 'NORMAL',
            'seed': self.seed,
            'device': self.device,
            'log_dir': str(self.log_dir),
        }
    
    def _final_evaluate(
        self, 
        agent, 
        num_episodes: int = None
    ) -> Tuple[float, float, float, float]:
        """
        Final evaluation against both weak and strong opponents.
        
        Returns:
            (win_rate_weak, reward_weak, win_rate_strong, reward_strong)
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes
        
        # Evaluate against weak
        reward_weak, win_rate_weak, _ = evaluate_agent(
            agent, self.eval_env_weak, num_episodes
        )
        
        # Evaluate against strong
        reward_strong, win_rate_strong, _ = evaluate_agent(
            agent, self.eval_env_strong, num_episodes
        )
        
        return win_rate_weak, reward_weak, win_rate_strong, reward_strong
    
    def _run_single_experiment(
        self,
        name: str,
        variant: str,
        category: str,
        config: TrainingConfig
    ) -> AblationResult:
        """Run a single ablation experiment."""
        print(f"Running: {name} ({variant})")
        print(f"Category: {category}")
        
        # Train
        agent, logger = train(config)
        
        # Final evaluation against both opponents
        win_weak, reward_weak, win_strong, reward_strong = self._final_evaluate(agent)
        
        # Get final training stats
        stats = logger.get_recent_stats(window=100)
        
        # Get best eval win rate from training
        best_eval = max(logger.eval_win_rates) if logger.eval_win_rates else 0.0
        
        result = AblationResult(
            name=name,
            variant=variant,
            category=category,
            final_train_win_rate=stats.get('win_rate', 0),
            final_train_reward=stats.get('avg_reward', 0),
            eval_win_rate_weak=win_weak,
            eval_reward_weak=reward_weak,
            eval_win_rate_strong=win_strong,
            eval_reward_strong=reward_strong,
            best_eval_win_rate=best_eval,
            config=asdict(config),
            log_dir=str(logger.exp_dir)
        )
        
        self.results.append(result)
        
        # Print summary
        print(f"Results for {name}:")
        print(f"  Training Win Rate (last 100): {stats.get('win_rate', 0)*100:.1f}%")
        print(f"  Eval vs Weak:   Win={win_weak*100:.1f}%, Reward={reward_weak:.2f}")
        print(f"  Eval vs Strong: Win={win_strong*100:.1f}%, Reward={reward_strong:.2f}")
        
        return result
    
    def run_dqn_variants(self) -> List[AblationResult]:
        """Run DQN variant ablations."""
        results = []
        
        # 1. Standard DQN
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name="DQN_standard",
            double_dqn=False,
            dueling=False,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "DQN Standard", "standard", "agent", config
        ))
        
        # 2. Double DQN
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name="DDQN",
            double_dqn=True,
            dueling=False,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Double DQN", "double", "agent", config
        ))
        
        # 3. Dueling DQN
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name="Dueling_DQN",
            double_dqn=False,
            dueling=True,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Dueling DQN", "dueling", "agent", config
        ))
        
        # 4. Double Dueling DQN
        config = TrainingConfig(
            agent_type="dqn",
            experiment_name="Double_Dueling_DQN",
            double_dqn=True,
            dueling=True,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Double Dueling DQN", "double_dueling", "agent", config
        ))
        
        return results
    
    def run_rainbow_variants(self) -> List[AblationResult]:
        """Run Rainbow component ablations."""
        results = []
        
        # 1. Full Rainbow (all components)
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_full",
            noisy=True,
            n_step=3,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Rainbow Full", "full", "rainbow_component", config
        ))
        
        # 2. Rainbow without Noisy Networks
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_no_noisy",
            noisy=False,
            n_step=3,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Rainbow No Noisy", "no_noisy", "rainbow_component", config
        ))
        
        # 3. Rainbow with 1-step (no multi-step returns)
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_1step",
            noisy=True,
            n_step=1,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Rainbow 1-Step", "no_nstep", "rainbow_component", config
        ))
        
        # 4. Rainbow without Noisy AND 1-step (minimal Rainbow)
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_minimal",
            noisy=False,
            n_step=1,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Rainbow Minimal", "minimal", "rainbow_component", config
        ))
        
        # 5. Rainbow with larger n-step
        config = TrainingConfig(
            agent_type="rainbow",
            experiment_name="Rainbow_5step",
            noisy=True,
            n_step=5,
            mixed_opponents=False,
            weak_opponent=True,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Rainbow 5-Step", "5step", "rainbow_component", config
        ))
        
        return results
    
    def run_opponent_strategy_ablation(self) -> List[AblationResult]:
        """Run opponent training strategy ablations using Rainbow."""
        results = []
        
        # Use Rainbow as the base agent for opponent strategy comparison
        base_rainbow_config = {
            'agent_type': 'rainbow',
            'noisy': False, 
            'n_step': 3,
        }
        
        # 1. Weak opponent only
        config = TrainingConfig(
            experiment_name="Rainbow_weak_only",
            mixed_opponents=False,
            weak_opponent=True,
            **base_rainbow_config,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Weak Only", "weak_only", "opponent_strategy", config
        ))
        
        # 2. Strong opponent only
        config = TrainingConfig(
            experiment_name="Rainbow_strong_only",
            mixed_opponents=False,
            weak_opponent=False,
            **base_rainbow_config,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Strong Only", "strong_only", "opponent_strategy", config
        ))
        
        # 3. Alternate (weak/strong each episode)
        config = TrainingConfig(
            experiment_name="Rainbow_alternate",
            mixed_opponents=True,
            opponent_mix_strategy="alternate",
            **base_rainbow_config,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Alternate", "alternate", "opponent_strategy", config
        ))
        
        # 4. Random (50/50 each episode)
        config = TrainingConfig(
            experiment_name="Rainbow_random",
            mixed_opponents=True,
            opponent_mix_strategy="random",
            **base_rainbow_config,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Random", "random", "opponent_strategy", config
        ))
        
        # 5. Curriculum (start weak, increase strong)
        config = TrainingConfig(
            experiment_name="Rainbow_curriculum",
            mixed_opponents=True,
            opponent_mix_strategy="curriculum",
            **base_rainbow_config,
            **self._base_config()
        )
        results.append(self._run_single_experiment(
            "Curriculum", "curriculum", "opponent_strategy", config
        ))
        
        return results
    
    def run_full_ablation(self) -> List[AblationResult]:
        """Run complete ablation study."""
        print("\n\n")
        print("COMPREHENSIVE ABLATION STUDY")
        print("\n")
        print(f"Episodes per experiment: {self.num_episodes}")
        print(f"Evaluation episodes: {self.eval_episodes}")
        print(f"Results will be saved to: {self.results_dir}")
        print("\n")
        
        # Phase 1: DQN Variants
        print("\n\n")
        print("PHASE 1: DQN VARIANTS")
        print("\n")
        self.run_dqn_variants()
        
        # Phase 2: Rainbow Component Ablation
        print("\n\n")
        print("PHASE 2: RAINBOW COMPONENT ABLATION")
        print("\n")
        self.run_rainbow_variants()
        
        # Phase 3: Opponent Strategy Ablation
        print("\n\n")
        print("PHASE 3: OPPONENT TRAINING STRATEGY ABLATION")
        print("\n")
        self.run_opponent_strategy_ablation()
        
        # Save and plot results
        self.save_results()
        self.generate_plots()
        self.print_summary()
        
        return self.results
    
    def save_results(self) -> None:
        """Save all results to JSON."""
        results_dict = []
        for r in self.results:
            results_dict.append({
                'name': r.name,
                'variant': r.variant,
                'category': r.category,
                'final_train_win_rate': r.final_train_win_rate,
                'final_train_reward': r.final_train_reward,
                'eval_win_rate_weak': r.eval_win_rate_weak,
                'eval_reward_weak': r.eval_reward_weak,
                'eval_win_rate_strong': r.eval_win_rate_strong,
                'eval_reward_strong': r.eval_reward_strong,
                'best_eval_win_rate': r.best_eval_win_rate,
                'log_dir': r.log_dir
            })
        
        with open(self.results_dir / 'ablation_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {self.results_dir / 'ablation_results.json'}")
    
    def generate_plots(self) -> None:
        """Generate comprehensive ablation plots."""
        sns.set_style("whitegrid")
        
        # 1. Agent Architecture Comparison
        self._plot_agent_comparison()
        
        # 2. Rainbow Component Ablation
        self._plot_rainbow_ablation()
        
        # 3. Opponent Strategy Comparison
        self._plot_opponent_strategy()
        
        # 4. Overall Comparison Heatmap
        self._plot_overall_heatmap()
        
        # 5. Weak vs Strong Performance Scatter
        self._plot_weak_vs_strong()
        
        print(f"\nPlots saved to: {self.results_dir}")
    
    def _plot_agent_comparison(self) -> None:
        """Plot DQN vs Rainbow architecture comparison."""
        agent_results = [r for r in self.results if r.category == "agent"]
        if not agent_results:
            return
        
        # Add Rainbow full for comparison
        rainbow_full = [r for r in self.results if r.variant == "full"]
        if rainbow_full:
            agent_results = agent_results + rainbow_full
        
        names = [r.name for r in agent_results]
        win_weak = [r.eval_win_rate_weak * 100 for r in agent_results]
        win_strong = [r.eval_win_rate_strong * 100 for r in agent_results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, win_weak, width, label='vs Weak', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, win_strong, width, label='vs Strong', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Agent Architecture', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Agent Architecture Comparison: Win Rate vs Weak and Strong Opponents', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'agent_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rainbow_ablation(self) -> None:
        """Plot Rainbow component ablation."""
        rainbow_results = [r for r in self.results if r.category == "rainbow_component"]
        if not rainbow_results:
            return
        
        names = [r.name for r in rainbow_results]
        win_weak = [r.eval_win_rate_weak * 100 for r in rainbow_results]
        win_strong = [r.eval_win_rate_strong * 100 for r in rainbow_results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, win_weak, width, label='vs Weak', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, win_strong, width, label='vs Strong', color='#9b59b6', alpha=0.8)
        
        ax.set_xlabel('Rainbow Variant', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Rainbow Component Ablation: Impact of Each Component', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'rainbow_ablation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_opponent_strategy(self) -> None:
        """Plot opponent training strategy comparison."""
        strategy_results = [r for r in self.results if r.category == "opponent_strategy"]
        if not strategy_results:
            return
        
        names = [r.name for r in strategy_results]
        win_weak = [r.eval_win_rate_weak * 100 for r in strategy_results]
        win_strong = [r.eval_win_rate_strong * 100 for r in strategy_results]
        
        # Calculate average performance
        avg_win = [(w + s) / 2 for w, s in zip(win_weak, win_strong)]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Grouped bar chart
        ax = axes[0]
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, win_weak, width, label='vs Weak', color='#1abc9c', alpha=0.8)
        bars2 = ax.bar(x + width/2, win_strong, width, label='vs Strong', color='#e67e22', alpha=0.8)
        
        ax.set_xlabel('Training Strategy', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Opponent Training Strategy Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        
        # Right: Average performance with generalization gap
        ax = axes[1]
        generalization_gap = [abs(w - s) for w, s in zip(win_weak, win_strong)]
        
        bars = ax.bar(names, avg_win, color='#34495e', alpha=0.8)
        ax.errorbar(names, avg_win, yerr=[g/2 for g in generalization_gap], 
                    fmt='none', color='red', capsize=5, label='Generalization Gap')
        
        ax.set_xlabel('Training Strategy', fontsize=12)
        ax.set_ylabel('Average Win Rate (%)', fontsize=12)
        ax.set_title('Average Performance & Generalization Gap', fontsize=14)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'opponent_strategy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_heatmap(self) -> None:
        """Plot overall comparison heatmap."""
        if not self.results:
            return
        
        # Prepare data for heatmap
        names = [r.name for r in self.results]
        metrics = ['Win vs Weak', 'Win vs Strong', 'Avg Win', 'Train Win']
        
        data = []
        for r in self.results:
            avg_win = (r.eval_win_rate_weak + r.eval_win_rate_strong) / 2
            data.append([
                r.eval_win_rate_weak * 100,
                r.eval_win_rate_strong * 100,
                avg_win * 100,
                r.final_train_win_rate * 100
            ])
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 12))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(names)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Win Rate (%)', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Overall Ablation Study Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'overall_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_weak_vs_strong(self) -> None:
        """Scatter plot of weak vs strong performance."""
        if not self.results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Define colors by category
        colors = {
            'agent': '#3498db',
            'rainbow_component': '#9b59b6',
            'opponent_strategy': '#e67e22'
        }
        
        for category in ['agent', 'rainbow_component', 'opponent_strategy']:
            cat_results = [r for r in self.results if r.category == category]
            if not cat_results:
                continue
            
            x = [r.eval_win_rate_weak * 100 for r in cat_results]
            y = [r.eval_win_rate_strong * 100 for r in cat_results]
            names = [r.name for r in cat_results]
            
            ax.scatter(x, y, c=colors[category], s=100, alpha=0.7, 
                      label=category.replace('_', ' ').title())
            
            for i, name in enumerate(names):
                ax.annotate(name, (x[i], y[i]), textcoords="offset points",
                           xytext=(5, 5), ha='left', fontsize=8)
        
        # Add diagonal line (equal performance)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal Performance')
        
        ax.set_xlabel('Win Rate vs Weak Opponent (%)', fontsize=12)
        ax.set_ylabel('Win Rate vs Strong Opponent (%)', fontsize=12)
        ax.set_title('Performance: Weak vs Strong Opponent', fontsize=14)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'weak_vs_strong_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self) -> None:
        """Print comprehensive summary of results."""
        print("ABLATION STUDY SUMMARY")
        
        # Group by category
        categories = ['agent', 'rainbow_component', 'opponent_strategy']
        category_names = ['Agent Architecture', 'Rainbow Components', 'Opponent Strategy']
        
        for cat, cat_name in zip(categories, category_names):
            cat_results = [r for r in self.results if r.category == cat]
            if not cat_results:
                continue
            
            print(f"\n{cat_name}:")
            print(f"{'Name':<25} {'Train Win':>10} {'vs Weak':>10} {'vs Strong':>10} {'Avg':>10}")
            
            for r in cat_results:
                avg = (r.eval_win_rate_weak + r.eval_win_rate_strong) / 2
                print(f"{r.name:<25} {r.final_train_win_rate*100:>9.1f}% "
                      f"{r.eval_win_rate_weak*100:>9.1f}% "
                      f"{r.eval_win_rate_strong*100:>9.1f}% "
                      f"{avg*100:>9.1f}%")
        
        # Find best performers
        print("BEST PERFORMERS")
        
        if self.results:
            # Best vs weak
            best_weak = max(self.results, key=lambda r: r.eval_win_rate_weak)
            print(f"Best vs Weak Opponent:   {best_weak.name} ({best_weak.eval_win_rate_weak*100:.1f}%)")
            
            # Best vs strong
            best_strong = max(self.results, key=lambda r: r.eval_win_rate_strong)
            print(f"Best vs Strong Opponent: {best_strong.name} ({best_strong.eval_win_rate_strong*100:.1f}%)")
            
            # Best average
            best_avg = max(self.results, 
                          key=lambda r: (r.eval_win_rate_weak + r.eval_win_rate_strong) / 2)
            avg_rate = (best_avg.eval_win_rate_weak + best_avg.eval_win_rate_strong) / 2
            print(f"Best Average:            {best_avg.name} ({avg_rate*100:.1f}%)")
            
            # Best generalization (smallest gap)
            best_gen = min(self.results, 
                          key=lambda r: abs(r.eval_win_rate_weak - r.eval_win_rate_strong))
            gap = abs(best_gen.eval_win_rate_weak - best_gen.eval_win_rate_strong)
            print(f"Best Generalization:     {best_gen.name} (gap: {gap*100:.1f}%)")
        
        print(f"Full results saved to: {self.results_dir}")
        
        # Save summary to file
        with open(self.results_dir / 'summary.txt', 'w') as f:
            f.write("ABLATION STUDY SUMMARY\n")
            
            for cat, cat_name in zip(categories, category_names):
                cat_results = [r for r in self.results if r.category == cat]
                if not cat_results:
                    continue
                
                f.write(f"\n{cat_name}:\n")
                f.write(f"{'Name':<25} {'Train Win':>10} {'vs Weak':>10} {'vs Strong':>10} {'Avg':>10}\n")
                
                for r in cat_results:
                    avg = (r.eval_win_rate_weak + r.eval_win_rate_strong) / 2
                    f.write(f"{r.name:<25} {r.final_train_win_rate*100:>9.1f}% "
                           f"{r.eval_win_rate_weak*100:>9.1f}% "
                           f"{r.eval_win_rate_strong*100:>9.1f}% "
                           f"{avg*100:>9.1f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ablation Study for DQN/Rainbow in Hockey Environment"
    )
    
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes per experiment")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--eval-freq", type=int, default=100,
                        help="Evaluation frequency during training")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs/ablation",
                        help="Directory for training logs")
    parser.add_argument("--results-dir", type=str, default="results/ablation",
                        help="Directory for results and plots")
    
    # Selective ablation options
    parser.add_argument("--dqn-only", action="store_true",
                        help="Run only DQN variant ablation")
    parser.add_argument("--rainbow-only", action="store_true",
                        help="Run only Rainbow component ablation")
    parser.add_argument("--opponent-only", action="store_true",
                        help="Run only opponent strategy ablation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced episodes (500)")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.episodes = 500
        args.eval_episodes = 20
        args.eval_freq = 100
    
    # Create ablation study
    study = AblationStudy(
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        eval_frequency=args.eval_freq,
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
        results_dir=args.results_dir
    )
    
    # Run selected ablations
    if args.dqn_only:
        study.run_dqn_variants()
    elif args.rainbow_only:
        study.run_rainbow_variants()
    elif args.opponent_only:
        study.run_opponent_strategy_ablation()
    else:
        study.run_full_ablation()
    
    # Always save and plot
    study.save_results()
    study.generate_plots()
    study.print_summary()


if __name__ == "__main__":
    main()
