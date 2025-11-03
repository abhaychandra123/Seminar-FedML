"""Visualization script for benchmarking results."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results(summary_path="results/benchmark_summary.json"):
    """Load benchmark results from JSON file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def plot_training_time_comparison(results, save_path="results/training_time.png"):
    """Plot training time comparison across strategies."""
    successful = [r for r in results["results"] if r.get("success")]
    
    if not successful:
        print("No successful results to plot.")
        return
    
    strategies = [r["strategy_name"] for r in successful]
    times = [r["elapsed_time"] for r in successful]
    
    # Sort by time
    sorted_pairs = sorted(zip(strategies, times), key=lambda x: x[1])
    strategies, times = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(strategies))
    bars = plt.bar(range(len(strategies)), times, color=colors)
    
    # Highlight top 3 and bottom 3
    if len(bars) >= 3:
        for i in range(3):
            bars[i].set_edgecolor('green')
            bars[i].set_linewidth(2)
            bars[-(i+1)].set_edgecolor('red')
            bars[-(i+1)].set_linewidth(2)
    
    plt.xlabel("Strategy", fontsize=12)
    plt.ylabel("Training Time (seconds)", fontsize=12)
    plt.title("Training Time Comparison Across Client Selection Strategies", fontsize=14)
    plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training time plot saved to: {save_path}")
    plt.close()


def plot_loss_accuracy_comparison(results, save_path="results/loss_accuracy.png"):
    """Plot loss and accuracy comparison across strategies."""
    successful = [r for r in results["results"] if r.get("success")]
    
    if not successful:
        print("No successful results to plot.")
        return
    
    strategies = [r["strategy_name"] for r in successful]
    losses = [r.get("final_loss", 0) for r in successful]
    accuracies = [r.get("final_accuracy", 0) for r in successful]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss comparison (sorted, lower is better)
    sorted_pairs = sorted(zip(strategies, losses), key=lambda x: x[1])
    strat_loss, loss_vals = zip(*sorted_pairs)
    
    colors_loss = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(strat_loss)))
    bars1 = ax1.barh(range(len(strat_loss)), loss_vals, color=colors_loss)
    ax1.set_yticks(range(len(strat_loss)))
    ax1.set_yticklabels(strat_loss)
    ax1.set_xlabel('Final Loss', fontsize=12)
    ax1.set_title('Final Loss by Strategy (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (bar, loss) in enumerate(zip(bars1, loss_vals)):
        ax1.text(loss, i, f' {loss:.4f}', va='center', fontsize=9)
    
    # Accuracy comparison (sorted, higher is better)
    sorted_pairs = sorted(zip(strategies, accuracies), key=lambda x: x[1], reverse=True)
    strat_acc, acc_vals = zip(*sorted_pairs)
    
    colors_acc = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strat_acc)))[::-1]
    bars2 = ax2.barh(range(len(strat_acc)), acc_vals, color=colors_acc)
    ax2.set_yticks(range(len(strat_acc)))
    ax2.set_yticklabels(strat_acc)
    ax2.set_xlabel('Final Accuracy', fontsize=12)
    ax2.set_title('Final Accuracy by Strategy (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (bar, acc) in enumerate(zip(bars2, acc_vals)):
        ax2.text(acc, i, f' {acc:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss and accuracy comparison saved to: {save_path}")
    plt.close()


def plot_metrics_evolution(results, save_path="results/metrics_evolution.png"):
    """Plot loss and accuracy evolution over rounds for top 5 performers in each metric."""
    successful = [r for r in results["results"] if r.get("success")]
    
    if not successful:
        print("No successful results to plot.")
        return
    
    # Check if metrics history is available
    has_history = any(r.get("metrics_history") for r in successful)
    if not has_history:
        print("No metrics history available for evolution plot.")
        return
    
    # Get top 3 by final loss (lower is better)
    sorted_by_loss = sorted(successful, key=lambda x: x.get("final_loss", float('inf')))[:3]
    
    # Get top 3 by final accuracy (higher is better)
    sorted_by_acc = sorted(successful, key=lambda x: x.get("final_accuracy", 0), reverse=True)[:3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss evolution for top 3 by loss - line plot
    colors_loss = plt.cm.tab10(np.linspace(0, 1, 3))
    for i, result in enumerate(sorted_by_loss):
        metrics_hist = result.get("metrics_history", [])
        if metrics_hist:
            # Remove duplicates by keeping only unique rounds
            unique_rounds = {}
            for m in metrics_hist:
                round_num = m["round"]
                if round_num not in unique_rounds:
                    unique_rounds[round_num] = m
            
            # Sort by round number
            sorted_metrics = sorted(unique_rounds.values(), key=lambda x: x["round"])
            rounds = [m["round"] for m in sorted_metrics]
            loss_hist = [m["loss"] for m in sorted_metrics]
            final_loss = result.get("final_loss", 0)
            ax1.plot(rounds, loss_hist, marker='o', markersize=6,
                       label=f"{result['strategy_name']} ({final_loss:.4f})", 
                       alpha=0.7, linewidth=2, color=colors_loss[i])
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Evolution - Top 3 Best Performers', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, title='Strategy (Final Loss)')
    ax1.grid(alpha=0.3)
    
    # Plot accuracy evolution for top 3 by accuracy - line plot
    colors_acc = plt.cm.tab10(np.linspace(0, 1, 3))
    for i, result in enumerate(sorted_by_acc):
        metrics_hist = result.get("metrics_history", [])
        if metrics_hist:
            # Remove duplicates by keeping only unique rounds
            unique_rounds = {}
            for m in metrics_hist:
                round_num = m["round"]
                if round_num not in unique_rounds:
                    unique_rounds[round_num] = m
            
            # Sort by round number
            sorted_metrics = sorted(unique_rounds.values(), key=lambda x: x["round"])
            rounds = [m["round"] for m in sorted_metrics]
            acc_hist = [m["accuracy"] for m in sorted_metrics]
            final_acc = result.get("final_accuracy", 0)
            ax2.plot(rounds, acc_hist, marker='s', markersize=6,
                       label=f"{result['strategy_name']} ({final_acc:.4f})", 
                       alpha=0.7, linewidth=2, color=colors_acc[i])
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Evolution - Top 3 Best Performers', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, title='Strategy (Final Accuracy)')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics evolution plot saved to: {save_path}")
    plt.close()


def plot_strategy_categories(results, save_path="results/strategy_categories.png"):
    """Plot strategies grouped by category."""
    successful = [r for r in results["results"] if r.get("success")]
    
    if not successful:
        print("No successful results to plot.")
        return
    
    # Categorize strategies
    categories = {
        "Data Sample-Based": ["num_samples", "high_grad_norm", "high_loss", "sum_loss"],
        "Model-Based": ["model_divergence", "same_sign", "update_direction", 
                       "local_change", "post_grad_norm", "grad_dot_product"],
        "Deadline-Based": ["hard_deadline", "soft_deadline"],
        "Baseline": ["random"]
    }
    
    category_times = {cat: [] for cat in categories}
    
    for result in successful:
        strategy = result["strategy_name"]
        time = result["elapsed_time"]
        
        for cat, strategies in categories.items():
            if strategy in strategies:
                category_times[cat].append(time)
                break
    
    # Calculate average time per category
    avg_times = {cat: np.mean(times) if times else 0 
                 for cat, times in category_times.items() if times}
    
    if not avg_times:
        print("No valid category data to plot.")
        return
    
    categories_list = list(avg_times.keys())
    times_list = list(avg_times.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    plt.bar(categories_list, times_list, color=colors[:len(categories_list)])
    
    plt.xlabel("Strategy Category", fontsize=12)
    plt.ylabel("Average Training Time (seconds)", fontsize=12)
    plt.title("Average Training Time by Strategy Category", fontsize=14)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category plot saved to: {save_path}")
    plt.close()


def plot_strategy_comparison_radar(results, save_path="results/strategy_radar.png"):
    """Create a radar chart comparing top 5 strategies across multiple metrics."""
    successful = [r for r in results["results"] if r.get("success")]
    
    if len(successful) < 3:
        print("Need at least 3 successful results for radar plot.")
        return
    
    # Select top 5 strategies by training time
    sorted_results = sorted(successful, key=lambda x: x["elapsed_time"])[:5]
    
    strategies = [r["strategy_name"] for r in sorted_results]
    
    # Extract real metrics from the results
    metrics = ["Speed", "Loss (inv)", "Accuracy", "Stability", "Efficiency"]
    num_metrics = len(metrics)
    
    # Compute real normalized scores (0-1) for each strategy
    scores = {}
    
    # Get min/max values for normalization
    all_times = [r["elapsed_time"] for r in sorted_results]
    all_losses = [r.get("final_loss", float('inf')) for r in sorted_results]
    all_accs = [r.get("final_accuracy", 0) for r in sorted_results]
    
    # Calculate stability from metrics history (std dev of loss)
    all_stabilities = []
    for r in sorted_results:
        metrics_hist = r.get("metrics_history", [])
        if metrics_hist and len(metrics_hist) > 1:
            losses = [m["loss"] for m in metrics_hist]
            stability = 1 / (1 + np.std(losses))  # Lower std = higher stability
        else:
            stability = 0.5
        all_stabilities.append(stability)
    
    min_time, max_time = min(all_times), max(all_times)
    min_loss, max_loss = min(all_losses), max(all_losses)
    min_acc, max_acc = min(all_accs), max(all_accs)
    
    for idx, result in enumerate(sorted_results):
        strategy = result["strategy_name"]
        time = result["elapsed_time"]
        loss = result.get("final_loss", float('inf'))
        acc = result.get("final_accuracy", 0)
        stability = all_stabilities[idx]
        
        # Normalize scores to 0-1
        # Speed: lower time = higher score (inverse)
        speed_score = 1 - (time - min_time) / (max_time - min_time + 1e-10)
        
        # Loss: lower loss = higher score (inverse)
        loss_score = 1 - (loss - min_loss) / (max_loss - min_loss + 1e-10)
        
        # Accuracy: higher accuracy = higher score (direct)
        acc_score = (acc - min_acc) / (max_acc - min_acc + 1e-10)
        
        # Efficiency: combination of speed and accuracy
        efficiency_score = (speed_score + acc_score) / 2
        
        scores[strategy] = [
            speed_score,
            loss_score,
            acc_score,
            stability,
            efficiency_score
        ]
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        values = scores[strategy]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title("Multi-Metric Strategy Comparison (Top 5)", fontsize=14, pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Radar plot saved to: {save_path}")
    print("  Note: All metrics are computed from real benchmark data.")
    print("  Stability = inverse of loss std dev, Efficiency = (Speed + Accuracy) / 2")
    plt.close()


def generate_summary_report(results, save_path="results/summary_report.txt"):
    """Generate a text summary report."""
    successful = [r for r in results["results"] if r.get("success")]
    failed = [r for r in results["results"] if not r.get("success")]
    
    with open(save_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("FEDERATED LEARNING CLIENT SELECTION BENCHMARK REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Configuration
        config = results["configuration"]
        f.write("Configuration:\n")
        f.write(f"  - Number of Clients: {config['num_clients']}\n")
        f.write(f"  - Number of Rounds: {config['num_rounds']}\n")
        f.write(f"  - Fraction Train: {config['fraction_train']}\n")
        f.write(f"  - Local Epochs: {config['local_epochs']}\n")
        f.write(f"  - Learning Rate: {config['lr']}\n\n")
        
        # Summary statistics
        f.write(f"Summary:\n")
        f.write(f"  - Total Strategies Tested: {len(results['results'])}\n")
        f.write(f"  - Successful: {len(successful)}\n")
        f.write(f"  - Failed: {len(failed)}\n\n")
        
        if successful:
            times = [r["elapsed_time"] for r in successful]
            losses = [r.get("final_loss", 0) for r in successful]
            accs = [r.get("final_accuracy", 0) for r in successful]
            
            f.write("Training Time Statistics:\n")
            f.write(f"  - Mean: {np.mean(times):.2f}s\n")
            f.write(f"  - Median: {np.median(times):.2f}s\n")
            f.write(f"  - Std Dev: {np.std(times):.2f}s\n")
            f.write(f"  - Min: {np.min(times):.2f}s\n")
            f.write(f"  - Max: {np.max(times):.2f}s\n\n")
            
            f.write("Final Loss Statistics:\n")
            f.write(f"  - Mean: {np.mean(losses):.4f}\n")
            f.write(f"  - Median: {np.median(losses):.4f}\n")
            f.write(f"  - Std Dev: {np.std(losses):.4f}\n")
            f.write(f"  - Best (Min): {np.min(losses):.4f}\n")
            f.write(f"  - Worst (Max): {np.max(losses):.4f}\n\n")
            
            f.write("Final Accuracy Statistics:\n")
            f.write(f"  - Mean: {np.mean(accs):.4f}\n")
            f.write(f"  - Median: {np.median(accs):.4f}\n")
            f.write(f"  - Std Dev: {np.std(accs):.4f}\n")
            f.write(f"  - Best (Max): {np.max(accs):.4f}\n")
            f.write(f"  - Worst (Min): {np.min(accs):.4f}\n\n")
            
            # Rankings
            sorted_results = sorted(successful, key=lambda x: x["elapsed_time"])
            f.write("Rankings (by Training Time):\n")
            f.write("-"*90 + "\n")
            f.write(f"{'Rank':<6} {'Strategy':<25} {'Time (s)':<12} {'Loss':<12} {'Accuracy':<12} {'Category'}\n")
            f.write("-"*90 + "\n")
            
            categories = {
                "num_samples": "Data Sample", "high_grad_norm": "Data Sample",
                "high_loss": "Data Sample", "sum_loss": "Data Sample",
                "model_divergence": "Model-Based", "same_sign": "Model-Based",
                "update_direction": "Model-Based", "local_change": "Model-Based",
                "post_grad_norm": "Model-Based", "grad_dot_product": "Model-Based",
                "hard_deadline": "Deadline", "soft_deadline": "Deadline",
                "random": "Baseline"
            }
            
            for i, result in enumerate(sorted_results, 1):
                strategy = result["strategy_name"]
                time = result["elapsed_time"]
                loss = result.get("final_loss", 0)
                acc = result.get("final_accuracy", 0)
                category = categories.get(strategy, "Unknown")
                f.write(f"{i:<6} {strategy:<25} {time:<12.2f} {loss:<12.4f} {acc:<12.4f} {category}\n")
            
            f.write("-"*90 + "\n\n")
        
        if failed:
            f.write("Failed Strategies:\n")
            for result in failed:
                f.write(f"  - {result['strategy_name']}: {result.get('error', 'Unknown error')}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Report generated: " + results["timestamp"] + "\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: {save_path}")


def main():
    """Generate all visualizations."""
    results_path = Path("results/benchmark_summary.json")
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run benchmark first: python -m fl_sim.benchmark")
        return
    
    print("\nGenerating visualizations...")
    results = load_results(results_path)
    
    # Generate plots
    plot_training_time_comparison(results)
    plot_loss_accuracy_comparison(results)
    plot_metrics_evolution(results)
    plot_strategy_categories(results)
    plot_strategy_comparison_radar(results)
    
    # Generate report
    generate_summary_report(results)
    
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - results/training_time.png")
    print("  - results/loss_accuracy.png")
    print("  - results/metrics_evolution.png")
    print("  - results/strategy_categories.png")
    print("  - results/strategy_radar.png")
    print("  - results/summary_report.txt")


if __name__ == "__main__":
    main()