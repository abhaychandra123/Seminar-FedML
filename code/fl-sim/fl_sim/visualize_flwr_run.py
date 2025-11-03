"""Visualization script for Flower simulation run output."""

import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List


def parse_flwr_log(log_file: str = "flwr_run_output.log") -> Dict:
    """Parse the Flower run output log file."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract train metrics per round
    train_pattern = r"Aggregated ClientApp-side Train Metrics:\s+INFO\s+:\s+{([^}]+(?:}[^}]*)*?)}"
    train_matches = re.findall(train_pattern, content, re.DOTALL)
    
    train_metrics_by_round = []
    
    if train_matches:
        # Parse the dictionary-like structure
        for match in train_matches:
            round_data = {}
            # Extract round numbers and their metrics
            round_blocks = re.finditer(r"(\d+):\s*{([^}]+)}", match)
            for block in round_blocks:
                round_num = int(block.group(1))
                metrics_str = block.group(2)
                
                metrics = {}
                # Extract each metric
                metric_matches = re.findall(r"'(\w+)':\s*'?([\d.e+-]+)'?", metrics_str)
                for metric_name, value in metric_matches:
                    try:
                        metrics[metric_name] = float(value)
                    except:
                        metrics[metric_name] = value
                
                round_data[round_num] = metrics
            
            if round_data:
                train_metrics_by_round.append(round_data)
    
    # Extract evaluate metrics if present
    eval_pattern = r"Aggregated ClientApp-side Evaluate Metrics:\s+INFO\s+:\s+{([^}]+(?:}[^}]*)*?)}"
    eval_matches = re.findall(eval_pattern, content, re.DOTALL)
    
    eval_metrics_by_round = []
    if eval_matches:
        for match in eval_matches:
            round_data = {}
            round_blocks = re.finditer(r"(\d+):\s*{([^}]+)}", match)
            for block in round_blocks:
                round_num = int(block.group(1))
                metrics_str = block.group(2)
                
                metrics = {}
                metric_matches = re.findall(r"'(\w+)':\s*'?([\d.e+-]+)'?", metrics_str)
                for metric_name, value in metric_matches:
                    try:
                        metrics[metric_name] = float(value)
                    except:
                        metrics[metric_name] = value
                
                round_data[round_num] = metrics
            
            if round_data:
                eval_metrics_by_round.append(round_data)
    
    return {
        "train_metrics": train_metrics_by_round,
        "eval_metrics": eval_metrics_by_round
    }


def aggregate_metrics(metrics_by_round: List[Dict]) -> Dict:
    """Aggregate metrics across all rounds."""
    
    if not metrics_by_round:
        return {}
    
    # Get all rounds
    all_metrics = {}
    
    for round_data in metrics_by_round:
        for round_num, metrics in round_data.items():
            if round_num not in all_metrics:
                all_metrics[round_num] = {}
            
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics[round_num]:
                    all_metrics[round_num][metric_name] = []
                all_metrics[round_num][metric_name].append(value)
    
    # Average across clients for each round
    averaged = {}
    for round_num, metrics in all_metrics.items():
        averaged[round_num] = {}
        for metric_name, values in metrics.items():
            averaged[round_num][metric_name] = np.mean(values)
    
    return averaged


def plot_training_progress(train_metrics: Dict, save_path: str = "results/flwr_training_progress.png"):
    """Plot training loss and accuracy over rounds."""
    
    if not train_metrics:
        print("No training metrics to plot.")
        return
    
    rounds = sorted(train_metrics.keys())
    train_loss = [train_metrics[r].get('train_loss', 0) for r in rounds]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(rounds, train_loss, marker='o', linewidth=2, markersize=8, color='#e74c3c', label='Train Loss')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training progress plot saved to: {save_path}")
    plt.close()


def plot_client_metrics(train_metrics: Dict, save_path: str = "results/flwr_client_metrics.png"):
    """Plot various client-side metrics over rounds."""
    
    if not train_metrics:
        print("No training metrics to plot.")
        return
    
    rounds = sorted(train_metrics.keys())
    
    # Extract metrics
    metrics_to_plot = {
        'Model Divergence': 'model_divergence',
        'Post-Training Grad Norm': 'post_grad_norm',
        'Same Sign %': 'same_sign_percentage',
        'Update Magnitude': 'update_magnitude',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (title, metric_key) in enumerate(metrics_to_plot.items()):
        values = [train_metrics[r].get(metric_key, 0) for r in rounds]
        
        axes[idx].plot(rounds, values, marker='o', linewidth=2, markersize=6, 
                      color=colors[idx], label=title)
        axes[idx].set_xlabel('Round', fontsize=11)
        axes[idx].set_ylabel(title, fontsize=11)
        axes[idx].set_title(f'{title} Evolution', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Client metrics plot saved to: {save_path}")
    plt.close()


def plot_system_metrics(train_metrics: Dict, save_path: str = "results/flwr_system_metrics.png"):
    """Plot system-related metrics like training time."""
    
    if not train_metrics:
        print("No training metrics to plot.")
        return
    
    rounds = sorted(train_metrics.keys())
    training_time = [train_metrics[r].get('training_time', 0) for r in rounds]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.bar(rounds, training_time, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Average Training Time per Round', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (r, t) in enumerate(zip(rounds, training_time)):
        ax.text(r, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ System metrics plot saved to: {save_path}")
    plt.close()


def plot_convergence_analysis(train_metrics: Dict, save_path: str = "results/flwr_convergence.png"):
    """Plot convergence analysis showing multiple metrics together."""
    
    if not train_metrics:
        print("No training metrics to plot.")
        return
    
    rounds = sorted(train_metrics.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Loss and Model Divergence
    train_loss = [train_metrics[r].get('train_loss', 0) for r in rounds]
    model_div = [train_metrics[r].get('model_divergence', 0) for r in rounds]
    
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(rounds, train_loss, marker='o', linewidth=2.5, markersize=7, 
                     color='#e74c3c', label='Train Loss')
    line2 = ax1_twin.plot(rounds, model_div, marker='s', linewidth=2.5, markersize=7,
                          color='#3498db', label='Model Divergence')
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12, color='#e74c3c')
    ax1_twin.set_ylabel('Model Divergence', fontsize=12, color='#3498db')
    ax1.set_title('Loss vs Model Divergence', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1_twin.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Right plot: Gradient Norms and Update Magnitude
    post_grad = [train_metrics[r].get('post_grad_norm', 0) for r in rounds]
    update_mag = [train_metrics[r].get('update_magnitude', 0) for r in rounds]
    
    ax2.plot(rounds, post_grad, marker='o', linewidth=2.5, markersize=7,
            color='#2ecc71', label='Post-Training Grad Norm')
    ax2.plot(rounds, update_mag, marker='^', linewidth=2.5, markersize=7,
            color='#f39c12', label='Update Magnitude')
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Gradient and Update Magnitudes', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence analysis plot saved to: {save_path}")
    plt.close()


def generate_summary_report(train_metrics: Dict, save_path: str = "results/flwr_summary.txt"):
    """Generate a text summary report."""
    
    if not train_metrics:
        print("No metrics to summarize.")
        return
    
    rounds = sorted(train_metrics.keys())
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FLOWER FEDERATED LEARNING SIMULATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Rounds: {len(rounds)}\n\n")
        
        # Training Loss Statistics
        train_losses = [train_metrics[r].get('train_loss', 0) for r in rounds]
        f.write("Training Loss:\n")
        f.write(f"  - Initial (Round 1): {train_losses[0]:.4f}\n")
        f.write(f"  - Final (Round {len(rounds)}): {train_losses[-1]:.4f}\n")
        f.write(f"  - Mean: {np.mean(train_losses):.4f}\n")
        f.write(f"  - Best (Min): {np.min(train_losses):.4f}\n")
        f.write(f"  - Std Dev: {np.std(train_losses):.4f}\n\n")
        
        # Model Divergence Statistics
        model_divs = [train_metrics[r].get('model_divergence', 0) for r in rounds]
        f.write("Model Divergence:\n")
        f.write(f"  - Initial: {model_divs[0]:.4f}\n")
        f.write(f"  - Final: {model_divs[-1]:.4f}\n")
        f.write(f"  - Mean: {np.mean(model_divs):.4f}\n")
        f.write(f"  - Std Dev: {np.std(model_divs):.4f}\n\n")
        
        # Training Time Statistics
        train_times = [train_metrics[r].get('training_time', 0) for r in rounds]
        f.write("Training Time (per round):\n")
        f.write(f"  - Mean: {np.mean(train_times):.2f}s\n")
        f.write(f"  - Min: {np.min(train_times):.2f}s\n")
        f.write(f"  - Max: {np.max(train_times):.2f}s\n")
        f.write(f"  - Total: {np.sum(train_times):.2f}s\n\n")
        
        # Same Sign Percentage
        same_sign = [train_metrics[r].get('same_sign_percentage', 0) for r in rounds]
        f.write("Same Sign Percentage (Model Alignment):\n")
        f.write(f"  - Initial: {same_sign[0]:.4f}\n")
        f.write(f"  - Final: {same_sign[-1]:.4f}\n")
        f.write(f"  - Mean: {np.mean(same_sign):.4f}\n\n")
        
        # Round-by-Round Details
        f.write("="*70 + "\n")
        f.write("ROUND-BY-ROUND DETAILS\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Round':<8} {'Loss':<12} {'Model Div':<12} {'Grad Norm':<12} {'Time (s)':<10}\n")
        f.write("-"*70 + "\n")
        
        for r in rounds:
            loss = train_metrics[r].get('train_loss', 0)
            div = train_metrics[r].get('model_divergence', 0)
            grad = train_metrics[r].get('post_grad_norm', 0)
            time = train_metrics[r].get('training_time', 0)
            f.write(f"{r:<8} {loss:<12.4f} {div:<12.4f} {grad:<12.4f} {time:<10.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"✓ Summary report saved to: {save_path}")


def main():
    """Main function to generate all visualizations."""
    
    log_file = "flwr_run_output.log"
    
    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found.")
        print("Please run 'flwr run . 2>&1 | tee flwr_run_output.log' first.")
        return
    
    print("\nParsing Flower simulation log...")
    data = parse_flwr_log(log_file)
    
    if not data["train_metrics"]:
        print("No training metrics found in log file.")
        return
    
    print("Aggregating metrics across clients...")
    train_metrics = aggregate_metrics(data["train_metrics"])
    
    print(f"\nFound metrics for {len(train_metrics)} rounds")
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_training_progress(train_metrics)
    plot_client_metrics(train_metrics)
    plot_system_metrics(train_metrics)
    plot_convergence_analysis(train_metrics)
    generate_summary_report(train_metrics)
    
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - results/flwr_training_progress.png")
    print("  - results/flwr_client_metrics.png")
    print("  - results/flwr_system_metrics.png")
    print("  - results/flwr_convergence.png")
    print("  - results/flwr_summary.txt")


if __name__ == "__main__":
    main()
