"""Benchmark script to compare all client selection strategies."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord
from flwr.simulation import start_simulation

from fl_sim.task import Net
from fl_sim.strategies import STRATEGY_REGISTRY


def run_strategy_simulation(
    strategy_name: str,
    strategy_kwargs: dict,
    num_clients: int = 10,
    num_rounds: int = 10,
    fraction_train: float = 0.5,
    local_epochs: int = 1,
    lr: float = 0.01,
) -> dict:
    """
    Run simulation for a single strategy.
    
    Args:
        strategy_name: Name of the strategy to test
        strategy_kwargs: Additional arguments for the strategy
        num_clients: Total number of clients
        num_rounds: Number of FL rounds
        fraction_train: Fraction of clients to select per round
        local_epochs: Number of local training epochs
        lr: Learning rate
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Running Strategy: {strategy_name.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Get strategy class
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # Create strategy instance
    strategy = strategy_class(
        fraction_train=fraction_train,
        **strategy_kwargs
    )
    
    # Load initial model
    global_model = Net()
    initial_arrays = ArrayRecord(global_model.state_dict())
    
    # Training configuration
    train_config = ConfigRecord({
        "lr": lr,
        "local-epochs": local_epochs,
    })
    
    try:
        # Note: This is a simplified version. For full Flower simulation,
        # you would use start_simulation with appropriate client_fn
        # Here we provide the framework structure
        
        # Create a mock grid (in actual implementation, this comes from Flower)
        from fl_sim.server_app import create_grid_mock
        grid = create_grid_mock(num_clients)
        
        # Run the strategy
        final_arrays, metrics_history = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=train_config,
            num_rounds=num_rounds,
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract final metrics and convert History to JSON-serializable format
        final_metrics = {}
        metrics_history_list = []
        
        if metrics_history and hasattr(metrics_history, 'metrics_distributed'):
            # History object stores data as:
            # - losses_distributed: [(round, loss), ...]
            # - metrics_distributed: {'metric_name': [(round, value), ...], ...}
            distributed_metrics = metrics_history.metrics_distributed
            losses_distributed = metrics_history.losses_distributed
            
            # Build a dictionary mapping round number to loss
            loss_by_round = {}
            if isinstance(losses_distributed, list):
                for round_num, loss_value in losses_distributed:
                    loss_by_round[round_num] = loss_value
            
            # Build a dictionary mapping round number to metrics
            # distributed_metrics is a dict like {'accuracy': [(round, value), ...]}
            metrics_by_round = {}
            if isinstance(distributed_metrics, dict):
                for metric_name, metric_values in distributed_metrics.items():
                    if isinstance(metric_values, list):
                        for round_num, metric_value in metric_values:
                            if round_num not in metrics_by_round:
                                metrics_by_round[round_num] = {}
                            metrics_by_round[round_num][metric_name] = metric_value
            
            # Get all unique rounds
            all_rounds = sorted(set(list(loss_by_round.keys()) + list(metrics_by_round.keys())))
            
            for round_num in all_rounds:
                round_metrics = {
                    "round": round_num + 1,  # 1-indexed for display
                    "loss": loss_by_round.get(round_num, 0.0),
                    "accuracy": 0.0,
                    "num_examples": 0
                }
                
                # Extract accuracy from distributed metrics
                if round_num in metrics_by_round:
                    metrics_dict = metrics_by_round[round_num]
                    if isinstance(metrics_dict, dict):
                        round_metrics["accuracy"] = metrics_dict.get("accuracy", 0.0)
                
                metrics_history_list.append(round_metrics)
            
            # Get final metrics (last round)
            if metrics_history_list:
                final_metrics = metrics_history_list[-1]
            
            # Get all unique rounds
            all_rounds = sorted(set(list(loss_by_round.keys()) + list(metrics_by_round.keys())))
            
            for round_num in all_rounds:
                round_metrics = {
                    "round": round_num + 1,  # 1-indexed for display
                    "loss": loss_by_round.get(round_num, 0.0),
                    "accuracy": 0.0,
                    "num_examples": 0
                }
                
                # Extract accuracy from distributed metrics
                if round_num in metrics_by_round:
                    metrics_dict = metrics_by_round[round_num]
                    if isinstance(metrics_dict, dict):
                        round_metrics["accuracy"] = metrics_dict.get("accuracy", 0.0)
                
                metrics_history_list.append(round_metrics)
            
            # Get final metrics (last round)
            if metrics_history_list:
                final_metrics = metrics_history_list[-1]
        
        # Collect results
        results = {
            "strategy_name": strategy_name,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "elapsed_time": elapsed_time,
            "success": True,
            "final_model_saved": True,
            "final_loss": final_metrics.get("loss", 0.0),
            "final_accuracy": final_metrics.get("accuracy", 0.0),
            "metrics_history": metrics_history_list,
        }
        
        # Save final model
        results_dir = Path("results") / strategy_name
        results_dir.mkdir(parents=True, exist_ok=True)
        model_path = results_dir / "final_model.pt"
        torch.save(final_arrays.to_torch_state_dict(), model_path)
        
        print(f"\n✓ Strategy completed in {elapsed_time:.2f}s")
        print(f"✓ Final Loss: {final_metrics.get('loss', 0.0):.4f}")
        print(f"✓ Final Accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
        print(f"✓ Final model saved to: {model_path}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Strategy failed with error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return {
            "strategy_name": strategy_name,
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
        }


def run_all_benchmarks(
    num_clients: int = 10,
    num_rounds: int = 10,
    fraction_train: float = 0.5,
    local_epochs: int = 1,
    lr: float = 0.01,
) -> dict:
    """
    Run benchmarks for all strategies.
    
    Args:
        num_clients: Total number of clients
        num_rounds: Number of FL rounds
        fraction_train: Fraction of clients to select
        local_epochs: Number of local epochs
        lr: Learning rate
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("FEDERATED LEARNING CLIENT SELECTION BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Total Clients: {num_clients}")
    print(f"  - FL Rounds: {num_rounds}")
    print(f"  - Fraction Train: {fraction_train}")
    print(f"  - Local Epochs: {local_epochs}")
    print(f"  - Learning Rate: {lr}")
    
    # Define strategies to test
    strategies_to_test = [
        ("random", {}),
        ("num_samples", {}),
        ("high_grad_norm", {}),
        ("high_loss", {}),
        ("sum_loss", {}),
        ("model_divergence", {}),
        ("same_sign", {}),
        ("update_direction", {}),
        ("local_change", {}),
        ("post_grad_norm", {}),
        ("grad_dot_product", {}),
        ("hard_deadline", {"deadline": 5.0}),
        ("soft_deadline", {"deadline": 3.0, "penalty_factor": 2.0}),
    ]
    
    all_results = []
    
    for strategy_name, strategy_kwargs in strategies_to_test:
        result = run_strategy_simulation(
            strategy_name=strategy_name,
            strategy_kwargs=strategy_kwargs,
            num_clients=num_clients,
            num_rounds=num_rounds,
            fraction_train=fraction_train,
            local_epochs=local_epochs,
            lr=lr,
        )
        all_results.append(result)
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": local_epochs,
            "lr": lr,
        },
        "results": all_results,
    }
    
    # Save summary
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nSummary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "-"*70)
    print(f"{'Strategy':<25} {'Status':<10} {'Time (s)':<12} {'Loss':<10} {'Acc':<10}")
    print("-"*70)
    
    for result in all_results:
        status = "✓ Success" if result.get("success") else "✗ Failed"
        elapsed = result.get("elapsed_time", 0)
        loss = result.get("final_loss", 0.0)
        acc = result.get("final_accuracy", 0.0)
        loss_str = f"{loss:.4f}" if result.get("success") else "N/A"
        acc_str = f"{acc:.4f}" if result.get("success") else "N/A"
        print(f"{result['strategy_name']:<25} {status:<10} {elapsed:<12.2f} {loss_str:<10} {acc_str:<10}")
    
    print("-"*70)
    
    return summary


def compare_strategies(summary_path: str = "results/benchmark_summary.json"):
    """
    Compare and analyze strategy results.
    
    Args:
        summary_path: Path to benchmark summary JSON
    """
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    
    successful_results = [r for r in summary["results"] if r.get("success")]
    
    if not successful_results:
        print("\nNo successful runs to compare.")
        return
    
    # Sort by different metrics
    sorted_by_time = sorted(successful_results, key=lambda x: x["elapsed_time"])
    sorted_by_loss = sorted(successful_results, key=lambda x: x.get("final_loss", float('inf')))
    sorted_by_acc = sorted(successful_results, key=lambda x: x.get("final_accuracy", 0), reverse=True)
    
    print("\n📊 Rankings by Training Time:")
    print("-"*70)
    for i, result in enumerate(sorted_by_time, 1):
        print(f"{i}. {result['strategy_name']:<25} {result['elapsed_time']:.2f}s")
    
    print("\n🏆 Rankings by Final Loss (Lower is Better):")
    print("-"*70)
    for i, result in enumerate(sorted_by_loss, 1):
        loss = result.get('final_loss', 0)
        print(f"{i}. {result['strategy_name']:<25} Loss: {loss:.4f}")
    
    print("\n🎯 Rankings by Final Accuracy (Higher is Better):")
    print("-"*70)
    for i, result in enumerate(sorted_by_acc, 1):
        acc = result.get('final_accuracy', 0)
        print(f"{i}. {result['strategy_name']:<25} Accuracy: {acc:.4f}")
    
    # Calculate statistics
    times = [r["elapsed_time"] for r in successful_results]
    losses = [r.get("final_loss", 0) for r in successful_results]
    accs = [r.get("final_accuracy", 0) for r in successful_results]
    
    print(f"\n📈 Statistics:")
    print(f"  Training Time:")
    print(f"    - Mean: {np.mean(times):.2f}s")
    print(f"    - Std Dev: {np.std(times):.2f}s")
    print(f"    - Min: {np.min(times):.2f}s")
    print(f"    - Max: {np.max(times):.2f}s")
    
    print(f"\n  Final Loss:")
    print(f"    - Mean: {np.mean(losses):.4f}")
    print(f"    - Std Dev: {np.std(losses):.4f}")
    print(f"    - Best: {np.min(losses):.4f}")
    print(f"    - Worst: {np.max(losses):.4f}")
    
    print(f"\n  Final Accuracy:")
    print(f"    - Mean: {np.mean(accs):.4f}")
    print(f"    - Std Dev: {np.std(accs):.4f}")
    print(f"    - Best: {np.max(accs):.4f}")
    print(f"    - Worst: {np.min(accs):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Federated Learning Client Selection Strategies"
    )
    parser.add_argument(
        "--num-clients", 
        type=int, 
        default=10,
        help="Number of clients in the federation"
    )
    parser.add_argument(
        "--num-rounds", 
        type=int, 
        default=10,
        help="Number of FL rounds"
    )
    parser.add_argument(
        "--fraction-train", 
        type=float, 
        default=0.5,
        help="Fraction of clients to select per round"
    )
    parser.add_argument(
        "--local-epochs", 
        type=int, 
        default=1,
        help="Number of local training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing results"
    )
    
    args = parser.parse_args()
    
    if args.compare_only:
        compare_strategies()
    else:
        summary = run_all_benchmarks(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            fraction_train=args.fraction_train,
            local_epochs=args.local_epochs,
            lr=args.lr,
        )
        compare_strategies()