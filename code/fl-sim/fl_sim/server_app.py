"""fl-sim: A Flower / PyTorch app with custom selection strategies."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_sim.task import Net
from fl_sim.strategies import get_strategy

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp with custom strategy support."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    
    # Check if custom strategy is specified
    strategy_name: str = context.run_config.get("strategy", "fedavg")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Custom strategies only work with MockGrid (benchmark.py)
    # For real Flower simulation, we use FedAvg
    if strategy_name != "fedavg":
        print(f"\n Custom strategy '{strategy_name}' is not yet supported with Flower simulation.")
        print("Using FedAvg instead. Custom strategies work with benchmark.py only.\n")
        strategy_name = "fedavg"
    
    # Use FedAvg strategy
    print(f"\nUsing FedAvg strategy")
    strategy = FedAvg(fraction_train=fraction_train)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.to_torch_state_dict() if hasattr(result, 'to_torch_state_dict') else result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("✓ Final model saved to: final_model.pt")


# Mock Grid for testing (used by benchmark.py)
class MockNode:
    """Mock client node for testing."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.partition_id = int(node_id.split("_")[1])


class MockGrid:
    """Mock Grid for testing strategies without full Flower infrastructure."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self._nodes = [f"client_{i}" for i in range(num_clients)]
    
    def nodes(self):
        """Return list of available node IDs."""
        return self._nodes
    
    def send(self, messages, method: str):
        """
        Mock send method that simulates client responses.
        
        Args:
            messages: List of (node_id, message) tuples
            method: "train" or "evaluate"
            
        Returns:
            Dictionary mapping node_id to mock response
        """
        import random
        from flwr.app import Message, MetricRecord, RecordDict, ArrayRecord
        
        results = {}
        
        for node_id, msg in messages:
            # Generate mock metrics based on method
            if method == "train":
                metrics_data = {
                    "train_loss": random.uniform(0.5, 2.5),
                    "num_examples": random.randint(100, 500),
                    "training_time": random.uniform(1.0, 6.0),
                    # Statistical metrics
                    "pre_grad_norm": random.uniform(0.1, 2.0),
                    "post_grad_norm": random.uniform(0.1, 2.0),
                    "model_divergence": random.uniform(0.5, 3.0),
                    "same_sign_percentage": random.uniform(0.4, 0.9),
                    "update_magnitude": random.uniform(0.3, 2.5),
                    "grad_dot_product": random.uniform(-1.0, 1.0),
                }
                
                # Return mock model (just perturb the input model slightly)
                state_dict = msg.content["arrays"].to_torch_state_dict()
                perturbed = {
                    k: v + torch.randn_like(v) * 0.01 
                    for k, v in state_dict.items()
                }
                
                # Create MetricRecord that can be accessed as dict
                metric_record = MetricRecord(metrics_data)
                # Store data attribute for easy access
                metric_record.data = metrics_data
                
                content = RecordDict({
                    "arrays": ArrayRecord(perturbed),
                    "metrics": metric_record
                })
                
            else:  # evaluate
                metrics_data = {
                    "eval_loss": random.uniform(0.5, 2.0),
                    "eval_acc": random.uniform(0.3, 0.9),
                    "num_examples": random.randint(50, 200),
                }
                
                metric_record = MetricRecord(metrics_data)
                metric_record.data = metrics_data
                
                content = RecordDict({"metrics": metric_record})
            
            # Create a reply message to the incoming message
            results[node_id] = Message(content, reply_to=msg)
        
        return results


def create_grid_mock(num_clients: int) -> MockGrid:
    """Create a mock Grid for testing."""
    return MockGrid(num_clients)