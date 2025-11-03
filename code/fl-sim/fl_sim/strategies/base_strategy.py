"""Base strategy for custom client selection in Federated Learning."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, RecordDict 
from flwr.serverapp import Grid
from flwr.common import Message
from flwr.server.history import History


class BaseSelectionStrategy(ABC):
    """Abstract base class for client selection strategies."""
    
    def __init__(
        self,
        fraction_train: float = 1.0,
        min_available_clients: int = 2,
    ):
        """
        Initialize the base selection strategy.
        
        Args:
            fraction_train: Fraction of clients to select per round
            min_available_clients: Minimum number of clients needed
        """
        self.fraction_train = fraction_train
        self.min_available_clients = min_available_clients
        self.current_round = 0
        self.history = History()
        self._final_parameters: Optional[ArrayRecord] = None
        self._previous_parameters: Optional[ArrayRecord] = None # To store w_global_t-1
        
    @abstractmethod
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """
        Compute utility score for a client.
        
        Args:
            client_id: Client identifier
            metrics: Dictionary of client metrics
            round_num: Current round number
            
        Returns:
            Utility score for the client
        """
        pass

    def get_final_parameters(self) -> Optional[ArrayRecord]:
        """Returns the final aggregated parameters."""
        return self._final_parameters
    
    def select_clients(
        self,
        client_metrics: Dict[str, Dict[str, float]],
        num_to_select: int,
    ) -> List[str]:
        """
        Select clients based on utility scores.
        
        Args:
            client_metrics: Dictionary mapping client IDs to their metrics
            num_to_select: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        utilities = {}
        for client_id, metrics in client_metrics.items():
            utilities[client_id] = self.compute_utility(
                client_id, metrics, self.current_round
            )
        
        sorted_clients = sorted(
            utilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected = [client_id for client_id, _ in sorted_clients[:num_to_select]]
        
        return selected
    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        train_config: ConfigRecord,
        num_rounds: int,
    ) -> Tuple[Optional[ArrayRecord], History]: # --- MODIFIED: Return type
        """
        Start the federated learning process.
        """
        current_arrays = initial_arrays
        self._final_parameters = initial_arrays
        self._previous_parameters = initial_arrays # Initialize for round 0
        
        config_dict = train_config
        
        for round_num in range(num_rounds):
            self.current_round = round_num
            print(f"\n[Round {round_num + 1}/{num_rounds}]")
            
            # Get available nodes - handle both MockGrid and real InMemoryGrid
            if hasattr(grid, 'nodes') and callable(grid.nodes):
                # MockGrid has nodes() method
                available_nodes = grid.nodes()
            else:
                # Real InMemoryGrid: generate node IDs based on num_supernodes
                # In Flower simulation, node IDs are typically 0, 1, 2, ...
                # We'll use the grid's internal structure or make assumptions
                # For now, assume node IDs from context or use a reasonable default
                try:
                    # Try to get from context if available
                    num_nodes = grid._num_nodes if hasattr(grid, '_num_nodes') else 10
                except:
                    # Fallback: use a reasonable default
                    num_nodes = 10
                available_nodes = list(range(num_nodes))
            
            num_available = len(available_nodes)
            
            if num_available < self.min_available_clients:
                print(f"Not enough clients available ({num_available} < {self.min_available_clients})")
                continue
            
            num_to_select = max(
                self.min_available_clients,
                int(num_available * self.fraction_train)
            )
            
            client_metrics = {}
            
            if round_num == 0:
                selected_nodes = np.random.choice(
                    available_nodes, 
                    size=num_to_select, 
                    replace=False
                ).tolist()
                print(f"[Round 0] Random selection: {len(selected_nodes)} clients")
            else:
                eval_messages = []
                for node_id in available_nodes:
                    eval_content = RecordDict({"arrays": current_arrays})
                    numeric_id = int(node_id.split("_")[-1]) if "_" in str(node_id) else int(node_id)
                    eval_msg = Message(eval_content, numeric_id, "evaluate")
                    eval_messages.append((node_id, eval_msg))
                
                eval_results = grid.send(eval_messages, "evaluate")
                
                for node_id, result in eval_results.items():
                    try:
                        metrics_dict = dict(result.content["metrics"])
                    except:
                        metrics_dict = {}
                    client_metrics[str(node_id)] = metrics_dict
                
                selected_nodes = self.select_clients(client_metrics, num_to_select)
                print(f"Selected {len(selected_nodes)} clients using {self.__class__.__name__}")

            
            train_messages = []
            for node_id in selected_nodes:
                # --- FIX: Add previous global model to config ---
                train_content_dict = {
                    "arrays": current_arrays,
                    "config": config_dict,
                    "previous_arrays": self._previous_parameters # Send w_global_t-1
                }
                train_content = RecordDict(train_content_dict)
                # ------------------------------------------------
                
                numeric_id = int(node_id.split("_")[-1]) if "_" in str(node_id) else int(node_id)
                train_msg = Message(train_content, numeric_id, "train")
                train_messages.append((node_id, train_msg))
            
            train_results = grid.send(train_messages, "train")
            
            # --- Update previous_arrays for the *next* round ---
            self._previous_parameters = current_arrays
            
            # Aggregate results
            current_arrays = self.aggregate(train_results)
            self._final_parameters = current_arrays 
            
            # Evaluate on all clients (for logging)
            eval_messages = []
            for node_id in available_nodes:
                eval_content = RecordDict({"arrays": current_arrays})
                numeric_id = int(node_id.split("_")[-1]) if "_" in str(node_id) else int(node_id)
                eval_msg = Message(eval_content, numeric_id, "evaluate")
                eval_messages.append((node_id, eval_msg))
            
            eval_results = grid.send(eval_messages, "evaluate")
            
            # Compute and log average metrics
            total_loss = 0.0
            total_acc = 0.0
            total_examples = 0
            
            for result in eval_results.values():
                try:
                    metrics = dict(result.content["metrics"])
                except:
                    metrics = {'eval_loss': 0, 'eval_acc': 0, 'num_examples': 1}

                num_examples = metrics.get("num_examples", 1)
                total_loss += metrics.get("eval_loss", 0) * num_examples
                total_acc += metrics.get("eval_acc", 0) * num_examples
                total_examples += num_examples
            
            avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
            avg_acc = total_acc / total_examples if total_examples > 0 else 0.0
            
            print(f"Round {round_num + 1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
            
            self.history.add_loss_distributed(round_num, avg_loss)
            # --- FIX: Corrected the signature for add_metrics_distributed ---
            self.history.add_metrics_distributed(
                server_round=round_num, 
                metrics={"accuracy": avg_acc}
            )
        
        # --- MODIFIED: Return both parameters and history ---
        return self._final_parameters, self.history
    
    def aggregate(self, results: Dict[str, Message]) -> ArrayRecord:
        """
        Aggregate model updates using FedAvg.
        """
        weights_list = []
        num_examples_list = []
        
        for result in results.values():
            arrays = result.content["arrays"]
            metric_obj = result.content["metrics"]
            
            try:
                metrics = dict(metric_obj)
            except:
                metrics = {'num_examples': 1}
            
            weights_list.append(arrays.to_torch_state_dict())
            num_examples_list.append(metrics.get("num_examples", 1))
        
        total_examples = sum(num_examples_list)
        
        if not weights_list or total_examples == 0:
            if hasattr(self, '_final_parameters') and self._final_parameters is not None:
                return self._final_parameters
            else:
                raise ValueError("Aggregation called with no results and no prior weights.")

        aggregated = {}
        
        for key in weights_list[0].keys():
            aggregated[key] = sum(
                w[key] * (n / total_examples)
                for w, n in zip(weights_list, num_examples_list)
            )
        
        aggregated_arrays = ArrayRecord(aggregated)
        return aggregated_arrays

