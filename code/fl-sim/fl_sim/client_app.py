"""fl-sim: A Flower / PyTorch app with enhanced metrics for client selection."""

import time
import torch
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_sim.task import Net, load_data
from fl_sim.task import test as test_fn
from fl_sim.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def get_gradient_vector(model, trainloader, device):
    """
    Compute L2 norm and the flattened vector of gradients on a sample.
    This corresponds to ∇w_i.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    if len(trainloader.dataset) == 0:
        return torch.tensor([]).to(device), 0.0
        
    sample_size = min(32, len(trainloader.dataset))
    sample_loader = torch.utils.data.DataLoader(
        trainloader.dataset, 
        batch_size=sample_size, 
        shuffle=True
    )
    
    try:
        batch = next(iter(sample_loader))
    except StopIteration:
        return torch.tensor([]).to(device), 0.0

    images = batch["img"].to(device)
    labels = batch["label"].to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_norm = 0.0
    grad_vector = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
            grad_vector.append(param.grad.data.flatten())
            
    grad_norm = grad_norm ** 0.5
    
    if not grad_vector:
        return torch.tensor([]).to(device), 0.0
        
    flat_grad_vector = torch.cat(grad_vector)
    return flat_grad_vector, grad_norm


def compute_model_divergence(local_state, global_state):
    """Compute L2 distance between local and global model."""
    divergence = 0.0
    for key in local_state.keys():
        divergence += torch.sum((local_state[key] - global_state[key]) ** 2).item()
    return divergence ** 0.5


def compute_same_sign_percentage(local_state, global_state):
    """Compute percentage of weights with same sign."""
    total_params = 0
    same_sign_count = 0
    
    for key in local_state.keys():
        local_tensor = local_state[key].flatten()
        global_tensor = global_state[key].flatten()
        
        same_sign = (torch.sign(local_tensor) == torch.sign(global_tensor)).sum().item()
        same_sign_count += same_sign
        total_params += local_tensor.numel()
    
    return same_sign_count / total_params if total_params > 0 else 0.0


def get_global_update_vector(global_state, prev_global_state):
    """
    Computes the global update vector (proxy for ∇w_bar) and its norm.
    This is (w_global_t - w_global_t-1).
    """
    global_update_vector = []
    norm = 0.0
    for key in global_state.keys():
        delta = (global_state[key] - prev_global_state[key]).flatten()
        norm += torch.sum(delta ** 2).item()
        global_update_vector.append(delta)
        
    if not global_update_vector:
        return torch.tensor([]), 0.0
        
    return torch.cat(global_update_vector), norm ** 0.5


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with enhanced metrics."""
    
    start_time = time.time()
    
    # Load models
    model = Net()
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Store initial state for post-training comparison
    initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    global_state_cpu = {k: v.cpu() for k, v in global_state.items()}

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)
    
    # === Note: Pre-training metrics are now done in evaluate() ===
    
    # Train
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    
    # === Simulate System Lag ===
    base_lag = 0.1 
    data_lag = len(trainloader.dataset) / 5000.0
    random_lag = np.random.uniform(0.0, 0.5)
    if partition_id % 2 != 0:
        total_lag = base_lag + data_lag + random_lag + 1.0
    else:
        total_lag = base_lag + data_lag + random_lag
    time.sleep(total_lag)
    
    # === Compute Post-Training Metrics ===
    final_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    
    # APP-FIVE
    model_divergence = compute_model_divergence(final_state, global_state_cpu)
    # APP-SIX
    same_sign_pct = compute_same_sign_percentage(final_state, global_state_cpu)
    # APP-EIGHT
    update_magnitude = compute_model_divergence(final_state, initial_state)
    # APP-NINE
    _, post_grad_norm = get_gradient_vector(model, trainloader, device) # Recalculate grad
    
    end_time = time.time()
    
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        # Standard metrics (use hyphens for Flower compatibility)
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),  # Flower expects hyphenated key
        "training_time": total_lag, # Use simulated lag
        
        # Post-training Statistical metrics
        "post_grad_norm": post_grad_norm,     # APP-NINE
        "model_divergence": model_divergence, # APP-FIVE
        "same_sign_percentage": same_sign_pct,# APP-SIX
        "update_magnitude": update_magnitude, # APP-EIGHT
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Evaluate the model on local data AND compute pre-training metrics.
    """
    model = Net()
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Load previous global state (w_global_t-1) ---
    # This is now sent by the server for metric calculation
    prev_global_state = msg.content["previous_arrays"].to_torch_state_dict()
    
    # --- Create CPU copies for metric calculations ---
    global_state_cpu = {k: v.cpu() for k, v in global_state.items()}
    prev_global_state_cpu = {k: v.cpu() for k, v in prev_global_state.items()}

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    # === Standard Evaluation ===
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # === Pre-Training Metrics (for next round's selection) ===
    
    # APP-TWO
    local_grad_vector, pre_grad_norm = get_gradient_vector(model, trainloader, device)
    
    # Proxy for global gradient: ∇w_bar ≈ (w_global_t - w_global_t-1)
    global_update_vector, global_update_norm = get_global_update_vector(
        global_state_cpu, prev_global_state_cpu
    )
    global_update_vector = global_update_vector.to(device)

    # APP-TEN (Eq. 11)
    if local_grad_vector.numel() > 0 and global_update_vector.numel() > 0:
        grad_dot_product = torch.dot(local_grad_vector, global_update_vector).item()
    else:
        grad_dot_product = 0.0

    # === Consolidate all metrics ===
    metrics = {
        # Standard evaluation metrics
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset), # Use val dataset size for eval (hyphenated for Flower)
        
        # Pre-training statistical metrics
        "pre_grad_norm": pre_grad_norm,      # APP-TWO
        "grad_dot_product": grad_dot_product, # APP-TEN (Eq. 11)
        "global_update_norm": global_update_norm, # For APP-SEVEN (Eq. 8)
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
