"""Phase 2 (decoupled MLP finetune) functions from refactor_training.py.

Moved on the asilomar branch. They used names that refactor_training.py got
from wildcard imports; the imports below restore them.
"""
import logging
import os

import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm, trange

from spnn.losses import LOSSES, mae_loss, mse_loss, nmae_loss
from spnn.models.factory import configure_embedding_module, configure_mlp_module
from spnn.training import output_dir


@torch.no_grad()
def compute_metrics_decoupled(mlp, embeddings, dataloader, device, loss_func, 
                              tqdm_prefix="train", log_batch_metrics=False, 
                              batch_metric_prefix=""):
    """
    Compute metrics on a dataset for decoupled training (pre-computed embeddings).
    
    Args:
        mlp: The MLP module
        embeddings: Pre-computed node embeddings
        dataloader: DataLoader for the dataset
        device: Device to run on
        loss_func: Loss function name
        tqdm_prefix: Prefix for progress bar
        log_batch_metrics: If True, log metrics for each batch
        batch_metric_prefix: Prefix for batch metrics (e.g., "test_batch_")
    
    Returns a dict with: loss, nmae, mse, mae
    """
    mlp.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_nmae = 0.0
    total_acc_1pct = 0
    total_acc_2pct = 0
    total_acc_5pct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc=f"Compute metrics {tqdm_prefix}"):
        srcs = batch[0]
        tars = batch[1]
        embd_srcs = embeddings[srcs].to(device, non_blocking=True)
        embd_tars = embeddings[tars].to(device, non_blocking=True)
        lengths = batch[2].to(device, non_blocking=True)
        batch_size = len(srcs)
        
        pred = mlp(embd_srcs, embd_tars, vn_emb=None)
        pred = pred.squeeze()
        
        # Compute metrics
        loss = LOSSES[loss_func](pred, lengths)
        batch_loss = loss.item()
        batch_mse = mse_loss(pred, lengths).item()
        batch_mae = mae_loss(pred, lengths).item()
        batch_nmae = nmae_loss(pred, lengths).item()
        
        pred_nmaes = torch.abs(pred - lengths) / lengths
        total_acc_1pct += (pred_nmaes < 0.01).sum().item()
        total_acc_2pct += (pred_nmaes < 0.02).sum().item()
        total_acc_5pct += (pred_nmaes < 0.05).sum().item()
        
        total_loss += batch_loss * batch_size
        total_mse += batch_mse * batch_size
        total_mae += batch_mae * batch_size
        total_nmae += batch_nmae * batch_size
        total_samples += batch_size
        
        # Log per-batch metrics if requested
        if log_batch_metrics:
            wandb.log({
                f'{batch_metric_prefix}loss': batch_loss,
                f'{batch_metric_prefix}mse': batch_mse,
                f'{batch_metric_prefix}mae': batch_mae,
                f'{batch_metric_prefix}nmae': batch_nmae,
            })
    
    mlp.train()
    
    return {
        'loss': total_loss / total_samples,
        'mse': total_mse / total_samples,
        'mae': total_mae / total_samples,
        'nmae': total_nmae / total_samples,
        'acc_1pct': total_acc_1pct / total_samples,
        'acc_2pct': total_acc_2pct / total_samples,
        'acc_5pct': total_acc_5pct / total_samples,
    }

def train_terrains_decoupled(train_dictionary,
                            model_config, 
                            layer_type, 
                            device,
                            prev_model_pth,
                            finetune_dataset_name,
                            activation='silu',
                            epochs=100, 
                            loss_func='mse_loss',
                            lr =0.001,
                            p=1, 
                            aggr='sum',
                            edge_attr=None, 
                            layer_norm=True,
                            new=True,
                            run_name=None,
                            wandb_tag=None,
                            wandb_config=None,
                            test_dictionary=None,
                            val_dictionary=None,
                            **kwargs):
    
    edge_dim=1
    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(embedding_config, 
                                                 layer_type, 
                                                 edge_dim=edge_dim,
                                                 new=new)
    print(embedding_module)
    prev_model_state_pth = os.path.join(prev_model_pth, 'final_model.pt')
    print("Loading from:", prev_model_state_pth)
    embedding_model_state = torch.load(prev_model_state_pth, map_location='cpu')
    print(embedding_model_state.keys())
    embedding_module.load_state_dict(embedding_model_state)
    embedding_module.to(torch.float)
    
    print(model_config)
    print(embedding_module)

    
    mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
    mlp = mlp.to(torch.float)
    mlp.to(device)

    for param in embedding_module.parameters():
        param.requires_grad =False
    
    ## Pre-process all datasets
    num_graphs = len(train_dictionary['graphs'])
    train_dictionary['embeddings'] = []

    for i in range(num_graphs):
        graph_data = train_dictionary['graphs'][i]
        if 'MLP' in layer_type:
            node_embeddings = embedding_module(graph_data.x)
        else:
            node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)
        
        train_dictionary['embeddings'].append(node_embeddings)
    
    # Pre-compute val/test embeddings (they share the same graph, created from test data split)
    for data_dict in [val_dictionary, test_dictionary]:
        if data_dict is not None:
            data_dict['embeddings'] = []
            for graph_data in data_dict['graphs']:
                if 'MLP' in layer_type:
                    node_embeddings = embedding_module(graph_data.x)
                else:
                    node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, edge_attr=graph_data.edge_attr)
                data_dict['embeddings'].append(node_embeddings)
        
    # Build wandb config with all relevant parameters
    base_config = {
        "learning_rate": lr,
        "epochs": epochs,
        "p": p,
        "previous_model_path": prev_model_pth,
        "layer_type": layer_type,
        "device": device,
        "finetune_dataset_name": finetune_dataset_name,
        "loss_func": loss_func,
        "aggr": aggr,
        "new": new,
    }
    # Merge with any additional config passed from training script
    if wandb_config:
        base_config.update(wandb_config)
    
    # Add Sparse GT config if using SparseGT layer type
    if layer_type == 'SparseGT' and hasattr(embedding_module, 'get_config_for_wandb'):
        base_config.update(embedding_module.get_config_for_wandb())
    
    # Initialize wandb first to get run ID
    run = wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'terrains'),
        dir=str(output_dir() / 'wandb'),
        name=run_name,
        tags=wandb_tag if wandb_tag else None,
        config=base_config
    )
    
    # Create log directory based on run_name and wandb run ID
    # Format: {run_name}_{wandb_id}/{finetune_dataset_name}
    train_filename = run_name if run_name else 'train'
    run_dir_name = f"{train_filename}_{run.id}"
    log_dir = os.path.join(output_dir(), 'runs', run_dir_name, finetune_dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Update wandb config with the log_dir
    wandb.config.update({"log_dir": log_dir})
    
    record_dir = os.path.join(log_dir, 'record')
    os.makedirs(record_dir, exist_ok=True)

    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file, force=True)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'loss function: {loss_func}')

    optimizer = AdamW(mlp.parameters(), lr=lr)

    for epoch in trange(epochs):
        total_loss = 0
        total_samples = 0
        for i in range(num_graphs):
            embeddings = train_dictionary['embeddings'][i]
            dataloader = train_dictionary['dataloaders'][i]
            total_samples = len(dataloader.dataset)
            for batch in tqdm(dataloader, desc=f"Main training loop for graph {i}"):
                srcs = batch[0]
                tars = batch[1]
                embd_srcs = embeddings[srcs].to(device, non_blocking=True)
                embd_tars = embeddings[tars].to(device, non_blocking=True)
                lengths = batch[2].to(device, non_blocking=True)

                pred = mlp(embd_srcs, embd_tars, vn_emb=None)
                pred=pred.squeeze()

                loss = LOSSES[loss_func](pred, lengths)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach()
                normalized_abs_err = nmae_loss(pred, lengths)

                wandb.log({'epoch_train_loss': loss, 
                        'epoch_total_loss': total_loss/total_samples,
                        'epoch_normalized_abs_err': normalized_abs_err})
        
        # Compute validation metrics after each epoch
        if val_dictionary is not None:
            val_metrics = compute_metrics_decoupled(
                mlp, val_dictionary['embeddings'][0], val_dictionary['dataloaders'][0],
                device, loss_func, tqdm_prefix="val"
            )
            wandb.log({
                'val_loss': val_metrics['loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_nmae': val_metrics['nmae'],
                'val_acc_1pct': val_metrics['acc_1pct'],
                'val_acc_2pct': val_metrics['acc_2pct'],
                'val_acc_5pct': val_metrics['acc_5pct'],
                'epoch': epoch,
            })

    # Compute final train metrics
    final_train_metrics = compute_metrics_decoupled(
        mlp, train_dictionary['embeddings'][0], train_dictionary['dataloaders'][0],
        device, loss_func, tqdm_prefix="train"
    )
    
    # Compute final test metrics if test data is provided
    final_test_metrics = None
    if test_dictionary is not None:
        final_test_metrics = compute_metrics_decoupled(
            mlp, test_dictionary['embeddings'][0], test_dictionary['dataloaders'][0],
            device, loss_func, tqdm_prefix="test", 
            log_batch_metrics=True, batch_metric_prefix="test_batch_"
        )
    
    # Log final metrics
    final_metrics = {
        'train_loss': final_train_metrics['loss'],
        'train_mse': final_train_metrics['mse'],
        'train_mae': final_train_metrics['mae'],
        'train_nmae': final_train_metrics['nmae'],
        'train_acc_1pct': final_train_metrics['acc_1pct'],
        'train_acc_2pct': final_train_metrics['acc_2pct'],
        'train_acc_5pct': final_train_metrics['acc_5pct'],
    }
    if final_test_metrics is not None:
        final_metrics.update({
            'test_loss': final_test_metrics['loss'],
            'test_mse': final_test_metrics['mse'],
            'test_mae': final_test_metrics['mae'],
            'test_nmae': final_test_metrics['nmae'],
            'test_acc_1pct': final_test_metrics['acc_1pct'],
            'test_acc_2pct': final_test_metrics['acc_2pct'],
            'test_acc_5pct': final_test_metrics['acc_5pct'],
        })
        logging.info(f'final test loss: {final_test_metrics["loss"]}')
        print("Final test loss:", final_test_metrics['loss'])
    
    wandb.log(final_metrics)
    logging.info(f'final training loss: {final_train_metrics["loss"]}')
    print("Final training loss:", final_train_metrics['loss'])
    path = os.path.join(record_dir, 'final_model.pt')
    print("saving model to:", path)
    torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                'mlp_state_dict': mlp.state_dict()}, path)
    wandb.finish()
    return embedding_module, mlp
