"""Evaluation metrics over a dataloader of (src, tar, length) pairs."""
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from .losses import LOSSES, mae_loss, mse_loss, nmae_loss


@torch.no_grad()
def compute_metrics(embedding_module, mlp, dataloader, graph_data, device, 
                    layer_type, siamese, p, loss_func, tqdm_prefix="train", 
                    log_batch_metrics=False, batch_metric_prefix=""):
    """
    Compute metrics on a dataset (train or test).
    
    Args:
        embedding_module: The embedding/GNN module
        mlp: The MLP module (can be None for siamese)
        dataloader: DataLoader for the dataset
        graph_data: Graph data containing node features and edges
        device: Device to run on
        layer_type: Type of layer ('MLP', 'TAGConv', etc.)
        siamese: Whether using siamese mode
        p: p-norm for siamese distance
        loss_func: Loss function name
        tqdm_prefix: Prefix for progress bar
        log_batch_metrics: If True, log metrics for each batch
        batch_metric_prefix: Prefix for batch metrics (e.g., "test_batch_")
    
    Returns a dict with: loss, nmae, mse, mae
    """
    embedding_module.eval()
    if mlp is not None:
        mlp.eval()
    
    # For GNN models, compute all node embeddings once instead of per-batch
    precomputed_embeddings = None
    if 'MLP' not in layer_type:
        precomputed_embeddings = embedding_module(
            graph_data.x, graph_data.edge_index, edge_attr=graph_data.edge_attr
        )
    
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_nmae = 0.0
    total_acc_1pct = 0
    total_acc_2pct = 0
    total_acc_5pct = 0
    total_samples = 0
    
    pred_results = []
    for batch in tqdm(dataloader, desc=f"Compute metrics {tqdm_prefix}"):
        srcs = batch[0].to(device, non_blocking=True)
        tars = batch[1].to(device, non_blocking=True)
        lengths = batch[2].to(device, non_blocking=True)
        batch_size = len(srcs)
        
        if 'MLP' in layer_type:
            src_embeddings = embedding_module(graph_data.x[srcs])
            tar_embeddings = embedding_module(graph_data.x[tars])
        else:
            src_embeddings = precomputed_embeddings[srcs]
            tar_embeddings = precomputed_embeddings[tars]
        
        if siamese:
            pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
        else:
            pred = mlp(src_embeddings, tar_embeddings, vn_emb=None)
            pred = pred.squeeze()
        
        # Compute metrics
        loss = LOSSES[loss_func](pred, lengths)
        batch_loss = loss.item()
        batch_mse = mse_loss(pred, lengths).item()
        batch_mae = mae_loss(pred, lengths).item()
        batch_nmae = nmae_loss(pred, lengths).item()
        
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

        pred_mses = torch.square(pred - lengths)
        pred_maes = torch.abs(pred - lengths)
        pred_nmaes = torch.abs(pred - lengths) / lengths
        total_acc_1pct += (pred_nmaes < 0.01).sum().item()
        total_acc_2pct += (pred_nmaes < 0.02).sum().item()
        total_acc_5pct += (pred_nmaes < 0.05).sum().item()
        # Generate predictions 
        result = (srcs.cpu().numpy(), 
                  tars.cpu().numpy(), 
                  pred.cpu().numpy(), 
                  lengths.cpu().numpy(), 
                  pred_mses.cpu().numpy(), 
                  pred_maes.cpu().numpy(), 
                  pred_nmaes.cpu().numpy()
        )
        pred_results.append(result)
    
    all_preds = np.array([np.concatenate(batch_results) for batch_results in zip(*pred_results)]).squeeze().T # (B*BS, 8)
    all_preds_df = pd.DataFrame(all_preds, columns=['srcs', 'tars', 'preds', 'lengths', 'pred_mses', 'pred_maes', 'pred_nmaes'])

    
    embedding_module.train()
    if mlp is not None:
        mlp.train()
    
    return {
        'loss': total_loss / total_samples,
        'mse': total_mse / total_samples,
        'mae': total_mae / total_samples,
        'nmae': total_nmae / total_samples,
        'acc_1pct': total_acc_1pct / total_samples,
        'acc_2pct': total_acc_2pct / total_samples,
        'acc_5pct': total_acc_5pct / total_samples,
        'all_preds_df': all_preds_df,
    }
