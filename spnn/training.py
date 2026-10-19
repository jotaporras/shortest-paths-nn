"""Single-stage training loop: embed nodes, predict ||h_s - h_t||_p, log to wandb."""
import copy
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

from .data import materialize_single_graph_batch
from .losses import LOSSES, mae_loss, mse_loss, nmae_loss
from .metrics import evaluate
from .models.factory import configure_embedding_module, configure_mlp_module

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def output_dir() -> Path:
    """Root for data, models and wandb files. Read at call time from TERRAIN_OUTPUT_DIR."""
    return Path(os.environ.get('TERRAIN_OUTPUT_DIR', REPO_ROOT))


def format_log_dir(output_dir, 
                   dataset_name, 
                   siamese, 
                   modelname, 
                   vn, 
                   aggr, 
                   loss_func, 
                   layer_type,
                   p,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'single_dataset', 
                           dataset_name)
    if not siamese:
        log_dir = os.path.join(log_dir, aggr)
    log_dir = os.path.join(log_dir, loss_func, modelname, trial)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


LR_SCHEDULES = ('constant', 'cosine')


# Scalar physical parameters that weight decay must not pull toward 0 (e33 review):
# the edge-kernel scale `rho` of GNNModel and the attention-bias slopes of SparseGT.
# With wd 0.1 at lr 3.5e-3 over 5,650 steps, decay alone multiplies them by 0.14.
NO_WEIGHT_DECAY_SUFFIXES = ('edge_kernel.rho', 'bias_slope')


def param_groups(named_parameters, weight_decay):
    """Split parameters into a decayed group and a no-decay group for AdamW."""
    decay, no_decay = [], []
    for name, p in named_parameters:
        (no_decay if name.endswith(NO_WEIGHT_DECAY_SUFFIXES) else decay).append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}]
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups


def lr_factor_fn(warmup_steps, total_steps, schedule='constant', min_ratio=1e-4):
    """Multiplier of the peak learning rate at optimizer step s (0-based), for LambdaLR.

    Linear warmup from 0 over warmup_steps, then constant or a cosine decay that
    reaches min_ratio x peak at the last step. Returns None for constant without
    warmup, so the caller creates no scheduler.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"Unknown lr_schedule '{schedule}'. Use one of {LR_SCHEDULES}.")
    if schedule == 'constant' and warmup_steps <= 0:
        return None
    decay_steps = max(1, total_steps - warmup_steps - 1)

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == 'constant':
            return 1.0
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


# train_dictionary = {'graphs': [g1, g2, g3, ....], 'dataloaders': [dl1, dl2, ....]}
def train_few_cross_terrain_case(train_dictionary,
                                model_config, 
                                layer_type, 
                                device,
                                epochs=100, 
                                loss_func='mse_loss',
                                lr =0.001,
                                base_log_dir=None,
                                siamese=True,
                                p=1, 
                                aggr='sum',
                                new=False,
                                finetune_from=None,
                                run_name=None,
                                wandb_tag=None,
                                wandb_config=None,
                                single_graph_full_batch=False,
                                test_dictionary=None,
                                val_dictionary=None,
                                early_stopping_patience=None,
                                seed=0,
                                weight_decay=0.01,
                                warmup_steps=0,
                                lr_schedule='constant',
                                on_epoch_end=None,
                                finish_wandb=True):
    """Train one model and log to wandb.

    weight_decay: AdamW weight decay (0.01 is the AdamW default used before).
    warmup_steps, lr_schedule: see lr_factor_fn; the defaults keep a constant rate.
    on_epoch_end(epoch, val_metrics) -> bool: called after each validation; True
        stops training (used by spnn.cli.tune for pruning).
    finish_wandb: False leaves the wandb run open for the caller to log more and finish.
    """
    if base_log_dir is None:
        base_log_dir = str(output_dir())
    # Seeds model init, dataloader shuffling and RPEARL random features.
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_graphs = len(train_dictionary['graphs'])
    edge_dim = 1
    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(embedding_config, 
                                                 layer_type, 
                                                 edge_dim=edge_dim,
                                                 new=new)
    print(embedding_module)
    if finetune_from:
        embedding_model_state = torch.load(finetune_from, map_location='cpu')
        embedding_module.load_state_dict(embedding_model_state)
    embedding_module.to(torch.float)
    embedding_module.to(device)
    mlp=None

    if siamese:
        # The pair head is the fixed p-norm, so model_config['mlp'] and aggr are not used.
        named_parameters = list(embedding_module.named_parameters())
    else:
        mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
        mlp = mlp.to(torch.float)
        mlp.to(device)
        named_parameters = list(embedding_module.named_parameters()) + list(mlp.named_parameters())
        print(mlp)

    optimizer = AdamW(param_groups(named_parameters, weight_decay), lr=lr, weight_decay=weight_decay)

    # Build wandb config with all relevant parameters
    base_config = {
        "seed": seed,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "lr_schedule": lr_schedule,
        "epochs": epochs,
        "siamese": siamese,
        "p": p,
        "layer_type": layer_type,
        "device": device,
        "loss_func": loss_func,
        "aggr": aggr,
        "new": new,
        "finetune_from": finetune_from,
        "single_graph_full_batch": single_graph_full_batch,
    }
    # Merge with any additional config passed from training script
    if wandb_config:
        base_config.update(wandb_config)
        # Extract just filenames for train/test resolution for easier reading
        base_config['train_resolution'] = os.path.basename(wandb_config['train_data'])
        base_config['test_resolution'] = os.path.basename(wandb_config['test_data'])
    
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
    # Format: {run_name}_{wandb_id} e.g., res04_phase1_abc123xy
    train_filename = run_name if run_name else 'train'
    run_dir_name = f"{train_filename}_{run.id}"
    log_dir = os.path.join(base_log_dir, 'runs', run_dir_name)
    if finetune_from:
        log_dir = os.path.join(log_dir, 'finetune')
    os.makedirs(log_dir, exist_ok=True)
    
    # Update wandb config with the log_dir
    wandb.config.update({"log_dir": log_dir})
    
    record_dir = os.path.join(log_dir, 'record')
    os.makedirs(record_dir, exist_ok=True)

    log_file = os.path.join(record_dir, 'training_log.log')
    # Write this run's log to its record dir without reconfiguring the root logger.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info(f'GNN layer: {layer_type}')
    logger.info(f'Number of epochs: {epochs}')
    logger.info(f'MLP aggregation: {aggr}')
    logger.info(f'Siamese? {siamese}')
    logger.info(f'loss function: {loss_func}')
    print("logging to....", record_dir)

    use_single_graph_full_batch = (
        single_graph_full_batch
        and siamese
        and num_graphs == 1
        and 'MLP' not in layer_type
    )
    if single_graph_full_batch and not use_single_graph_full_batch:
        msg = (
            "single_graph_full_batch requested, but it only applies to "
            "single-graph, non-MLP siamese training. Falling back to batch mode."
        )
        print(msg)
        logger.info(msg)

    if use_single_graph_full_batch:
        total_steps = epochs
    else:
        total_steps = epochs * sum(len(dl) for dl in train_dictionary['dataloaders'])
    factor = lr_factor_fn(warmup_steps, total_steps, lr_schedule)
    scheduler = LambdaLR(optimizer, factor) if factor is not None else None

    full_batch_graph = None
    full_batch_srcs = None
    full_batch_tars = None
    full_batch_lengths = None
    if use_single_graph_full_batch:
        full_batch_graph = train_dictionary['graphs'][0].to(device)
        full_batch_srcs, full_batch_tars, full_batch_lengths = materialize_single_graph_batch(
            train_dictionary['dataloaders'][0].dataset,
            device,
        )
        msg = (
            f"Using full-batch single-graph siamese training with "
            f"{len(full_batch_srcs)} source-target pairs."
        )
        print(msg)
        logger.info(msg)
    
    best_val_nmae = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    global_step = 0
    for epoch in trange(epochs):
        if use_single_graph_full_batch:
            optimizer.zero_grad()
            node_embeddings = embedding_module(
                full_batch_graph.x,
                full_batch_graph.edge_index,
                edge_attr=full_batch_graph.edge_attr,
            )
            src_embeddings = node_embeddings[full_batch_srcs]
            tar_embeddings = node_embeddings[full_batch_tars]
            pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
            loss = LOSSES[loss_func](pred, full_batch_lengths)
            loss.backward()
            optimizer.step()
            step_log = {}
            if scheduler is not None:
                step_log['lr'] = scheduler.get_last_lr()[0]
                scheduler.step()

            with torch.no_grad():
                train_mse = mse_loss(pred, full_batch_lengths).item()
                train_mae = mae_loss(pred, full_batch_lengths).item()
                train_nmae = nmae_loss(pred, full_batch_lengths).item()

            wandb.log({
                'train_loss': loss.item(),
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_nmae': train_nmae,
                'epoch': epoch,
                'global_step': global_step,
                **step_log,
            }, step=global_step)

            global_step += 1
        else:
            for i in range(num_graphs):
                graph_data = train_dictionary['graphs'][i].to(device)
                dataloader = train_dictionary['dataloaders'][i]
                for batch in tqdm(dataloader, desc=f"Main training loop for graph {i}"):
                    srcs = batch[0].to(device, non_blocking=True)
                    tars = batch[1].to(device, non_blocking=True)
                    lengths = batch[2].to(device, non_blocking=True)
                    
                    if 'MLP' in layer_type:
                        src_embeddings = embedding_module(graph_data.x[srcs])
                        tar_embeddings = embedding_module(graph_data.x[tars])
                    else:
                        node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, edge_attr=graph_data.edge_attr)
                        src_embeddings = node_embeddings[srcs]
                        tar_embeddings = node_embeddings[tars]
                        
                    if siamese:
                        pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
                    else:
                        pred = mlp(src_embeddings, tar_embeddings, vn_emb=None)
                        pred = pred.squeeze()
                    
                    loss = LOSSES[loss_func](pred, lengths)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    step_log = {}
                    if scheduler is not None:
                        step_log['lr'] = scheduler.get_last_lr()[0]
                        scheduler.step()

                    # Log metrics every batch
                    with torch.no_grad():
                        train_mse = mse_loss(pred, lengths).item()
                        train_mae = mae_loss(pred, lengths).item()
                        train_nmae = nmae_loss(pred, lengths).item()

                    wandb.log({
                        'train_loss': loss.item(),
                        'train_mse': train_mse,
                        'train_mae': train_mae,
                        'train_nmae': train_nmae,
                        'epoch': epoch,
                        'global_step': global_step,
                        **step_log,
                    }, step=global_step)
                    
                    global_step += 1
        
        # Compute validation metrics after each epoch
        if val_dictionary is not None:
            val_metrics = evaluate(
                embedding_module, mlp, val_dictionary, device, layer_type, siamese, p,
                loss_func, tqdm_prefix="val"
            )
            cur_val_nmae = val_metrics['nmae']
            if cur_val_nmae < best_val_nmae:
                best_val_nmae = cur_val_nmae
                epochs_without_improvement = 0
                best_model_state = {
                    'embedding': copy.deepcopy(embedding_module.state_dict()),
                    'mlp': copy.deepcopy(mlp.state_dict()) if mlp is not None else None,
                }
            else:
                epochs_without_improvement += 1
            wandb.log({
                'val_loss': val_metrics['loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_nmae': val_metrics['nmae'],
                'val_acc_1pct': val_metrics['acc_1pct'],
                'val_acc_2pct': val_metrics['acc_2pct'],
                'val_acc_5pct': val_metrics['acc_5pct'],
                'best_val_nmae': best_val_nmae,
                'epoch': epoch,
            }, step=global_step)

            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break
            if on_epoch_end is not None and on_epoch_end(epoch, val_metrics):
                print(f"Stopped by on_epoch_end at epoch {epoch}")
                wandb.run.summary['stopped_by_callback_epoch'] = epoch
                break

    # Restore best model for final evaluation
    if best_model_state is not None:
        embedding_module.load_state_dict(best_model_state['embedding'])
        if mlp is not None and best_model_state['mlp'] is not None:
            mlp.load_state_dict(best_model_state['mlp'])
        print(f"Restored best model (val_nmae={best_val_nmae:.6f}) for final evaluation")

    # Compute final train metrics
    final_train_metrics = evaluate(
        embedding_module, mlp, train_dictionary, device, layer_type, siamese, p,
        loss_func, tqdm_prefix="train"
    )
    
    # Compute final test metrics if test data is provided
    final_test_metrics = None
    if test_dictionary is not None:
        final_test_metrics = evaluate(
            embedding_module, mlp, test_dictionary, device, layer_type, siamese, p,
            loss_func, tqdm_prefix="test", log_batch_metrics=True,
            batch_metric_prefix="test_batch_"
        )
    
    # Log final metrics (use final_ prefix to distinguish from batch metrics)
    final_metrics = {
        'final_train_loss': final_train_metrics['loss'],
        'final_train_mse': final_train_metrics['mse'],
        'final_train_mae': final_train_metrics['mae'],
        'final_train_nmae': final_train_metrics['nmae'],
        'final_train_acc_1pct': final_train_metrics['acc_1pct'],
        'final_train_acc_2pct': final_train_metrics['acc_2pct'],
        'final_train_acc_5pct': final_train_metrics['acc_5pct'],
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
        # Save predictions to CSV
        preds_path = os.path.join(log_dir, 'preds.csv')
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        final_test_metrics['all_preds_df'].to_csv(preds_path, index=False)
        logger.info(f'Saved predictions to {preds_path}')
        logger.info(f'final test loss: {final_test_metrics["loss"]}')
        print("Final test loss:", final_test_metrics['loss'])
    
    # Per-batch test logs advance wandb's own step; a log at an older step is dropped.
    global_step = max(global_step, wandb.run.step)
    wandb.log(final_metrics, step=global_step)
    for k, v in final_metrics.items():
        wandb.run.summary[k] = v
    wandb.run.summary['p'] = p
    wandb.run.summary['best_val_nmae'] = best_val_nmae
    logger.info(f'final training loss: {final_train_metrics["loss"]}')
    print("Final training loss:", final_train_metrics['loss'])
    logger.removeHandler(file_handler)
    file_handler.close()
    os.makedirs(log_dir, exist_ok=True)
    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(embedding_module.state_dict(), path)
        if finish_wandb:
            wandb.finish()
        return embedding_module
    else:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        if finish_wandb:
            wandb.finish()
        return embedding_module, mlp
