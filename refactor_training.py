import argparse
from tqdm import tqdm, trange
from torch.optim import Adam, AdamW
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from src.baselines import *
from src.loss_funcs import *
from src.custom_models import SparseGTWithRPEARL
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
from pathlib import Path
import csv
import logging
import copy

import time

ACTIVATION_ALIASES = {
    'relu': 'ReLU',
    'lrelu': 'LeakyReLU',
    'silu': 'SiLU'
}


def normalize_activation(activation: str) -> str:
    if not activation:
        return activation
    return ACTIVATION_ALIASES.get(activation.lower(), activation)

from torch.profiler import profile, record_function, ProfilerActivity

import wandb

wandb.login()

MSE = nn.MSELoss()

REPO_ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def compute_metrics(embedding_module, mlp, dataloader, graph_data, device, 
                    layer_type, siamese, p, loss_func):
    """
    Compute metrics on a dataset (train or test).
    
    Returns a dict with: loss, nmae, mse, mae
    """
    embedding_module.eval()
    if mlp is not None:
        mlp.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_nmae = 0.0
    total_samples = 0
    
    pred_results = []
    for batch in dataloader:
        srcs = batch[0].to(device, non_blocking=True)
        tars = batch[1].to(device, non_blocking=True)
        lengths = batch[2].to(device, non_blocking=True)
        batch_size = len(srcs)
        
        if 'MLP' in layer_type:
            src_embeddings = embedding_module(graph_data.x[srcs])
            tar_embeddings = embedding_module(graph_data.x[tars])
        else:
            node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, 
                                               edge_attr=graph_data.edge_attr)
            src_embeddings = node_embeddings[srcs]
            tar_embeddings = node_embeddings[tars]
        
        if siamese:
            pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
        else:
            pred = mlp(src_embeddings, tar_embeddings, vn_emb=None)
            pred = pred.squeeze()
        
        # Compute metrics
        loss = globals()[loss_func](pred, lengths)
        total_loss += loss.item() * batch_size
        total_mse += mse_loss(pred, lengths).item() * batch_size
        total_mae += mae_loss(pred, lengths).item() * batch_size
        total_nmae += nmae_loss(pred, lengths).item() * batch_size
        total_samples += batch_size

        pred_mses = torch.square(pred - lengths)
        pred_maes = torch.abs(pred - lengths)
        pred_nmaes = torch.abs(pred - lengths) / lengths
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
        'all_preds_df': all_preds_df,
    }


@torch.no_grad()
def compute_metrics_decoupled(mlp, embeddings, dataloader, device, loss_func):
    """
    Compute metrics on a dataset for decoupled training (pre-computed embeddings).
    
    Returns a dict with: loss, nmae, mse, mae
    """
    mlp.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_nmae = 0.0
    total_samples = 0
    
    for batch in dataloader:
        srcs = batch[0]
        tars = batch[1]
        embd_srcs = embeddings[srcs].to(device, non_blocking=True)
        embd_tars = embeddings[tars].to(device, non_blocking=True)
        lengths = batch[2].to(device, non_blocking=True)
        batch_size = len(srcs)
        
        pred = mlp(embd_srcs, embd_tars, vn_emb=None)
        pred = pred.squeeze()
        
        # Compute metrics
        loss = globals()[loss_func](pred, lengths)
        total_loss += loss.item() * batch_size
        total_mse += mse_loss(pred, lengths).item() * batch_size
        total_mae += mae_loss(pred, lengths).item() * batch_size
        total_nmae += nmae_loss(pred, lengths).item() * batch_size
        total_samples += batch_size
    
    mlp.train()
    
    return {
        'loss': total_loss / total_samples,
        'mse': total_mse / total_samples,
        'mae': total_mae / total_samples,
        'nmae': total_nmae / total_samples,
    }
output_dir = Path(os.environ.get('TERRAIN_OUTPUT_DIR', REPO_ROOT))

sparse_tensor = ToSparseTensor()
virtual_node_transform = VirtualNode()

class SingleGraphShortestPathDataset(Dataset):
    def __init__(self, sources, targets, lengths):
        self.sources = sources
        self.targets = targets
        self.lengths = lengths

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.lengths[idx]

class TerrainPatchesDataset(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self.x.size(0)
        if key == 'tar':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'], dtype=torch.int)
    tars = torch.tensor(data['tars'],  dtype=torch.int)
    lengths = torch.tensor(data['lengths'])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    l2 = torch.norm(node_features[srcs] - node_features[tars], dim=1, p=2)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

def debug_dataset(data, n=100):
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'][:n])
    tars = torch.tensor(data['tars'][:n])
    lengths = torch.tensor(data['lengths'][:n])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    l2 = torch.norm(node_features[srcs] - node_features[tars], dim=1, p=2)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

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


def configure_embedding_module(model_config, 
                               layer_type,
                               edge_dim=1,
                               new=True):
    print(model_config)
    print(new)
    embedding_config = model_config['constr']
    layer_norm = model_config['layer_norm']
    dropout = model_config['dropout']
    activation = model_config['activation']
    activation_for_gnn = normalize_activation(activation)
    if not new and 'MLP' in layer_type:
        embedding_module = initialize_mlp(**embedding_config, 
                                          activation=activation, 
                                          layer_norm=layer_norm, 
                                          dropout=dropout)
    elif 'MLP' in layer_type and new:
        embedding_module = NewMLP(**embedding_config, add_norm=layer_norm)
    elif layer_type == 'SparseGT':
        # SparseGT with RPEARL positional encodings
        sparse_gt_config = model_config.get('sparse_gt', {})
        embedding_module = SparseGTWithRPEARL(
            input_dim=embedding_config.get('input', 3),
            hidden_dim=sparse_gt_config.get('hidden_dim', embedding_config.get('hidden', 64)),
            output_dim=embedding_config.get('output', 64),
            num_layers=sparse_gt_config.get('num_layers', 3),
            num_heads=sparse_gt_config.get('num_heads', 4),
            num_hops=sparse_gt_config.get('num_hops', 2),
            rpearl_samples=sparse_gt_config.get('rpearl_samples', 30),
            rpearl_num_layers=sparse_gt_config.get('rpearl_num_layers', 3),
            dropout=sparse_gt_config.get('dropout', 0.3),
            attn_dropout=sparse_gt_config.get('attn_dropout', 0.1),
        )
    else:
        embedding_module = GNNModel(layer_type=layer_type, 
                                    edge_dim=edge_dim, 
                                    activation=activation_for_gnn, 
                                    layer_norm=layer_norm, 
                                    **embedding_config)
    return embedding_module

    
def configure_mlp_module(mlp_config, 
                        aggr='sum', 
                        new=True):
    model_config = copy.deepcopy(mlp_config['constr'])
    layer_norm=mlp_config['layer_norm']
    dropout=mlp_config['dropout']
    
    if aggr == 'combine':
        model_config['input'] = model_config['input'] * 3
    elif aggr == 'concat' or aggr == 'sum+diff':
        model_config['input'] = model_config['input'] * 2 
    if not new:
        mlp_nn = initialize_mlp(**model_config, layer_norm=False, dropout=dropout, activation='lrelu')
    else:
        mlp_nn = NewMLP(**model_config, add_norm=layer_norm)
    mlp = MLPBaseline1(mlp_nn, aggr=aggr)
    return mlp

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
    embedding_module.to(torch.double)
    
    print(model_config)
    print(embedding_module)

    
    mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
    mlp = mlp.to(torch.double)
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
    
    # Pre-compute test embeddings if test data is provided
    if test_dictionary is not None:
        num_test_graphs = len(test_dictionary['graphs'])
        test_dictionary['embeddings'] = []
        for i in range(num_test_graphs):
            graph_data = test_dictionary['graphs'][i]
            if 'MLP' in layer_type:
                node_embeddings = embedding_module(graph_data.x)
            else:
                node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, edge_attr=graph_data.edge_attr)
            test_dictionary['embeddings'].append(node_embeddings)
        
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
        dir=str(output_dir / 'wandb'),
        name=run_name,
        tags=wandb_tag if wandb_tag else None,
        config=base_config
    )
    
    # Create log directory based on run_name and wandb run ID
    # Format: {run_name}_{wandb_id}/{finetune_dataset_name}
    train_filename = run_name if run_name else 'train'
    run_dir_name = f"{train_filename}_{run.id}"
    log_dir = os.path.join(output_dir, 'runs', run_dir_name, finetune_dataset_name)
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
            for batch in dataloader:
                srcs = batch[0]
                tars = batch[1]
                embd_srcs = embeddings[srcs].to(device, non_blocking=True)
                embd_tars = embeddings[tars].to(device, non_blocking=True)
                lengths = batch[2].to(device, non_blocking=True)

                pred = mlp(embd_srcs, embd_tars, vn_emb=None)
                pred=pred.squeeze()

                loss = globals()[loss_func](pred, lengths)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach()
                normalized_abs_err = globals()['nmae_loss'](pred, lengths)

                wandb.log({'epoch_train_loss': loss, 
                        'epoch_total_loss': total_loss/total_samples,
                        'epoch_normalized_abs_err': normalized_abs_err})

    # Compute final train metrics
    final_train_metrics = compute_metrics_decoupled(
        mlp, train_dictionary['embeddings'][0], train_dictionary['dataloaders'][0],
        device, loss_func
    )
    
    # Compute final test metrics if test data is provided
    final_test_metrics = None
    if test_dictionary is not None:
        final_test_metrics = compute_metrics_decoupled(
            mlp, test_dictionary['embeddings'][0], test_dictionary['dataloaders'][0],
            device, loss_func
        )
    
    # Log final metrics
    final_metrics = {
        'train_loss': final_train_metrics['loss'],
        'train_mse': final_train_metrics['mse'],
        'train_mae': final_train_metrics['mae'],
        'train_nmae': final_train_metrics['nmae'],
    }
    if final_test_metrics is not None:
        final_metrics.update({
            'test_loss': final_test_metrics['loss'],
            'test_mse': final_test_metrics['mse'],
            'test_mae': final_test_metrics['mae'],
            'test_nmae': final_test_metrics['nmae'],
        })
    wandb.log(final_metrics)
    
    logging.info(f'final training loss: {final_train_metrics["loss"]}')
    if final_test_metrics is not None:
        logging.info(f'final test loss: {final_test_metrics["loss"]}')

    print("Final training loss:", final_train_metrics['loss'])
    if final_test_metrics is not None:
        print("Final test loss:", final_test_metrics['loss'])
    path = os.path.join(record_dir, 'final_model.pt')
    print("saving model to:", path)
    torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                'mlp_state_dict': mlp.state_dict()}, path)
    wandb.finish()
    return embedding_module, mlp


# train_dictionary = {'graphs': [g1, g2, g3, ....], 'dataloaders': [dl1, dl2, ....]}
def train_few_cross_terrain_case(train_dictionary,
                                model_config, 
                                layer_type, 
                                device,
                                activation='silu',
                                epochs=100, 
                                loss_func='mse_loss',
                                lr =0.001,
                                base_log_dir=str(output_dir),
                                siamese=True,
                                p=1, 
                                aggr='sum',
                                new=False,
                                finetune_from=None,
                                run_name=None,
                                wandb_tag=None,
                                wandb_config=None,
                                test_dictionary=None):
    torch.manual_seed(0)
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
    embedding_module.to(torch.double)
    embedding_module.to(device)
    print(embedding_module)    
    mlp=None

    if siamese:
        parameters = embedding_module.parameters() #Thisis the one were using
    else:
        mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
        mlp = mlp.to(torch.double)
        mlp.to(device)
        parameters = list(embedding_module.parameters()) + list(mlp.parameters())
        print(mlp)
    
    optimizer = AdamW(parameters, lr=lr)

    # Build wandb config with all relevant parameters
    base_config = {
        "learning_rate": lr,
        "epochs": epochs,
        "siamese": siamese,
        "p": p,
        "layer_type": layer_type,
        "device": device,
        "loss_func": loss_func,
        "aggr": aggr,
        "new": new,
        "finetune_from": finetune_from,
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
        project='terrains',
        dir=str(output_dir / 'wandb'),
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
    logging.basicConfig(level=logging.INFO, filename=log_file, force=True)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'Siamese? {siamese}')
    logging.info(f'loss function: {loss_func}')
    print("logging to....", record_dir)
    
    global_step = 0
    for epoch in trange(epochs):
        for i in range(num_graphs):
            graph_data = train_dictionary['graphs'][i].to(device)
            dataloader = train_dictionary['dataloaders'][i]
            for batch in dataloader:
                srcs = batch[0].to(device, non_blocking=True)
                tars = batch[1].to(device, non_blocking=True)
                lengths = batch[2].to(device, non_blocking=True)
                batch_size = len(srcs)
                
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
                
                loss = globals()[loss_func](pred, lengths)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
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
                }, step=global_step)
                
                global_step += 1

    # Compute final train metrics
    graph_data = train_dictionary['graphs'][0].to(device)
    final_train_metrics = compute_metrics(
        embedding_module, mlp, train_dictionary['dataloaders'][0],
        graph_data, device, layer_type, siamese, p, loss_func
    )
    
    # Compute final test metrics if test data is provided
    final_test_metrics = None
    if test_dictionary is not None:
        test_graph_data = test_dictionary['graphs'][0].to(device)
        final_test_metrics = compute_metrics(
            embedding_module, mlp, test_dictionary['dataloaders'][0],
            test_graph_data, device, layer_type, siamese, p, loss_func
        )
    
    # Log final metrics (use final_ prefix to distinguish from batch metrics)
    final_metrics = {
        'final_train_loss': final_train_metrics['loss'],
        'final_train_mse': final_train_metrics['mse'],
        'final_train_mae': final_train_metrics['mae'],
        'final_train_nmae': final_train_metrics['nmae'],
    }
    if final_test_metrics is not None:
        final_metrics.update({
            'test_loss': final_test_metrics['loss'],
            'test_mse': final_test_metrics['mse'],
            'test_mae': final_test_metrics['mae'],
            'test_nmae': final_test_metrics['nmae'],
        })
    wandb.log(final_metrics, step=global_step)

    # Save predictions to CSV and log as wandb artifact
    if final_test_metrics is not None and 'all_preds_df' in final_test_metrics:
        preds_path = os.path.join(log_dir, 'preds.csv')
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        final_test_metrics['all_preds_df'].to_csv(preds_path, index=False)
        logging.info(f'Saved predictions to {preds_path}')
        
        # Log as wandb artifact
        # disable to avoid clutter
        # artifact = wandb.Artifact(name='predictions', type='predictions')
        # artifact.add_file(preds_path)
        # wandb.log_artifact(artifact)
    
    logging.info(f'final training loss: {final_train_metrics["loss"]}')
    if final_test_metrics is not None:
        logging.info(f'final test loss: {final_test_metrics["loss"]}')

    print("Final training loss:", final_train_metrics['loss'])
    if final_test_metrics is not None:
        print("Final test loss:", final_test_metrics['loss'])
    print("siamese", siamese)
    os.makedirs(log_dir, exist_ok=True)
    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(embedding_module.state_dict(), path)
        wandb.finish()
        return embedding_module
    else:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        wandb.finish()
        return embedding_module, mlp
