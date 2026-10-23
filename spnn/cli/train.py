"""Train one model per config entry on a terrain graph, evaluate on a test terrain.

Usage: python -m spnn.cli.train --dataset-name ... --train-data ... --test-data ... --config ...
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from ..data import PairBatchLoader, npz_to_dataset, split_dataset_for_validation
from ..graphs.spectral import laplacian_pe
from ..training import LR_SCHEDULES, format_log_dir, output_dir, train_few_cross_terrain_case


def get_artificial_datasets(res=2):
    dataset_names = []
    train_data_pths = []
    amps = [1.0, 2.0, 4.0, 6.0, 8.0,9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    for a in amps:
        # TODO this is hardcoded fix it
        pth = str(output_dir() / 'data' / 'artificial' / 'change-heights' / f'amp-{a}-res-{res}-train-50k.npz')
        name =  f'artificial/change-heights/amp-{a}-res-{res}-train-50k'
        dataset_names.append(name)
        train_data_pths.append(pth)

    return dataset_names, train_data_pths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0, choices=[0],
                        help='Virtual nodes are not implemented; only 0 is accepted')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--aggr', type=str, default='max')
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer-type', type=str)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: --trial if it is an integer, else 0)')
    parser.add_argument('--p', type=int, default=1 )
    parser.add_argument('--finetune', type=int, default=0, choices=[0],
                        help='Has no effect; use --finetune-from. Only 0 is accepted')
    parser.add_argument('--include-edge-attr', type=int, default=0)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--finetune-from', type=str, default='none')
    parser.add_argument('--artificial', action='store_true')
    parser.add_argument('--single-graph-full-batch', action='store_true')
    parser.add_argument('--val-fraction', type=float, default=0.005,
                        help='Fraction of training pairs held out for checkpoint selection')
    parser.add_argument('--val-pairs', type=int, default=None,
                        help='Number of training pairs held out for validation; overrides --val-fraction')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay (0.01 is the AdamW default)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Linear learning-rate warmup from 0 over this many optimizer steps')
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=list(LR_SCHEDULES),
                        help='After warmup: constant, or cosine decay to 1e-4 x lr at the last step')
    parser.add_argument('--fast-loader', action='store_true',
                        help='Slice batches from tensors on the device instead of a DataLoader. '
                             'Same batches, about 2.5x faster steps for SparseGT at batch 16,384')
    parser.add_argument('--wandb-tag', type=str, nargs='+', default=None)
    parser.add_argument('--resolution', type=int, default=None,
                        help='Graph resolution to log directly to wandb/config')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Stop training if val_mse does not improve for this many epochs.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom wandb run name. Defaults to terrain-graph-{layer_type}-{resolution}-stage1')
    parser.add_argument('--rpearl-embedding-mode', type=str, default=None,
                        choices=['random', 'data', 'lappe'],
                        help='Positional encoding mode for SparseGT (overrides config). lappe = precomputed '
                             'Laplacian eigenvectors appended to the node features (e35)')
    parser.add_argument('--lappe-k', type=int, default=None,
                        help='Number of Laplacian eigenvectors in lappe mode (overrides sparse_gt.lappe_k)')
    parser.add_argument('--lappe-cache-dir', type=str, default=None,
                        help='Eigenvector cache; default $SPNN_LAPPE_CACHE or ~/data/spnn/lappe')

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    finetune_from=None if args.finetune_from == 'none' else args.finetune_from
    trial = args.trial
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(trial) if trial is not None and trial.isdigit() else 0

    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    if args.artificial:
        dataset_names, train_data_pths = get_artificial_datasets(res=1)
        num_datasets = len(dataset_names)
    else:
        dataset_names = [args.dataset_name]
        train_name = Path(args.train_data)
        if train_name.is_absolute():
            train_pth = train_name
        elif train_name.suffix:
            train_pth = output_dir() / 'data' / train_name
        else:
            train_pth = output_dir() / 'data' / f'{args.train_data}.npz'
        train_data_pths = [str(train_pth)]
        num_datasets = 1
    
    for i in range(num_datasets):
        dataset_name = dataset_names[i]
        train_data_pth = train_data_pths[i]
        print("Now training:", dataset_name)
        print("Train data:", train_data_pth)
        for modelname in model_configs:
            test_file = os.path.join(output_dir(), 'data', args.test_data)
            

            train_data = np.load(train_data_pth, allow_pickle=True)
            test_data = np.load(test_file, allow_pickle=True)

            train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
            test_dataset_full, test_node_features, test_edge_index = npz_to_dataset(test_data)

            # Hold out training pairs for checkpoint selection; the test pairs stay unseen.
            val_dataset, train_dataset = split_dataset_for_validation(
                train_dataset, val_fraction=args.val_fraction, seed=42, n_val=args.val_pairs)
            test_dataset = test_dataset_full
            print(f"Split train data: {len(val_dataset)} validation samples, {len(train_dataset)} train samples")

            train_edge_attr = None 
            test_edge_attr = None
            if args.include_edge_attr:
                train_edge_attr = train_data['distances']
                test_edge_attr = test_data['distances']
            print("Number of train nodes:", len(train_node_features))
            print("Number of test nodes:", len(test_node_features))
            if args.fast_loader:
                train_dataloader = PairBatchLoader(train_dataset, args.batch_size, shuffle=True, device=args.device)
                val_dataloader = PairBatchLoader(val_dataset, args.batch_size, device=args.device)
                test_dataloader = PairBatchLoader(test_dataset, args.batch_size, device=args.device)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            if train_edge_attr is not None:
                edge_attr = torch.tensor(train_edge_attr, dtype=torch.float)
                edge_attr = edge_attr.unsqueeze(-1)
            else:
                edge_attr = None
            
            if test_edge_attr is not None:
                test_edge_attr_tensor = torch.tensor(test_edge_attr, dtype=torch.float)
                test_edge_attr_tensor = test_edge_attr_tensor.unsqueeze(-1)
            else:
                test_edge_attr_tensor = None
            
            graph_data = Data(x=train_node_features, edge_index=train_edge_index, edge_attr=edge_attr)        
            test_graph_data = Data(x=test_node_features, edge_index=test_edge_index, edge_attr=test_edge_attr_tensor)
            train_dictionary = {'graphs': [graph_data], 'dataloaders': [train_dataloader]}
            val_dictionary = {'graphs': [graph_data], 'dataloaders': [val_dataloader]}
            test_dictionary = {'graphs': [test_graph_data], 'dataloaders': [test_dataloader]}

            log_dir = format_log_dir(output_dir(), 
                                    dataset_name, 
                                    siamese, 
                                    modelname, 
                                    vn, 
                                    aggr, 
                                    args.loss, 
                                    args.layer_type,
                                    args.p,
                                    args.trial)
            config=model_configs[modelname]

            if args.rpearl_embedding_mode is not None:
                config.setdefault('gnn', {}).setdefault('sparse_gt', {})['embedding_mode'] = args.rpearl_embedding_mode
            sgt_config = config.get('gnn', {}).get('sparse_gt', {})
            if args.lappe_k is not None:
                sgt_config['lappe_k'] = args.lappe_k
            lappe_info = {}
            if args.layer_type == 'SparseGT' and sgt_config.get('embedding_mode') == 'lappe':
                # Append the eigenvectors to the node features; the model splits them off again.
                lappe_k = sgt_config.setdefault('lappe_k', 16)
                for label, graph, src in (('train', graph_data, train_data_pth), ('test', test_graph_data, test_file)):
                    t0 = time.time()
                    V, info = laplacian_pe(graph.num_nodes, graph.edge_index.numpy(), lappe_k,
                                           cache_dir=args.lappe_cache_dir, source=src)
                    graph.x = torch.cat([graph.x, torch.from_numpy(V)], dim=1)
                    lappe_info[f'lappe_{label}_eig_seconds'] = info['seconds']
                    lappe_info[f'lappe_{label}_eig_peak_rss_gb'] = info['peak_rss_gb']
                    lappe_info[f'lappe_{label}_eig_k_computed'] = info['k']
                    lappe_info[f'lappe_{label}_eig_host'] = info['host']
                    lappe_info[f'lappe_{label}_eigenvalues'] = info['eigenvalues'][:lappe_k + 1]
                    lappe_info[f'lappe_{label}_load_seconds'] = round(time.time() - t0, 2)
                    print(f"lappe {label}: {lappe_k} eigenvectors from {info['cache']} "
                          f"(computed in {info['seconds']} s on {info['host']}, loaded in {time.time() - t0:.1f} s)")
                lappe_info['lappe_k'] = lappe_k

            print(modelname, config)

            # Build wandb config with all command-line arguments
            wandb_config = {
                "dataset_name": dataset_name,
                "train_data": args.train_data,
                "test_data": args.test_data,
                "batch_size": args.batch_size,
                "config_file": args.config,
                "vn": vn,
                "include_edge_attr": args.include_edge_attr,
                "trial": trial,
                "artificial": args.artificial,
                "model_config_name": modelname,
            }
            if args.rpearl_embedding_mode is not None:
                wandb_config["rpearl_embedding_mode"] = args.rpearl_embedding_mode
            wandb_config.update(lappe_info)
            if args.resolution is not None:
                wandb_config["resolution"] = args.resolution
            
            # Extract resolution from dataset_name (e.g., "norway/res17" -> "res17")
            resolution_label = f"res{args.resolution:02d}" if args.resolution is not None else (
                dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
            )
            
            train_few_cross_terrain_case(train_dictionary=train_dictionary,
                                        seed = seed,
                                        model_config = config,
                                        layer_type = args.layer_type,
                                        device = args.device,
                                        epochs = args.epochs,
                                        lr= args.lr,
                                        loss_func=args.loss,
                                        aggr = aggr, 
                                        base_log_dir=log_dir,
                                        p = args.p,
                                        siamese=siamese,
                                        finetune_from=finetune_from,
                                        new=args.new,
                                        run_name=args.run_name or f"terrain-graph-{args.layer_type}-{resolution_label}-stage1",
                                        wandb_tag=args.wandb_tag,
                                        wandb_config=wandb_config,
                                        single_graph_full_batch=args.single_graph_full_batch,
                                        test_dictionary=test_dictionary,
                                        val_dictionary=val_dictionary,
                                        early_stopping_patience=args.early_stopping_patience,
                                        weight_decay=args.weight_decay,
                                        warmup_steps=args.warmup_steps,
                                        lr_schedule=args.lr_schedule)
        
if __name__=='__main__':
    main()

