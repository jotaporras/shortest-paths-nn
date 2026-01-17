import argparse
from tqdm import tqdm, trange
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData

from src.baselines import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
from pathlib import Path
import csv

from refactor_training import *

REPO_ROOT = Path(__file__).resolve().parent
output_dir = Path(os.environ.get('TERRAIN_OUTPUT_DIR', REPO_ROOT))

def prepare_single_terrain_dataset(train_data, batch_size):

    train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)

    train_edge_attr = train_data['distances']
    edge_attr = torch.tensor(train_edge_attr)
    edge_attr = edge_attr.unsqueeze(-1)
    graph_data = Data(x=train_node_features, edge_index=train_edge_index, edge_attr=edge_attr) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(graph_data)
    return graph_data, train_dataloader

def get_artificial_datasets(layer_type, trial, res=1, new=True):
    dataset_names = []
    train_data_pths = []
    finetuning_files = []
    amps = [1.0, 2.0, 4.0, 6.0, 8.0,9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    for a in amps:
        pth = str(output_dir / 'data' / 'artificial' / 'change-heights' / f'amp-{a}-res-{res}-train-50k.npz')
        name = str(
            output_dir
            / 'models'
            / 'single_dataset'
            / 'artificial'
            / 'change-heights'
            / f'amp-{a}-res-{res}-train-50k'
            / layer_type
            / 'no-vn'
            / 'siamese'
            / 'p-1'
            / 'mse_loss'
            / '<modelname>'
            / trial
        )
        if new and layer_type =='MLP':
            name = os.path.join(name, 'new')
        dataset_name = f'amp-{a}-res-{res}-train-50k'
        dataset_names.append(dataset_name)
        train_data_pths.append(pth)
        finetuning_files.append(name)

    return dataset_names, train_data_pths, finetuning_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--aggr', type=str, default='max')
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer-type', type=str)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--p', type=int, default=1 )
    parser.add_argument('--finetune-from', type=str, nargs='+')
    parser.add_argument('--include-edge-attr', type=int, default=0)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--single-terrain-per-model', action='store_true')
    parser.add_argument('--artificial', action='store_true')
    parser.add_argument('--wandb-tag', type=str, nargs='+', default=None)
    parser.add_argument('--test-data', type=str, default=None, help='Optional test data file for evaluation')

    args = parser.parse_args()

    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    trial = args.trial

    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)

    # If we are working with artificial datasets, get artificial dataset names. 
    # Train all artificial datasets sequentially together. This tends to be fast so it is 
    # easy to throw into a for-loop for training.  
    if args.artificial:
        dataset_names, train_data_pths, finetuning_from = get_artificial_datasets(layer_type = args.layer_type, 
                                                                                  trial=args.trial, 
                                                                                  new=args.new,
                                                                                  res=1)
        print(train_data_pths)
        num_datasets = len(dataset_names)
    else:
        dataset_names = [args.dataset_name]
        train_name = Path(args.train_data)
        if train_name.is_absolute():
            train_pth = train_name
        elif train_name.suffix:
            train_pth = output_dir / 'data' / train_name
        else:
            default_ext = '.npz' if args.single_terrain_per_model else '.pt'
            train_pth = output_dir / 'data' / f'{args.train_data}{default_ext}'
        train_data_pths = [str(train_pth)]
        finetuning_from = args.finetune_from
        num_datasets = 1

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        train_data_pth = train_data_pths[i]
        prev_model_file_pth = finetuning_from[i]
        for modelname in model_configs:

            # Provide train set dataloaders 
            if args.single_terrain_per_model:
                train_data = np.load(train_data_pth)
                graph_data, train_dataloader = prepare_single_terrain_dataset(train_data, args.batch_size)
                train_dictionary = {'graphs': [graph_data], 'dataloaders': [train_dataloader]}
            else:
                train_data = torch.load(train_data_pth)
                train_dictionary = {'graphs': train_data['graphs'], 'dataloaders': []}
                for dataset in train_data['datasets']:
                    train_dictionary['dataloaders'].append(DataLoader(dataset, batch_size = args.batch_size, shuffle=True))
            
            
            config=model_configs[modelname]
            print(modelname, config)
            model_to_finetune_from_pth = prev_model_file_pth.replace('<modelname>', modelname)
            
            # Build wandb config with all command-line arguments
            wandb_config = {
                "dataset_name": dataset_name,
                "train_data": args.train_data,
                "batch_size": args.batch_size,
                "config_file": args.config,
                "siamese": siamese,
                "vn": vn,
                "include_edge_attr": args.include_edge_attr,
                "trial": trial,
                "single_terrain_per_model": args.single_terrain_per_model,
                "artificial": args.artificial,
                "model_config_name": modelname,
                "finetune_from": prev_model_file_pth,
            }
            
            # Extract resolution from dataset_name (e.g., "norway/res17" -> "res17")
            res_part = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
            
            # Handle test data if provided (for validation and final evaluation)
            val_dictionary = None
            test_dictionary = None
            if args.test_data is not None:
                test_file = os.path.join(output_dir, 'data', args.test_data) if not Path(args.test_data).is_absolute() else args.test_data
                test_data = np.load(test_file, allow_pickle=True)
                test_dataset_full, test_node_features, test_edge_index = npz_to_dataset(test_data)
                
                # Split test dataset into validation (0.5%) and test (99.5%)
                val_dataset, test_dataset = split_dataset_for_validation(test_dataset_full, val_fraction=0.005, seed=42)
                print(f"Split test data: {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
                
                test_edge_attr = None
                if args.include_edge_attr:
                    test_edge_attr = test_data['distances']
                
                if test_edge_attr is not None:
                    test_edge_attr_tensor = torch.tensor(test_edge_attr)
                    test_edge_attr_tensor = test_edge_attr_tensor.unsqueeze(-1)
                else:
                    test_edge_attr_tensor = None
                
                test_graph_data = Data(x=test_node_features, edge_index=test_edge_index, edge_attr=test_edge_attr_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                val_dictionary = {'graphs': [test_graph_data], 'dataloaders': [val_dataloader]}
                test_dictionary = {'graphs': [test_graph_data], 'dataloaders': [test_dataloader]}
            
            train_terrains_decoupled(train_dictionary = train_dictionary,
                                    model_config = config, 
                                    layer_type = args.layer_type, 
                                    device = args.device,
                                    prev_model_pth = model_to_finetune_from_pth,
                                    finetune_dataset_name = dataset_name,
                                    epochs=args.epochs, 
                                    loss_func=args.loss,
                                    lr =args.lr,
                                    p=args.p, 
                                    aggr=aggr, 
                                    new=args.new,
                                    run_name=f"terrain-graph-{args.layer_type}-{res_part}-stage2",
                                    wandb_tag=args.wandb_tag,
                                    wandb_config=wandb_config,
                                    test_dictionary=test_dictionary,
                                    val_dictionary=val_dictionary)
    
if __name__=='__main__':
    main()

