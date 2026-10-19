"""Optuna studies of SparseGT (e32, docs/2026-10-e32-sgt-tuning.md).

--space optimizer (default) searches the training recipe with a fixed architecture (study e32-sgt2).
--space architecture searches the SparseGT architecture with the recipe fixed by the --lr,
--warmup-steps, --lr-schedule, --batch-size, --weight-decay, --dropout and --attn-dropout flags
(study e32-arch).

One process is one worker. It loads the training graph and the test set once and
runs trials in-process. Workers share a SQLite study, so several can run at once
and a stopped study resumes when a worker starts again.

Usage:
    python -m spnn.cli.tune --study-name e32-sgt2 --storage ~/runs/e32/e32-sgt2.db \
        --train-data ~/data/spnn/e31/norway_grid_s16.npz \
        --test-data ~/data/spnn/e32/full_test-004_100k.npz \
        --config configs/e31-sparse-gt-k2.yml --device cuda --timeout 9000 --gpu-tag 0
    python -m spnn.cli.tune --space architecture --study-name e32-arch ... \
        --lr 3.5e-3 --warmup-steps 300 --lr-schedule constant --batch-size 8192 \
        --weight-decay 0.1 --dropout 0.1 --attn-dropout 0.2
"""
import argparse
import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
import yaml
from torch_geometric.data import Data

from ..data import PairBatchLoader, npz_to_dataset, split_dataset_for_validation
from ..metrics import evaluate
from ..training import train_few_cross_terrain_case

LAYER_TYPE = 'SparseGT'
P = 1
LOSS = 'mse_loss'
VAL_SEED = 42  # same split seed as spnn.cli.train
METRIC_KEYS = ('loss', 'mse', 'mae', 'nmae', 'acc_1pct', 'acc_2pct', 'acc_5pct')


RECIPE_KEYS = ('lr', 'warmup_steps', 'lr_schedule', 'batch_size', 'weight_decay', 'dropout', 'attn_dropout')
ARCH_KEYS = ('num_layers', 'hidden_dim', 'num_heads', 'num_hops', 'rpearl_num_layers', 'pe_k')


def optimizer_space(trial, args):
    """The e32-sgt2 search space: the training recipe. The architecture comes from --config."""
    params = {
        'lr': trial.suggest_float('lr', 3e-4, 1e-2, log=True),
        'warmup_steps': trial.suggest_categorical('warmup_steps', [0, 300, 1000]),
        'lr_schedule': trial.suggest_categorical('lr_schedule', ['constant', 'cosine']),
        'dropout': trial.suggest_categorical('dropout', [0.0, 0.1, 0.3, 0.5]),
        'attn_dropout': trial.suggest_categorical('attn_dropout', [0.0, 0.2]),
        'batch_size': trial.suggest_categorical('batch_size', [8192, 16384]),
        'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 0.01, 0.1]),
    }
    return params, dict(params), {}


def architecture_space(trial, args):
    """The e32-arch search space: the SparseGT architecture. The recipe comes from the CLI flags."""
    params = {
        'num_layers': trial.suggest_categorical('num_layers', [3, 4, 6, 8]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8]),
        'num_hops': trial.suggest_categorical('num_hops', [2, 4, 6]),
        'rpearl_num_layers': trial.suggest_categorical('rpearl_num_layers', [2, 4, 6]),
        'pe_k': trial.suggest_categorical('pe_k', [1, 2, 3]),
    }
    recipe = {k: getattr(args, k) for k in RECIPE_KEYS}
    return params, recipe, {k: params[k] for k in ARCH_KEYS}


# name -> fn(trial, args) returning (suggested params, training recipe, sparse_gt config overrides).
SEARCH_SPACES = {'optimizer': optimizer_space, 'architecture': architecture_space}


def check_recipe_flags(args):
    """The recipe flags must all be set for --space architecture and must be unset otherwise."""
    given = [k for k in RECIPE_KEYS if getattr(args, k) is not None]
    if args.space == 'architecture' and len(given) != len(RECIPE_KEYS):
        missing = [f"--{k.replace('_', '-')}" for k in RECIPE_KEYS if k not in given]
        raise SystemExit(f'--space architecture needs the fixed recipe; missing {missing}')
    if args.space == 'optimizer' and given:
        raise SystemExit(f'--space optimizer searches the recipe; remove {given}')


def build_config(base_config, recipe, arch):
    """Copy of base_config with the recipe dropouts and the architecture overrides applied."""
    config = copy.deepcopy(base_config)
    sgt = config['gnn']['sparse_gt']
    sgt['dropout'] = recipe['dropout']
    sgt['attn_dropout'] = recipe['attn_dropout']
    sgt['embedding_mode'] = 'data'
    sgt.update(arch)
    if 'hidden_dim' in arch:
        # The node embedding width (gnn.constr.output) follows the attention width.
        config['gnn']['constr']['hidden'] = arch['hidden_dim']
        config['gnn']['constr']['output'] = arch['hidden_dim']
    return config


def load_graph(path, device):
    """Pair dataset and a PyG graph (xyz features, 3D edge length as edge_attr) on device."""
    data = np.load(path)
    dataset, node_features, edge_index = npz_to_dataset(data)
    edge_attr = torch.tensor(data['distances'], dtype=torch.float).unsqueeze(-1)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device)
    return dataset, graph


def should_stop(epoch, val_nmae, abort_after_epoch, abort_nmae):
    """Reason to stop a trial regardless of the pruner, or None."""
    if not math.isfinite(val_nmae):
        return 'non-finite val nmae'
    if epoch >= abort_after_epoch and val_nmae > abort_nmae:
        return f'val nmae {val_nmae:.3f} > {abort_nmae} at epoch {epoch}'
    return None


def make_objective(args, base_config, train_dataset, val_dataset, train_graph, test_dictionary):
    device = args.device
    out = Path(args.out)
    (out / 'trials').mkdir(parents=True, exist_ok=True)
    val_loader = PairBatchLoader(val_dataset, args.eval_batch_size, device=device)

    def objective(trial):
        params, recipe, arch = SEARCH_SPACES[args.space](trial, args)
        config = build_config(base_config, recipe, arch)
        train_loader = PairBatchLoader(train_dataset, recipe['batch_size'], shuffle=True, device=device)
        state = {'best': float('inf'), 'best_epoch': None, 'curve': [], 'stop_reason': None}

        def on_epoch_end(epoch, val_metrics):
            val_nmae = val_metrics['nmae']
            state['curve'].append(val_nmae)
            if val_nmae < state['best']:
                state['best'], state['best_epoch'] = val_nmae, epoch
            reason = should_stop(epoch, val_nmae, args.abort_after_epoch, args.abort_nmae)
            if reason is None:
                trial.report(val_nmae, epoch)
                if trial.should_prune():
                    reason = 'median pruner'
            if reason is not None:
                state['stop_reason'] = reason
                print(f"trial {trial.number}: pruned at epoch {epoch} ({reason})", flush=True)
                return True
            return False

        run_name = f'{args.run_prefix}-t{trial.number}'
        wandb_config = {
            'train_data': args.train_data,
            'test_data': args.test_data,
            'batch_size': recipe['batch_size'],
            'search_space': args.space,
            'config_file': args.config,
            'study_name': args.study_name,
            'trial_number': trial.number,
            'gpu_tag': args.gpu_tag,
            'val_pairs': len(val_dataset),
            'rpearl_embedding_mode': 'data',
            **{f'trial_{k}': v for k, v in params.items()},
            **{f'recipe_{k}': v for k, v in recipe.items()},
        }
        print(f"trial {trial.number}: start {params}", flush=True)
        start = time.time()
        status = 'fail'
        record = {'trial': trial.number, 'study': args.study_name, 'gpu': args.gpu_tag, 'space': args.space,
                  'params': params, 'recipe': recipe}
        module = None
        try:
            try:
                module = train_few_cross_terrain_case(
                    train_dictionary={'graphs': [train_graph], 'dataloaders': [train_loader]},
                    val_dictionary={'graphs': [train_graph], 'dataloaders': [val_loader]},
                    test_dictionary=None,
                    model_config=config,
                    layer_type=LAYER_TYPE,
                    device=device,
                    epochs=args.epochs,
                    loss_func=LOSS,
                    lr=recipe['lr'],
                    base_log_dir=str(out / 'models'),
                    siamese=True,
                    p=P,
                    aggr='sum+diff',
                    new=True,
                    run_name=run_name,
                    wandb_tag=args.wandb_tags,
                    wandb_config=wandb_config,
                    seed=args.seed,
                    weight_decay=recipe['weight_decay'],
                    warmup_steps=recipe['warmup_steps'],
                    lr_schedule=recipe['lr_schedule'],
                    on_epoch_end=on_epoch_end,
                    finish_wandb=False,
                )
            except torch.cuda.OutOfMemoryError as e:
                # An architecture that does not fit on the GPU is pruned, not failed, so TPE learns from it.
                state['stop_reason'] = f'CUDA out of memory: {str(e).splitlines()[0]}'
                print(f"trial {trial.number}: pruned ({state['stop_reason']})", flush=True)
            pruned = state['stop_reason'] is not None
            status = 'pruned' if pruned else 'complete'
            test_metrics = None
            if not pruned:
                t_test = time.time()
                metrics = evaluate(module, None, test_dictionary, device, LAYER_TYPE, True, P, LOSS,
                                   tqdm_prefix='small test')
                test_metrics = {k: metrics[k] for k in METRIC_KEYS}
                test_metrics['num_pairs'] = metrics['num_pairs']
                test_metrics['seconds'] = time.time() - t_test
                for k in METRIC_KEYS:
                    wandb.run.summary[f'small_test_{k}'] = test_metrics[k]
            record.update({
                'best_val_nmae': state['best'],
                'best_epoch': state['best_epoch'],
                'epochs_run': len(state['curve']),
                'stop_reason': state['stop_reason'],
                'small_test': test_metrics,
                'val_nmae_curve': state['curve'],
                'wandb_run': wandb.run.url if wandb.run is not None else None,
            })
            if wandb.run is not None:
                wandb.run.summary['trial_state'] = status
                wandb.run.summary['trial_best_epoch'] = state['best_epoch']
        finally:
            record['state'] = status
            record['wall_seconds'] = time.time() - start
            if wandb.run is not None:
                wandb.run.summary['trial_state'] = status
                wandb.finish(exit_code=0 if status != 'fail' else 1)
            path = out / 'trials' / f't{trial.number:03d}.json'
            path.write_text(json.dumps(record, indent=1))
            del module
            gc.collect()
            torch.cuda.empty_cache()
            print(f"trial {trial.number}: {status} best val nmae {state['best']:.5f} "
                  f"(epoch {state['best_epoch']}) in {record['wall_seconds']:.0f} s -> {path}", flush=True)

        if status == 'pruned':
            raise optuna.TrialPruned(state['stop_reason'])
        return state['best']

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study-name', required=True)
    parser.add_argument('--storage', required=True, help='SQLite file shared by the workers')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--test-data', required=True, help='Evaluated once per completed trial')
    parser.add_argument('--space', choices=sorted(SEARCH_SPACES), default='optimizer',
                        help='optimizer: search the recipe; architecture: search sparse_gt keys, fixed recipe')
    parser.add_argument('--config', required=True,
                        help='SparseGT config; the dropouts and the searched keys are overridden')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Seconds after which this worker starts no new trial')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Study-wide cap on complete + pruned trials')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu-tag', default='0', help='Worker label; also offsets the sampler seed')
    parser.add_argument('--out', default=str(Path.home() / 'runs' / 'e32'),
                        help='trials/t<NNN>.json and models/ go here')
    parser.add_argument('--val-pairs', type=int, default=25000)
    parser.add_argument('--eval-batch-size', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=1, help='Model seed, as --trial 1 in e31')
    parser.add_argument('--sampler-seed', type=int, default=0,
                        help='TPE seed; the worker adds its integer --gpu-tag so workers differ')
    parser.add_argument('--n-startup-trials', type=int, default=5, help='Random TPE trials')
    parser.add_argument('--pruner-startup-trials', type=int, default=None,
                        help='MedianPruner n_startup_trials; default --n-startup-trials')
    parser.add_argument('--pruner-warmup-epochs', type=int, default=15)
    parser.add_argument('--abort-after-epoch', type=int, default=30)
    parser.add_argument('--abort-nmae', type=float, default=0.5)
    parser.add_argument('--wandb-project', default='terrains-dev')
    parser.add_argument('--wandb-tags', nargs='+', default=['e32', 'e32-sgt2'])
    parser.add_argument('--run-prefix', default='e32')
    recipe = parser.add_argument_group('fixed recipe, --space architecture only')
    recipe.add_argument('--lr', type=float)
    recipe.add_argument('--warmup-steps', type=int)
    recipe.add_argument('--lr-schedule', choices=['constant', 'cosine'])
    recipe.add_argument('--batch-size', type=int)
    recipe.add_argument('--weight-decay', type=float)
    recipe.add_argument('--dropout', type=float, help='sparse_gt.dropout')
    recipe.add_argument('--attn-dropout', type=float, help='sparse_gt.attn_dropout')
    args = parser.parse_args()
    check_recipe_flags(args)

    os.environ['WANDB_PROJECT'] = args.wandb_project
    t0 = time.time()
    train_full, train_graph = load_graph(args.train_data, args.device)
    val_dataset, train_dataset = split_dataset_for_validation(train_full, seed=VAL_SEED, n_val=args.val_pairs)
    test_dataset, test_graph = load_graph(args.test_data, args.device)
    test_dictionary = {'graphs': [test_graph],
                       'dataloaders': [PairBatchLoader(test_dataset, args.eval_batch_size, device=args.device)]}
    print(f"loaded data in {time.time() - t0:.1f} s: {len(train_dataset)} train / {len(val_dataset)} val pairs "
          f"on {train_graph.num_nodes} nodes, {len(test_dataset)} test pairs on {test_graph.num_nodes} nodes",
          flush=True)
    with open(args.config) as f:
        configs = yaml.safe_load(f)
    if len(configs) != 1:
        raise ValueError(f'{args.config} must hold one model config, found {list(configs)}')
    base_config = next(iter(configs.values()))

    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{Path(args.storage).expanduser()}',
        engine_kwargs={'connect_args': {'timeout': 300}},
    )
    worker_offset = int(args.gpu_tag) if args.gpu_tag.isdigit() else 0
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed + worker_offset,
                                         n_startup_trials=args.n_startup_trials, constant_liar=True)
    pruner_startup = args.n_startup_trials if args.pruner_startup_trials is None else args.pruner_startup_trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup,
                                         n_warmup_steps=args.pruner_warmup_epochs)
    study = optuna.create_study(study_name=args.study_name, storage=storage, direction='minimize',
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    callbacks = []
    if args.n_trials is not None:
        callbacks.append(optuna.study.MaxTrialsCallback(
            args.n_trials, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)))
    objective = make_objective(args, base_config, train_dataset, val_dataset, train_graph, test_dictionary)
    study.optimize(objective, timeout=args.timeout, callbacks=callbacks, catch=(Exception,), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        print(f"best trial {study.best_trial.number}: {study.best_value:.5f} {study.best_params}", flush=True)
    print(f"worker {args.gpu_tag} finished after {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
