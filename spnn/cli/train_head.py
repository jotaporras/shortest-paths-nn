"""Stage 2 of Chen et al. (ICML 2025): freeze a stage-1 embedding model and train an MLP pair head.

The stage-1 checkpoint (an e34/e35 run directory with models/**/final_model.pt) is loaded and frozen. The
stage-2 graph (the --train-data npz) and the test graph are embedded once. The head takes the two node
embeddings, aggregates them (default sum+diff, as the legacy Phase 2 and Chen's M-CTR) and predicts the
distance; it is trained with AdamW on the stage-2 pairs, validated on held-out pairs, and the best-validation
head is evaluated on the test pairs.

Outputs in --out: <run>.json (final test/train metrics and per-epoch validation, e34 keys), <run>_head.pt,
preds/<run>__<test>.npy (float16 km, test-pair order, as scripts/e34c_eval_cross.py) and metrics/<run>__<test>.json.

Example (e36, SparseGT trained on grid stride 12, head trained on the stride-4 grid pairs):
    python -m spnn.cli.train_head --embedding-run ~/runs/e34/norway_grid_s12_sgt2 --model sgt2 \
        --train-data ~/data/spnn/e31/norway_grid_s04.npz --test-data ~/data/spnn/full_test-004.npz \
        --out ~/runs/e36 --run norway_grid_s12_sgt2_head-s04 --device cuda
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..graphs.spectral import laplacian_pe
from ..models.factory import configure_embedding_module
from ..models.mlp import MLPBaseline1, NewMLP

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
MODEL_SPEC = {
    'mlp': ('MLP', 'configs/e31-mlp.yml'),
    'tagconv': ('TAGConv', 'configs/e31-tagconv.yml'),
    'sgt2': ('SparseGT', 'configs/e32-sparse-gt-k2.yml'),
    'sgt2lap': ('SparseGT', 'configs/e35-sparse-gt-k2-lappe.yml'),
}


def load_embedding(run_dir, model, device, lappe_k=None):
    layer_type, cfg_path = MODEL_SPEC[model]
    cfg = next(iter(yaml.safe_load(open(REPO / cfg_path)).values()))
    if layer_type == 'SparseGT':
        sgt = cfg['gnn'].setdefault('sparse_gt', {})
        sgt['embedding_mode'] = 'lappe' if model == 'sgt2lap' else 'data'
        if model == 'sgt2lap' and lappe_k:
            sgt['lappe_k'] = lappe_k
    net = configure_embedding_module(cfg['gnn'], layer_type, edge_dim=1, new=True)
    weights = next(Path(run_dir).glob('models/**/final_model.pt'))
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    for p in net.parameters():
        p.requires_grad_(False)
    return net.to(device).eval(), layer_type, str(weights)


@torch.no_grad()
def embed(net, layer_type, graph, device):
    x = torch.tensor(graph['node_features'], dtype=torch.float, device=device)
    if layer_type == 'MLP':
        return net(x)
    if getattr(net, 'embedding_mode', None) == 'lappe':
        V, _ = laplacian_pe(len(x), graph['edge_index'], net.lappe_k)
        x = torch.cat([x, torch.from_numpy(V).to(device)], dim=1)
    ei = torch.tensor(graph['edge_index'], dtype=torch.long, device=device)
    w = torch.tensor(graph['distances'], dtype=torch.float, device=device).unsqueeze(-1)
    return net(x, ei, edge_attr=w)


@torch.no_grad()
def predict(head, E, srcs, tars, chunk=500_000):
    head.eval()
    out = torch.empty(len(srcs), device=E.device)
    for i in range(0, len(srcs), chunk):
        out[i:i + chunk] = head(E[srcs[i:i + chunk]], E[tars[i:i + chunk]]).squeeze(-1)
    head.train()
    return out


def metrics(pred, true, prefix=''):
    nz = true > 0
    rel = (pred[nz] - true[nz]).abs() / true[nz]
    err = pred[nz] - true[nz]
    return {f'{prefix}loss': float((err ** 2).mean()), f'{prefix}mse': float((err ** 2).mean()),
            f'{prefix}mae': float(err.abs().mean()), f'{prefix}nmae': float(rel.mean()),
            f'{prefix}acc_1pct': float((rel < 0.01).float().mean()), f'{prefix}acc_2pct': float((rel < 0.02).float().mean()),
            f'{prefix}acc_5pct': float((rel < 0.05).float().mean()), f'{prefix}bias': float(err.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--embedding-run', required=True, help='stage-1 run directory (holds models/**/final_model.pt)')
    ap.add_argument('--model', required=True, choices=list(MODEL_SPEC))
    ap.add_argument('--lappe-k', type=int, default=None)
    ap.add_argument('--train-data', required=True, help='stage-2 pairs with their graph (npz)')
    ap.add_argument('--test-data', required=True)
    ap.add_argument('--test-name', default='grid', help='label of the test set in the preds/metrics file names')
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', required=True, help='run name, used for the output files')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--aggr', default='sum+diff')
    ap.add_argument('--layer-norm', action='store_true')
    ap.add_argument('--val-pairs', type=int, default=25000)
    ap.add_argument('--train-pairs', type=int, default=None, help='subsample this many stage-2 pairs (after the val split)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out).expanduser()
    (out / 'preds').mkdir(parents=True, exist_ok=True)
    (out / 'metrics').mkdir(exist_ok=True)
    t0 = time.time()

    net, layer_type, weights = load_embedding(Path(args.embedding_run).expanduser(), args.model, device, args.lappe_k)
    logger.info('stage-1 weights %s', weights)
    train = dict(np.load(Path(args.train_data).expanduser()))
    test = dict(np.load(Path(args.test_data).expanduser()))
    E_train = embed(net, layer_type, train, device).float()
    E_test = embed(net, layer_type, test, device).float()
    logger.info('embedded train graph %d nodes, test graph %d nodes, dim %d, %.0fs',
                len(E_train), len(E_test), E_train.shape[1], time.time() - t0)
    del net
    torch.cuda.empty_cache()

    # pairs: validation split first, then an optional subsample of the rest
    srcs = torch.tensor(train['srcs'], dtype=torch.long)
    tars = torch.tensor(train['tars'], dtype=torch.long)
    lengths = torch.tensor(train['lengths'], dtype=torch.float)
    keep = lengths > 0
    srcs, tars, lengths = srcs[keep], tars[keep], lengths[keep]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(srcs), generator=g)
    val_idx, rest = perm[:args.val_pairs], perm[args.val_pairs:]
    if args.train_pairs:
        rest = rest[:args.train_pairs]
    tr = [t[rest].to(device) for t in (srcs, tars, lengths)]
    va = [t[val_idx].to(device) for t in (srcs, tars, lengths)]
    te_s = torch.tensor(test['srcs'], dtype=torch.long, device=device)
    te_t = torch.tensor(test['tars'], dtype=torch.long, device=device)
    te_l = torch.tensor(test['lengths'], dtype=torch.float, device=device)
    logger.info('pairs: train %d, val %d, test %d', len(tr[0]), len(va[0]), len(te_s))

    dim = E_train.shape[1]
    in_dim = {'sum+diff': 2, 'concat': 2, 'combine': 3}.get(args.aggr, 1) * dim
    head = MLPBaseline1(NewMLP(in_dim, 1, args.hidden, args.layers, add_norm=args.layer_norm), aggr=args.aggr).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(tr[0])
    best = dict(val_nmae=float('inf'), epoch=-1, state=None)
    val_log = []
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        total = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i + args.batch_size]
            pred = head(E_train[tr[0][b]], E_train[tr[1][b]]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, tr[2][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        vm = metrics(predict(head, E_train, va[0], va[1]), va[2], 'val_')
        if vm['val_nmae'] < best['val_nmae']:
            best = dict(val_nmae=vm['val_nmae'], epoch=epoch, state={k: v.detach().clone() for k, v in head.state_dict().items()})
        vm.update(epoch=epoch, train_loss=total / n, best_val_nmae=best['val_nmae'])
        val_log.append(vm)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info('epoch %d train mse %.5f val nmae %.4f%% (best %.4f%% at %d)', epoch, total / n,
                        100 * vm['val_nmae'], 100 * best['val_nmae'], best['epoch'])

    head.load_state_dict(best['state'])
    train_m = metrics(predict(head, E_train, tr[0], tr[1]), tr[2], 'final_train_')
    pred_test = predict(head, E_test, te_s, te_t)
    test_m = metrics(pred_test, te_l, 'test_')
    final = {**train_m, **test_m, 'best_epoch': best['epoch'], 'best_val_nmae': best['val_nmae'],
             'wall_seconds': round(time.time() - t0, 1), 'stage1_weights': weights, 'args': vars(args)}
    (out / f'{args.run}.json').write_text(json.dumps({'final': final, 'val': val_log}, indent=1))
    torch.save(head.state_dict(), out / f'{args.run}_head.pt')
    np.save(out / 'preds' / f'{args.run}__{args.test_name}.npy', pred_test.cpu().numpy().astype(np.float16))
    m = {k[5:]: v for k, v in test_m.items()}
    m.update(run=args.run, model=args.model, test=args.test_name, n_pairs=int((te_l > 0).sum()))
    (out / 'metrics' / f'{args.run}__{args.test_name}.json').write_text(json.dumps(m, indent=1))
    logger.info('DONE %s: test nmae %.4f%% acc2 %.2f%% bias %+.4f km, train nmae %.4f%%, best epoch %d, %.0fs',
                args.run, 100 * test_m['test_nmae'], 100 * test_m['test_acc_2pct'], test_m['test_bias'],
                100 * train_m['final_train_nmae'], best['epoch'], time.time() - t0)


if __name__ == '__main__':
    main()
