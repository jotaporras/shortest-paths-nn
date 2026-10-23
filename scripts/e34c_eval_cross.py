"""e34c: evaluate every e34 model on every scheme's test set (90 models x 5 test sets).

For each run <runs>/norway_<scheme>_s<stride>_<model> the final_model.pt (best-validation checkpoint) is
loaded, the test graph is embedded once, and the prediction for every test pair is the L1 distance between
the two node embeddings (the siamese p=1 head of e34). Metrics follow spnn.metrics.compute_metrics: pairs
with zero true length are left out of nmae and the acc fractions.

Usage: python scripts/e34c_eval_cross.py --gpu 0 --models sgt2 [--schemes ...] [--tests ...]
Output: <out>/metrics/<run>__<test>.json and <out>/preds/<run>__<test>.npy (float16 predictions in km,
in test-pair order). Existing outputs are skipped. On the grid test set the nmae is compared with the
e34 run JSON as a check of the evaluation code.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spnn.graphs.spectral import laplacian_pe  # noqa: E402
from spnn.models.factory import configure_embedding_module  # noqa: E402

MODEL_SPEC = {
    'mlp': ('MLP', 'configs/e31-mlp.yml'),
    'tagconv': ('TAGConv', 'configs/e31-tagconv.yml'),
    'sgt2': ('SparseGT', 'configs/e32-sparse-gt-k2.yml'),
    # e35 Laplacian-PE model; LAPPE_K (env) must equal the k the runs were trained with.
    'sgt2lap': ('SparseGT', 'configs/e35-sparse-gt-k2-lappe.yml'),
}
LAPPE_K = int(os.environ.get('LAPPE_K', 0)) or None
SCHEMES = ['grid', 'uniform', 'poisson-disk', 'slope-adaptive', 'quadtree']
STRIDES = [32, 24, 16, 12, 8, 4]
TESTS = {'grid': '/home/jporras/data/spnn/full_test-004.npz',
         **{s: f'/home/jporras/data/spnn/e34c/norway_{s}_test-004.npz' for s in SCHEMES[1:]}}


def load_model(run_dir, model, device):
    layer_type, cfg_path = MODEL_SPEC[model]
    cfg = next(iter(yaml.safe_load(open(REPO / cfg_path)).values()))
    if layer_type == 'SparseGT':
        sgt = cfg['gnn'].setdefault('sparse_gt', {})
        if model == 'sgt2lap':
            sgt['embedding_mode'] = 'lappe'
            if LAPPE_K is not None:
                sgt['lappe_k'] = LAPPE_K
        else:
            sgt['embedding_mode'] = 'data'
    net = configure_embedding_module(cfg['gnn'], layer_type, edge_dim=1, new=True)
    weights = next(run_dir.glob('models/**/final_model.pt'))
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    return net.to(device).eval(), layer_type


@torch.no_grad()
def embed(net, layer_type, graph, device):
    x = torch.tensor(graph['node_features'], dtype=torch.float, device=device)
    if layer_type == 'MLP':
        return net(x)
    if getattr(net, 'embedding_mode', None) == 'lappe':
        V, _ = laplacian_pe(len(x), graph['edge_index'], net.lappe_k)   # cached, same file as training
        x = torch.cat([x, torch.from_numpy(V).to(device)], dim=1)
    ei = torch.tensor(graph['edge_index'], dtype=torch.long, device=device)
    w = torch.tensor(graph['distances'], dtype=torch.float, device=device).unsqueeze(-1)
    return net(x, ei, edge_attr=w)


@torch.no_grad()
def predict(E, srcs, tars, device, chunk=500_000):
    out = np.empty(len(srcs), dtype=np.float32)
    s = torch.tensor(srcs, device=device)
    t = torch.tensor(tars, device=device)
    for i in range(0, len(srcs), chunk):
        out[i:i + chunk] = (E[s[i:i + chunk]] - E[t[i:i + chunk]]).abs().sum(1).float().cpu().numpy()
    return out


def metrics(pred, true, srcs):
    nz = true > 0
    rel = np.abs(pred[nz] - true[nz]) / true[nz]
    per_src = {int(s): float(rel[srcs[nz] == s].mean()) for s in np.unique(srcs)}
    return dict(nmae=float(rel.mean()), acc_1pct=float((rel < 0.01).mean()), acc_2pct=float((rel < 0.02).mean()),
                acc_5pct=float((rel < 0.05).mean()), mae=float(np.abs(pred[nz] - true[nz]).mean()),
                mse=float(((pred[nz] - true[nz]) ** 2).mean()), bias=float((pred[nz] - true[nz]).mean()),
                n_pairs=int(nz.sum()), nmae_per_source=per_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--models', nargs='+', default=list(MODEL_SPEC))
    ap.add_argument('--schemes', nargs='+', default=SCHEMES, help='training schemes of the models')
    ap.add_argument('--strides', nargs='+', type=int, default=STRIDES)
    ap.add_argument('--tests', nargs='+', default=list(TESTS))
    ap.add_argument('--runs', default='/home/jporras/runs/e34')
    ap.add_argument('--out', default='/home/jporras/runs/e34c')
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'
    runs, out = Path(args.runs), Path(args.out)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    (out / 'preds').mkdir(exist_ok=True)

    for test_name in args.tests:
        graph = dict(np.load(TESTS[test_name]))
        srcs, tars, true = graph['srcs'], graph['tars'], graph['lengths']
        print(f'== test {test_name}: {len(graph["node_features"])} nodes, {len(true)} pairs', flush=True)
        for model in args.models:
            for scheme in args.schemes:
                for stride in args.strides:
                    run = f'norway_{scheme}_s{stride:02d}_{model}'
                    mpath = out / 'metrics' / f'{run}__{test_name}.json'
                    if mpath.exists() or not (runs / f'{run}.json').exists():
                        continue
                    t0 = time.time()
                    net, layer_type = load_model(runs / run, model, device)
                    E = embed(net, layer_type, graph, device)
                    pred = predict(E, srcs, tars, device)
                    m = metrics(pred, true, srcs)
                    m.update(run=run, train_scheme=scheme, train_stride=stride, model=model, test=test_name,
                             seconds=round(time.time() - t0, 1))
                    note = ''
                    if test_name == 'grid':
                        ref = json.load(open(runs / f'{run}.json'))['final']['test_nmae']
                        m['e34_test_nmae'] = ref
                        note = f' (e34 JSON {ref * 100:.3f}, diff {(m["nmae"] - ref) * 100:+.3f})'
                    np.save(out / 'preds' / f'{run}__{test_name}.npy', pred.astype(np.float16))
                    mpath.write_text(json.dumps(m, indent=1))
                    print(f'{run} on {test_name}: nmae {m["nmae"] * 100:.3f}% acc2 {m["acc_2pct"] * 100:.1f}% '
                          f'{m["seconds"]}s{note}', flush=True)
                    del net, E
                    torch.cuda.empty_cache()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
