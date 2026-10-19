"""Shortest-path pair datasets on a single graph."""
import math

import torch
from torch.utils.data import Dataset, Subset


class SingleGraphShortestPathDataset(Dataset):
    def __init__(self, sources, targets, lengths):
        self.sources = sources
        self.targets = targets
        self.lengths = lengths

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.lengths[idx]

def _pair_tensors(dataset):
    """(sources, targets, lengths) of a SingleGraphShortestPathDataset or a Subset of one."""
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        base = dataset.dataset
        return base.sources[indices], base.targets[indices], base.lengths[indices]
    return dataset.sources, dataset.targets, dataset.lengths


class PairBatchLoader:
    """Batches of (srcs, tars, lengths) sliced from tensors, with no per-pair collate.

    It yields the same batches as DataLoader(dataset, batch_size, shuffle) with the
    default sampler, because the shuffle seed is drawn from the global torch RNG in
    the same way as RandomSampler. DataLoader collates one pair at a time, which
    costs about 0.2 s per batch of 16,384 pairs.
    """

    def __init__(self, dataset, batch_size, shuffle=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        tensors = _pair_tensors(dataset)
        self.tensors = tuple(t.to(device) for t in tensors) if device is not None else tensors

    def __len__(self):
        return math.ceil(len(self.tensors[0]) / self.batch_size)

    def __iter__(self):
        n = len(self.tensors[0])
        order = None
        # DataLoader draws a worker base seed from the global RNG on every iter();
        # draw it too, so the RNG stream (and later random ops) stays the same.
        torch.empty((), dtype=torch.int64).random_()
        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            order = torch.randperm(n, generator=generator).to(self.tensors[0].device)
        for start in range(0, n, self.batch_size):
            if order is not None:
                idx = order[start:start + self.batch_size]
                yield tuple(t[idx] for t in self.tensors)
            else:
                yield tuple(t[start:start + self.batch_size] for t in self.tensors)


def materialize_single_graph_batch(dataset, device):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        sources = base_dataset.sources[indices]
        targets = base_dataset.targets[indices]
        lengths = base_dataset.lengths[indices]
    else:
        sources = dataset.sources
        targets = dataset.targets
        lengths = dataset.lengths

    return (
        sources.to(device=device, non_blocking=True).long(),
        targets.to(device=device, non_blocking=True).long(),
        lengths.to(device=device, non_blocking=True),
    )

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'], dtype=torch.int)
    tars = torch.tensor(data['tars'],  dtype=torch.int)
    lengths = torch.tensor(data['lengths'], dtype=torch.float)
    node_features = torch.tensor(data['node_features'], dtype=torch.float)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

def split_dataset_for_validation(dataset, val_fraction=0.005, seed=42, n_val=None):
    """Split a random validation subset off a dataset.

    Args:
        dataset: A SingleGraphShortestPathDataset instance
        val_fraction: Fraction of data to use for validation (default 0.005 = 0.5%)
        seed: Random seed for reproducibility
        n_val: Number of validation pairs; overrides val_fraction when given
        
    Returns:
        val_dataset: Subset for validation
        rest_dataset: Subset with the remaining pairs
    """
    n = len(dataset)
    if n_val is None:
        n_val = max(1, int(n * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[:n_val]
    rest_indices = indices[n_val:]
    
    val_dataset = Subset(dataset, val_indices)
    rest_dataset = Subset(dataset, rest_indices)
    return val_dataset, rest_dataset
