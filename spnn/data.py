"""Shortest-path pair datasets on a single graph."""
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

def split_dataset_for_validation(dataset, val_fraction=0.005, seed=42):
    """Split a random validation subset off a dataset.
    
    Args:
        dataset: A SingleGraphShortestPathDataset instance
        val_fraction: Fraction of data to use for validation (default 0.005 = 0.5%)
        seed: Random seed for reproducibility
        
    Returns:
        val_dataset: Subset for validation
        rest_dataset: Subset with the remaining pairs
    """
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[:n_val]
    rest_indices = indices[n_val:]
    
    val_dataset = Subset(dataset, val_indices)
    rest_dataset = Subset(dataset, rest_indices)
    return val_dataset, rest_dataset
