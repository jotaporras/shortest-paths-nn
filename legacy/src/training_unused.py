"""Unused helpers from refactor_training.py (moved on the asilomar branch)."""
import torch
from torch_geometric.data import Data

from spnn.data import SingleGraphShortestPathDataset


class TerrainPatchesDataset(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self.x.size(0)
        if key == 'tar':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def debug_dataset(data, n=100):
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'][:n])
    tars = torch.tensor(data['tars'][:n])
    lengths = torch.tensor(data['lengths'][:n], dtype=torch.float)
    node_features = torch.tensor(data['node_features'], dtype=torch.float)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index
