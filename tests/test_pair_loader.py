"""PairBatchLoader must yield the same batches as the DataLoader it replaces."""
import torch
from torch.utils.data import DataLoader

from spnn.data import PairBatchLoader, SingleGraphShortestPathDataset, split_dataset_for_validation


def _dataset(n=1000):
    g = torch.Generator().manual_seed(1)
    srcs = torch.randint(0, 50, (n,), generator=g, dtype=torch.int)
    tars = torch.randint(0, 50, (n,), generator=g, dtype=torch.int)
    lengths = torch.rand(n, generator=g)
    return SingleGraphShortestPathDataset(srcs, tars, lengths)


def _batches(loader, epochs, seed):
    torch.manual_seed(seed)
    epochs_out = [[tuple(t.clone() for t in b) for b in loader] for _ in range(epochs)]
    # The global RNG must end in the same state, so later random ops are unchanged.
    epochs_out.append([(torch.rand(4),)])
    return epochs_out


def test_same_batches_as_dataloader():
    val, train = split_dataset_for_validation(_dataset(), seed=42, n_val=100)
    assert len(val) == 100 and len(train) == 900
    for subset, shuffle in ((train, True), (val, False)):
        ref = _batches(DataLoader(subset, batch_size=64, shuffle=shuffle), 2, seed=3)
        new = _batches(PairBatchLoader(subset, 64, shuffle=shuffle), 2, seed=3)
        assert len(PairBatchLoader(subset, 64)) == len(ref[0])
        for ref_epoch, new_epoch in zip(ref, new):
            assert len(ref_epoch) == len(new_epoch)
            for rb, nb in zip(ref_epoch, new_epoch):
                for rt, nt in zip(rb, nb):
                    assert rt.dtype == nt.dtype
                    assert torch.equal(rt, nt)
