"""
Own implementation of structure-based sparse GT with k-hop attention windows. Copied from manifold-transformers repo.

Implementation might have some overlap with exphormer.py, and we might merge later.
"""

# filter the annoying warnings about sparse CSR ops.
import warnings
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import utils
from torch_geometric.data import Data

from manifold_transformers.exphormer_optimized import create_expander_graph_cached
warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class NaiveKHopGraphAttention(nn.Module):
    """Dense edge-index based k-hop graph attention with multi-head projections.

    Note: Supports separate attn_dropout parameter for API parity, though
    dense naive variant does not drop attention logits explicitly.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hops: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_hops = num_hops

        assert self.d_model % self.num_heads == 0

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        # Keep for API parity; currently unused in naive attention path.
        self.attn_dropout_p = attn_dropout

    def forward(
        self, data: Data, attn_window: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes attention over edges provided by a COO window.

        Args:
            data: Graph data to attend over.
            attn_window: Edge indices of shape (2, E).

        Returns:
            Projected representation of shape (N, d_model).
        """
        if isinstance(data, Data):
            X = data.x
            if attn_window is None:
                attn_window = data.edge_index
        elif isinstance(data, List):
            raise ValueError(
                "List of graphs not yet supported. Pass each graph individually."
            )

        N, _ = X.shape
        _, num_edges = attn_window.shape

        src_idx = attn_window[0, :]
        dst_idx = attn_window[1, :]

        QX = self.Q(X)
        KX = self.K(X)
        VX = self.V(X)

        QX_src = QX[src_idx, :]
        KX_dst = KX[dst_idx, :]
        VX_dst = VX[dst_idx, :]

        head_dim = self.d_model // self.num_heads
        QX_src = QX_src.view(num_edges, self.num_heads, head_dim)
        KX_dst = KX_dst.view(num_edges, self.num_heads, head_dim)
        VX_dst = VX_dst.view(num_edges, self.num_heads, head_dim)

        unnormalized_weights = (QX_src * KX_dst).sum(dim=-1)
        unnormalized_weights = unnormalized_weights / (head_dim**0.5)

        attn_weights = utils.softmax(
            src=unnormalized_weights, index=src_idx, dim=0, num_nodes=N
        )

        weighted_values = attn_weights.unsqueeze(-1) * VX_dst
        weighted_values = weighted_values.view(num_edges, self.d_model)

        attn_out = utils.scatter(
            src=weighted_values, index=src_idx, dim=0, reduce="sum", dim_size=N
        )

        attn_out = self.ln1(attn_out)
        attn_out = self.dropout(attn_out)

        trf_layer_out = self.out_proj(attn_out)
        trf_layer_out = self.ln2(trf_layer_out)
        trf_layer_out = self.dropout(trf_layer_out)

        return trf_layer_out


def _mha_sparse_attention(
    QX: torch.Tensor,
    KX: torch.Tensor,
    VX: torch.Tensor,
    A_csr: torch.Tensor,
    dropout=None,
) -> torch.Tensor:
    """Sparse multi-head attention using CSR adjacency with per-head sampled_addmm.

    Args:
        QX: Query tensor of shape (H, N, Fh).
        KX: Key tensor of shape (H, N, Fh).
        VX: Value tensor of shape (H, N, Fh).
        A_csr: Sparse CSR adjacency of shape (N, N).
        dropout: SparseCSRDropout module to apply to the unnormalized weights.

    Returns:
        Aggregated tensor of shape (N, H*Fh).
    """
    H, N, F_head = QX.shape
    outputs: list[torch.Tensor] = []

    # HACK: Ideally I would do this by batching//all heads & batches in one pass, but
    # Backprop complains, so we have to do it one head at a time.
    for i in range(H):
        KX_T_i = KX[i].transpose(0, 1)
        unnormalized = torch.sparse.sampled_addmm(
            input=A_csr, mat1=QX[i], mat2=KX_T_i, beta=0.0
        )

        if dropout is not None:
            unnormalized = dropout(unnormalized)

        crow = unnormalized.crow_indices()
        col = unnormalized.col_indices()
        row_counts = crow[1:] - crow[:-1]
        row_index = torch.arange(N, device=QX.device).repeat_interleave(row_counts)

        scores = unnormalized.values() / (F_head**0.5)
        probs = utils.softmax(src=scores, index=row_index, dim=0, num_nodes=N)

        B_h = torch.sparse_csr_tensor(
            crow_indices=crow, col_indices=col, values=probs, size=(N, N)
        )

        out_h = torch.sparse.mm(B_h, VX[i])
        outputs.append(out_h)

    weighted_values = torch.stack(outputs, dim=0)
    attn_out = weighted_values.permute(1, 0, 2).reshape(N, H * F_head)
    torch.utils.checkpoint.checkpoint(lambda: attn_out)

    return attn_out


class SparseCSRDropout(nn.Module):
    def __init__(self, p: float = 0.5, set_to_neg_inf: bool = False):
        """
        Args:
            p: Dropout probability.
            set_to_neg_inf: If True, set the dropped out values to -inf for logit-space dropout.
        """
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.set_to_neg_inf = set_to_neg_inf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse_csr:
            raise TypeError("Input must be a sparse CSR tensor")
        if not self.training or self.p == 0.0:
            return x

        keep = 1.0 - self.p
        v = x.values()
        if self.set_to_neg_inf:
            crow = x.crow_indices()
            N = x.size(0)
            row_counts = crow[1:] - crow[:-1]
            row_index = torch.arange(N, device=v.device).repeat_interleave(row_counts)
            mask = torch.rand_like(v) < keep
            kept_per_row = utils.scatter(
                mask.to(v.dtype), index=row_index, dim=0, reduce="sum", dim_size=N
            )
            need_force = (kept_per_row == 0) & (row_counts > 0)
            rand_vals = torch.rand_like(v)
            max_per_row = utils.scatter(
                rand_vals, row_index, dim=0, reduce="max", dim_size=N
            )
            is_force = need_force[row_index] & (rand_vals == max_per_row[row_index])
            mask = mask | is_force
            new_values = v.masked_fill(~mask, float("-inf"))
        else:
            if keep == 0.0:
                new_values = torch.zeros_like(v)
            else:
                mask = (torch.rand_like(v) < keep).to(v.dtype)
                new_values = (v * mask) / keep

        return torch.sparse_csr_tensor(
            crow_indices=x.crow_indices(),
            col_indices=x.col_indices(),
            values=new_values,
            size=x.size(),
            dtype=x.dtype,
            device=x.device,
        )


class SparseKHopGraphAttention(nn.Module):
    """Sparse CSR-based k-hop graph attention with multi-head projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hops: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_hops = num_hops

        assert self.d_model % self.num_heads == 0

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        # Attention dropout on sparse logits (separate from layer dropout)
        self.attn_dropout = SparseCSRDropout(p=attn_dropout, set_to_neg_inf=True)

        d_ff=d_model*3
        self.out_proj_1 = nn.Linear(d_model, d_ff)
        self.out_proj_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, data: Data, attn_window: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes attention using CSR adjacency derived from the window.

        Args:
            data: Graph data to attend over.
            attn_window: Edge indices of shape (2, E).

        Returns:
            Projected representation of shape (N, d_model).
        """
        if isinstance(data, Data):
            X = data.x
            if attn_window is None:
                attn_window = data.edge_index
        elif isinstance(data, List):
            raise ValueError(
                "List of graphs not yet supported. Pass each graph individually."
            )

        N, _ = X.shape

        QX = self.Q(X)
        KX = self.K(X)
        VX = self.V(X)

        head_dim = self.d_model // self.num_heads
        QH = QX.view(N, self.num_heads, head_dim).permute(1, 0, 2)
        KH = KX.view(N, self.num_heads, head_dim).permute(1, 0, 2)
        VH = VX.view(N, self.num_heads, head_dim).permute(1, 0, 2)

        values = torch.ones(attn_window.shape[1], device=X.device)
        A = torch.sparse_coo_tensor(
            indices=attn_window, values=values, size=(N, N)
        ).coalesce()
        A_csr = A.to_sparse_csr()

        attn_out = _mha_sparse_attention(QH, KH, VH, A_csr, dropout=self.attn_dropout)

        attn_out = self.ln1(attn_out)
        attn_out = self.dropout(attn_out)

        trf_layer_out = self.out_proj_1(attn_out)
        trf_layer_out = F.relu(trf_layer_out)
        trf_layer_out = self.dropout(trf_layer_out)
        trf_layer_out = self.out_proj_2(trf_layer_out)
        trf_layer_out = self.ln2(trf_layer_out)
        trf_layer_out = self.dropout(trf_layer_out)

        return trf_layer_out


class KHopGTModel(nn.Module):
    """Stacked k-hop attention encoder with selectable attention algorithm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hops: int,
        num_layers: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.0,
        attn_algorithm: str = "sparse",
        random_graph: str = "Random-d2",
        random_graph_degree: int = 5,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding_dim = d_model
        self.random_graph = random_graph
        self.random_graph_degree = random_graph_degree

        if self.random_graph not in {"Random-d", "Random-d2", "None"}:
            raise ValueError("random_graph must be 'Random-d', 'Random-d2', or 'None'")
        if self.random_graph != "None" and self.random_graph_degree <= 0:
            raise ValueError("random_graph_degree must be positive when a random graph is selected")

        if attn_algorithm == "naive":
            layer_cls: type[nn.Module] = NaiveKHopGraphAttention
        elif attn_algorithm == "sparse":
            layer_cls = SparseKHopGraphAttention
        else:
            raise ValueError("attn_algorithm must be 'naive' or 'sparse'")

        self.attention_layers = nn.ModuleList(
            [
                layer_cls(
                    d_model,
                    num_heads,
                    num_hops,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.nonlinearity = nn.ReLU()

    def _construct_augmented_edge_index(
        self,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Construct the augmented edge index for the random graph.
        Args:
            num_nodes: Number of nodes in the graph.
            dtype: Data type of the edge index.
            device: Device of the edge index.
            layer_idx: Index of the layer (0-indexed).

        Returns:
            Augmented edge index of shape (2, E).
            If no random graph is selected, return an empty tensor.
        """
        if self.random_graph == "None":
            return torch.empty(
                (2, 0),
                dtype=dtype,
                device=device,
            )
        random_edges = create_expander_graph_cached(
            num_nodes=num_nodes,
            degree=self.random_graph_degree,
            algorithm=self.random_graph,
            device=device,
            layer_idx=layer_idx,
        )
        if random_edges.numel() == 0:
            return random_edges
        random_edges = utils.to_undirected(random_edges, num_nodes=num_nodes)
        values = torch.ones(random_edges.shape[1], device=random_edges.device)
        sparse = torch.sparse_coo_tensor(
            indices=random_edges,
            values=values,
            size=(num_nodes, num_nodes),
        ).coalesce()
        return sparse.indices()

    def _get_attn_window(self, data: Data, num_hops):
        """
        Given graph data, construct the attention window for the given number of hops,
        and return the corresponding COO edge_index.
        """
        if num_hops == 0:
            return torch.empty((2, 0), dtype=data.edge_index.dtype, device=data.edge_index.device)
        if num_hops <= 1:
            return data.edge_index

        # Create diagonal indices for self-loops
        self_loops = (
            torch.arange(data.x.shape[0], device=data.edge_index.device)
            .unsqueeze(0)
            .repeat(2, 1)
        )
        edge_index = torch.cat([self_loops, data.edge_index], dim=1)  # data.edge_index
        N = data.x.shape[0]

        adjacency_values = torch.ones(edge_index.shape[1], device=edge_index.device)
        A_coo = torch.sparse_coo_tensor(
            indices=edge_index, values=adjacency_values, size=(N, N)
        )
        A_csr = A_coo.to_sparse_csr()

        k_hop_adjacency = A_csr
        for k in range(num_hops - 1):
            k_hop_adjacency = k_hop_adjacency @ A_csr

        attn_window = k_hop_adjacency.to_sparse_coo().indices()

        return attn_window

    def forward(self, data: Data):
        ##TODO this should be handled at the data level.
        num_nodes = data.x.shape[0]
        dtype = data.edge_index.dtype
        device = data.edge_index.device

        attn_window = self._get_attn_window(data, self.num_hops)

        data_l = data
        for i, attn_layer in enumerate(self.attention_layers[:-1]):
            augmented_edge_index = self._construct_augmented_edge_index(num_nodes, dtype, device, layer_idx=i)
            
            if augmented_edge_index.numel() > 0:
                combined_edges = torch.cat([attn_window, augmented_edge_index], dim=1)
                values = torch.ones(combined_edges.shape[1], device=device)
                attn_window_expanded = torch.sparse_coo_tensor(
                    indices=combined_edges,
                    values=values,
                    size=(num_nodes, num_nodes),
                ).coalesce().indices()
            else:
                attn_window_expanded = attn_window

            attn_out = attn_layer(data_l, attn_window=attn_window_expanded)
            X = data_l.x + attn_out
            X = self.nonlinearity(X)
            data_l = Data(x=X, edge_index=data_l.edge_index)

        augmented_edge_index = self._construct_augmented_edge_index(num_nodes, dtype, device, layer_idx=len(self.attention_layers) - 1)
        
        if augmented_edge_index.numel() > 0:
            combined_edges = torch.cat([attn_window, augmented_edge_index], dim=1)
            values = torch.ones(combined_edges.shape[1], device=device)
            attn_window_expanded = torch.sparse_coo_tensor(
                indices=combined_edges,
                values=values,
                size=(num_nodes, num_nodes),
            ).coalesce().indices()
        else:
            attn_window_expanded = attn_window

        attn_out = self.attention_layers[-1](data_l, attn_window=attn_window_expanded)
        X = data_l.x + attn_out

        return X
