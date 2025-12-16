"""
Custom models for Sparse GT + RPEARL integration.

This module combines:
- Sparse GT: K-hop sparse attention backbone
- RPEARL: Random GNN positional encodings

Adapted from manifold-transformers repo for terrain graph experiments.
"""

import warnings
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv
from torch_geometric import utils

# Filter sparse CSR warnings
warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


# =============================================================================
# GCN for Positional Encodings (TAGConv-based)
# =============================================================================

class GCN(nn.Module):
    """
    A simple TAG-based graph convolutional backbone that returns node embeddings.

    Args:
        in_channels (int): Number of input features per node
        hidden_channels (int): Number of hidden features per node
        num_layers (int): Number of convolution layers (must be >= 2)
        skip_connection (bool): Whether to use skip connections
        k (int): Order of TAGConv polynomial (K)

    Returns:
        torch.Tensor: Node embeddings of shape [num_nodes, hidden_channels]
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        skip_connection=False,
        dropout=0.5,
        k: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN requires at least 2 layers.")

        self.convs = nn.ModuleList()
        self.k = k
        self.convs.append(TAGConv(in_channels, hidden_channels, K=self.k))
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.skip_connection = skip_connection
        self.embedding_dim = hidden_channels

    def forward(self, data: Data):
        """
        Forward pass through the GCN.

        Args:
            data (Data): PyTorch Geometric Data object containing node features (x)
                        and edge indices (edge_index)

        Returns:
            torch.Tensor: Output node embeddings [num_nodes, hidden_channels]
        """
        x0, edge_index = data.x, data.edge_index
        x_prev = x0
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x_prev, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            if self.skip_connection and i > 0:
                x = x + x_prev
            x_prev = x
        x = self.convs[-1](x, edge_index)
        return x


# =============================================================================
# RPEARL: Random GNN Positional Encodings
# =============================================================================

class RandomGNNPositionalEncodings(nn.Module):
    """
    Random graph positional encodings (R-PEARL).

    Args:
        pe_hidden_channels (int): Hidden dimension for the GCN
        pe_num_layers (int): Number of layers in the GCN
        d_model (int): Output dimension
        num_samples (int): Number of random samples (M) to use
    """

    def __init__(
        self, pe_hidden_channels, pe_num_layers, d_model, num_samples=30, dropout=0.1
    ):
        super().__init__()
        # Create a GCN that takes 1-dimensional random features
        self.pe_gcn = GCN(
            1, pe_hidden_channels, pe_num_layers, skip_connection=True, dropout=dropout
        )
        # Add a final projection to ensure output is d_model dimensions
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.M = num_samples

    def forward(self, data):
        X, edge_index = data.x, data.edge_index

        # Generate random node embeddings for positional encoding
        num_nodes = X.shape[0]
        Q = torch.randn((num_nodes, self.M), device=X.device, dtype=X.dtype)

        # Process random embeddings individually through GCN
        P_m = []

        for i in range(self.M):
            def _pe_block(q_col, edge_idx, _dummy):
                q_data = Data(x=q_col.unsqueeze(-1), edge_index=edge_idx)
                pe_local = self.pe_gcn(q_data)
                pe_local = self.dropout(pe_local)
                pe_local = self.output_projection(pe_local)
                return pe_local

            dummy = Q.new_ones(1, requires_grad=True)
            pe = checkpoint(_pe_block, Q[:, i], edge_index, dummy, use_reentrant=False)
            P_m.append(pe)

        P = torch.stack(P_m, dim=-1)
        pooled_pe = P.mean(dim=-1)
        pooled_pe = self.batch_norm(pooled_pe)
        return pooled_pe


# =============================================================================
# Sparse Attention Components
# =============================================================================

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

        d_ff = d_model * 3
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

        values = torch.ones(attn_window.shape[1], device=X.device, dtype=X.dtype)
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


# =============================================================================
# Sparse GT Backbone (Simplified - No Expander Graphs)
# =============================================================================

class KHopGTModel(nn.Module):
    """Stacked k-hop attention encoder (simplified, no expander graphs)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hops: int,
        num_layers: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding_dim = d_model

        self.attention_layers = nn.ModuleList(
            [
                SparseKHopGraphAttention(
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
        edge_index = torch.cat([self_loops, data.edge_index], dim=1)
        N = data.x.shape[0]

        adjacency_values = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=data.x.dtype)
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
        attn_window = self._get_attn_window(data, self.num_hops)

        data_l = data
        for attn_layer in self.attention_layers[:-1]:
            attn_out = attn_layer(data_l, attn_window=attn_window)
            X = data_l.x + attn_out
            X = self.nonlinearity(X)
            data_l = Data(x=X, edge_index=data_l.edge_index)

        attn_out = self.attention_layers[-1](data_l, attn_window=attn_window)
        X = data_l.x + attn_out

        return X


# =============================================================================
# Combined Model: Sparse GT with RPEARL
# =============================================================================

class SparseGTWithRPEARL(nn.Module):
    """
    Combined Sparse GT backbone with RPEARL positional encodings.
    
    Follows the simple interface expected by terrain graph training:
    forward(x, edge_index, edge_attr=None, batch=None) -> node_embeddings
    
    Args:
        input_dim (int): Input node feature dimension
        hidden_dim (int): Hidden dimension (d_model for attention)
        output_dim (int): Output embedding dimension
        num_layers (int): Number of sparse attention layers
        num_heads (int): Number of attention heads
        num_hops (int): K-hop neighborhood size for attention window
        rpearl_samples (int): Number of random samples M for RPEARL
        rpearl_num_layers (int): Number of GCN layers in RPEARL
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        num_hops: int = 2,
        rpearl_samples: int = 30,
        rpearl_num_layers: int = 3,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hops = num_hops
        self.rpearl_samples = rpearl_samples
        self.rpearl_num_layers = rpearl_num_layers
        self.dropout_p = dropout
        self.attn_dropout_p = attn_dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # RPEARL positional encodings
        self.positional_encoding = RandomGNNPositionalEncodings(
            pe_hidden_channels=hidden_dim,
            pe_num_layers=rpearl_num_layers,
            d_model=hidden_dim,
            num_samples=rpearl_samples,
            dropout=dropout,
        )
        
        # Sparse GT backbone
        self.backbone = KHopGTModel(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_hops=num_hops,
            num_layers=num_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # For compatibility with training code
        self.hidden_channels = output_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass compatible with terrain graph training interface.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (unused, for interface compatibility)
            batch: Batch indices (unused, for interface compatibility)
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Ensure float32 dtype (dataset may provide float64)
        x = x.float()
        
        # Create Data object for internal processing
        data = Data(x=x, edge_index=edge_index)
        
        # Project input to hidden dimension
        x_proj = self.input_projection(x)
        
        # Compute RPEARL positional encodings
        pe = self.positional_encoding(data)
        
        # Add positional encodings to projected input
        x_with_pe = x_proj + pe
        
        # Create data with PE-augmented features
        data_with_pe = Data(x=x_with_pe, edge_index=edge_index)
        
        # Run through Sparse GT backbone
        embeddings = self.backbone(data_with_pe)
        
        # Add residual connection from PE-augmented input
        embeddings = x_with_pe + embeddings
        
        # Project to output dimension
        output = self.output_projection(embeddings)
        
        return output
    
    def get_config_for_wandb(self):
        """Return configuration dict for wandb logging with sparse_gt_ prefix."""
        return {
            "sparse_gt_input_dim": self.input_dim,
            "sparse_gt_hidden_dim": self.hidden_dim,
            "sparse_gt_output_dim": self.output_dim,
            "sparse_gt_num_layers": self.num_layers,
            "sparse_gt_num_heads": self.num_heads,
            "sparse_gt_num_hops": self.num_hops,
            "sparse_gt_rpearl_samples": self.rpearl_samples,
            "sparse_gt_rpearl_num_layers": self.rpearl_num_layers,
            "sparse_gt_dropout": self.dropout_p,
            "sparse_gt_attn_dropout": self.attn_dropout_p,
        }

