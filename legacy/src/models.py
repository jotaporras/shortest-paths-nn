"""
Imported from manifold-transformers on Dec 15.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.nn import (
    TAGConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_adj, to_undirected

from .exphormer_optimized import ExphormerOptimized
from .linkx import LINKXBackbone
from .sparse_gt import KHopGTModel


class GraphModel(nn.Module):
    """A modular graph model for different types of Graph ML tasks."""

    def __init__(
        self,
        positional_encoding: Optional[nn.Module],
        backbone: nn.Module,
        head: nn.Module,
        input_projection: Optional[nn.Module] = None,
        positional_encoding_dim: int = None,
    ):
        super().__init__()

        if positional_encoding_dim is None:
            raise ValueError("positional_encoding_dim must always be specified.")

        if positional_encoding:
            self.positional_encoding = positional_encoding
        else:
            self.positional_encoding = GraphNullOp(dim=positional_encoding_dim)
        self.backbone = backbone
        self.head = head
        if input_projection is not None:
            self.input_projection = input_projection
        else:
            self.input_projection = nn.Identity()

    def forward(self, data):
        x_d = data.x
        pe = self.positional_encoding(data)
        x_d_proj = self.input_projection(x_d)
        x_p = x_d_proj + pe
        data_w_pe = Data(
            x=x_p, edge_index=data.edge_index, batch=getattr(data, "batch", None)
        )

        embeddings = self.backbone(data_w_pe)

        # Only add residual if dimensions match
        if pe.shape[-1] == embeddings.shape[-1]:
            embeddings = x_p + embeddings

        data_w_embeddings = Data(
            x=embeddings, edge_index=data.edge_index, batch=getattr(data, "batch", None)
        )

        model_output = self.head(embeddings, data_w_embeddings)

        return model_output


class GraphNullOp(nn.Module):
    """Utility positional encoding that emits zeros in the desired feature space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, data):
        zeros = torch.zeros(
            data.x.size(0),
            self.dim,
            device=data.x.device,
            dtype=data.x.dtype,
        )
        return zeros


@dataclass
class BackboneComponents:
    """Container returned by backbone factories."""

    model: nn.Module
    input_dim: int
    input_projection: Optional[nn.Module] = None


class CustomSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, attn_dropout=0.25):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.head_dim = self.d_model // self.nhead

        self.QKV = nn.Linear(d_model, d_model * 3)
        self.ff_out = nn.Linear(self.d_model, self.d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, X: torch.tensor) -> torch.tensor:
        # TODO support batches. Right now it's tailored for the PyG single Data object.
        N, C = X.shape

        X = X.unsqueeze(
            0
        )  # induce batch dimension. SDPA infers algorithm from batch size (which is 1 in this single-graph case.)

        projections = self.QKV(X)
        QX, KX, VX = projections.chunk(3, dim=-1)

        # [...,N,H,dim_h]
        QXH = QX.view(-1, N, self.nhead, self.head_dim).transpose(
            1, 2
        )  # [H,N,head_dim] transposed T and self.nhead to match SDPA expectations.
        KXH = KX.view(-1, N, self.nhead, self.head_dim).transpose(1, 2)
        VXH = VX.view(-1, N, self.nhead, self.head_dim).transpose(1, 2)

        # [H,N,head_dim]
        Y = F.scaled_dot_product_attention(
            QXH,
            KXH,
            VXH,
            attn_mask=None,
            dropout_p=self.attn_dropout,
            is_causal=False,
        )
        # Y = Y.transpose(0,1).contiguous().view(N,self.d_model) # no batch impl
        Y = Y.transpose(1, 2).reshape(N, self.d_model)
        Y = self.ff_out(Y)
        Y = self.ff_dropout(Y)
        Y = Y.squeeze(0)  # Remove batch dimension.
        return Y


class CustomTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, attn_dropout=0.25):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = CustomSelfAttention(
            d_model, nhead, dim_feedforward, dropout, attn_dropout
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        X = X + self.attention(self.ln1(X))
        X_attn = X
        X = self.ff2(F.relu(self.ff1(self.ln2(X))))
        X = X_attn + self.dropout_out(X)

        return X


class DenseGraphTransformerBackbone(nn.Module):
    """
    A N^2 dense graph transformer backbone that computes attention pairwise over all nodes.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout=0.1,
        attn_dropout=0.25,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                # nn.TransformerEncoderLayer(
                #     d_model, nhead, dim_feedforward, dropout=dropout
                # )
                CustomTransformerEncoderBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # self.layer_norm = nn.LayerNorm(d_model)
        self.embedding_dim = d_model

    def forward(self, data):
        X = data.x
        for layer in self.encoder_layers:
            X = X + layer(X)
        return X


## Positional Encoding Models
class DataGNNPositionalEncodings(nn.Module):
    """
    Graph positional encodings using the graph's true node features.

    Args:
        pe_hidden_channels (int): Hidden dimension for the GCN
        pe_num_layers (int): Number of layers in the GCN
        d_model (int): Output dimension
    """

    def __init__(
        self, in_features, pe_hidden_channels, pe_num_layers, d_model, dropout=0.1
    ):
        super().__init__()
        # Create a GCN that takes d_model features and outputs d_model features
        self.pe_gcn = GCN(
            in_features,
            pe_hidden_channels,
            pe_num_layers,
            skip_connection=True,
            dropout=dropout,
        )
        # Add a final projection to ensure output is d_model dimensions
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        pe = self.pe_gcn(data)
        pe = self.dropout(pe)
        pe = self.output_projection(pe)
        return pe

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
        # self.layer_norm = nn.LayerNorm(d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.M = num_samples

    def forward(self, data):
        X, edge_index = data.x, data.edge_index

        # Generate random node embeddings for positional encoding
        num_nodes = X.shape[0]
        Q = torch.randn((num_nodes, self.M), device=X.device)

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
            pe = checkpoint(_pe_block, Q[:, i], edge_index, dummy,use_reentrant=False)
            P_m.append(pe)
        # checkpoint

        P = torch.stack(P_m, dim=-1)
        pooled_pe = P.mean(dim=-1)
        # pooled_pe = self.layer_norm(pooled_pe)
        pooled_pe = self.batch_norm(pooled_pe)
        return pooled_pe


class LaplacianPositionalEncodings(nn.Module):
    """
    Laplacian eigenvector positional encodings.

    Args:
        d_model (int): Output dimension for the positional encodings
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, data):
        num_nodes = data.num_nodes

        undirected_edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)
        A = to_dense_adj(undirected_edge_index, max_num_nodes=num_nodes)[0]

        D = A.sum(dim=1)
        L = torch.diag(D) - A
        L = L + 1e-5 * torch.eye(num_nodes, device=L.device)
        L = (L + L.T) / 2

        _, eigenvectors = torch.linalg.eigh(L)

        top_eigenvectors = eigenvectors[:, -self.d_model :]

        return top_eigenvectors


## Backbone Models


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
            # self.norms.append(nn.BatchNorm1d(hidden_channels))
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


class MLPBackbone(nn.Module):
    def __init__(
        self, in_features: int, hidden_dim: int, num_layers: int, dropout: float = 0.5
    ):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)]
        for _ in range(max(0, num_layers - 1)):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.embedding_dim = hidden_dim

    def forward(self, data: Data):
        return self.net(data.x)


class SparseTransformerConGT(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hops: int,
        num_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding_dim = d_model

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        out_channels = d_model // num_heads
        self.attention_layers = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=d_model,
                    out_channels=out_channels,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.nonlinearity = nn.ReLU()

    def _get_attn_window(self, data: Data, num_hops):
        if num_hops <= 1:
            return data.edge_index

        self_loops = (
            torch.arange(data.x.shape[0], device=data.edge_index.device)
            .unsqueeze(0)
            .repeat(2, 1)
        )
        edge_index = torch.cat([self_loops, data.edge_index], dim=1)
        N = data.x.shape[0]

        adjacency_values = torch.ones(edge_index.shape[1], device=edge_index.device)
        A_coo = torch.sparse_coo_tensor(
            indices=edge_index, values=adjacency_values, size=(N, N)
        )
        A_csr = A_coo.to_sparse_csr()

        k_hop_adjacency = A_csr
        for _ in range(num_hops - 1):
            k_hop_adjacency = k_hop_adjacency @ A_csr

        attn_window = k_hop_adjacency.to_sparse_coo().indices()
        return attn_window

    def forward(self, data: Data):
        undirected_edge_index = to_undirected(data.edge_index)
        data_proj = Data(x=data.x, edge_index=undirected_edge_index)

        attn_window = self._get_attn_window(data_proj, self.num_hops)
        data_l = data_proj
        for attn_layer in self.attention_layers[:-1]:
            attn_out = attn_layer(data_l.x, attn_window)
            X = data_l.x + attn_out
            X = self.nonlinearity(X)
            data_l = Data(x=X, edge_index=data_l.edge_index)

        attn_out = self.attention_layers[-1](data_l.x, attn_window)
        X = data_l.x + attn_out
        return X


## Model heads
class NodeClassifier(nn.Module):
    """
    Node classification head that takes node embeddings and outputs class logits.

    Args:
        backbone (nn.Module): The backbone model that generates node embeddings
        num_classes (int): Number of output classes
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, embedding_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings, data=None):
        """
        Forward pass for node classification.

        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            data: Unused for node classification (kept for interface consistency)

        Returns:
            torch.Tensor: Node classification logits [num_nodes, num_classes]
        """
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)
        return logits


class GraphClassifier(nn.Module):
    """
    Graph classification head that takes node embeddings, pools them, and outputs class logits.

    Args:
        embedding_dim (int): The dimension of the node embeddings.
        num_classes (int): Number of output classes
        pooling (str): Pooling method ('mean', 'max', 'sum'). Defaults to 'mean'.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, embedding_dim, num_classes, pooling="mean", dropout=0.1):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings, data):
        """
        Forward pass for graph classification.

        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            data: Graph data object with batch attribute for pooling

        Returns:
            torch.Tensor: Graph classification logits [batch_size, num_classes]
        """

        # Pool node embeddings to get graph-level representation
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(
                embeddings.shape[0], dtype=torch.long, device=embeddings.device
            )
        )

        if self.pooling == "mean":
            graph_embeddings = global_mean_pool(embeddings, batch)
        elif self.pooling == "max":
            graph_embeddings = global_max_pool(embeddings, batch)
        elif self.pooling == "sum":
            graph_embeddings = global_add_pool(embeddings, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        graph_embeddings = self.dropout(graph_embeddings)
        logits = self.classifier(graph_embeddings)
        return logits


class LaplacianGraphTransformer(nn.Module):
    """
    #TODO: This code is legacy, and mixes backbone and positional encodings. 

    A graph transformer backbone using Laplacian eigenvector positional encodings.

    This model:
    1. Computes the graph Laplacian eigenvectors as positional encodings
    2. Adds these encodings to the input node features
    3. Applies transformer encoder layers for self-attention
    4. Returns node embeddings

    Args:
        d_model (int): Dimension of transformer features and node embeddings
        nhead (int): Number of attention heads
        dim_feedforward (int): Dimension of the feedforward network in transformer
        num_layers (int): Number of transformer encoder layers
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=512,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.embedding_dim = d_model

    def forward(self, data: Data):
        """
        Forward pass through the Laplacian Graph Transformer.

        Args:
            data (Data): PyTorch Geometric Data object containing node features (x)
                         and edge indices (edge_index)

        Returns:
            torch.Tensor: Output node embeddings [num_nodes, d_model]
        """
        X = data.x
        num_pos_enc = X.shape[-1]
        num_nodes = data.num_nodes

        # Compute Laplacian eigenvector positional encodings
        # 1. Create undirected graph representation
        undirected_edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)
        A = to_dense_adj(undirected_edge_index, max_num_nodes=num_nodes)[0]

        # 2. Compute graph Laplacian
        D = A.sum(dim=1)
        L = torch.diag(D) - A
        # Add small epsilon to diagonal and ensure symmetry
        L = L + 1e-5 * torch.eye(num_nodes, device=L.device)
        L = (L + L.T) / 2

        # 3. Compute eigenvectors (sorted by ascending eigenvalues)
        _, eigenvectors = torch.linalg.eigh(L)

        # 4. Select the eigenvectors corresponding to the largest eigenvalues
        # (these are the last eigenvectors when sorted in ascending order)
        top_eigenvectors = eigenvectors[:, -num_pos_enc:]

        # ToDO might need to change if different batching.
        if top_eigenvectors.shape[-1] < num_pos_enc:
            top_eigenvectors = torch.cat(
                [
                    top_eigenvectors,
                    torch.zeros(
                        num_nodes,
                        num_pos_enc - top_eigenvectors.shape[-1],
                        device=top_eigenvectors.device,
                    ),
                ],
                dim=-1,
            )

        # Add positional encodings to input features
        X = X + top_eigenvectors
        X = self.layer_norm(X)

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            X = layer(X)  # Self-attention

        return X


def _make_linear_projection(in_features: int, target_dim: int) -> Optional[nn.Module]:
    if in_features == target_dim:
        return None
    return nn.Linear(in_features, target_dim)


def _build_gcn_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config.get("gcn_in_channels", config["in_features"])
    projection = _make_linear_projection(config["in_features"], input_dim)
    backbone = GCN(
        in_channels=input_dim,
        hidden_channels=config["gcn_hidden_channels"],
        num_layers=config["gcn_num_layers"],
        skip_connection=config.get("gcn_skip_connection", False),
        dropout=config["dropout"],
        k=config.get("gcn_k", 3),
    )
    return BackboneComponents(model=backbone, input_dim=input_dim, input_projection=projection)


def _build_dense_gt_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["dense_d_model"]
    projection = _make_linear_projection(config["in_features"], input_dim)
    backbone = DenseGraphTransformerBackbone(
        d_model=config["dense_d_model"],
        nhead=config["dense_nhead"],
        dim_feedforward=config["dense_dim_feedforward"],
        num_layers=config["dense_transformer_num_layers"],
        dropout=config["dropout"],
        attn_dropout=config.get("dense_attn_dropout", 0.15),
    )
    return BackboneComponents(model=backbone, input_dim=input_dim, input_projection=projection)


def _build_sparse_gt_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["sparse_gt_d_model"]
    projection = _make_linear_projection(config["in_features"], input_dim)
    backbone = KHopGTModel(
        d_model=config["sparse_gt_d_model"],
        num_heads=config["sparse_gt_nhead"],
        num_hops=config["sparse_gt_num_hops"],
        num_layers=config["sparse_gt_num_layers"],
        dropout=config["dropout"],
        attn_dropout=config.get("sparse_gt_attn_dropout", 0.25),
        attn_algorithm=config.get("sparse_gt_attn_algorithm", "sparse"),
        random_graph=config.get("sparse_gt_random_graph", "Random-d2"),
        random_graph_degree=config.get("sparse_gt_random_graph_degree", 5),
    )
    return BackboneComponents(model=backbone, input_dim=input_dim, input_projection=projection)


def _build_sparse_gt_pyg_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["sparse_gt_d_model"]
    projection = _make_linear_projection(config["in_features"], input_dim)
    backbone = SparseTransformerConGT(
        d_model=config["sparse_gt_d_model"],
        num_heads=config["sparse_gt_nhead"],
        num_hops=config["sparse_gt_num_hops"],
        num_layers=config["sparse_gt_num_layers"],
        dropout=config["dropout"],
    )
    return BackboneComponents(model=backbone, input_dim=input_dim, input_projection=projection)


def _build_exphormer_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["in_features"]
    backbone = ExphormerOptimized(
        in_features=config["in_features"],
        d_model=config["exphormer_d_model"],
        nhead=config["exphormer_nhead"],
        dim_feedforward=config["exphormer_dim_feedforward"],
        num_layers=config["exphormer_num_layers"],
        exp_degree=config.get("exphormer_exp_degree", 5),
        exp_algorithm=config.get("exphormer_exp_algorithm", "Random-d"),
        dropout=config["dropout"],
    )
    return BackboneComponents(model=backbone, input_dim=input_dim)


def _build_transformer_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["transformer_d_model"]
    projection = _make_linear_projection(config["in_features"], input_dim)
    backbone = LaplacianGraphTransformer(
        d_model=config["transformer_d_model"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_layers=config["transformer_num_layers"],
        dropout=config["dropout"],
    )
    return BackboneComponents(model=backbone, input_dim=input_dim, input_projection=projection)


def _build_linkx_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["in_features"]
    backbone = LINKXBackbone(
        in_features=config["in_features"],
        hidden_dim=config["linkx_hidden_dim"],
        x_num_layers=config["linkx_x_num_layers"],
        a_num_layers=config["linkx_a_num_layers"],
        dropout=config["dropout"],
    )
    return BackboneComponents(model=backbone, input_dim=input_dim)


def _build_mlp_backbone(config: Dict[str, Any]) -> BackboneComponents:
    input_dim = config["in_features"]
    backbone = MLPBackbone(
        in_features=config["in_features"],
        hidden_dim=config["mlp_hidden_dim"],
        num_layers=config["mlp_num_layers"],
        dropout=config["dropout"],
    )
    return BackboneComponents(model=backbone, input_dim=input_dim)


BACKBONE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], BackboneComponents]] = {
    "dense_gt": _build_dense_gt_backbone,
    "exphormer": _build_exphormer_backbone,
    "gcn": _build_gcn_backbone,
    "linkx": _build_linkx_backbone,
    "mlp": _build_mlp_backbone,
    "sparse_gt": _build_sparse_gt_backbone,
    "sparse_gt_pyg": _build_sparse_gt_pyg_backbone,
    "transformer": _build_transformer_backbone,
}

BACKBONE_CHOICES: Tuple[str, ...] = tuple(BACKBONE_REGISTRY.keys())


def _build_no_positional_encoding(
    config: Dict[str, Any], *, target_dim: int, backbone_name: str
) -> Optional[nn.Module]:
    return None


def _build_random_positional_encoding(
    config: Dict[str, Any], *, target_dim: int, backbone_name: str
) -> nn.Module:
    if backbone_name in {"sparse_gt", "sparse_gt_pyg"}:
        hidden_channels = config["sparse_gt_d_model"]
        num_layers = config.get("sparse_gt_posenc_num_layers", 5)
        num_samples = config.get("sparse_gt_posenc_num_samples", 5)
    else:
        hidden_channels = config["rpearl_hidden_channels"]
        num_layers = config["rpearl_num_layers"]
        num_samples = config["rpearl_num_samples"]

    return RandomGNNPositionalEncodings(
        pe_hidden_channels=hidden_channels,
        pe_num_layers=num_layers,
        d_model=target_dim,
        num_samples=num_samples,
        dropout=config["dropout"],
    )


def _build_data_positional_encoding(
    config: Dict[str, Any], *, target_dim: int, backbone_name: str
) -> nn.Module:
    if backbone_name in {"sparse_gt", "sparse_gt_pyg"}:
        hidden_channels = config["sparse_gt_d_model"]
        num_layers = config.get("sparse_gt_posenc_num_layers", 5)
    else:
        hidden_channels = config["data_posenc_hidden_channels"]
        num_layers = config["data_posenc_num_layers"]

    return DataGNNPositionalEncodings(
        in_features=config["in_features"],
        pe_hidden_channels=hidden_channels,
        pe_num_layers=num_layers,
        d_model=target_dim,
        dropout=config["dropout"],
    )


POSENC_REGISTRY: Dict[str, Callable[..., Optional[nn.Module]]] = {
    "none": _build_no_positional_encoding,
    "rpearl": _build_random_positional_encoding,
    "data": _build_data_positional_encoding,
}

POSENC_CHOICES: Tuple[str, ...] = tuple(POSENC_REGISTRY.keys())

def create_model(task, config, backbone, posenc):
    """
    Factory function to create models based on backbone/positional encoding choices.
    """
    if not backbone:
        raise ValueError("The 'backbone' argument is required.")
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone}'.")

    if posenc is None:
        raise ValueError("The 'posenc' argument is required.")

    backbone_bundle = BACKBONE_REGISTRY[backbone](config)
    positional_encoding = POSENC_REGISTRY[posenc](
        config, target_dim=backbone_bundle.input_dim, backbone_name=backbone
    )

    if task == "node_classification":
        head = NodeClassifier(
            embedding_dim=backbone_bundle.model.embedding_dim,
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
    elif task == "graph_classification":
        head = GraphClassifier(
            embedding_dim=backbone_bundle.model.embedding_dim,
            num_classes=config["num_classes"],
            pooling=config.get("pooling", "mean"),
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    model = GraphModel(
        positional_encoding=positional_encoding,
        backbone=backbone_bundle.model,
        head=head,
        input_projection=backbone_bundle.input_projection,
        positional_encoding_dim=backbone_bundle.input_dim,
    )

    return model
