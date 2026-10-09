"""Build the embedding module and the MLP pair head from a model config."""
import copy

from .gnn import GNNModel
from .mlp import MLPBaseline1, NewMLP, initialize_mlp
from .sparse_gt import SparseGTWithRPEARL

ACTIVATION_ALIASES = {
    'relu': 'ReLU',
    'lrelu': 'LeakyReLU',
    'silu': 'SiLU'
}


def normalize_activation(activation: str) -> str:
    if not activation:
        return activation
    return ACTIVATION_ALIASES.get(activation.lower(), activation)


def configure_embedding_module(model_config, 
                               layer_type,
                               edge_dim=1,
                               new=True):
    embedding_config = model_config['constr']
    layer_norm = model_config['layer_norm']
    dropout = model_config['dropout']
    activation = model_config['activation']
    activation_for_gnn = normalize_activation(activation)
    if not new and 'MLP' in layer_type:
        embedding_module = initialize_mlp(**embedding_config, 
                                          activation=activation, 
                                          layer_norm=layer_norm, 
                                          dropout=dropout)
    elif 'MLP' in layer_type and new:
        embedding_module = NewMLP(**embedding_config, add_norm=layer_norm)
    elif layer_type == 'SparseGT':
        # SparseGT with RPEARL positional encodings
        sparse_gt_config = model_config.get('sparse_gt', {})
        embedding_module = SparseGTWithRPEARL(
            input_dim=embedding_config.get('input', 3),
            hidden_dim=sparse_gt_config.get('hidden_dim', embedding_config.get('hidden', 64)),
            output_dim=embedding_config.get('output', 64),
            num_layers=sparse_gt_config.get('num_layers', 3),
            num_heads=sparse_gt_config.get('num_heads', 4),
            num_hops=sparse_gt_config.get('num_hops', 2),
            rpearl_samples=sparse_gt_config.get('rpearl_samples', 30),
            rpearl_num_layers=sparse_gt_config.get('rpearl_num_layers', 3),
            dropout=sparse_gt_config.get('dropout', 0.3),
            attn_dropout=sparse_gt_config.get('attn_dropout', 0.1),
            embedding_mode=sparse_gt_config.get('embedding_mode', 'random'),
            pe_k=sparse_gt_config.get('pe_k', 3),
        )
    else:
        embedding_module = GNNModel(layer_type=layer_type, 
                                    edge_dim=edge_dim, 
                                    activation=activation_for_gnn, 
                                    layer_norm=layer_norm, 
                                    **embedding_config)
    return embedding_module

    
def configure_mlp_module(mlp_config, 
                        aggr='sum', 
                        new=True):
    model_config = copy.deepcopy(mlp_config['constr'])
    layer_norm=mlp_config['layer_norm']
    dropout=mlp_config['dropout']
    
    if aggr == 'combine':
        model_config['input'] = model_config['input'] * 3
    elif aggr == 'concat' or aggr == 'sum+diff':
        model_config['input'] = model_config['input'] * 2 
    if not new:
        mlp_nn = initialize_mlp(**model_config, layer_norm=False, dropout=dropout, activation='lrelu')
    else:
        mlp_nn = NewMLP(**model_config, add_norm=layer_norm)
    mlp = MLPBaseline1(mlp_nn, aggr=aggr)
    return mlp
