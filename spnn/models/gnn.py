"""GNN embedding model and the graph layers it can use."""
import torch.nn as nn
from torch.nn import LeakyReLU, ReLU, Sigmoid, SiLU
from torch_geometric.nn import (GATConv, GATv2Conv, GCN, GCN2Conv, GCNConv, GeneralConv, GINConv,
                                MessagePassing, TAGConv, TransformerConv, aggr)
from torch_geometric.nn.norm import LayerNorm

# Layers that support the edge_dim parameter
LAYERS_WITH_EDGE_DIM = {'GATConv', 'GATv2Conv', 'TransformerConv', 'GeneralConv', 
                        'GeneralConvMaxAttention', 'GeneralConvMinAttention', 
                        'GeneralConvMultiAttention'}

def create_graph_layer(layer_type, in_channels, out_channels, edge_dim=None, K=None):
    """
    Factory function to create graph convolutional layers with appropriate parameters.
    
    Some layers (GCNConv, TAGConv, GINConv) don't support edge_dim, while others
    (GATConv, GATv2Conv, TransformerConv, GeneralConv) do.
    
    Args:
        layer_type: String name of the layer class (e.g., 'GATConv', 'GCNConv')
        in_channels: Number of input features
        out_channels: Number of output features
        edge_dim: Edge feature dimension (only passed to layers that support it)
        K: Number of hops for TAGConv (None: the PyG default, 3)
    
    Returns:
        Instantiated graph layer
    """
    layer_class = LAYERS[layer_type]
    
    if layer_type == 'TAGConv' and K is not None:
        return layer_class(in_channels, out_channels, K=K)
    if layer_type in LAYERS_WITH_EDGE_DIM and edge_dim is not None:
        return layer_class(in_channels, out_channels, edge_dim=edge_dim)
    else:
        return layer_class(in_channels, out_channels)



class GINLayer(nn.Module):
    def __init__(self, input=3, output=20, eps=0.001):
        super(GINLayer, self).__init__()
        self.nn = nn.Sequential(nn.Linear(input, output), 
                                nn.ReLU(),
                                nn.Linear(output, output))
        self.layer = GINConv(self.nn, eps=0.001)

    def forward(self, x, edge_index):
        output = self.layer(x, edge_index)
        return output


class GCNLayer(MessagePassing):
    def __init__(self, input, output, edge_dim, aggr='add'):
        super(GCNLayer, self).__init__(aggr=aggr)  # "add" aggregation
        self.layer = GCN(input, output, num_layers=1)
    
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GeneralConvMaxAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMaxAttention, self).__init__()
        self.layer = GeneralConv(in_channels=input, 
                                out_channels=output,
                                aggr='max',
                                in_edge_channels=edge_dim,
                                attention=True,
                                l2_normalize=False)
        
        
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output

class GeneralConvMinAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMinAttention, self).__init__()
        self.aggregation = aggr.MultiAggregation([ 'min'])
        self.layer = GeneralConv(in_channels=input, 
                                 out_channels=output, 
                                 aggr='min', 
                                 in_edge_channels=edge_dim,
                                 attention=True, 
                                 l2_normalize=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output

class GeneralConvMultiAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMultiAttention, self).__init__()
        self.aggregation = aggr.MultiAggregation(['mean', 'max', 'sum', 'min'])
        self.layer = GeneralConv(in_channels=input, 
                                 out_channels=output, 
                                 aggr='min', 
                                 in_edge_channels=edge_dim,
                                 attention=True, 
                                 l2_normalize=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output
    
class CNNLayer(nn.Module):
    def __init__(self, input=3, output=20, sz=25, **kwargs):
        super(CNNLayer, self).__init__()
        self.layer = nn.Conv2d(in_channels=input, 
                                 out_channels=output, 
                                 kernel_size=1)
        self.size = sz
        self.in_channels = input

        self.out_channels = output
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        output = self.layer(x)
        return output


# Layer classes that `layer_type` may name. GNNModel used to look these up in
# globals(), so this registry lists every class that lookup could reach.
LAYERS = {cls.__name__: cls for cls in (
    GATConv, GATv2Conv, GCN, GCN2Conv, GCNConv, GeneralConv, GINConv, TAGConv, TransformerConv,
    GINLayer, GCNLayer, GeneralConvMaxAttention, GeneralConvMinAttention,
    GeneralConvMultiAttention, CNNLayer,
)}

# Activation classes by name, after normalize_activation() maps 'lrelu' etc.
ACTIVATIONS = {cls.__name__: cls for cls in (ReLU, LeakyReLU, Sigmoid, SiLU)}

'''

New and improved version of graph neural network with a linear output layer. 

'''  
class GNNModel(nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU',
                 edge_dim=None, layer_norm=False, K=None, **kwargs):
        super(GNNModel, self).__init__()

        # Initialize the first layer using factory function
        self.initial = create_graph_layer(layer_type, input, hidden, edge_dim=edge_dim, K=K)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([
            create_graph_layer(layer_type, hidden, hidden, edge_dim=edge_dim, K=K)
            for _ in range(layers - 1)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden, output)

        # activation function
        self.activation = ACTIVATIONS[activation]()

        self.layer_type = layer_type
        self.hidden_channels = hidden
        if layer_norm:
            self.norm = LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Only pass edge_attr to layers that support it
        use_edge_attr = self.layer_type in LAYERS_WITH_EDGE_DIM
        layer_edge_attr = edge_attr if use_edge_attr else None

        x = self.initial(x, edge_index) if not use_edge_attr else self.initial(x, edge_index, edge_attr=layer_edge_attr)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index) if not use_edge_attr else layer(x, edge_index, edge_attr=layer_edge_attr)
            x = self.activation(x)
            if hasattr(self, 'norm'):
                x = self.norm(x)
        if self.layer_type=='CNNLayer':
            x = x.flatten(start_dim=2, end_dim=3).mT.flatten(start_dim=0, end_dim=1)
        x = self.output(x)
        return x
