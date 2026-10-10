"""GNN layer types removed from spnn/models/gnn.py on the asilomar branch.

None of them ran inside GNNModel (checked 2026-10-09):
- GCNLayer: create_graph_layer did not pass the required edge_dim.
- GeneralConv{Max,Min,Multi}Attention: shape error in GeneralConv attention
  (edge_attr.unsqueeze(-1) gives a 3-D edge tensor). The Multi variant also
  set aggr='min' and never used its MultiAggregation.
- CNNLayer: Conv2d on (num_nodes, features) input.
"""
import torch.nn as nn
from torch_geometric.nn import GCN, GeneralConv, MessagePassing, aggr


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
