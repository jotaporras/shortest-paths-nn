"""Unused model classes from src/baselines.py (moved on the asilomar branch).

Virtual-node models, dense transformers and helpers that no live code calls.
`create_graph_layer` and `LAYERS_WITH_EDGE_DIM` now live in spnn.models.gnn,
and the activation lookup uses spnn.models.gnn.ACTIVATIONS.
"""
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import TransformerConv, global_add_pool

from spnn.models.gnn import ACTIVATIONS, LAYERS_WITH_EDGE_DIM, create_graph_layer

def get_coords(batch):
    graph = batch[3]
    source_coord = graph.pos[batch[0]]
    target_coord = graph.pos[batch[1]]
    return source_coord, target_coord


class MLPBaseline0(nn.Module):
    def __init__(self, siamese: nn.Module, final: nn.Module, max=False):
        super(MLPBaseline0, self).__init__()
        self.siamese = siamese
        self.final = final
        self.max = max
        print("max?", self.max)
    
    def init_weights(self):
        for m in self.siamese:
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0.01)
        for m in self.final:
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0.01)
    
    def forward(self, input1, input2):
        out1 = self.siamese(input1)
        out2 = self.siamese(input2)
        if not self.max:
            embd = out1 + out2
        else:
            embd = torch.max(out1, out2)
        return self.final(embd)

class CNN_Final_VN_Model(torch.nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                    layer_type='GATConv', activation='LeakyReLU', batches=False, 
                    edge_dim=None, **kwargs):
        super(CNN_Final_VN_Model, self).__init__()
        # Initialize the first layer using factory function
        self.initial = create_graph_layer(layer_type, input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([
            create_graph_layer(layer_type, hidden, hidden, edge_dim=edge_dim) 
            for _ in range(layers - 1)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden, output)
        #self.output = graph_layer(hidden, output)

        # activation function
        self.activation = ACTIVATIONS[activation]()

        self.layer_type = layer_type
        self.hidden_channels = hidden

        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 2):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, output), torch.nn.ReLU()))
                

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x = data.x
        # edge_index = data.edge_index

        # Only pass edge_attr to layers that support it
        use_edge_attr = self.layer_type in LAYERS_WITH_EDGE_DIM
        layer_edge_attr = edge_attr if use_edge_attr else None

        # Don't pass edge_attr to the initial layer if it doesn't support it
        x = self.initial(x, edge_index) if not use_edge_attr else self.initial(x, edge_index, edge_attr=layer_edge_attr)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index) if not use_edge_attr else layer(x, edge_index, edge_attr=layer_edge_attr)
            x = self.activation(x)
        
        if batch == None:
            num_nodes = x.size()[2]**2
            out = x.reshape(1, self.hidden_channels, num_nodes).squeeze().T
            vn_emb = torch.sum(x, dim = 0)
            # vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
            # vn_emb  = global_add_pool(x, None, size=1) + vn_emb
        else: 
            out = x.flatten(start_dim=2, end_dim=3).mT.flatten(start_dim=0, end_dim=1)
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
            vn_emb = global_add_pool(out, batch.batch) + vn_emb
            
        for mlp_layer in self.mlp_virtualnode_list:
            vn_emb = mlp_layer(vn_emb)
        x = self.output(out)
        return x, vn_emb

class GNN_Final_VN_Model(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                    layer_type='GATConv', activation='LeakyReLU', batches=False, 
                    edge_dim=None, **kwargs):
        super(GNN_Final_VN_Model, self).__init__()
        # Initialize the first layer using factory function
        self.initial = create_graph_layer(layer_type, input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([
            create_graph_layer(layer_type, hidden, hidden, edge_dim=edge_dim) 
            for _ in range(layers - 1)
        ])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = ACTIVATIONS[activation]()

        self.layer_type = layer_type

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 2):
            # if batches:
            #     self.mlp_virtualnode_list.append(
            #         torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
            #                             torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            # else:
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, output), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Only pass edge_attr to layers that support it
        use_edge_attr = self.layer_type in LAYERS_WITH_EDGE_DIM
        layer_edge_attr = edge_attr if use_edge_attr else None

        out = self.initial(x, edge_index) if not use_edge_attr else self.initial(x, edge_index, edge_attr=layer_edge_attr)
        
        for layer in self.module_list:            
            out = layer(out, edge_index) if not use_edge_attr else layer(out, edge_index, edge_attr=layer_edge_attr)
            out = self.activation(out)
        
        #vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
        if batch == None:
            vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
            vn_emb  = global_add_pool(out, None, size=1) + vn_emb
        else: 
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
            vn_emb = global_add_pool(out, batch.batch) + vn_emb
        
        for mlp_layer in self.mlp_virtualnode_list:
            vn_emb = mlp_layer(vn_emb)
        out = self.output(out)
        return out, vn_emb

class GNN_VN_Model(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', batches=False, 
                 edge_dim=None, **kwargs):
        super(GNN_VN_Model, self).__init__()

        torch.manual_seed(1234567)
        # Initialize the first layer using factory function
        self.initial = create_graph_layer(layer_type, input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([
            create_graph_layer(layer_type, hidden, hidden, edge_dim=edge_dim) 
            for _ in range(layers - 1)
        ])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = ACTIVATIONS[activation]()

        self.layer_type = layer_type

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 1):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Only pass edge_attr to layers that support it
        use_edge_attr = self.layer_type in LAYERS_WITH_EDGE_DIM
        layer_edge_attr = edge_attr if use_edge_attr else None

        out = self.initial(x, edge_index) if not use_edge_attr else self.initial(x, edge_index, edge_attr=layer_edge_attr)
        
        if batch == None:
            vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
        else: 
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
        
        for layer in self.module_list:

            out = out + vn_emb[batch.batch] if batch != None else out + vn_emb
            
            out = layer(out, edge_index) if not use_edge_attr else layer(out, edge_index, edge_attr=layer_edge_attr)
            if batch == None:
                vn_emb  = global_add_pool(out, None, size=1) + vn_emb
            else:
                vn_emb = global_add_pool(out, batch.batch) + vn_emb
            for mlp_layer in self.mlp_virtualnode_list:
                vn_emb = mlp_layer(vn_emb)
        out = self.output(out)
        return out

class GNN_VN_Hierarchical(torch.nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', batches=False, num_vn=1, 
                 n=100, m = 100, edge_dim=None, **kwargs):
        super(GNN_VN_Hierarchical, self).__init__()

        torch.manual_seed(1234567)
        # Initialize the first layer using factory function
        self.initial = create_graph_layer(layer_type, input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([
            create_graph_layer(layer_type, hidden, hidden, edge_dim=edge_dim) 
            for _ in range(layers - 1)
        ])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = ACTIVATIONS[activation]()

        self.layer_type = layer_type

        # number of virtual nodes
        self.num_vn = num_vn
        

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 1):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, h_blocks, h_levels, h_num):
        out = self.initial(x, edge_index)
        vn_direct = self.virtualnode_embedding(torch.zeros(h_num).to(edge_index.dtype).to(edge_index.device))
        vn_root = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))

        for layer in self.module_list:
            # Get information from virtual nodes
            out = out + vn_direct[h_blocks]
            out = layer(out, edge_index)

            # Get information from real nodes + root virtual node
            vn_direct = global_add_pool(out, h_blocks) + vn_direct
            vn_direct = vn_direct + vn_root

            # Root VN gets information from vn_direct
            vn_root = global_add_pool(vn_direct, None, size=1) + vn_root
            
            for mlp_layer in self.mlp_virtualnode_list:
                vn_direct = mlp_layer(vn_direct)
                vn_root = mlp_layer(vn_root)
        return out
        

"""
Graph transformer model. 
"""
class GraphTransformer(torch.nn.Module):
    def __init__(self, input, hidden, output, layers,
                 heads=2, dropout=0.3, **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, layers + 1):
            if i < layers:
                out_channels = hidden // heads
                concat = True
            else:
                out_channels = output
                concat = False
            conv = TransformerConv(input, out_channels, heads,
                                   concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            input = hidden

            if i < layers:
                self.norms.append(torch.nn.LayerNorm(hidden))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)

class Transformer(torch.nn.Module):
    def __init__(self, input, hidden, output, layers, heads=2, dropout=0.3, **kwargs):
        super(Transformer, self).__init__()
        # project to high dimensions
        self.projection_layer = nn.Linear(in_features=input, out_features=hidden)
        #self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers)])
        self.layer = nn.TransformerEncoderLayer(d_model = input, nhead=heads, batch_first=True)
        self.final_layer = nn.Linear(in_features=input, out_features=output)
        self.activation = LeakyReLU()

    def forward(self, x, edge_index):
        out = self.activation(self.layer(x)).squeeze()
        out = self.final_layer(out)
        return out
