"""MLP modules: the pair head for non-siamese runs and the MLP embedding model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim, add_norm=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        if add_norm:
            self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        return F.silu(x)

class NewMLP(nn.Module):
    def __init__(self, input, output, hidden, layers, add_norm=True, edge_dim=None):
        input_dim = input 
        out_dim = output
        hid_dim = hidden
        n_layers = layers
        super().__init__()
        self.layers = nn.ModuleList(
            [LinearLayer(input_dim, hid_dim, add_norm)] +
            [LinearLayer(hid_dim, hid_dim, add_norm) for _ in range(n_layers - 2)]
        )
        self.out_proj = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)

 
# Baseline 0
def initialize_mlp(input, 
                   hidden, 
                   output, 
                   layers, 
                   dropout=True, 
                   layer_norm=True, 
                   activation='relu', 
                   **kwargs):
    if layers == 1:
        hidden=output
    if activation == 'relu' or activation == 'ReLU':
        func = nn.ReLU
    elif activation =='lrelu' or activation == 'LeakyReLU':
        func = nn.LeakyReLU
    elif activation=='sigmoid':
        func = nn.Sigmoid
    elif activation =='softplus':
        func = nn.Softplus
    elif activation == 'silu' or activation == 'SiLU':
        func = nn.SiLU
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input, hidden))
    if layer_norm:
        phi_layers.append(nn.LayerNorm(hidden))
    if dropout:
        phi_layers.append(nn.Dropout(p=0.30))
    phi_layers.append(func())
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden, hidden))
            if layer_norm:
                phi_layers.append(nn.LayerNorm(hidden))
            if dropout:
                phi_layers.append(nn.Dropout(p=0.30))
            phi_layers.append(func())
        else:
            phi_layers.append(nn.Linear(hidden, output))

    phi = nn.Sequential(*phi_layers)
    return phi

class MLPBaseline1(nn.Module):
    def __init__(self, mlp: nn.Module, max=True, aggr='max'):
        super(MLPBaseline1, self).__init__()
        self.mlp = mlp
        self.aggr=aggr
    
    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                # torch.nn.init.constant_(m.weight, 0.01)
                # torch.nn.init.constant_(m.bias, 0.01)
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.normal_(m.bias, std=0.01)
    
    def forward(self, input1, input2, vn_emb=None, batch=False):
        if self.aggr == 'max':
            embd = torch.max(input1, input2)
        elif self.aggr== 'sum':
            embd = input1 + input2
        elif self.aggr == 'min':
            embd = torch.min(input1, input2)
        elif self.aggr == 'combine':
            embd = torch.hstack((input1 + input2, torch.max(input1, input2), torch.min(input1, input2)))
        elif self.aggr == 'concat':
            embd = torch.hstack((input1, input2))
        elif self.aggr == 'sum+diff':
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2)))
        elif self.aggr == 'sum+diff+vn' and batch:
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2), vn_emb))
        elif self.aggr == 'sum+diff+vn':
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2), vn_emb.repeat(input1.size()[0], 1)))
        elif self.aggr == 'diff':
            embd = torch.abs(input1- input2)
        elif self.aggr == 'diff-no-abs':
            embd = input1 - input2
        return self.mlp(embd)
