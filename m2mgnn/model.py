import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter
import numpy as np
from torch_geometric.utils import remove_self_loops


class M2M2_layer(nn.Module):
    def __init__(self, in_feat, out_feat, c, dropout,  temperature=1):
        super(M2M2_layer, self).__init__()

        self.lin = nn.Linear(in_feat, out_feat, bias=False)
        self.att = nn.Linear(out_feat, c, bias=False)
        self.para_lin = self.lin.parameters()
        self.para_att = self.att.parameters()

        self.temperature = temperature
        self.c = c
        self.dropout = dropout
        self.reg = None

    def forward(self, x, edge_index):
        x = self.lin(x)
        row, col = edge_index

        bin_rela = F.relu(0.5*x[row] + x[col])
        bin_rela = self.att(bin_rela)
        bin_rela = F.softmax(bin_rela/self.temperature, dim=1)

        self.reg = np.sqrt(self.c)/bin_rela.size(0)*torch.linalg.vector_norm(bin_rela.sum(dim=0), 2) - 1

        # deg = degree(row, num_nodes=x.size(0))
        # deg_inv = 1 / deg
        # deg_inv[deg_inv == float('inf')] = 0.

        x_j = torch.cat([x[col] * bin_rela[:, i].view(-1, 1) for i in range(self.c)], dim=1)
        out = scatter(x_j, row, dim=0, dim_size=x.size(0))
        # out = out * deg_inv.view(-1, 1)
        return out


class M2MGNN(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_layers, dropout, c, beta, dropout2, temperature=1, remove_self_loof=True):
        super(M2MGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lin1 = nn.Linear(in_feat, hid_feat*c)
        self.lin2 = nn.Linear(hid_feat * c, out_feat)
        self.norms.append(nn.LayerNorm(hid_feat*c))
        self.remove_self_loop = remove_self_loof

        for i in range(num_layers):
            self.convs.append(M2M2_layer(hid_feat * c, hid_feat, c, dropout, temperature))
            self.norms.append(nn.LayerNorm(hid_feat * c))
        self.params1 = list(self.lin2.parameters()) + list(self.lin1.parameters())
        self.params2 = list(self.convs.parameters()) + list(self.norms.parameters())

        self.dropout = dropout
        self.dropout2 = dropout2
        self.num_layers = num_layers
        self.beta = beta
        self.reg = None

    def forward(self, x, edge_index):
        if self.remove_self_loop == True:
            edge_index, _ = remove_self_loops(edge_index)

        self.reg = 0
        if self.dropout2 != 0:
            x = F.dropout(x, p=self.dropout2, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.norms[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        ego = x

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.norms[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1-self.beta) * x + self.beta * ego
            self.reg = self.reg + self.convs[i].reg

        x = self.lin2(x)
        self.reg = self.reg / self.num_layers
        return x
