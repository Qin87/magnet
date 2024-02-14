import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP


class ChebBen1(nn.Module):
    def __init__(self, input_dim, nhid, out_dim,  dropout, layer, K):
        super(ChebBen1, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, out_dim, K)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        return x

class ChebBen2(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer, K):
        super(ChebBen2, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, hid_dim, K)
        self.conv2 = ChebConv(hid_dim, out_dim, K)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        return x

class ChebBenX(torch.nn.Module):

    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer, K):
        super(ChebBenX, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, hid_dim, K)
        self.conv2 = ChebConv(hid_dim, out_dim, K)
        self.convx = nn.ModuleList([ChebConv(hid_dim, hid_dim, K) for _ in range(layer - 2)])

        self.reg_params = list(self.conv1.parameters())+ list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_Cheb(nfeat, nhid, nclass, dropout, nlayer, K):
    if nlayer == 1:
        model = ChebBen1(nfeat, nhid, nclass, dropout, nlayer, K)
    elif nlayer == 2:
        model = ChebBen2(nfeat, nhid, nclass, dropout, nlayer, K)
    else:
        model = ChebBenX(nfeat, nhid, nclass, dropout, nlayer, K)
    return model