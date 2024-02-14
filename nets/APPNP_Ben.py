import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

class APPNP_ModelBen1(nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout, layer=1,alpha=0.1):
        super(APPNP_ModelBen1, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, out_dim)
        self.conv1 = APPNP(K=10, alpha=alpha)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        return x

class APPNP_ModelBen2(nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout, layer=2, alpha=0.1):
        super(APPNP_ModelBen2, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, nhid)
        self.line2 = nn.Linear(nhid, out_dim)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.line2(x)
        x = self.conv2(x, edge_index)

        return x

class APPNP_ModelBenX(nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout, layer=3, alpha=0.1):
        super(APPNP_ModelBenX, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, nhid)
        self.line2 = nn.Linear(nhid, out_dim)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.line3 = nn.Linear(nhid, nhid)
        self.convx= nn.ModuleList([APPNP(K=10, alpha=alpha) for _ in range(layer-2)])

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.line2(x)
        x = self.conv2(x, edge_index)

        return x

def create_APPNP(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1):
    if nlayer == 1:
        model = APPNP_ModelBen1(nfeat, nhid, nclass, dropout,nlayer,alpha)
    elif nlayer == 2:
        model = APPNP_ModelBen2(nfeat, nhid, nclass, dropout,nlayer,alpha)
    else:
        model = APPNP_ModelBenX(nfeat, nhid, nclass, dropout,nlayer,alpha)
    return model