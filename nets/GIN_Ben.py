import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP


class GIN_ModelBen1(nn.Module):
    def __init__(self, input_dim, nhid, out_dim,  dropout, layer=2):
        super(GIN_ModelBen1, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, out_dim)
        self.conv1 = GINConv(self.line1)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
        x = self.conv1(x, edge_index)
        print(x)

        return x

class GIN_ModelBen2(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer=2):
        super(GIN_ModelBen2, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, hid_dim)
        self.line2 = nn.Linear(hid_dim, out_dim)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # return x
        return F.log_softmax(x, dim=1)

class GIN_ModelBenX(torch.nn.Module):

    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super(GIN_ModelBenX, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, nhid)
        self.line2 = nn.Linear(nhid, out_dim)
        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)
        self.convx = nn.ModuleList([GINConv(nn.Linear(nhid, nhid)) for _ in range(layer-2)])

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.conv2(x, edge_index))      # I should desert this line
        x = self.conv2(x, edge_index)      # I should desert this line
        # print(x.shape)

        return x

def create_GIN(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = GIN_ModelBen1(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = GIN_ModelBen2(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = GIN_ModelBenX(nfeat, nhid, nclass, dropout, nlayer)
    return model