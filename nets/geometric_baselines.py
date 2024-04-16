import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP, JumpingKnowledge
from torch.nn import Parameter

####################################################################
# Link Prediction Models
####################################################################
'''
def pairwise_similar(x):
    x = torch.tanh(x)
    xx = torch.exp(torch.matmul(x, x.T))
    xx = xx - torch.diag(torch.diag(xx, 0))
    return xx, torch.sum(xx, 1)+1e-8
'''


class APPNP_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha=0.1, dropout=False, K=1):
        super(APPNP_Link, self).__init__()
        self.dropout = dropout

        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=K, alpha=alpha)
        self.conv2 = APPNP(K=K, alpha=alpha)

        self.linear = nn.Linear(filter_num * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class GIN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False):
        super(GIN_Link, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)
        self.linear = nn.Linear(filter_num * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class GCN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False):
        super(GCN_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.linear = nn.Linear(filter_num * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class Cheb_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout=False):
        super(Cheb_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.linear = nn.Linear(filter_num * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class SAGE_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False):
        super(SAGE_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        # self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.linear = nn.Linear(filter_num * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class GAT_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout=False):
        super(GAT_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num * heads, filter_num, heads=heads)
        self.linear = nn.Linear(filter_num * heads * 2, out_dim)

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:, 0]], x[index[:, 1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


'''
####################################################################
# Link Prediction Models in old versions of the paper
####################################################################
class Sym_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(Sym_Link, self).__init__()
        self.dropout = dropout
        self.conv11 = GCNConv(input_dim, filter_num)
        self.conv12 = GCNConv(input_dim, filter_num)
        self.conv13 = GCNConv(input_dim, filter_num)

        self.conv21 = GCNConv(filter_num*3, filter_num)
        self.conv22 = GCNConv(filter_num*3, filter_num)
        self.conv23 = GCNConv(filter_num*3, filter_num)

        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, positive, negative):
        x1 = self.conv11(x, edge_index)
        x2 = self.conv12(x, edge_in, in_w)
        x3 = self.conv13(x, edge_out, out_w)
        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x1 = self.conv21(x, edge_index)
        x2 = self.conv21(x, edge_in, in_w)
        x3 = self.conv23(x, edge_out, out_w)
        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class APPNP_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha = 0.1, dropout = False, K=1):
        super(APPNP_Link, self).__init__()
        self.dropout = dropout

        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=K, alpha=alpha)
        self.conv2 = APPNP(K=K, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GIN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GIN_Link, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GCN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GCN_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class Cheb_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout = False):
        super(Cheb_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class SAGE_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(SAGE_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GAT_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout = False):
        super(GAT_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num*heads, filter_num, heads=heads)
        self.Conv = nn.Conv1d(filter_num*heads, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)
'''


####################################################################
# Node Classification Models
####################################################################


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout=False, layer=2):
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num * heads, filter_num, heads=heads)
        self.Conv = nn.Conv1d(filter_num * heads, out_dim, kernel_size=1)
        self.layer = layer
        if layer == 3:
            self.conv3 = GATConv(filter_num * heads, filter_num, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class SAGEModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super(SAGEModel, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = SAGEConv(filter_num, filter_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)

class SAGEModelBen(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super(SAGEModelBen, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = SAGEConv(filter_num, filter_num)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)

class SAGEModelBen1(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hid_dim, dropout=False, layer=2):
        super(SAGEModelBen1, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        # self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        #
        # self.reg_params = list(self.conv1.parameters())
        # self.non_reg_params = self.conv2.parameters()

        self.layer = layer
        if layer == 3:
            self.conv3 = SAGEConv(hid_dim, hid_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = GCNConv(filter_num, filter_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)  # adds a singleton dimension at the beginning of the tensor x.
        x = x.permute((0, 2,
                       1))  # If the original shape of x was [batch_size, original_dim1, original_dim2], the result of this permutation will be [batch_size, original_dim2, original_dim1].
        x = self.Conv(
            x)  # applies a convolutional operation (assuming self.Conv is a convolutional layer) to the tensor x
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class ChebModelBen(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout=False, layer=2):
        super(ChebModelBen, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = ChebConv(filter_num, filter_num, K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class ChebModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout=False, layer=2):
        super(ChebModel, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = ChebConv(filter_num, filter_num, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class APPNP_ModelBen(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha=0.1, dropout=False, layer=3):
        super(APPNP_ModelBen, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = APPNP(K=10, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.line3(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class APPNP_Model(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha=0.1, dropout=False, layer=3):
        super(APPNP_Model, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = APPNP(K=10, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.line3(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class GIN_ModelBen2(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hid_dim, dropout=False, layer=2):
        super(GIN_ModelBen2, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, hid_dim)
        self.line2 = nn.Linear(hid_dim, out_dim)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(hid_dim, hid_dim)
            self.conv3 = GINConv(self.line3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class GIN_Model(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super(GIN_Model, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = GINConv(self.line3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)


class GATModelBen(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout=False, layer=2):
        super(GATModelBen, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num * heads, filter_num, heads=heads)
        self.Conv = nn.Conv1d(filter_num * heads, out_dim, kernel_size=1)
        self.layer = layer
        if layer == 3:
            self.conv3 = GATConv(filter_num * heads, filter_num, heads=heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)

class GCNModelBen(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super(GCNModelBen, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = GCNConv(filter_num, filter_num)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)      # adds a singleton dimension at the beginning of the tensor x.
        x = x.permute((0, 2, 1))    # If the original shape of x was [batch_size, original_dim1, original_dim2], the result of this permutation will be [batch_size, original_dim2, original_dim1].
        x = self.Conv(x)    # applies a convolutional operation (assuming self.Conv is a convolutional layer) to the tensor x
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)

    class SAGEModelBen(torch.nn.Module):
        def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
            super(SAGEModelBen, self).__init__()
            self.dropout = dropout
            self.conv1 = SAGEConv(input_dim, filter_num)
            self.conv2 = SAGEConv(filter_num, filter_num)
            self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

            self.layer = layer
            if layer == 3:
                self.conv3 = SAGEConv(filter_num, filter_num)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)

            if self.layer == 3:
                x = self.conv3(x, edge_index)
                x = F.relu(x)

            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)
            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Conv(x)
            x = x.permute((0, 2, 1)).squeeze()

            return F.log_softmax(x, dim=1)

class GCN_JKNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, layer):

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.lin1 = torch.nn.Linear(nhid, nclass)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=nhid,
                                   num_layers=2
                                   )

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)   # without is better
        return F.log_softmax(x, dim=1)

class GCN_JKNet2(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):

        super(GCN_JKNet2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.lin1 = torch.nn.Linear(nhid, nclass)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=nhid,
                                   num_layers=2
                                   )

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)   # without is better
        return F.log_softmax(x, dim=1)

def create_JK(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = GCN_JKNet(nfeat, nhid, nclass, dropout,nlayer)
    elif nlayer == 2:
        model = StandGCN2BN(nfeat, nhid, nclass, dropout,nlayer)
    else:
        model = StandGCNXBN(nfeat, nhid, nclass, dropout,nlayer)

    return model
from torch_geometric.nn import MessagePassing, APPNP
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, args):
    # def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
