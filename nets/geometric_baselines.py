import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import triu
from torch.nn import Linear, ModuleList, init
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP, JumpingKnowledge
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum
from torch_sparse import mul


# from dgl.python.dgl import add_self_loop
from nets.gcn import gcn_norm

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

def get_conv(conv_type, input_dim, output_dim, alpha):      # from Rossi(LoG)
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
        # return DirGCNConv(input_dim, output_dim)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class DirGCNConv(torch.nn.Module):
    # def __init__(self, input_dim, output_dim, alpha):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    # def initialize_parameters(self):
    #     init.kaiming_uniform_(self.lin_src_to_dst.weight, a=init.calculate_gain('relu'))
    #     if self.lin_src_to_dst.bias is not None:
    #         init.zeros_(self.lin_src_to_dst.bias)

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")     # this is key: improve from 57 to 72

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")  #

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )

def count_upper_triangle_edges(self):
    row, col, _ = self.coo()
    upper_triangle_mask = row < col
    upper_triangle_count = upper_triangle_mask.sum().item()
    return upper_triangle_count

class DirGCNConv_2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(DirGCNConv_2, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)

        self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])
        # nn.ModuleList([DirGCNConv(in_dim, out_dim) for _ in range(20)])
        self.alpha = nn.Parameter(torch.ones(1) * args.alphaDir, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1) * args.betaDir, requires_grad=False)
        self.gama = nn.Parameter(torch.ones(1) * args.gamaDir, requires_grad=False)
        # self.alpha = args.alphaDir
        # self.beta = args.betaDir
        # self.gama = args.gamaDir

        self.norm_list = []


        self.adj_norm, self.adj_t_norm = None, None

        # self
        self.adj_norm_in2, self.adj_norm_out2, self.adj_norm_in2_in, self.adj_norm_out2_out = None, None, None, None

        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge is not None:
            input_dim_jk = output_dim * 3 if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.lin = Linear(input_dim_jk, output_dim)


    def forward(self, x, edge_index):
        if self.adj_norm is None:
            from torch_geometric.utils import add_self_loops
            # edge_index, _ = add_self_loops(edge_index, fill_value=1)      # TODO
            # edge_index, _ = remove_self_loops(edge_index)      # TODO
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")     # this is key: improve from 57 to 72

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")  #

            print('edge number(A, At):', sparse_triu(self.adj_norm), sparse_triu(self.adj_t_norm))

        if self.adj_norm_in2 is None:
            self.adj_norm_in2 = directed_norm(self.adj_norm @ self.adj_t_norm, rm_gen_sLoop=False)
            self.adj_norm_out2 = directed_norm(self.adj_t_norm @ self.adj_norm, rm_gen_sLoop=False)

            self.adj_norm_in2_in = get_norm_adj(self.adj_norm @ self.adj_norm, norm="dir")
            self.adj_norm_out2_out = get_norm_adj(self.adj_t_norm @ self.adj_t_norm, norm="dir")
            self.norm_list = [self.adj_norm_in2, self.adj_norm_out2, self.adj_norm_in2_in, self.adj_norm_out2_out]
            print('AAt, AtA, AA, AtAt: ',
                  sparse_triu(self.adj_norm_in2, k=1),
                  sparse_triu(self.adj_norm_out2, k=1),
                  sparse_triu(self.adj_norm_in2_in, k=1),
                  sparse_triu(self.adj_norm_out2_out, k=1))
            Union_A_AA, Intersect_A_AA = share_edge(self.adj_norm_in2_in, self.adj_norm)
            print('Union: Insetction(A and AA):', len(Union_A_AA), len(Intersect_A_AA))
            Union_A_AAt,  Intersect_A_AAt= share_edge(self.adj_norm_in2, self.adj_norm)
            print('Union: Insetction (A and AAt):', len(Union_A_AAt), len(Intersect_A_AAt))

            # intersect_A_AA = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            # self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        # # xs = []
        # # x_lin = self.lin_src_to_dst(x)

        out1 = 1*(self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x))
        out2 = 1*(self.beta * self.linx[0](self.norm_list[0] @ x) + (1 - self.beta) * self.linx[1](self.norm_list[1] @ x))
        out3 = 1*(self.gama * self.linx[2](self.norm_list[2] @ x) + (1 - self.gama) * self.linx[3](self.norm_list[3] @ x))

        xs = [out1, out2, out3]

        if self.jumping_knowledge_inner is not None:
            x = self.jump(xs)
            x = self.lin(x)
        else:
            x = out1 + out2 + out3


        return x

def filter_upper_triangle(edges):
    """Filter edges to include only those in the upper triangle."""
    return {edge for edge in edges if edge[0] < edge[1]}
    # return torch.tensor(edge for edge in edges if edge[0] < edge[1])
def tensor_to_tuple(tensor):
    return tuple(map(int, tensor.cpu().numpy()))
def share_edge(m1, m2):
    import torch
    row1 = m1.storage.row()
    row2 = m2.storage.row()
    # new_row = torch.cat(row1, row2)
    new_row = torch.cat((row1, row2), dim=0)
    col1 = torch.tensor(m1.storage.col())
    col2 = torch.tensor(m2.storage.col())
    new_col = torch.cat((col1, col2), dim=0)

    union_edges = torch.stack([new_row, new_col], dim=1)
    unique_edges = torch.unique(union_edges, dim=0)

        # Convert tensors to sets of tuples for intersection
    edges1 = torch.stack([row1, col1], dim=1)
    edges2 = torch.stack([row2, col2], dim=1)
    set1 = set(map(tuple, edges1.tolist()))
    set2 = set(map(tuple, edges2.tolist()))

    # Find the intersection
    intersection = set1.intersection(set2)

    # Convert the result back to a tensor
    intersection_tensor = torch.tensor(list(intersection))
    intersection_tensor = filter_upper_triangle(intersection_tensor)
    # intersection_tensor = torch.tensor(list(intersection_tensor), dtype=torch.int64)
    unique_edges = filter_upper_triangle(unique_edges)
    # unique_edges = torch.tensor(list(unique_edges), dtype=torch.int64)

    return unique_edges, intersection_tensor



def sparse_triu(sparse_matrix, k=0):
    # count the non-zero edges in upper triangle, that what GCN takes in.
    row = sparse_matrix.storage.row()
    col = sparse_matrix.storage.col()
    values = sparse_matrix.storage.value()

    # Create mask for upper triangular elements
    mask = col - row >= k

    # Apply mask
    new_values = values[mask]

    return (new_values != 0).sum().item()



class DirGCNConv_Qin(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv_Qin, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight):
        device = edge_index.device
        num_nodes = edge_index.max().item() + 1

        adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to(device)

        # Perform sparse matrix multiplication
        out = torch.sparse.mm(adj_matrix, x)

        return self.lin_src_to_dst(out)

def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))

# def remove_self_loops(adj):
#     """Remove self-loops from the adjacency matrix."""
#     mask = adj.row() != adj.col()
#     adj = adj.index_select(mask)
#     return adj

# def remove_self_loops(adj):
#     """Remove self-loops from the adjacency matrix."""
#     row, col, value = adj.coo()
#     mask = row != col
#     row = row[mask]
#     col = col[mask]
#     value = value[mask] if value is not None else None
#     adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=adj.sparse_sizes())
#     return adj

def directed_norm(adj, rm_gen_sLoop=False):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    # print(type(adj))
    if rm_gen_sLoop:
        adj = remove_self_loops(adj)
    if not adj.is_cuda:
        adj = adj.cuda()
        # print("Moved adj to CUDA")
    in_deg = sparsesum(adj, dim=0)
    # print(f"in_deg device: {in_deg.device}")

    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)
    # print(f"in_deg_inv_sqrt device: {in_deg_inv_sqrt.device}")

    out_deg = sparsesum(adj, dim=1)
    # print(f"out_deg device: {out_deg.device}")

    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)
    # print(f"out_deg_inv_sqrt device: {out_deg_inv_sqrt.device}")

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj

def get_model(num_features,  n_cls, args):
    return GNN(
        num_features=num_features,
        hidden_dim=args.feat_dim,
        num_layers=args.layer,
        num_classes=n_cls,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alphaDir,
        learn_alpha=args.learn_alpha,
    )

class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )

class GNN(torch.nn.Module):     # from Rossi(LoG paper)
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1/2,
        learn_alpha=False,
    ):
        super(GNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):

        super(GCN_JKNet, self).__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.feat_dim
        hidden_dim = nhid
        normalize = args.BN_model
        dropout = args.dropout

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([DirGCNConv_2(nfeat, output_dim, args)])
        else:
            self.convs = ModuleList([DirGCNConv_2(nfeat, nhid, args)])
            for _ in range(layer - 2):
                self.convs.append(DirGCNConv_2(nhid, nhid, args))
            self.convs.append(DirGCNConv_2(nhid, output_dim, args))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * layer if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return x


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
