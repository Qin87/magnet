import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

# from nets.DGCN import DGCNConv
from nets.DiGCN import InceptionBlock
from nets.Sym_Reg import DGCNConv


class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DiG_Simple1(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim,  dropout, layer=1):
        super(DiG_Simple1, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

        # # # type2
        # self.reg_params = list(self.conv1.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple2(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer=2):
        super(DiG_Simple2, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

        # # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_SimpleX(torch.nn.Module):
    def __init__(self, input_dim,  hid_dim, out_dim, dropout, layer=3):
        super(DiG_SimpleX, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, out_dim)
        self.convx = nn.ModuleList([DIGCNConv(hid_dim, hid_dim) for _ in range(layer-2)])
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

        # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.relu(self.conv2(x, edge_index))      # I should desert this line
        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple1BN(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim,  dropout, layer=1):
        super(DiG_Simple1BN, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

        # # # type2
        # self.reg_params = list(self.conv1.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x
class DiG_Simple2BN(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer=2):
        super(DiG_Simple2BN, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        # type1
        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

        # # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x
class DiG_SimpleXBN(torch.nn.Module):
    def __init__(self, input_dim,  hid_dim, out_dim, dropout, layer=3):
        super(DiG_SimpleXBN, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, out_dim)
        self.convx = nn.ModuleList([DIGCNConv(hid_dim, hid_dim) for _ in range(layer-2)])
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)

        # type1
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

        # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        return x

def create_DiGSimple(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiG_Simple1BN(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiG_Simple2BN(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiG_SimpleXBN(nfeat, nhid, nclass, dropout, nlayer)
    return model

class DiGCN_IB_2(torch.nn.Module):   # very slow to improve!----------so delete
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, num_classes)
        self._dropout = dropout

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_1BN, self).__init__()
        self.ib1 = InceptionBlock(num_features, num_classes)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(num_classes)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2BN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, num_classes)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_Sym(torch.nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Sym, self).__init__()
        self.ib1 = InceptionBlock(input_dim, nhid)
        self.ib2 = InceptionBlock(nhid, out_dim)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        # symx1 += self.bias1
        # symx2 += self.bias1
        # symx3 += self.bias1

        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2 + symx
        x = self.batch_norm1(x)
        x = F.relu(x)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        # symx1 += self.bias2
        # symx2 += self.bias2
        # symx3 += self.bias2

        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2 + symx
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_SymCat(torch.nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_SymCat, self).__init__()
        self.ib1 = InceptionBlock(input_dim, nhid)
        self.ib2 = InceptionBlock(nhid, out_dim)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)


        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = torch.cat((x, symx), dim=-1)

        # x = self.Conv2(x)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat(torch.nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_1BN_SymCat, self).__init__()
        self.ib1 = InceptionBlock(input_dim, nhid)
        self.ib2 = InceptionBlock(nhid, out_dim)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)


        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        x = self.batch_norm1(x)
        # x = F.relu(x)

        # symx = self.lin2(x)
        # symx1 = self.gconv(symx, edge_index)
        # symx2 = self.gconv(symx, edge_in, in_w)
        # symx3 = self.gconv(symx, edge_out, out_w)
        #
        # symx = symx1 + symx2 + symx3
        # # symx = self.batch_norm1(symx)
        # # symx = F.relu(symx)
        #
        # x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = x0 + x1 + x2
        # x = torch.cat((x, symx), dim=-1)
        #
        # # x = self.Conv2(x)
        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv2(x)  # with this block or without, almost the same result
        # x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2MixBN_SymCat, self).__init__()
        self.ib1 = InceptionBlock(input_dim, nhid)
        self.ib2 = InceptionBlock(nhid, out_dim)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)


        symx = symx1 + symx2 + symx3

        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        # symx = self.lin2(x)
        # symx1 = self.gconv(symx, edge_index)
        # symx2 = self.gconv(symx, edge_in, in_w)
        # symx3 = self.gconv(symx, edge_out, out_w)
        #
        # symx = symx1 + symx2 + symx3
        # # symx = self.batch_norm1(symx)
        # # symx = F.relu(symx)
        #
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        # x = torch.cat((x, symx), dim=-1)
        #
        # # x = self.Conv2(x)
        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv2(x)  # with this block or without, almost the same result
        # x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_Ben(torch.nn.Module):    #  obviously worse than DiGCN_IB_2BN
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Ben, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(2*hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 2, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(2*hidden)
        self.batch_norm2 = nn.BatchNorm1d(2*num_classes)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = 0.5*(x1 + x2)
        x = 2*(x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = 0.5 * (x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        return x

class DiGCN_IB_2BN_Ben_cat(torch.nn.Module):    #  obviously worse than DiGCN_IB_2BN
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Ben_cat, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(3*hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 3, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(3*hidden)
        self.batch_norm2 = nn.BatchNorm1d(3*num_classes)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = 2*(x1 + x2)
        # x = torch.cat((x0, x), axis=-1)
        x = torch.cat((x0, x1, x2), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = 2 * (x1 + x2)
        # x = torch.cat((x0, x), axis=-1)
        x = torch.cat((x0, x1, x2), axis=-1)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        return x
class DiGCN_IB_2BN_Ben2(torch.nn.Module):    #  obviously worse than DiGCN_IB_2BN
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Ben2, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(2*hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 2, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(2*hidden)
        self.batch_norm2 = nn.BatchNorm1d(2*num_classes)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = 2*(x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = 2 * (x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        return x
class DiGCN_IB_XBN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_XBN, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, num_classes)
        self._dropout = dropout
        # self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden)

        self.layer = layer
        self.ibx=nn.ModuleList([InceptionBlock(hidden,hidden) for _ in range(layer-2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = self.batch_norm1(x)

        for iter_layer in self.ibx:
            x0, x1, x2 = iter_layer(x, edge_index, edge_weight, edge_index2, edge_weight2)
            x = x0 + x1 + x2
            x = self.batch_norm3(x)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0 + x1 + x2
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

def create_DiG_IB(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiGCN_IB_1BN(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_Sym(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiGCN_IB_1BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
        # model = DiGCN_IB_2BN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_SymCat(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiGCN_IB_1BN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        # model = DiGCN_IB_2BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
        model = DiGCN_IB_2BN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_MixIB_SymCat(nfeat, nhid, nclass, dropout, nlayer):
    # if nlayer == 1:
    #     model = DiGCN_IB_1BN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    # elif nlayer == 2:
        # model = DiGCN_IB_2BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    model = DiGCN_IB_2MixBN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    # else:
    #     model = DiGCN_IB_XBN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    return model
