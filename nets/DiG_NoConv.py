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
from nets.DiGCN import InceptionBlock, InceptionBlock4batch, InceptionBlock_Qin
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

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class DiG_Simple1BN_nhid(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim,  dropout, layer=1):
        super(DiG_Simple1BN_nhid, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        x = self.conv1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple2BN_nhid(nn.Module):
    def __init__(self, input_dim, nhid, ncls, dropout, layer=2):
        super(DiG_Simple2BN_nhid, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)
        self.Conv = nn.Conv1d(nhid, ncls, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))      # no BN here is better
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()


        return x

class DiG_SimpleXBN_nhid(torch.nn.Module):
    def __init__(self, input_dim,  hid_dim, out_dim, dropout, layer=3):
        super(DiG_SimpleXBN_nhid, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, hid_dim)
        self.convx = nn.ModuleList([DIGCNConv(hid_dim, hid_dim) for _ in range(layer-2)])
        self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        self.batch_norm2 = nn.BatchNorm1d(hid_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)

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


    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))      # no BN here is better
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)


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
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

def create_DiGSimple(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiG_Simple1BN(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiG_Simple2BN(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiG_SimpleXBN(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiGSimple_nhid(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiG_Simple1BN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiG_Simple2BN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiG_SimpleXBN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model


class DiGCN_IB_1BN_nhid(torch.nn.Module):
    def __init__(self, num_features, nhid, n_cls, dropout=0.5, layer=1):
        super(DiGCN_IB_1BN_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(n_cls)
        self.Conv = nn.Conv1d(nhid, n_cls, kernel_size=1)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_batch(torch.nn.Module):
    '''
    for large dataset, using small batches not the whole graph
    '''
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=1, batch_size=1024):
        super(DiGCN_IB_1BN_batch, self).__init__()
        self.ib1 = InceptionBlock4batch(num_features, num_classes)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(num_classes)
        self.batch_size = batch_size

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_1BN_batch_nhid(torch.nn.Module):
    '''
    for large dataset, using small batches not the whole graph
    '''

    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=1, batch_size=1024):
        super(DiGCN_IB_1BN_batch_nhid, self).__init__()
        self.ib1 = InceptionBlock4batch(num_features, hidden)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_size = batch_size
        self.Conv1 = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()

            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_2BN_nhid(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, hidden)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.reg_params = list(self.ib1.parameters())+list(self.ib2.parameters())
        self.non_reg_params = (self.Conv)

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_batch(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2, batch_size=1000):
        super(DiGCN_IB_2BN_batch, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, num_classes)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_size = batch_size

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        # batch_size = 1000  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_2BN_batch_nhid(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2, batch_size=1000):
        super(DiGCN_IB_2BN_batch_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, hidden)
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_size = batch_size

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        # batch_size = 1000  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x

class DiGCN_IB_2BN_samebatch(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2, batch_size=1000):
        super(DiGCN_IB_2BN_samebatch, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, num_classes)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_size = batch_size

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batatch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batatch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x

class DiGCN_IB_1BN_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_1BN_Sym, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, out_dim)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(out_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, out_dim, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x= x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1))
        # x = x.squeeze(0)
        x = self.batch_norm1(x)     # keep it is better performance

        x = F.dropout(x, p=self._dropout, training=self.training)   # only dropout during training   keep this is better
        return x
class DiGCN_IB_1BN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_1BN_Sym_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(out_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx


        x= x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm1(x)     # keep it is better performance

        x = F.dropout(x, p=self._dropout, training=self.training)   # only dropout during training   keep this is better
        return x

class DiGCN_IB_1BN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion

    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=1, batch_size=1024):
        super(DiGCN_IB_1BN_Sym_batch_nhid, self).__init__()
        self.batch_size = batch_size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                    (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                       (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)
            x_batch = self.batch_norm2(x_batch)

            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_1BN_Sym_batchConvOut(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''

    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=1, batch_size=1024):
        super(DiGCN_IB_1BN_Sym_batchConvOut, self).__init__()
        self.batch_size = batch_size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        # self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        # x = self.batch_norm1(x)        # without this is better performance
        x = x.unsqueeze(0)  # ?
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm2(x)  # ?

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2BN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index1 confusion

    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2BN_Sym_batch_nhid, self).__init__()
        self.batch_size = batch_size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)


        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(x_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        x = x.unsqueeze(0)  # ?
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self._dropout, training=self.training)       # keep is better performance
        return x
class DiGCN_IB_2BN_Sym(torch.nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Sym, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_Sym_nhid(torch.nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Sym_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_XBN_Sym, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self.ibx = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layer in self.linx:
            symx = iter_layer(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            x = x + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_XBN_Sym_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self.ibx = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layer in self.linx:
            symx = iter_layer(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            x = x + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion

    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3, batch_size=1024):
        super(DiGCN_IB_XBN_Sym_batch_nhid, self).__init__()
        self.batch_size = batch_size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer-2)])
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            # symx1 = self.gconv(symx, edge_index)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)
        x = DiGx + symx
        x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layerlinx,  iter_layeribx in zip(self.linx, self.ibx):
            outputs = []
            sym_outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                        (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                edge_index_batch = edge_index[:, mask] - start_idx

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                           (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                edge_in_batch = edge_in[:, mask_in]
                edge_in_batch = edge_in_batch - start_idx
                in_w_batch = in_w[mask_in]
                mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                            (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                edge_out_batch = edge_out[:, mask_out]
                edge_out_batch = edge_out_batch - start_idx
                out_w_batch = out_w[mask_out]

                # Forward pass for the current batch
                symx = iter_layerlinx(batch_x)
                # symx1 = self.gconv(symx, edge_index)
                symx1 = self.gconv(symx, edge_index_batch)
                symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                symx_batch = symx1 + symx2 + symx3
                sym_outputs.append(symx_batch)

                x_batch = iter_layeribx(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                
                # x_batch = self.batch_norm1(x_batch)       # without is better
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            DiGx = torch.cat(outputs, dim=0)
            symx = torch.cat(sym_outputs, dim=0)
            x = DiGx + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        x = x.unsqueeze(0)  # ?
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    all ib ends with nhid
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_SymCat_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3
        symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)
        x = torch.cat((x, symx), dim=-1)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)

        # x = self.Conv2(x)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        return x

class DiGCN_IB_2BN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    only has nhid version
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2BN_SymCat_batch_nhid, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)


        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm2(symx_batch)

            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm2(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv2(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        # x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_XBN_SymCat_nhid, self).__init__()
        self.layer= layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer - 2)])
        # self.ibx = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        # self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])
        # self.linx = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)       # without this is faster
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_lin, iter_ib in zip(self.linx, self.ibx):
            symx = iter_lin(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            DiGx = iter_ib(x, edge_index_tuple, edge_weight_tuple)
            x = torch.cat((DiGx, symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)  # with this block or without, almost the same result
            x = x.permute((0, 2, 1)).squeeze()
            # x = self.batch_norm1(x)       # without this is better
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm2(x)     # keep this is better

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_SymCat_1ibx_nhid(torch.nn.Module):
    '''
    revised for edge_index confusionx
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_XBN_SymCat_1ibx_nhid, self).__init__()
        self.layer= layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        # self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer - 2)])
        self.ibx = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        # self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])
        # self.linx = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)       # without this is faster
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # for iter_lin, iter_ib in zip(self.linx, self.ibx):
        for iter_lin in self.linx:
            symx = iter_lin(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            x = torch.cat((x, symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)  # with this block or without, almost the same result
            x = x.permute((0, 2, 1)).squeeze()
            # x = self.batch_norm1(x)       # without this is better
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm2(x)     # keep this is better

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    only has nhid version
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3, batch_size=1024):
        super(DiGCN_IB_XBN_SymCat_batch_nhid, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self.ibx = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        DiGx = []
        Symx =[]
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            Symx.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx.append(DiGx_batch)

        Symx = torch.cat(Symx, dim=0)
        DiGx = torch.cat(DiGx, dim=0)
        x = torch.cat((DiGx, Symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_lin in self.linx:
            Symx_x = []
            DiGx_x = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                        (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                edge_index_batch = edge_index[:, mask] - start_idx

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                           (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                edge_in_batch = edge_in[:, mask_in]
                edge_in_batch = edge_in_batch - start_idx
                in_w_batch = in_w[mask_in]
                mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                            (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                edge_out_batch = edge_out[:, mask_out]
                edge_out_batch = edge_out_batch - start_idx
                out_w_batch = out_w[mask_out]

                symx = iter_lin(batch_x)
                symx1 = self.gconv(symx, edge_index_batch)
                symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                symx_batch = symx1 + symx2 + symx3
                Symx_x.append(symx_batch)

                DiGx_batch = self.ibx(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                DiGx_x.append(DiGx_batch)
            Symx = torch.cat(Symx_x, dim=0)
            DiGx = torch.cat(DiGx_x, dim=0)
            x = torch.cat((DiGx, Symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)
            x = x.permute((0, 2, 1)).squeeze()

            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        Symx_2 = []
        DiGx_2 = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            Symx_2.append(symx_batch)

            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_2.append(DiGx_batch)

        Symx = torch.cat(Symx_2, dim=0)
        DiGx = torch.cat(DiGx_2, dim=0)
        x = torch.cat((DiGx, Symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm2(x)     # keep this is better

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index comfusion
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=1):
        super(DiGCN_IB_1BN_SymCat_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        # self.Conv1 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)     # very bad
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        symx = self.batch_norm1(symx)

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        DiGx = self.batch_norm1(DiGx)
        x = torch.cat((DiGx, symx), dim=-1)
        x = self.batch_norm2(x)     # with this and BN1 is better
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)     # keep this is better!

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index comfusion

    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=1,  batch_size=1024):
        super(DiGCN_IB_1BN_SymCat_batch_nhid, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)
        self.batch_normx= nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm2(x_batch)

            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm2(x_batch)
            # x_batch = F.relu(x_batch)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        # x = self.batch_normx(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv1(x)  # with this block or without, almost the same result
        # x = x.permute((0, 2, 1)).squeeze()
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat_batchConvOut(torch.nn.Module):
    '''
    revised for edge_index comfusion
    move all conv outside of batch
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=1,  batch_size=1024):
        super(DiGCN_IB_1BN_SymCat_batchConvOut, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            # x_batch = x_batch.unsqueeze(0)
            # x_batch = x_batch.permute((0, 2, 1))
            # x_batch = self.Conv1(x_batch)
            # x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            # x_batch = F.relu(x_batch)
            # if self._dropout > 0:
            #     x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)     # is this necessary?
        return x

class DiGCN_IB_2MixBN_SymCat_nhid(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2MixBN_SymCat_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)     # keep both it and the endBN is better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2MixBN_SymCat(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2MixBN_SymCat, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, out_dim)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)

        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # keep both it and the endBN is better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_batch, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, out_dim)
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

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        DiGx = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)
            
            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx.append(DiGx_batch)
        x= torch.cat(DiGx, dim=0)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2MixBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_batch_nhid, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)
        self.Conv = nn.Conv1d(2 * nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better

            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        DiGx = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            DiGx.append(DiGx_batch)
        x = torch.cat(DiGx, dim=0)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2MixBN_SymCat_Sym, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
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

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    nhid
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2):
        super(DiGCN_IB_2MixBN_SymCat_Sym_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters()) + list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiGCN_IB_2MixBN_SymCat_Sym_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_Sym_batch, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in] - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=2, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid, self).__init__()
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters()) + list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
        # x = torch.cat(outputs, dim=0)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in] - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        # x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Convx(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiGCN_IB_3MixBN_SymCat_Sym_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3, batch_size=1024):
        super(DiGCN_IB_3MixBN_SymCat_Sym_batch, self).__init__()
        self.layer = layer
        self.batch_size = batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])


        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # second layer --addSym
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2_(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)

        # more than 3 layer
        DiGx =[]
        if self.layer > 3:
            for iter_layer in self.ibx:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                x_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                

                x_batch = F.relu(x_batch)
                if self._dropout > 0:
                    x_batch = F.dropout(x_batch, self._dropout, training=self.training)
                DiGx.append(x_batch)

        x = torch.cat(DiGx, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_3MixBN_SymCat(torch.nn.Module):
        '''
        revised for edge_index confusion
        first layer is cat(Sym, DiGib), second layer is DiGib
        '''
        def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
            super(DiGCN_IB_3MixBN_SymCat, self).__init__()
            self.layer = layer
            self.ib1 = InceptionBlock_Qin(input_dim, nhid)
            self.ib2 = InceptionBlock_Qin(nhid, nhid)
            self._dropout = dropout
            self.batch_norm1 = nn.BatchNorm1d(nhid)
            self.batch_norm2 = nn.BatchNorm1d(nhid)
            self.batch_norm3 = nn.BatchNorm1d(out_dim)

            self.gconv = DGCNConv()
            self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
            self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

            self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
            self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
            # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
            if self.layer >3:
                self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

            self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
            self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

            nn.init.zeros_(self.bias1)
            nn.init.zeros_(self.bias2)

            self.reg_params = list(self.ib1.parameters())
            self.non_reg_params = self.ib2.parameters()

        def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
            # first layer---Sym + DiG
            symx = self.lin1(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
            x = torch.cat((x, symx), dim=-1)
            #
            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Conv1(x)  # with this block or without, almost the same result
            x = x.permute((0, 2, 1)).squeeze()
            x = self.batch_norm1(x)     # with this is a bit better
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

            # second layer --DiGib
            x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

            # x = self.batch_norm2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self._dropout, training=self.training)

            # more than 3 layer
            if self.layer > 3:
                for iter_layer in self.linx:
                    symx = iter_layer(x)
                    symx1 = self.gconv(symx, edge_index)
                    symx2 = self.gconv(symx, edge_in, in_w)
                    symx3 = self.gconv(symx, edge_out, out_w)
                    x = symx1 + symx2 + symx3

                    # x = self.batch_norm2(x)  # without this is better performance
                    x = F.relu(x)
                    if self._dropout > 0:
                        x = F.dropout(x, self._dropout, training=self.training)

            # third layer
            symx = self.lin2(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            x = symx1 + symx2 + symx3

            x = self.batch_norm3(x)       # keep this is better performance
            # x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

            return x


class DiGCN_IB_3MixBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_3MixBN_SymCat_nhid, self).__init__()
        self.layer = layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)
        self.Convx = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --DiGib
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                symx = iter_layer(x)
                symx1 = self.gconv(symx, edge_index)
                symx2 = self.gconv(symx, edge_in, in_w)
                symx3 = self.gconv(symx, edge_out, out_w)
                x = symx1 + symx2 + symx3

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Convx(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()


        x = self.batch_norm3(x)  # keep this is better performance
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x


class DiGCN_IB_3MixBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3, batch_size=1000):
        super(DiGCN_IB_3MixBN_SymCat_batch_nhid, self).__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --DiGib
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                sym_outputs = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    batch_x = x[start_idx:end_idx]

                    mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                    edge_index_batch = edge_index[:, mask] - start_idx

                    DiGedge_indexi_batch = ()
                    DiGedge_weighti_batch = ()
                    for i in range(len(edge_index_tuple)):
                        mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                                  (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                        edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                        edge_weighti_batch = edge_weight_tuple[i][mask_i]

                        DiGedge_indexi_batch += (edge_indexi_batch,)
                        DiGedge_weighti_batch += (edge_weighti_batch,)

                    mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                               (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                    edge_in_batch = edge_in[:, mask_in]
                    edge_in_batch = edge_in_batch - start_idx
                    in_w_batch = in_w[mask_in]
                    mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                                (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                    edge_out_batch = edge_out[:, mask_out]
                    edge_out_batch = edge_out_batch - start_idx
                    out_w_batch = out_w[mask_out]

                    # Forward pass for the current batch
                    symx = iter_layer(batch_x)
                    symx1 = self.gconv(symx, edge_index_batch)
                    symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                    symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                    symx_batch = symx1 + symx2 + symx3
                    sym_outputs.append(symx_batch)

                x = torch.cat(sym_outputs, dim=0)

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            x_batch = symx1 + symx2 + symx3

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            sym_outputs.append(x_batch)

        x = torch.cat(sym_outputs, dim=0)

        x = self.batch_norm3(x)  # keep this is better performance
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x
class DiGCN_IB_3MixBN_SymCat_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3, batch_size=1000):
        super(DiGCN_IB_3MixBN_SymCat_batch, self).__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            # mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
            #         (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            # edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --DiGib
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            # mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
            #         (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            # edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
            #            (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            # edge_in_batch = edge_in[:, mask_in]
            # edge_in_batch = edge_in_batch - start_idx
            # in_w_batch = in_w[mask_in]
            # mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
            #             (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            # edge_out_batch = edge_out[:, mask_out]
            # edge_out_batch = edge_out_batch - start_idx
            # out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                sym_outputs = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    batch_x = x[start_idx:end_idx]

                    mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                    edge_index_batch = edge_index[:, mask] - start_idx

                    DiGedge_indexi_batch = ()
                    DiGedge_weighti_batch = ()
                    for i in range(len(edge_index_tuple)):
                        mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                                  (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                        edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                        edge_weighti_batch = edge_weight_tuple[i][mask_i]
        
                        DiGedge_indexi_batch += (edge_indexi_batch,)
                        DiGedge_weighti_batch += (edge_weighti_batch,)

                    mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                               (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                    edge_in_batch = edge_in[:, mask_in]
                    edge_in_batch = edge_in_batch - start_idx
                    in_w_batch = in_w[mask_in]
                    mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                                (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                    edge_out_batch = edge_out[:, mask_out]
                    edge_out_batch = edge_out_batch - start_idx
                    out_w_batch = out_w[mask_out]

                    # Forward pass for the current batch
                    symx = iter_layer(batch_x)
                    symx1 = self.gconv(symx, edge_index_batch)
                    symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                    symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                    symx_batch = symx1 + symx2 + symx3
                    sym_outputs.append(symx_batch)

                x = torch.cat(sym_outputs, dim=0)

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)

        x = self.batch_norm3(x)  # keep this is better performance
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x

class DiGCN_IB_3MixBN_SymCat_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_3MixBN_SymCat_Sym_nhid, self).__init__()
        self.layer = layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --addSym
        symx = self.lin2_(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        x=symx
        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.ibx:
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)

                # x = self.batch_norm2(x)
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3

        x = self.batch_norm3(x)     # with this, much better than without
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x
class DiGCN_IB_3MixBN_SymCat_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, input_dim, nhid, out_dim, dropout=0.5, layer=3):
        super(DiGCN_IB_3MixBN_SymCat_Sym, self).__init__()
        self.layer = layer
        self.ib1 = InceptionBlock_Qin(input_dim, nhid)
        self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Qin(nhid, nhid) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --addSym
        symx = self.lin2_(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        x=symx
        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.ibx:
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)
                # x = self.batch_norm2(x)
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3

        x = self.batch_norm3(x)     # with this, much better than without
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x


class DiGCN_IB_2BN_Ben(torch.nn.Module):    #  obviously worse than DiGCN_IB_2BN
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_2BN_Ben, self).__init__()
        self.ib1old = InceptionBlock(num_features, hidden)
        self.ib2old = InceptionBlock(2 * hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 2, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(2*hidden)
        self.batch_norm2 = nn.BatchNorm1d(2*num_classes)

        self.reg_params = list(self.ib1old.parameters())
        self.non_reg_params = self.ib2old.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1old(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = 0.5*(x1 + x2)
        x = 2*(x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2old(x, edge_index, edge_weight, edge_index2, edge_weight2)
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
        self.ib1old = InceptionBlock(num_features, hidden)
        self.ib2old = InceptionBlock(3 * hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 3, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(3*hidden)
        self.batch_norm2 = nn.BatchNorm1d(3*num_classes)

        self.reg_params = list(self.ib1old.parameters())
        self.non_reg_params = self.ib2old.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1old(x, edge_index, edge_weight, edge_index2, edge_weight2)
        # x = 2*(x1 + x2)
        # x = torch.cat((x0, x), axis=-1)
        x = torch.cat((x0, x1, x2), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2old(x,  edge_index, edge_weight, edge_index2, edge_weight2)
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
        self.ib1old = InceptionBlock(num_features, hidden)
        self.ib2old = InceptionBlock_Qin(2 * hidden, num_classes)
        self.Conv = nn.Conv1d(num_classes * 2, num_classes, kernel_size=1)
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(2*hidden)
        self.batch_norm2 = nn.BatchNorm1d(2*num_classes)

        self.reg_params = list(self.ib1old.parameters())
        self.non_reg_params = self.ib2old.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1old(x,  edge_index, edge_weight, edge_index2, edge_weight2)
        x = 2*(x1 + x2)
        # x = torch.cat(x0, x)
        x = torch.cat((x0, x), axis=-1)
        x = self.batch_norm1(x)
        x0, x1, x2 = self.ib2old(x,  edge_index, edge_weight, edge_index2, edge_weight2)
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
class DiGCN_IB_XBN_nhid(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2):
        super(DiGCN_IB_XBN_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, num_classes)
        self._dropout = dropout
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden)

        self.layer = layer
        self.ibx=nn.ModuleList([InceptionBlock_Qin(hidden,hidden) for _ in range(layer-2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)

        for iter_layer in self.ibx:
            x = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
            x = self.batch_norm3(x)

        x = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = self.batch_norm2(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_batch_nhid(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2, batch_size=1000):
        super(DiGCN_IB_XBN_batch_nhid, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, hidden)
        self._dropout = dropout
        self.batch_size = batch_size
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden)

        self.layer = layer
        self.ibx = nn.ModuleList([InceptionBlock_Qin(hidden, hidden) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = self.batch_norm1(DiGx_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        for iter_layer in self.ibx:
            num_samples = features.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                # Forward pass for the current batch
                DiGx_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

                x_batch = self.batch_norm3(DiGx_batch)
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            x = torch.cat(outputs, dim=0)

        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)
            x_batch = self.batch_norm2(x_batch)

            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_batch(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer=2, batch_size=1000):
        super(DiGCN_IB_XBN_batch, self).__init__()
        self.ib1 = InceptionBlock_Qin(num_features, hidden)
        self.ib2 = InceptionBlock_Qin(hidden, num_classes)
        self._dropout = dropout
        self.batch_size = batch_size
        # self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden)

        self.layer = layer
        self.ibx=nn.ModuleList([InceptionBlock_Qin(hidden,hidden) for _ in range(layer-2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)


            # Forward pass for the current batch
            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        for iter_layer in self.ibx:
            num_samples = features.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                # Forward pass for the current batch
                DiGx_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                
                x_batch = self.batch_norm3(x_batch)
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            x = torch.cat(outputs, dim=0)

        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

def create_DiG_IB_nhid(nfeat, nhid, nclass, dropout, nlayer):

    if nlayer == 1:
        model = DiGCN_IB_1BN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_batch(nfeat, nhid, nclass, dropout, nlayer, batchSize):
    if nlayer == 1:
        model = DiGCN_IB_1BN_batch(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_batch(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    else:
        model = DiGCN_IB_XBN_batch(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    return model

def create_DiG_IB_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize):
    if nlayer == 1:
        model = DiGCN_IB_1BN_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    else:
        model = DiGCN_IB_XBN_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    return model

def create_DiG_IB_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_Sym(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_XBN_Sym(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    else:
        model = DiGCN_IB_XBN_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batchSize)
    return model

def create_DiG_IB_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer, ibx1):
    '''
    revised for edge_index confusion
    only has nhid version,
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        if ibx1:
            model = DiGCN_IB_XBN_SymCat_1ibx_nhid(nfeat, nhid, nclass, dropout, nlayer)
        else:
            model = DiGCN_IB_XBN_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_IB_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_XBN_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model

def create_DiG_IB_SymCat_batchConvOut(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
        model = DiGCN_IB_1BN_SymCat_batchConvOut(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    elif nlayer == 2:
        model = DiGCN_IB_2BN_SymCat_batchConvOut(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_XBN_SymCat_batchConvOut(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model

def create_DiG_MixIB_SymCat(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_3MixBN_SymCat(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_MixIB_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_3MixBN_SymCat_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_MixIB_SymCat_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batch_size:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_3MixBN_SymCat_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model

def create_DiG_MixIB_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batch_size:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_3MixBN_SymCat_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model

def create_DiG_MixIB_SymCat_Sym(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_MixIB_SymCat_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer):
    '''
    revised for edge_index confusion
    all main layer end with nhid
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_nhid(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_DiG_MixIB_SymCat_Sym_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_batch(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model

def create_DiG_MixIB_SymCat_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    if nlayer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif nlayer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_batch_nhid(nfeat, nhid, nclass, dropout, nlayer, batch_size)
    return model