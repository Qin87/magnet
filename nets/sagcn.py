from typing import Optional

import torch
import torch_geometric
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.nn.dense.linear import Linear

from nets.SAGCN2 import SAGCN2


class SAGCN(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: Optional[bool] = None,
            normalize: bool = True,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # Main transformation for message passing
        self.lin = Linear(in_channels, out_channels, bias=False)

        # Add a second linear layer for the skip connection
        self.lin_skip = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin_skip.reset_parameters()  # Reset parameters for the skip connection
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x_transformed = self.lin(x)  # Transform for message passing

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x_transformed, edge_weight=edge_weight)

        # Add skip connection with linear transformation
        # out = out + self.lin_skip(x)

        if self.bias is not None:
            out = out + self.bias

        return out

def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from typing import Union, Tuple, Optional, List
from torch_sparse import SparseTensor, matmul as spmm
from torch_geometric.typing import Adj, Size, OptPairTensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing

class NormalizedSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = True,  # TODO False originally
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                f"support lazy initialization with "
                                f"`project=True`")
            # self.lin = Linear(in_channels[0], in_channels[0], bias=True)
            # self.lin = Linear(in_channels[0], out_channels, bias=True)      # TODO Qin
            self.lin = torch_geometric.nn.dense.linear.Linear(in_channels[0], out_channels, bias=False,
                              weight_initializer='glorot')

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        # self.lin = Linear(in_channels, out_channels, bias=False)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def normalize_adj(self, edge_index: Adj, num_nodes: Optional[int] = None) -> Adj:
        # Convert edge_index to SparseTensor if it isn't already
        if isinstance(edge_index, Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        else:
            adj_t = edge_index

        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)
        # Calculate degree matrix D
        deg = adj_t.sum(dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        # Normalize adjacency matrix (D^(-1/2)AD^(-1/2))
        adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
        adj_t = adj_t * deg_inv_sqrt.view(1, -1)

        return adj_t

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # Normalize adjacency matrix
        if isinstance(edge_index, Tensor):
            num_nodes = x[0].size(0)
            edge_index = self.normalize_adj(edge_index, num_nodes)

        # x = self.lin_l(x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        # out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

import torch.nn as nn
class SAGCNXBN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super().__init__()
        self.is_add_self_loops = True  # Qin
        GCNConv = SAGCN2
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)


        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())  # no effect to layer=1,
        self.non_reg_params = self.conv2.parameters()

        self.layer = nlayer

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index, edge_weight)
        # x = self.mlp1(x)
        if self.layer == 1:
            return x
        # x = self.batch_norm1(x)
        x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                # x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index, edge_weight)
                # x= self.batch_norm3(x)
                x = F.relu(x)

        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = self.batch_norm2(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x