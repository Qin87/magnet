"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/sage_conv.py
"""

from typing import Union, Tuple, Optional

# from torch_geometric.nn.conv.sage_conv import NormalizedSAGEConv
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add

from nets.sagcn import SAGCN, NormalizedSAGEConv
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian

from nets.src2.sage_qin import SAGEConv_QinNov
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, SAGEConv
class SAGEConv_SHA(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        # kwargs.setdefault('aggr', 'mean')
        # super().__init__(**kwargs)
        # super().__init__(aggr='mean')
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        # Add this line to declare edge_weight support
        self.propagate_type = {'x': OptPairTensor, 'edge_weight': OptTensor}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_l1 = Linear(in_channels[0], in_channels[0], bias=bias)
        self.l1 = Linear(in_channels[0], in_channels[0], bias=bias)
        if self.root_weight:
            self.temp_weight = Linear(in_channels[1], out_channels, bias=False)
            self.temp_weight1 = Linear(in_channels[1], in_channels[1], bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.temp_weight.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)


        # propagate_type: (x: OptPairTensor)
        x = self.l1(x[0])
        edge_index, edge_weight = inci_norm(  # yapf: disable
            edge_index, edge_weight,
             # self.flow,
            # x.dtype
        )   #  Qin

        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        # out = self.propagate(edge_index, size=size, x=x, norm=edge_weight)
        out = self.propagate(edge_index, x=x, norm=edge_weight)
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l1(out)  #  Qin delete
        out = self.lin_l(out)

        x_r = x[1]
        # out_r = self.temp_weight(x_r)

        out_r = self.temp_weight1(x_r)  #  remove this Qin
        out_r = self.temp_weight(out_r)

        if self.root_weight and x_r is not None:
            out += out_r

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    # def message(self, x_j: Tensor) -> Tensor:
    #     return x_j
    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:  #  from GCN
    #     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)
    # def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    #     return spmm(adj_t, x, reduce=self.aggr)

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)



    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SAGEConv_Qin(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')  # Use mean for SAGE default
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        # Linear transformations
        self.lin_l = Linear(in_channels, out_channels, bias=False)
        self.lin_r = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        edge_index, edge_attr = inci_norm(  # yapf: disable
            edge_index, edge_weight
        )

        # Transform node features
        x1= self.lin_l(x[0])

        # Propagate
        out = self.propagate(edge_index, x=x1, edge_weight=edge_attr)

        # Apply skip-connection with right transformation
        x_r = self.lin_r(x[1])
        # out += x_r

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

def inci_norm(edge_index, edge_weight):
    """
    Normalize edge weights using GCN-style normalization
    """

    num_nodes = edge_index.max().item() + 1
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]

    # Compute node degrees
    # deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    # Compute D^(-1/2)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Normalize edge weights
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


