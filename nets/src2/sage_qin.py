from typing import Union, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor
from torch_scatter import scatter_add


class SAGEConv_QinNov(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

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

    def __norm__(self, edge_index, num_nodes, edge_weight):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv = 1.0 / deg
        norm = deg_inv[row] * edge_weight

        return edge_index, norm

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x = self.l1(x[0])

        # Normalize edge weights
        edge_index, edge_weight = self.__norm__(
            edge_index, x.size(self.node_dim), edge_weight
        )

        # Use edge_attr instead of norm
        out = self.propagate(edge_index, x=x, edge_attr=edge_weight, size=size)

        out = self.lin_l1(out)
        out = self.lin_l(out)

        x_r = x[1]
        out_r = self.temp_weight1(x_r)
        out_r = self.temp_weight(out_r)

        if self.root_weight and x_r is not None:
            out += out_r

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)