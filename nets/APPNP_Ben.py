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

#########################################################################
###  above is all wasted
#########################################################################
class APPNP1LayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super(APPNP1LayerWithGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP2LayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim,num_classes, dropout, alpha, K):
        super(APPNP2LayerWithGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNPXLayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super(APPNPXLayerWithGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.convx = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        # self.convx = GCNConv(hidden_dim, hidden_dim)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        for iter_layer in self.convx:
            # x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class APPNP1BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super(APPNP1BN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP2BN(nn.Module):
    def __init__(self, in_features, hidden_dim,num_classes, dropout, alpha, K):
        super(APPNP2BN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNPXBN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super(APPNPXBN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.convx = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        # self.convx = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))

        for iter_layer in self.convx:
            # x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index)))

        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP1Simp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super(APPNP1Simp_BN, self).__init__()
        self.line1 = nn.Linear(in_features, num_classes)
        self.conv1 = APPNP(K=10, alpha=alpha)

        self.batch_norm1 = nn.BatchNorm1d(num_classes)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.batch_norm1(self.conv1(x, edge_index))
        return x


class APPNP2Simp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super(APPNP2Simp_BN, self).__init__()
        self.line1 = nn.Linear(in_features, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, num_classes)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.line2(x)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        return x


class APPNPXSimp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super(APPNPXSimp_BN, self).__init__()

        self.line1 = nn.Linear(in_features, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, num_classes)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.linex = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        self.convx = nn.ModuleList([APPNP(K=10, alpha=alpha) for _ in range(layer - 2)])

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))

        for iter_layer, line_layer in zip(self.convx, self.linex):
            x = line_layer(x)
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index)))

        x = self.line2(x)
        x = self.batch_norm2(self.conv2(x, edge_index))
        return x

def create_APPNPGGPT_0(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1LayerWithGCN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2LayerWithGCN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXLayerWithGCN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model

def create_APPNPGGPT(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1BN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2BN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXBN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model

def create_APPNPSimp(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1Simp_BN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2Simp_BN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXSimp_BN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model