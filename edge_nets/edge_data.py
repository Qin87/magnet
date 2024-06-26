import sys
import torch
import numpy as np
import pickle as pk
import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.data import Data
from torch import Tensor
from torch_sparse import SparseTensor, coalesce
# from stellargraph.data import EdgeSplitter    # can't install Ben
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling, dropout_adj
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import scipy
import os
from joblib import Parallel, delayed
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add


def sub_adj(edge_index, prob, seed):
    sub_train, sub_test = train_test_split(edge_index.T, test_size=prob, random_state=seed)
    sub_train, sub_val = train_test_split(sub_train, test_size=0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T


def edges_positive(edge_index):
    # return true edges and reverse edges
    return edge_index, edge_index[[1, 0]]


def edges_negative(edge_index):
    from torch_geometric.utils import to_undirected

    size = edge_index.max().item() + 1
    adj = np.zeros((size, size), dtype=np.int8)
    adj[edge_index[0], edge_index[1]] = 1
    x, y = np.where((adj - adj.T) < 0)

    reverse = torch.from_numpy(np.c_[x[:, np.newaxis], y[:, np.newaxis]])
    undirected_index = to_undirected(edge_index)
    negative = negative_sampling(undirected_index, num_neg_samples=edge_index[0].shape[0], force_undirected=False)

    _from_, _to_ = negative[0].unsqueeze(0), negative[1].unsqueeze(0)
    neg_index = torch.cat((_from_, _to_), axis=0)
    # neg_index = torch.cat((reverse.T, neg_index), axis = 1)
    # print(edge_index.shape, reverse.shape, neg_index.shape)
    return reverse.T, neg_index


def split_negative(edge_index, prob, seed, neg_sampling=True):
    reverse, neg_index = edges_negative(edge_index)
    if neg_sampling:
        neg_index = torch.cat((reverse, neg_index), axis=1)
    else:
        neg_index = reverse

    sub_train, sub_test = train_test_split(neg_index.T, test_size=prob, random_state=seed)
    sub_train, sub_val = train_test_split(sub_train, test_size=0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T


def label_pairs_gen(pos, neg):
    pairs = torch.cat((pos, neg), axis=-1)
    label = np.r_[np.ones(len(pos[0])), np.zeros(len(neg[0]))]
    return pairs, label


def generate_dataset_2class(edge_index, splits=10, test_prob=0.6):
    # this function doesn't consider the connectivity during removing edges for validation/testing
    from torch_geometric.utils import to_undirected
    datasets = {}

    for i in range(splits):
        train, val, test = sub_adj(edge_index, prob=test_prob, seed=i * 10)
        train_neg, val_neg, test_neg = split_negative(edge_index, seed=i * 10, prob=test_prob)
        ############################################
        # training data
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i] = {}

        datasets[i]['graph'] = train
        datasets[i]['undirected'] = to_undirected(train).numpy().T

        rng = np.random.default_rng(i)

        datasets[i]['train'] = {}
        pairs, label = label_pairs_gen(train, train_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['train']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['train']['label'] = label[perm]

        ############################################
        # validation data
        ############################################
        # positive edges, reverse edges, negative edges

        datasets[i]['validate'] = {}
        pairs, label = label_pairs_gen(val, val_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['validate']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['validate']['label'] = label[perm]
        ############################################
        # test data
        ############################################
        # positive edges, reverse edges, negative edges

        datasets[i]['test'] = {}
        pairs, label = label_pairs_gen(test, test_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['test']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['test']['label'] = label[perm]
    return datasets


# in-out degree calculation
# def in_out_degree(edge_index, size):
#     A = coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])), shape=(size, size), dtype=np.float32).tocsr()
#     out_degree = np.sum(A, axis=0).T
#     in_degree = np.sum(A, axis=1)
#     degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
#     return degree

def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix((np.ones(len(edge_index)), (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis = 0).T
    in_degree = np.sum(np.abs(A), axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def undirected_label2directed_label(adj, edge_pairs, task):
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = edge_pairs.copy()
    counter = 0
    for i, e in enumerate(edge_pairs):  # directed edges
        if adj[e[0], e[1]] + adj[e[1], e[0]] > 0:  # exists an edge
            if adj[e[0], e[1]] > 0:
                if adj[e[1], e[0]] == 0:  # rule out undirected edges
                    if counter % 2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                else:
                    new_edge_pairs[i] = [e[0], e[1]]
                    labels[i] = -1
            else:  # the other direction, and not an undirected edge
                if counter % 2 == 0:
                    labels[i] = 0
                    new_edge_pairs[i] = [e[1], e[0]]
                    counter += 1
                else:
                    labels[i] = 1
                    new_edge_pairs[i] = [e[0], e[1]]
                    counter += 1
        else:  # negative edges
            labels[i] = 2
            new_edge_pairs[i] = [e[0], e[1]]

    if task != 2:
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        rng = np.random.default_rng(1000)
        neg_half = rng.choice(neg, size=len(neg) - np.sum(labels == 0), replace=False)
        labels[neg_half] = -1
    return new_edge_pairs[labels >= 0], labels[labels >= 0]


def generate_dataset_3class(edge_index, size, save_path, splits=10, probs=[0.15, 0.05], task=2, label_dim=2):
    print(os.getcwd())
    print(sys.argv[0])
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_directory)
    save_file = save_path + 'task' + str(task) + 'dim' + str(label_dim) + 'prob' + str(int(probs[0] * 100)) + '_' + str(int(probs[1] * 100)) + '.pk'
    if os.path.exists(save_file):
        print('File exists!')
        d_results = pk.load(open(save_file, 'rb'))
        return d_results

    row, col = edge_index[0], edge_index[1]
    # adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    # A_dense = np.array(adj.todense())
    # edge_num = np.sum(A_dense)
    # print( "undirected rate:", 1.0*np.sum(A_dense * A_dense.T)/edge_num )

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    # G = nx.from_scipy_sparse_matrix(A)  # create an undirected graph based on the adjacency
    G = nx.from_scipy_sparse_array(A)  # create an undirected graph based on the adjacency

    def iteration(ind):
        datasets = {}
        edge_splitter_test = EdgeSplitter(G)
        G_test, ids_test, _ = edge_splitter_test.train_test_split(p=float(probs[0]), method="global", keep_connected=True, seed=ind)
        ids_test, labels_test = undirected_label2directed_label(A, ids_test, task)

        edge_splitter_val = EdgeSplitter(G_test)
        G_val, ids_val, _ = edge_splitter_val.train_test_split(p=float(probs[1]), method="global", keep_connected=True, seed=ind)
        ids_val, labels_val = undirected_label2directed_label(A, ids_val, task)

        edge_splitter_train = EdgeSplitter(G_val)
        _, ids_train, _ = edge_splitter_train.train_test_split(p=0.99, method="global", keep_connected=False, seed=ind)
        ids_train, labels_train = undirected_label2directed_label(A, ids_train, task)

        # observation after removing edges for training/validation/testing
        edges = [e for e in G_val.edges]
        # convert back to directed graph
        oberved_edges = np.zeros((len(edges), 2), dtype=np.int32)
        undirected_edges = np.zeros((2 * len(G.edges), 2), dtype=np.int32)

        for i, e in enumerate(edges):
            if A[e[0], e[1]] > 0:
                oberved_edges[i, 0] = int(e[0])
                oberved_edges[i, 1] = int(e[1])
            if A[e[1], e[0]] > 0:
                oberved_edges[i, 0] = int(e[1])
                oberved_edges[i, 1] = int(e[0])

        for i, e in enumerate(G.edges):
            if A[e[0], e[1]] > 0 or A[e[1], e[0]] > 0:
                undirected_edges[i, :] = [int(e[1]), e[0]]
                undirected_edges[i + len(edges), :] = [int(e[0]), e[1]]
        if label_dim == 2:
            ids_train = ids_train[labels_train < 2]
            labels_train = labels_train[labels_train < 2]
            ids_test = ids_test[labels_test < 2]
            labels_test = labels_test[labels_test < 2]
            ids_val = ids_val[labels_val < 2]
            labels_val = labels_val[labels_val < 2]
        ############################################
        # training data
        ############################################
        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(oberved_edges.T).long()
        datasets[ind]['undirected'] = undirected_edges

        datasets[ind]['train'] = {}
        datasets[ind]['train']['pairs'] = ids_train
        datasets[ind]['train']['label'] = labels_train
        ############################################
        # validation data
        ############################################
        datasets[ind]['validate'] = {}
        datasets[ind]['validate']['pairs'] = ids_val
        datasets[ind]['validate']['label'] = labels_val
        ############################################
        # test data
        ############################################
        datasets[ind]['test'] = {}
        datasets[ind]['test']['pairs'] = ids_test
        datasets[ind]['test']['label'] = labels_test
        return datasets

    # use larger n_jobs if the number of cpus is enough
    try:
        p_data = Parallel(n_jobs=4)(delayed(iteration)(ind) for ind in range(10))
    except:
        p_data = Parallel(n_jobs=1)(delayed(iteration)(ind) for ind in range(10))

    d_results = {}
    for ind in p_data:
        split = list(ind.keys())[0]
        d_results[split] = ind[split]

    if os.path.isdir(save_path) == False:
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print('Folder exists!')

    if os.path.exists(save_file) == False:
        try:
            pk.dump(d_results, open(save_file, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
        except FileExistsError:
            print('File exists!')

    return d_results


#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
# def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
# I think edge_weight shouldn't be in the parameter. so I changes it.
#     from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
#     from torch_scatter import scatter_add
#
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                      device=edge_index.device)
#     fill_value = 1
#     edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
#     row, col = edge_index
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     deg_inv = deg.pow(-1)
#     deg_inv[deg_inv == float('inf')] = 0
#     p = deg_inv[row] * edge_weight
#
#     # personalized pagerank p
#     p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
#     p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
#     p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
#     p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
#     p_v[0:num_nodes,num_nodes] = alpha
#     p_v[num_nodes,num_nodes] = 0.0
#     p_ppr = p_v
#
#     eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
#     eig_value = torch.from_numpy(eig_value.real)
#     left_vector = torch.from_numpy(left_vector.real)
#     val, ind = eig_value.sort(descending=True)
#
#     pi = left_vector[:,ind[0]] # choose the largest eig vector
#     pi = pi[0:num_nodes]
#     p_ppr = p_dense
#     pi = pi/pi.sum()  # norm pi
#
#     # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
#     assert len(pi[pi<0]) == 0
#
#     pi_inv_sqrt = pi.pow(-0.5)
#     pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
#     pi_inv_sqrt = pi_inv_sqrt.diag()
#     pi_sqrt = pi.pow(0.5)
#     pi_sqrt[pi_sqrt == float('inf')] = 0
#     pi_sqrt = pi_sqrt.diag()
#
#     # L_appr  Ben__L is a matrix of n*n, non zero is edges, value of L is edge weight,
#     L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
#
#     # make nan to 0
#     L[torch.isnan(L)] = 0
#
#     # transfer dense L to sparse
#     L_indices = torch.nonzero(L,as_tuple=False).t()
#     L_values = L[L_indices[0], L_indices[1]]
#     edge_index = L_indices
#     edge_weight = L_values
#
#     # row normalization
#     row, col = edge_index
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype):
#     edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
#     fill_value = 1
#     edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
#     row, col = edge_index
#     # print(row.shape, edge_weight.shape, num_nodes)   # torch.Size([623]) torch.Size([623]) 222
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     deg_inv = deg.pow(-1)
#     deg_inv[deg_inv == float('inf')] = 0
#     p = deg_inv[row] * edge_weight
#
#     # personalized pagerank p
#     p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
#     p_v = torch.zeros(torch.Size([num_nodes + 1, num_nodes + 1]))
#     p_v[0:num_nodes, 0:num_nodes] = (1 - alpha) * p_dense
#     p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
#     p_v[0:num_nodes, num_nodes] = alpha
#     p_v[num_nodes, num_nodes] = 0.0
#     p_ppr = p_v
#
#     eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(), left=True, right=False)
#     eig_value = torch.from_numpy(eig_value.real)
#     left_vector = torch.from_numpy(left_vector.real)
#     val, ind = eig_value.sort(descending=True)
#
#     pi = left_vector[:, ind[0]]  # choose the largest eig vector
#     pi = pi[0:num_nodes]
#     p_ppr = p_dense
#     pi = pi / pi.sum()  # norm pi
#
#     # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
#     assert len(pi[pi < 0]) == 0
#
#     pi_inv_sqrt = pi.pow(-0.5)
#     pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
#     pi_inv_sqrt = pi_inv_sqrt.diag()
#     pi_sqrt = pi.pow(0.5)
#     pi_sqrt[pi_sqrt == float('inf')] = 0
#     pi_sqrt = pi_sqrt.diag()
#
#     # 将所有涉及的张量移动到同一个设备上
#     device = edge_index.device
#     p_ppr = p_ppr.to(device)
#     pi_inv_sqrt = pi_inv_sqrt.to(device)
#     pi_sqrt = pi_sqrt.to(device)
#
#     # L_appr  Ben__L is a matrix of n*n, non zero is edges, value of L is edge weight,
#     L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
#
#     # make nan to 0
#     L[torch.isnan(L)] = 0
#
#     # transfer dense L to sparse
#     L_indices = torch.nonzero(L, as_tuple=False).t()
#     L_values = L[L_indices[0], L_indices[1]]
#     edge_index = L_indices
#     edge_weight = L_values
#
#     # row normalization
#     row, col = edge_index
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def union_edge_index(edge_index):
    # Concatenate the original edge_index with its reverse
    union = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Remove duplicates
    union = torch.unique(union, dim=1)

    return union
def Qin_get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    """
    based on get_appr_directed_adj, all weights to 1
    Args:
        alpha:
        edge_index:
        num_nodes:
        dtype:
        edge_weight:

    Returns:

    """

    device = edge_index.device
    fill_value = 1
    edge_index, _ = add_self_loops(edge_index.long(), fill_value=fill_value, num_nodes=num_nodes)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_weight = edge_weight.to(device)
    edge_index = edge_index.to(device)

    return edge_index,  edge_weight

def WCJ_get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, W_degree=0, edge_weight=None):
    """
    based on get_appr_directed_adj, all weights to 1
    Args:
        alpha:
        edge_index:
        num_nodes:
        dtype:
        edge_weight:

    Returns:

    """

    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg0 = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)  # row degree
    deg1 = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes).to(device)  # col degree
    deg2 = deg0 + deg1

    # plt.hist(deg0.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of  degree0:Tel')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if W_degree == 0:  # in-degree
        edge_weight = deg0[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg0")
    elif W_degree == 1:     # out-degree
        edge_weight = deg1[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg1")
    elif W_degree == 2:     # total-degree
        edge_weight = deg2[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg2")
    elif W_degree == 3:     # random number in [1,100]
        edge_weight = torch.randint(1, 101, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 300:     # random number in [1,10000]
        edge_weight = torch.randint(1, 10001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 30000:     # random number in [1,1000000]
        edge_weight = torch.randint(1, 1000001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 400:  # random number in [0.001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.999 + 0.001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 40000:  # random number in [0.00001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.99999 + 0.00001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 4:  # random number in [0.1,1]       # random number in [0.1,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.9 + 0.1
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 5:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == 50:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        edge_weight = torch.abs(torch.sin(edge_weight))
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == -3:   # three peaks
        edge_weight = trimodal_distribution(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -2:   # two peaks
        edge_weight = trimodal_distribution2(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -4:   # two peaks
        edge_weight = trimodal_distribution4(edge_index.size(1), edge_index.device, dtype)
    else:
        NotImplementedError('Not Implemented edge-weight type')
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_weight = edge_weight.to(device)
    edge_index = edge_index.to(device)

    return edge_index,  edge_weight

def WCJ_get_appr_directed_adj0(alpha, edge_index, num_nodes, dtype, W_degree=0, edge_weight=None):
    """
    based on get_appr_directed_adj, all weights to 1
    Args:
        alpha:
        edge_index:
        num_nodes:
        dtype:
        edge_weight:

    Returns:

    """

    device = edge_index.device
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    row, col = edge_index
    deg0 = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)      # row degree
    deg1 = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes).to(device)      # col degree
    deg2 = deg0 + deg1
    deg_inv = deg0.pow(-1).to(device)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight



    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense().to(device)
    #
    # p_v = torch.zeros(torch.Size([num_nodes + 1, num_nodes + 1])).to(device)  # dummy node
    # p_v[0:num_nodes, 0:num_nodes] = (1 - alpha) * p_dense  # original P
    # p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    # p_v[0:num_nodes, num_nodes] = alpha
    # p_v[num_nodes, num_nodes] = 0.0
    # p_ppr = p_v.cpu()
    p_ppr = p_dense.to(device)
    L = (p_ppr + p_ppr.t()) / 2.0  # a bit time consuming

    # L = p_dense       # Qin revise
    # # make nan to 0
    # L[torch.isnan(L)] = 0
    #
    # # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()     # the indices of all nonzero elements in the input tensor L, arranged as a tensor where each column represents the indices of a nonzero element
    edge_index = L_indices.to(device)      # their transformed edges of this symmetric_A of digraph

    # edge_weight= torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if W_degree == 0:  # in-degree
        edge_weight = deg0[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg0")
    elif W_degree == 1:     # out-degree
        edge_weight = deg1[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg1")
    elif W_degree == 2:     # total-degree
        edge_weight = deg2[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg2")
    elif W_degree == 3:     # random number in [1,100]
        edge_weight = torch.randint(1, 101, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 300:     # random number in [1,10000]
        edge_weight = torch.randint(1, 10001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 30000:     # random number in [1,1000000]
        edge_weight = torch.randint(1, 1000001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 400:  # random number in [0.001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.999 + 0.001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 40000:  # random number in [0.00001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.99999 + 0.00001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 4:  # random number in [0.1,1]       # random number in [0.1,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.9 + 0.1
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 5:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        # random_values = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device)
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == 50:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        edge_weight = torch.abs(torch.sin(edge_weight))
        # random_values = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device)
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == -3:   # three peaks
        edge_weight = trimodal_distribution(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -2:   # two peaks
        edge_weight = trimodal_distribution2(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -4:   # two peaks
        edge_weight = trimodal_distribution4(edge_index.size(1), edge_index.device, dtype)
    else:
        print("not implemented!")
        sys.exit()


    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_weight = edge_weight.to(device)

    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Normalized Edge weight range: [{min_val}, {max_val}]")

    # return edge_index,  torch.ones((edge_index.size(1), ), dtype=dtype,device=edge_index.device)
    return edge_index,  edge_weight


def Qin_get_appr_directed_adj0(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    """
    based on get_appr_directed_adj, all weights to 1, this is equal to GCN(norm inside GCNConV is False, and better than GCN with norm)
    QinDiG worked for telegram
        alpha:
        edge_index:
        num_nodes:
        dtype:
        edge_weight:

    Returns:

    """

    device = edge_index.device
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
    edge_index = edge_index.to(device)

    return edge_index,  edge_weight


def get_appr_directed_adj0(alpha, edge_index, num_nodes, dtype, edge_weight=None):   # TODO to delete(old DiG)
    device = edge_index.device
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)
    deg_inv = deg.pow(-1).to(device)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense().to(device)
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1])).to(device)     # dummy node
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense      # original P
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v.cpu()   # for p_ppr.numpy()  # this is new P with one dummy node

    p_ppr_sparse = csr_matrix(p_ppr.numpy())
    eig_value, left_vector = scipy.linalg.eig(p_ppr_sparse.toarray(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real).to(device)     # Qin ask: why only real?       # converting a NumPy array containing eigenvalues to a PyTorch tensor
    left_vector = torch.from_numpy(left_vector.real).to(device)
    val, ind = eig_value.sort(descending=True)
    #  sort the tensor eig_value in descending order and also get the corresponding indices of the sorted elements

    pi = left_vector[:,ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]    # X+1 back to X  # remove the dummy node
    p_ppr = p_dense.to(device)
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0
    pi_inv_sqrt = pi.pow(-0.5)      # (183,) to (183,)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag().to(device)     # (183,) to (183, 183)
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag().to(device)

    # L_appr   # actually, L_appr= I-L, so this L is the equivalent their version of symmetric_A of digraph
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0       # a bit time consuming
    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()     # the indices of all nonzero elements in the input tensor L, arranged as a tensor where each column represents the indices of a nonzero element
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices      # their transformed edges of this symmetric_A of digraph
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  #



def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):       # TODO get back, new DiG, name remove 2
    device = edge_index.device


    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)
    deg_inv = deg.pow(-1).to(device)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # plt.hist(deg.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of degree:Cora')       # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense().to(device)
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1])).to(device)     # dummy node
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense      # original P
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v.cpu()   # for p_ppr.numpy()  # this is new P with one dummy node

    p_ppr_sparse = csr_matrix(p_ppr.numpy())
    eig_value, left_vector = scipy.linalg.eig(p_ppr_sparse.toarray(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real).to(device)     # Qin ask: why only real?       # converting a NumPy array containing eigenvalues to a PyTorch tensor
    left_vector = torch.from_numpy(left_vector.real).to(device)
    val, ind = eig_value.sort(descending=True)
    #  sort the tensor eig_value in descending order and also get the corresponding indices of the sorted elements
    #
    pi = left_vector[:,ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]    # X+1 back to X  # remove the dummy node
    p_ppr = p_dense.to(device)
    pi = pi/pi.sum()  # norm pi
    #
    # # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0
    pi_inv_sqrt = pi.pow(-0.5)      # (183,) to (183,)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag().to(device)     # (183,) to (183, 183)
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag().to(device)

    # L_appr   # actually, L_appr= I-L, so this L is the equivalent their version of symmetric_A of digraph
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0       # a bit time consuming
    L[torch.isnan(L)] = 0    # make nan to 0
    # L[torch.isnan(L)] = 1  # make nan to 1   # TODO delete it after testing(Qin use 1, original is 0)---worse

    # L = (p_ppr + p_ppr.t()) / 2.0       # TODO delete it after testing

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()     # the indices of all nonzero elements in the input tensor L, arranged as a tensor where each column represents the indices of a nonzero element
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices      # their transformed edges of this symmetric_A of digraph
    edge_weight = L_values

    # edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    # edge_weight= torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (0.1 - 0.00001) + 0.0001   # TODO delete this just for test
    # perm = torch.randperm(edge_weight.size(0), device=edge_weight.device)
    # edge_weight = edge_weight[perm]

    # edge_weight = torch.abs(torch.sin(edge_weight))     # TODO delete
    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Original Edge weight range: [{min_val}, {max_val}]")

    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of  DiG edge weights_F1=(WebKB/Cornell)')       # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Normalized Edge weight range: [{min_val}, {max_val}]")

    # delete TODO
    # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    # edge_weight = torch.cat([edge_weight, torch.ones((num_nodes,), device=edge_weight.device)])
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    return edge_index, edge_weight

import torch
import torch.distributions as dist
def trimodal_distribution(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 1.0, 1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001, 0.1, 100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(3, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples

def trimodal_distribution4(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 0.01, 900, 5000.0], device=device, dtype=dtype)
    stds = torch.tensor([10, 100, 1000, 5000], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(4, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples

def trimodal_distribution2(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001,  1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001,  100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(2, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples

def Qin_get_second_directed_adj(edge_index, num_nodes, dtype):
    """
    no change based on get_second_directed_adj
    Args:
        edge_index:
        num_nodes:
        dtype:

    Returns:

    """
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight      # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)    # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())   # both source from which is non_zero in L_out

    L_in_hat = L_in   # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0        # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight      # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)    # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())   # both source from which is non_zero in L_out

    L_in_hat = L_in   # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0        # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_third_directed_adj(edge_index, num_nodes, dtype):
    device = edge_index.device
    edge_weight1 = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight1 = add_self_loops(
        edge_index, edge_weight1, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight1, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight1      # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)    # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())   # both source from which is non_zero in L_out

    L_in_hat = L_in   # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0        # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index2 = L_indices.to(device)
    edge_weight2 = L_values.to(device)

    # row normalization
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight2 = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]
    #########################
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight2  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index2, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0  # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index3 = L_indices.to(device)
    edge_weight3 = L_values.to(device)

    # row normalization
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight3 = deg_inv_sqrt[row] * edge_weight3 * deg_inv_sqrt[col]

    return (edge_index2, edge_index3), (edge_weight2,edge_weight3)

def get_4th_directed_adj(edge_index, num_nodes, dtype):
    device = edge_index.device
    edge_weight1 = torch.ones((edge_index.size(1),), dtype=dtype,device=edge_index.device)

    fill_value = 1
    edge_index, edge_weight1 = add_self_loops(
        edge_index, edge_weight1, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight1, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight1      # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)    # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())   # both source from which is non_zero in L_out

    L_in_hat = L_in   # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0        # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index2 = L_indices.to(device)
    edge_weight2 = L_values.to(device)

    # row normalization
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight2 = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]
    #########################3rd
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight2  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index2, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0  # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index3 = L_indices.to(device)
    edge_weight3 = L_values.to(device)

    # row normalization
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight3 = deg_inv_sqrt[row] * edge_weight3 * deg_inv_sqrt[col]

    ############################################### 4th
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight3  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index3, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0  # Qin learn: this is intersection
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index4 = L_indices.to(device)
    edge_weight4 = L_values.to(device)

    # row normalization
    row, col = edge_index4
    deg = scatter_add(edge_weight4, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight4 = deg_inv_sqrt[row] * edge_weight4 * deg_inv_sqrt[col]


    return (edge_index2, edge_index3, edge_index4), (edge_weight2,edge_weight3, edge_weight4)


def get_third_directed_adj_union(edge_index, num_nodes, dtype):
    device = edge_index.device
    edge_weight1 = torch.ones((edge_index.size(1),), dtype=dtype,
                              device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight1 = add_self_loops(
        edge_index, edge_weight1, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight1, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight1  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True
    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index2 = L_indices.to(device)
    edge_weight2 = L_values.to(device)

    # row normalization
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight2 = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]
    #########################
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight2  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index2, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True
    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index3 = L_indices.to(device)
    edge_weight3 = L_values.to(device)

    # row normalization
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5).to(device)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight3 = deg_inv_sqrt[row] * edge_weight3 * deg_inv_sqrt[col]

    # return edge_index2, edge_weight2, edge_index3, edge_weight3
    return (edge_index2,edge_index3), (edge_weight2, edge_weight3)

def get_4th_directed_adj_union(edge_index, num_nodes, dtype):
    device = edge_index.device
    edge_weight1 = torch.ones((edge_index.size(1),), dtype=dtype,
                              device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight1 = add_self_loops(
        edge_index, edge_weight1, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight1, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight1  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True
    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index2 = L_indices.to(device)
    edge_weight2 = L_values.to(device)

    # row normalization
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight2 = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]
    #########################
    row, col = edge_index2
    deg = scatter_add(edge_weight2, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight2  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index2, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True
    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index3 = L_indices.to(device)
    edge_weight3 = L_values.to(device)

    # row normalization
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5).to(device)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight3 = deg_inv_sqrt[row] * edge_weight3 * deg_inv_sqrt[col]

    #################################   4th
    row, col = edge_index3
    deg = scatter_add(edge_weight3, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight3  # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index3, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)  # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())  # both source from which is non_zero in L_out

    L_in_hat = L_in  # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True
    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index4 = L_indices.to(device)
    edge_weight4 = L_values.to(device)

    # row normalization
    row, col = edge_index4
    deg = scatter_add(edge_weight4, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5).to(device)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight4 = deg_inv_sqrt[row] * edge_weight4 * deg_inv_sqrt[col]


    return (edge_index2,edge_index3,edge_index4), (edge_weight2, edge_weight3, edge_weight4)

def get_second_directed_adj_union(edge_index, num_nodes, dtype):
    '''
    Qin change to get union
    Args:
        edge_index:
        num_nodes:
        dtype:

    Returns:

    '''
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight      # all this is same with get directed adj
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)    # both point to which is non_zero in L_in
    L_out = torch.mm(p_dense, p_dense.t())   # both source from which is non_zero in L_out

    L_in_hat = L_in   # because L_in and L_out are symmetric, so their T is the same as them
    L_out_hat = L_out

    mask1 = (L_out != 0) & (L_in_hat == 0)  # Create boolean mask
    L_in_hat[mask1] = L_out[mask1]  # Update L_in_hat where mask is True

    mask2 = (L_in != 0) & (L_out_hat == 0)
    L_out_hat[mask2] = L_in[mask2]
    true_count = mask1.to(torch.int).sum().item()
    true_count2 = mask2.to(torch.int).sum().item()
    non_zero_in = L_in_hat.nonzero().size(0)
    non_zero_out = L_out_hat.nonzero().size(0)

    print(true_count, true_count2, non_zero_in,non_zero_out )

    # L_in_hat[(L_out != 0).to(torch.bool) & L_in_hat == 0] = L_out        # Qin learn: this is intersection
    # L_out_hat[(L_in != 0).to(torch.bool) & L_out_hat == 0] = L_in        # Qin learn: this is intersection
    # L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0        # 2-order symmetric_A

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def to_undirected(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    return edge_index, edge_weight


def to_undirectedBen(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    # print(edge_index, edge_index.shape)

    edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1])]

    set_edges = set(edges)

    unique_edges = list(set_edges)

    num_list = [0] * len(edges)
    history = []
    count = 0
    for i in range(len(edges)):
        if edges[i] in history:
            num_list[i] += 1
            count += 1
            # print("Duplicate: ", edges[i], num_list[i], count)
        else:
            history.append(edges[i])

    edge_index0 = [i[0] for i in unique_edges]
    edge_index1 = [i[1] for i in unique_edges]
    edge_index = torch.tensor([edge_index0, edge_index1])
    # print(edge_index, edge_index.shape)

    return edge_index





def remove_dupEdge(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    return edge_index, edge_weight


def link_prediction_evaluation(out_val, out_test, y_val, y_test):
    r"""Evaluates link prediction results.

    Args:
        out_val: (torch.FloatTensor) Log probabilities of validation edge output, with 2 or 3 columns.
        out_test: (torch.FloatTensor) Log probabilities of test edge output, with 2 or 3 columns.
        y_val: (torch.LongTensor) Validation edge labels (with 2 or 3 possible values).
        y_test: (torch.LongTensor) Test edge labels (with 2 or 3 possible values).

    :rtype:
        result_array: (np.array) Array of evaluation results, with shape (2, 5).
    """
    out = torch.exp(out_val).detach().to('cpu').numpy()
    y_val = y_val.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis=1)
    val_acc_full = accuracy_score(pred_label, y_val)
    # two-class evaluation
    out = out[y_val < 2, :2]
    y_val = y_val[y_val < 2]

    prob = out[:, 0] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    val_auc = roc_auc_score(y_val, prob)
    pred_label = np.argmax(out, axis=1)
    val_acc = accuracy_score(pred_label, y_val)
    val_f1_macro = f1_score(pred_label, y_val, average='macro')
    val_f1_micro = f1_score(pred_label, y_val, average='micro')

    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis=1)
    test_acc_full = accuracy_score(pred_label, y_test)
    # two-class evaluation
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]

    prob = out[:, 0] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis=1)
    test_acc = accuracy_score(pred_label, y_test)
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    return [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
            [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro]]

def organize4edgePred(new_x, edges, sampling_src_idx,neighbor_dist_list):
    '''

    Args:
        new_x:
        edges:
        sampling_src_idx:
        neighbor_dist_list:

    Returns:

    '''


    return data