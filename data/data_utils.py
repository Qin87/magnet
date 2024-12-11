import os
import random
from torch_geometric.datasets import QM9, MalNetTiny
import scipy
from torch_geometric.data import download_url
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)
import torch_geometric.transforms as transforms

from torch_geometric.datasets import Actor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor

from utils import get_norm_adj

try:
    import dgl
    from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset, FraudDataset, \
    FlickrDataset, YelpDataset, RedditDataset
except:
    print("dgl not imported, install chardet!")
import torch
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS

from data.Citation import citation_datasets
from data.preprocess import load_syn

def keep_all_data(edge_index, label, n_data, n_cls, train_mask):
    device = edge_index.device
    class_num_list = n_data
    data_train_mask = train_mask

    index_list = torch.arange(len(train_mask)).to(device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    train_node_mask = train_mask.to(device)

    edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)   # Qin revise May16
    return class_num_list, data_train_mask, idx_info, train_node_mask, edge_mask

def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

def load_snap_patents_mat(n_classes=5, root="dataset/"):
    dataset_drive_url = {"snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"}
    splits_drive_url = {"snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"}

    # Build dataset folder
    if not os.path.exists(f"{root}snap_patents"):
        os.mkdir(f"{root}snap_patents")

    # Download the data
    if not os.path.exists(f"{root}snap_patents/snap_patents.mat"):
        p = dataset_drive_url["snap-patents"]
        print(f"Snap patents url: {p}")
        gdown.download(
            id=dataset_drive_url["snap-patents"],
            output=f"{root}snap_patents/snap_patents.mat",
            quiet=False,
        )

    # Get data
    fulldata = scipy.io.loadmat(f"{root}snap_patents/snap_patents.mat")
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)
    num_nodes = int(fulldata["num_nodes"])
    years = fulldata["years"].flatten()
    label = even_quantile_labels(years, n_classes, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    # Download splits
    name = "snap-patents"
    if not os.path.exists(f"{root}snap_patents/{name}-splits.npy"):
        assert name in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[name],
            output=f"{root}snap_patents/{name}-splits.npy",
            quiet=False,
        )

    # Get splits
    splits_lst = np.load(f"{root}snap_patents/{name}-splits.npy", allow_pickle=True)
    train_mask, val_mask, test_mask = process_fixed_splits(splits_lst, num_nodes)
    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    dataset = DummyDataset(data, n_classes)

    return dataset

# adapting
# https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L221
# load splits from here https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
def process_fixed_splits(splits_lst, num_nodes):
    n_splits = len(splits_lst)
    train_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    for i in range(n_splits):
        train_mask[splits_lst[i]["train"], i] = 1
        val_mask[splits_lst[i]["valid"], i] = 1
        test_mask[splits_lst[i]["test"], i] = 1
    return train_mask, val_mask, test_mask

def scaled_edges(edge_index, num_nodes):
    inci_norm = 'dir'
    rm_gen_sLoop = False


    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # adj_norm = get_norm_adj(adj, norm=inci_norm)  # this is key: improve from 57 to 72

    adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
    # adj_t_norm = get_norm_adj(adj_t, norm=inci_norm)

    # adj_norm_in_in = get_norm_adj(adj @ adj, norm=inci_norm, rm_gen_sLoop=rm_gen_sLoop)

    adj_in_in = adj_t @ adj_t
    # Convert SparseTensor to edge_index
    # edge_index = adj_in_in.coo()[:2]
    row, col = adj_in_in.coo()[:2]
    edge_index = torch.stack([row, col], dim=0).to(torch.long)

    return edge_index

class DummyDataset(object):
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
def load_directedData(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_func, subset = args.Dataset.split('/')[0], args.Dataset.split('/')[1]
    if load_func in ['malnet']:
        dataset = MalNetTiny(root=args.data_path, split='train')

        # Access the first graph in the dataset
        # data = dataset[0]
        #
        # # Print some information about the dataset
        # print(f'Dataset: {dataset}:')
        # print('====================')
        # print(f'Number of graphs: {len(dataset)}')
        # print(f'Number of features: {dataset.num_features}')
        # print(f'Number of classes: {dataset.num_classes}')
        #
        # # Print information about the first graph
        # print('\nFirst graph:')
        # print('====================')
        # print(f'Number of nodes: {data.num_nodes}')
        # print(f'Number of edges: {data.num_edges}')
        # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        # print(f'Contains self-loops: {data.contains_self_loops()}')
        # print(f'Is undirected: {data.is_undirected()}')
        # return dataset
    elif load_func in ['arxiv-year']:
        path = args.data_path
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name="ogbn-arxiv")
        y = even_quantile_labels(dataset._data.node_year.flatten().numpy(), nclasses=5, verbose=False)
        # dataset._data.y = torch.as_tensor(y).reshape(-1, 1)
        dataset._data.y = torch.as_tensor(y)

        # if name in ["arxiv-year"]:
            # Datasets from https://arxiv.org/pdf/2110.14446.pdf have five splits stored
            # in https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
        split_number = 10
        num_nodes = y.shape[0]
        github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
        split_file_name = f"{load_func}-splits.npy"
        local_dir = os.path.join(path, load_func.replace("-", "_"), "raw")

        download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
        splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
        split_idx = splits[split_number % len(splits)]

        dataset._data.train_mask = get_mask(split_idx["train"], num_nodes)
        dataset._data.val_mask = get_mask(split_idx["valid"], num_nodes)
        dataset._data.test_mask = get_mask(split_idx["test"], num_nodes)

        # return train_mask, val_mask, test_mask
        # Tran, val and test masks are required during preprocessing. Setting them here to dummy values as
        # they are overwritten later for this dataset (see get_dataset_split function below)
        # dataset._data.train_mask, dataset._data.val_mask, dataset._data.test_mask = 0, 0, 0
        # Create directory for this dataset
        # os.makedirs(os.path.join(path, name.replace("-", "_"), "raw"), exist_ok=True)
    elif load_func in ['snap-patents']:
        dataset = load_snap_patents_mat(n_classes=5, root=args.data_path)
    elif load_func == "Cora" or load_func == "CiteSeer" or load_func == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(args.data_path, load_func, transform=T.NormalizeFeatures(), split='full')

    elif load_func in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(name=load_func, transform=transforms.ToSparseTensor(), root=args.data_path)
        # evaluator = Evaluator(name=name)
        split_idx = dataset.get_idx_split()
        dataset.data.train_mask = get_mask(split_idx["train"], dataset.data.num_nodes)
        dataset.data.val_mask = get_mask(split_idx["valid"], dataset.data.num_nodes)
        dataset.data.test_mask = get_mask(split_idx["test"], dataset.data.num_nodes)
        num_train_nodes = dataset._data.train_mask.sum().item()
        num_train_nodes0 = dataset._data.val_mask.sum().item()
        num_train_nodes1 = dataset._data.test_mask.sum().item()
        dataset.y = dataset.y.squeeze()
        dataset._data.y = dataset._data.y.squeeze()
        print(num_train_nodes, num_train_nodes0, num_train_nodes1)

    elif load_func in ["directed-roman-empire"]:
        dataset = DirectedHeterophilousGraphDataset(name=load_func, transform=transforms.NormalizeFeatures(), root=args.data_path)
    elif load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)

    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        if subset not in ['crocodile']:
            dataset = load_func(root=args.data_path, name=subset)
        else:
            dataset = load_func(root=args.data_path, name=subset, geom_gcn_preprocess=False)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path, is_undirected=False)
    elif load_func == 'WikiCS_U':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)        # get undirected
    elif load_func == 'cora_ml':
        dataset = citation_datasets(root='./cora_ml.npz')
    elif load_func == 'citeseer':
        dataset = citation_datasets(root='./citeseer_npz.npz')
    elif load_func in ['film']:
        dataset = Actor(root='../data/film', transform=T.NormalizeFeatures())

    elif load_func == 'dgl':    # Ben
        subset = subset.lower()
        dataset = load_dgl_graph(subset)
    elif load_func == 'telegram':
        dataset = load_syn(root='./telegram')
    else:
        # dataset = load_syn(args.data_path + load_func+ '/'+ subset, None)
        dataset = load_syn(load_func+ '/'+ subset, None)

    return dataset

def load_dgl_graph(subset):
    if subset == 'citeseer':    # Nodes: 3327, Edges: 9228, Number of Classes: 6
        return CiteseerGraphDataset(reverse_edge=False)
    elif subset == 'cora':  # Nodes: 2708, Edges: 10556, Number of Classes: 7
        return CoraGraphDataset(reverse_edge=False)
    elif subset == 'pubmed':    # Nodes: 19717, Edges: 88651
        dataset = PubmedGraphDataset(reverse_edge=False)
    elif subset== 'coauthor-cs':   # bidirected
        dataset = CoauthorCSDataset()
    elif subset== 'coauthor-ph':   # bidirected
        dataset = CoauthorPhysicsDataset()
    elif subset == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif subset == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif subset == 'reddit':
        dataset = RedditDataset()
    elif subset == 'fyelp':
        dataset = FraudDataset('yelp')
    elif subset == 'famazon':
        dataset = FraudDataset('amazon')
    elif subset == 'flickr':
        dataset = FlickrDataset()
    elif subset == 'yelp':
        dataset = YelpDataset()
    # all below not working
    elif subset == 'aifb':  # Nodes: 7262, Edges: 48810 (including reverse edges)
        dataset = dgl.data.rdf.AIFBDataset(insert_reverse=False)    # don't have data_x  #
        #  assortative , node classification
    elif subset =='mutag':  # Nodes: 27163, Edges: 148100 (including reverse edges), 2 class
        dataset = dgl.data.rdf.MUTAGDataset(insert_reverse=False)   # for graph classification
    elif subset == 'bgs':   # Nodes: 94806,  Edges: 672884 (including reverse edges), 2 class
        dataset = dgl.data.rdf.BGSDataset(insert_reverse=False)     # not work to load
    elif subset == 'am':   # Nodes: 881680  Edges: 5668682 (including reverse edges)
        dataset = dgl.data.rdf.AMDataset(insert_reverse=False)
    else:
        raise NotImplementedError
    return dataset

def random_planetoid_splits(data, y, train_ratio=0.7, val_ratio=0.1, percls_trn=20,  val_lb=30, num_splits=10, Flag=1):
    # Set new random planetoid splits based on provided ratios
    num_node = y.size()[0]
    data.train_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)
    data.val_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)
    data.test_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)

    for split_idx in range(num_splits):
        for i in range(y.max().item() + 1):
            index = (y == i).nonzero().view(-1)

            if Flag == 1:
                train_size = percls_trn
                val_size = val_lb
            else:       # If Flag is 0, use ratio split
                total = index.size(0)
                train_size = int(train_ratio * total)
                val_size = int(val_ratio * total)

            train_indices = index[:train_size]
            val_indices = index[train_size:train_size + val_size]
            test_indices = index[train_size + val_size:]

            # Assign masks
            data.train_mask[train_indices, split_idx] = 1
            data.val_mask[val_indices, split_idx] = 1
            data.test_mask[test_indices, split_idx] = 1
        index = index[torch.randperm(index.size(0))]

    return data

import os.path as osp
from typing import Callable, Optional
try:
    import gdown
except:
    pass
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
class DirectedHeterophilousGraphDataset(InMemoryDataset):
    r"""The directed heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace("-", "_")
        assert self.name in [
            "directed_roman_empire",
            "directed_amazon_ratings",
            "directed_questions",
        ]

        self.url = {
            "directed_roman_empire": "https://drive.google.com/uc?id=1atonwA1YqKMV3xWS7T04dRgfmDrsyRj8",
            "directed_amazon_ratings": "https://drive.google.com/uc?id=12Cyw0oZXLjPrebCficporBcIKiAgU5kc",
            "directed_questions": "https://drive.google.com/uc?id=1EnOvBehgLN3uCAQBXrGzGB1d3-aXS2Lk",
        }

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        gdown.download(self.url[self.name], f"{self.raw_dir}/{self.name}.npz", fuzzy=True)

    def process(self):
        raw = np.load(self.raw_paths[0], "r")
        x = torch.from_numpy(raw["node_features"])
        y = torch.from_numpy(raw["node_labels"])
        edge_index = torch.from_numpy(raw["edges"]).t().contiguous()
        train_mask = torch.from_numpy(raw["train_masks"]).t().contiguous()
        val_mask = torch.from_numpy(raw["val_masks"]).t().contiguous()
        test_mask = torch.from_numpy(raw["test_masks"]).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

def set_device(args):
    cuda_device = args.GPUdevice
    if torch.cuda.is_available():
        print("cuda Device Index:", cuda_device)
        device = torch.device("cuda:%d" % cuda_device)
    else:
        print("cuda is not available, using CPU.")
        device = torch.device("cpu")
    if args.CPU:
        device = torch.device("cpu")
        print("args.CPU true, using CPU.")

    return device

def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def even_quantile_labels(vals, nclasses, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


import numpy as np
from collections import defaultdict


def find_max_spanning_tree(edge_index, num_nodes, weights=None):
    """
    Find the maximum spanning tree of a graph using modified Kruskal's algorithm.

    Parameters:
    edge_index: numpy array or list of shape (2, num_edges) containing edges
    num_nodes: int, number of nodes in the graph
    weights: numpy array or list of shape (num_edges,) containing edge weights
            If None, all weights are set to 1

    Returns:
    mst_edges: list of edges in the maximum spanning tree
    """

    # Convert edge_index to list of edges with weights
    if weights is None:
        weights = np.ones(edge_index.shape[1])

    edges = [(edge_index[0, i], edge_index[1, i], weights[i])
             for i in range(edge_index.shape[1])]

    # Sort edges by weight in descending order
    edges.sort(key=lambda x: x[2], reverse=True)

    # Initialize disjoint set data structure
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Build maximum spanning tree
    mst_edges = []
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, w))
            if len(mst_edges) == num_nodes - 1:
                break

    return mst_edges


# Example usage
def example():
    # Example graph
    edge_index = np.array([[0, 0, 1, 1, 2],
                           [1, 2, 2, 3, 3]])
    num_nodes = 4
    weights = np.array([4, 3, 5, 2, 1])

    mst = find_max_spanning_tree(edge_index, num_nodes, weights)
    print("Maximum Spanning Tree edges:", mst)
    return mst