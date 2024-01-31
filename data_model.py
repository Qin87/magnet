import os

import torch

from nets import create_gcn, create_gat, create_sage
import os.path as osp

from data_utils import load_directedData, get_dataset
from nets.DGCN import SymModel
from nets.DiGCN import DiModel, DiGCN_IB
from nets.geometric_baselines import GIN_ModelBen, ChebModelBen, APPNP_ModelBen


def CreatModel(args, num_features, n_cls, data_x,device):
    if args.net == 'GCN':
        model = create_gcn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.layer)
    elif args.net == 'GAT':
        model = create_gat(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.layer)
    elif args.net == "SAGE":
        model = create_sage(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.layer)
    elif args.net == 'GIN':
        model = GIN_ModelBen(data_x.size(-1), n_cls, filter_num=args.num_filter,
                             dropout=args.dropout, layer=args.layer)
    elif args.net == 'Cheb':
        model = ChebModelBen(data_x.size(-1), n_cls, K=args.K,
                             filter_num=args.num_filter, dropout=args.dropout,
                             layer=args.layer).to(device)
    elif args.net == 'APPNP':
        model = APPNP_ModelBen(data_x.size(-1), n_cls,
                               filter_num=args.num_filter, alpha=args.alpha,
                               dropout=args.dropout, layer=args.layer).to(device)
    elif args.net == 'DiG':
        if not args.net[-2:] == 'ib':
            model = DiModel(data_x.size(-1), n_cls, filter_num=args.num_filter,
                            dropout=args.dropout, layer=args.layer).to(device)
        else:
            model = DiGCN_IB(data_x.size(-1), hidden=args.num_filter,
                             n_cls=n_cls, dropout=args.dropout,
                             layer=args.layer).to(device)

    elif args.net == 'SymDiGCN':
        model = SymModel(data_x.size(-1), n_cls, filter_num=args.num_filter,
                         dropout=args.dropout, layer=args.layer).to(device)
    else:
        raise NotImplementedError("Not Implemented Architecture!")
    try:
        print(model)  # # StandGCN2((conv1): GCNConv(3703, 64)  (conv2): GCNConv(64, 6))
    except:
        pass
    return model


def load_dataset(args,device):
    if args.IsDirectedData:
        dataset = load_directedData(args)
    else:
        path = args.data_path
        path = osp.join(path, args.undirect_dataset)
        dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print("Dataset is ", dataset, "\nChosen from DirectedData: ", args.IsDirectedData)

    # if os.path.isdir(log_path) is False:
    #     os.makedirs(log_path)

    data = dataset[0]
    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        data.edge_weight = torch.FloatTensor(data.edge_weight)
    except:
        data.edge_weight = None

    # if args.to_undirected:
    #     data.edge_index = to_undirected(data.edge_index)

    # copy GraphSHA
    if args.IsDirectedData and args.Direct_dataset.split('/')[0].startswith('dgl'):
        edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        data_y = data.ndata['label']
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (
            data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
        data_x = data.ndata['feat']
        dataset_num_features = data_x.shape[1]
    # elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
    elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_x = data.x
        dataset_num_features = dataset.num_features

        data_y = data_y.long()
        n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
        n_cls = torch.tensor(n_cls).to(device)

        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio,
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls),
                                                                    labeling_ratio=0.1,
                                                                    all_idx=[i for i in range(data.x.shape[0])],
                                                                    all_label=data.y.cpu().detach().numpy(),
                                                                    nclass=n_cls)

        data_train_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_maskOrigin[train_idx] = True
        data_val_maskOrigin[valid_idx] = True
        data_test_maskOrigin[test_idx] = True
        train_idx = data_train_maskOrigin.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]
    elif dataset == 'Amazon-Photo':
        pass
    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(),data.test_mask.clone())
        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    # IsDirectedGraph = test_directed(edges)
    # print("This is directed graph: ", IsDirectedGraph)
    # print("data_x", data_x.shape)  # [11701, 300])


    data = data.to(device)
    data_x = data_x.to(device)
    data_y = data_y.long().to(device)
    edges = edges.to(device)
    # dataset_num_features = dataset_num_features.to(device)
    data_train_maskOrigin = data_train_maskOrigin.to(device)
    data_val_maskOrigin = data_val_maskOrigin.to(device)
    data_test_maskOrigin = data_test_maskOrigin.to(device)
    return data, data_x, data_y, edges, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin