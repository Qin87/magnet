import os
from datetime import datetime
import torch
from nets.geometric_baselines import GCN_JKNet, GPRGNN
from nets.models import JKNet, create_MLP, create_SGC, create_pgnn, GPRGNNNet1
# sys.path.append('./Signum_quaternion/QuaNet_node_prediction_one_laplacian_Qin')
# sys.path.append('./Signum_quaternion/')
# print('sys path is',sys.path)

from nets.Signum_quaternion import QuaNet_node_prediction_one_laplacian_Qin
from nets.Signum import SigMaNet_node_prediction_one_laplacian_Qin
from edge_nets.edge_data import to_undirectedBen
from gens import test_directed
from nets import create_gcn, create_gat, create_sage

from data.data_utils import random_planetoid_splits, load_directedData
from nets.APPNP_Ben import APPNP_Model, ChebModel, SymModel
# from nets.DGCN import SymModel
# from nets.DiGCN import DiModel, DiGCN_IB
from nets.DiG_NoConv import (create_DiG_MixIB_SymCat_Sym_nhid,
                             create_DiG_MixIB_SymCat_nhid, create_DiG_IB_SymCat_nhid, create_DiG_IB_Sym_nhid, create_DiG_IB_Sym_nhid_para,
                             create_DiG_IB_nhid_para, create_DiSAGESimple_nhid, create_Di_IB_nhid, Si_IB_XBN_nhid, DiGCN_IB_XBN_nhid_para)
# from nets.DiG_NoConv import  create_DiG_IB
from nets.GIN_Ben import create_GIN
from nets.Sym_Reg import create_SymReg_add, create_SymReg_para_add
# from nets.UGCL import UGCL_Model_Qin
from nets.sparse_magnet import ChebNet_Ben, ChebNet_BenQin
import torch.nn.init as init

def init_model(model):
    # Initialize weights and biases of all parameters
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                init.xavier_uniform_(param)  # Initialize weights using Xavier initialization
            else:
                init.constant_(param, 0)  # Initialize biases to zero
        elif 'bias' in name:
            init.constant_(param, 0)  # Initialize biases to zero for bias parameters
    # Initialize parameters of batch normalization layers
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_parameters()  # Res

def CreatModel(args, num_features, n_cls, data_x,device):
    if args.net == 'pgnn':
        model = create_pgnn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout, layer =args.layer)
    elif args.net == 'mlp':
        model = create_MLP(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer)
    elif args.net == 'sgc':
        model = create_SGC(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer,K=args.K)

    elif args.net == 'jk':
        model = JKNet(in_channels=num_features,
                        out_channels=n_cls,
                        num_hid=args.feat_dim,
                        K=args.K,
                        alpha=args.alpha,
                        dropout=args.dropout,layer=args.layer)
    elif args.net == 'gprgnn':
        # model = GPRGNNNet1_Qin(in_channels=num_features,      # a bit worse
        model = GPRGNNNet1(in_channels=num_features,
                            out_channels=n_cls,
                            num_hid=args.feat_dim,
                            ppnp=args.ppnp,
                            K=args.K,
                            alpha=args.alpha,
                            Init=args.Init,
                            Gamma=args.Gamma,
                            dprate=args.dprate,
                            dropout=args.dropout)

    elif args.net == 'GIN':
        model = create_GIN(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
    elif args.net == 'Cheb':
        model = ChebModel(num_features, n_cls, K=args.K,filter_num=args.feat_dim, dropout=args.dropout,layer=args.layer).to(device)
        # model = create_Cheb(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, K=args.K).to(device)
    elif args.net == 'JKNet':
        model = GCN_JKNet(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, layer=args.layer)
    elif args.net == 'GPRGNN':
        model = GPRGNN(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, args= args)
    elif args.net == 'APPNP':
        model = APPNP_Model(num_features, n_cls,args.feat_dim, alpha=args.alpha,dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith(('Di', 'Qi', 'Wi', 'Ci')):        # GCN  -->  SAGE
        if args.net[-2:] not in ['i2', 'u2', 'i3', 'u3', 'i4', 'u4']:
            model = create_DiSAGESimple_nhid(args.net[2], num_features, n_cls, args).to(device)     # Jun22
        else:
            if args.net[3:].startswith(('Sym', 'Qym')):
                if args.net[6:].startswith('Cat'):
                    if args.net[9:].startswith('Mix'):
                        if args.net[12:].startswith(('Sym', 'Qym')):
                            model = create_DiG_MixIB_SymCat_Sym_nhid(args.net[2], num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                        else:
                            model = create_DiG_MixIB_SymCat_nhid(args.net[2],num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                    else:
                        model = create_DiG_IB_SymCat_nhid(args.net[2], num_features, args.feat_dim, n_cls, args.dropout, args.layer, args.ibx1).to(device)
                else:
                    if args.paraD:
                        model = create_DiG_IB_Sym_nhid_para(args.net[2], num_features,  n_cls, args).to(device)
                    else:
                        model = create_DiG_IB_Sym_nhid(args.net[2], num_features,  n_cls, args).to(device)
            else:
                if args.paraD:
                    model = DiGCN_IB_XBN_nhid_para(args.net[2], num_features,  n_cls,  args).to(device)
                else:
                    if args.net.startswith('Ci'):
                        model = Si_IB_XBN_nhid(args.net[2], num_features, n_cls, args=args).to(device)
                    else:
                        model = create_Di_IB_nhid(m=args.net[2], nfeat=num_features, nclass=n_cls, args=args).to(device)
    elif args.net.startswith(('Sym', 'Qym')):
        # model = create_SymReg(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        model = SymModel(num_features, n_cls, filter_num=args.feat_dim,dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith(('addSym', 'addQym')):
        if not args.net.endswith('para'):
            model = create_SymReg_add(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        else:
            model = create_SymReg_para_add(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
    elif args.net.startswith('Mag'):
        if args.net[3:].startswith('Qin'):
            model = ChebNet_BenQin(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                                activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
        else:
            model = ChebNet_Ben(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                            activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
    elif args.net.startswith('Sig'):
        model = SigMaNet_node_prediction_one_laplacian_Qin(num_features, K=args.K, hidden=args.feat_dim, label_dim=n_cls,i_complex=args.i_complex, layer=args.layer,
                                                           activation=args.activation,follow_math=args.follow_math, gcn=args.gcn, net_flow=args.netflow, unwind=True).to(device)
    elif args.net.startswith('Qua'):
        model = QuaNet_node_prediction_one_laplacian_Qin(device, num_features, K=args.K, hidden=args.feat_dim, label_dim=n_cls,
                                                     layer=args.layer, unwind=True,
                                                     quaternion_weights=args.qua_weights, quaternion_bias=args.qua_bias).to(device)

    else:
        if args.net == 'GCN':
            model = create_gcn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, norm= args.gcnconv_norm)
        elif args.net == 'GAT':
            model = create_gat(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, head=args.heads)
        elif args.net == "SAGE":
            model = create_sage(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout,nlayer=args.layer)
        else:
            raise NotImplementedError("Not Implemented Architecture!"+ args.net)
    model = model.to(device)
    init_model(model)
    return model

def get_name(args, IsDirectedGraph):
    dataset_to_print = args.Dataset.replace('/', '_')
    if not IsDirectedGraph:
        dataset_to_print = dataset_to_print + 'Undire'
    else:
        dataset_to_print = dataset_to_print + 'Direct'
    if args.net.startswith('Wi'):
        net_to_print = args.net + str(args.W_degree) + '_'
    elif args.net.startswith('Mag'):
        net_to_print = args.net + str(args.q)
    else:
        net_to_print = args.net
    if args.net[1:3] == 'iA' or args.net == 'GAT':
        net_to_print = net_to_print + '_Head' + str(args.heads)

    if args.net[1:].startswith('iG'):
        if args.paraD:
            net_to_print = net_to_print + 'paraD' + str(args.coeflr)
    if args.feat_dim != 64:
        net_to_print = net_to_print + str(args.feat_dim) + 'hid_'

    return net_to_print, dataset_to_print


def log_file(net_to_print, dataset_to_print, args):
    log_file_name = 'QymNorm_NoSelfLoop' + dataset_to_print+'_'+net_to_print+'_lay'+str(args.layer)+'_lr'+str(args.lr)+'_NoImp'+str(args.NotImproved)+'q'+str(args.q)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name_with_timestamp = f"{log_file_name}_{timestamp}.log"

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp

def load_dataset(args):
    dataset = load_directedData(args)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    data = dataset[0]
    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        edges_weight = torch.FloatTensor(data.edge_weight)
    except:
        edges_weight = None

    # copy GraphSHA
    if args.Dataset.split('/')[0].startswith('dgl'):
        edge_types = data.etypes
        print("Available edge types:", edge_types)
        num_edge_types = len(data.etypes)

        if num_edge_types == 1:
            # Only one edge type, retrieve edges normally
            edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        else:
            # Multiple edge types
            print("Edge types:", data.etypes)
            all_src = []
            all_dst = []

            for etype in data.etypes:
                src, dst = data.edges(etype=etype)
                all_src.append(src)
                all_dst.append(dst)

            # Concatenate all source and destination nodes
            all_src = torch.cat(all_src)
            all_dst = torch.cat(all_dst)

            # Combine source and destination to form edges
            edges = torch.stack([all_src, all_dst])
        data_y = data.ndata['label']
        print(data.ndata.keys())
        try:
            data_x = data.ndata['feat']
        except:
            data_x = data.ndata['feature']
        if args.Dataset.split('/')[1].startswith(('reddit', 'yelp')):
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())

        elif args.Dataset.split('/')[1].startswith(('Fyelp', 'Famazon')):
            data = random_planetoid_splits(data, data_y, train_ratio=0.7, val_ratio=0.1, Flag=0)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
        else:
            data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        # for multi-label dataset
        if args.Dataset.split('/')[1].startswith('yelp'):
            data_y = data_y[:, 1]
        dataset_num_features = data_x.shape[1]

    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        if len(data.train_mask.shape) > 1 and data.train_mask.size(-1) == 10:
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
        else:
            data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    IsDirectedGraph = test_directed(edges)        # time consuming
    # IsDirectedGraph = args.IsDirectedData
    print("This is directed graph: ", IsDirectedGraph)
    print("data_x", data_x.shape)  # [11701, 300])

    if IsDirectedGraph and args.to_undirected:
        edges = to_undirectedBen(edges)
        IsDirectedGraph = False
        print("Converted to undirected data")

    return data_x, data_y, edges, edges_weight, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph