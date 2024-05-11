################################
# compared to All2Main.py, this version to ensure that when I stop the process half way, it still could print the result.
################################
import signal
import statistics
import sys
import time

import random
import numpy as np
import torch
import torch.nn.functional as F

from args import parse_args
from data_utils import get_idx_info, make_longtailed_data_remove, keep_all_data
from edge_nets.Edge_DiG_ import edge_prediction
from edge_nets.edge_data import get_appr_directed_adj, get_second_directed_adj, get_second_directed_adj_union, get_third_directed_adj, get_third_directed_adj_union, get_4th_directed_adj, \
    get_4th_directed_adj_union
from gens import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, \
    sampling_idx_individual_dst, neighbor_sampling_BiEdge, neighbor_sampling_BiEdge_bidegree, \
    neighbor_sampling_bidegree, neighbor_sampling_bidegreeOrigin, neighbor_sampling_bidegree_variant1, \
    neighbor_sampling_bidegree_variant2, neighbor_sampling_reverse, neighbor_sampling_bidegree_variant2_1, \
    neighbor_sampling_bidegree_variant2_0, neighbor_sampling_bidegree_variant2_1_
from data_model import CreatModel, load_dataset, log_file
from nets.src2 import laplacian
from nets.src2.quaternion_laplacian import process_quaternion_laplacian
from preprocess import F_in_out_Qin, F_in_out
from utils import CrossEntropy
from sklearn.metrics import balanced_accuracy_score, f1_score
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist

import warnings

from utils0.perturb import composite_perturb
from torch_geometric.data import Data
warnings.filterwarnings("ignore")

def signal_handler(sig, frame):
    global end_time
    end_time = time.time()
    print("Process interrupted!")
    calculate_time()
    log_results()
    sys.exit(0)

def calculate_time():
    global start_time, end_time
    if start_time is not None and end_time is not None:
        elapsed_time = end_time - start_time
        print("Total time: {:.2f} seconds".format(elapsed_time))

def log_results():
    global start_time, end_time
    if start_time is not None and end_time is not None:
        with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
            elapsed_time = end_time - start_time
            print("Total time: {:.2f} seconds".format(elapsed_time), file=log_file)
            if len(macro_F1) > 1:
                average = statistics.mean(macro_F1)
                std_dev = statistics.stdev(macro_F1)
                print(net_to_print, args.layer, str(args.to_undirected), 'Macro F1: ', f"{average:.3f}±{std_dev:.2f},{len(macro_F1):2d}splits", file=log_file)
                print(net_to_print, args.layer, str(args.to_undirected), 'Macro F1: ', f"{average:.3f}±{std_dev:.2f},{len(macro_F1):2d}splits")
            elif len(macro_F1) == 1:
                print(net_to_print, args.layer, str(args.to_undirected), 'Macro F1: ', f"{macro_F1[0]:.3f}", file=log_file)
                print(net_to_print, args.layer, str(args.to_undirected), 'Macro F1: ', f"{macro_F1[0]:.3f}")
            else:
                print("not a single split is finished")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
def train_UGCL(pos_edges, neg_edges, size, train_index, val_index):
    model.train()

    if args.graph:
        pos_edges1, neg_edges1 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
        pos_edges2, neg_edges2 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
    else:
        pos_edges1, neg_edges1 = pos_edges, neg_edges
        pos_edges2, neg_edges2 = pos_edges, neg_edges

    if args.laplacian:
        q1, q2 = random.sample(np.arange(0, 0.5, 0.1).tolist(), 2)
        q1, q2 = np.pi * q1, np.pi * q2
    else:
        q1, q2 = args.q, args.q

    if args.composite:
        pos_edges1, neg_edges1 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
        pos_edges2, neg_edges2 = pos_edges, neg_edges
        q2 = random.sample(np.arange(0, 0.5, 0.1).tolist(), 1)[0]
        q1, q2 = args.q, np.pi * q2
    ### Augmenting
    ####################################################################################

    z1 = model(X_real, X_img, q1, pos_edges1, neg_edges1, args, size, train_index)
    z2 = model(X_real, X_img, q2, pos_edges2, neg_edges2, args, size, train_index)
    contrastive_loss = model.contrastive_loss(z1, z2, batch_size=args.batch_size)
    label_loss = model.label_loss(z1, z2, data_y[data_train_mask])
    train_loss = args.loss_weight * contrastive_loss + label_loss

    train_loss.backward()

    model.eval()
    z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, val_index)
    z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, val_index)
    val_loss = model.label_loss(z1, z2, data_y[data_val_mask])

    optimizer.step()
    scheduler.step(val_loss, epoch)


def train(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real, norm_imag,
          X_img_i, X_img_j, X_img_k,norm_img_i,norm_img_j, norm_img_k, Quaedge_index):

    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    new_edge_index=None
    new_x = None
    new_y = None
    new_y_train = None
    try:
        model.train()
    except:
        pass

    optimizer.zero_grad()
    if args.AugDirect == 0:
        if args.net.startswith('Sym') or args.net.startswith('addSym'):
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
            # out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)
        elif args.net.startswith('DiG'):
            if args.net[3:].startswith('Sym'):
                # out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, new_SparseEdges, edge_weight)
                out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight,SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)  # (1,5,183)
            # out = out.permute(2, 1, 0).squeeze()        # (183,5)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        elif args.net.startswith('Qua'):  # TODO might change
            out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
        else:
            out = model(data_x, edges)
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])
        criterion(out[data_train_mask], data_y[data_train_mask]).backward()
        # print('Aug', args.AugDirect, ',edges', edges.shape[1], ',x', data_x.shape[0])
    else:
        if epoch > args.warmup:
            # identifying source samples
            prev_out_local = prev_out[train_idx]
            sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, args.tau, args.max, args.no_mask)

            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

            # type 1
            sampling_src_idx = sampling_src_idx.to(torch.long).to(data_y.device)  # Ben for GPU error
            _new_y = data_y[sampling_src_idx].clone()
            new_y_train = torch.cat((data_y[data_train_mask], _new_y), dim=0)
            new_y = torch.cat((data_y, _new_y), dim=0)

            if args.AugDirect == 1:
                # new_edge_index = neighbor_sampling(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 100:
                data100 = Data(x=torch.tensor(new_x, dtype=torch.float),
                            edge_index=torch.tensor(edges, dtype=torch.long),
                            y=torch.tensor(new_y, dtype=torch.float))
                new_edge_index = edge_prediction(args, data100, sampling_src_idx, neighbor_dist_list)
                # new_edge_index = edge_prediction_test()
                # pass
            elif args.AugDirect == -1:
                # new_edge_index = neighbor_sampling_reverse(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_reverse(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)

            elif args.AugDirect == 2:
                # new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges[:, train_edge_mask],
                #                                           sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 4:
                # new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges[:, train_edge_mask],,sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 20:
                # type 1
                # new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)  # has two types

            elif args.AugDirect == 21:
                # new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 22:
                # new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 23:
                # new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)

            elif args.AugDirect == 231:
                new_edge_index = neighbor_sampling_bidegree_variant2_1(args, data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 2311:
                new_edge_index = neighbor_sampling_bidegree_variant2_1_(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)

            elif args.AugDirect == 230:
                new_edge_index = neighbor_sampling_bidegree_variant2_0(data_x.size(0), edges, sampling_src_idx, neighbor_dist_list)


            else:
                raise NotImplementedError



            # print('Aug', args.AugDirect, ',edges', new_edge_index.shape[1], ',x',new_x.shape[0])
        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
            new_edge_index = duplicate_neighbor(data_x.size(0), edges[:,train_edge_mask], sampling_src_idx)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

            # type 1
            sampling_src_idx = sampling_src_idx.to(torch.long).to(data_y.device)  # Ben for GPU error
            _new_y = data_y[sampling_src_idx].clone()
            new_y_train = torch.cat((data_y[data_train_mask], _new_y), dim=0)
            # out = model(new_x, new_edge_index)
            # Sym_edges = torch.cat([edges, new_edge_index], dim=1)
            # Sym_edges = torch.unique(Sym_edges, dim=1)
            new_y = torch.cat((data_y, _new_y), dim=0)


        if args.net.startswith('Sym') or args.net.startswith('addSym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(new_edge_index, new_y.size(-1), data.edge_weight)  # all edge and all y, not only train
            # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(new_edge_index, new_y.size(-1), data.edge_weight)  # all edge and all
            # y, not only train
            # out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)  # all edges(aug+all edges)
            out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight)  # all edges(aug+all edges)
        elif args.net.startswith('DiG'):
            edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(), new_y.size(-1), new_x.dtype)
            edge_index1 = edge_index1.to(device)
            edge_weights1 = edge_weights1.to(device)
            if args.net[-2:] == 'ib' or args.net[-2:] == 'ub':
                if args.net[-2:] == 'ib':
                    edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index2, edge_weights2 = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                edge_index2 = edge_index2.to(device)
                edge_weights2 = edge_weights2.to(device)
                new_SparseEdges = (edge_index1, edge_index2)
                edge_weight = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            elif args.net[-2:] == 'i3' or args.net[-2:] == 'u3':
                if args.net[-2:] == 'i3':
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                new_SparseEdges = (edge_index1,)+ edge_index_tuple      # typo in Apr17
                edge_weight = (edge_weights1,)+edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            elif args.net[-2:] == 'i4' or args.net[-2:] == 'u4':
                if args.net[-2:] == 'i4':
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                new_SparseEdges = (edge_index1,) + edge_index_tuple     # typo in Apr17
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            else:
                new_SparseEdges = edge_index1
                edge_weight = edge_weights1
            del edge_index1, edge_weights1
            if args.net[3:].startswith('Sym'):
                data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(new_edge_index, new_y.size(-1), data.edge_weight)
                # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges, data_y.size(-1), data.edge_weight)
                out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, new_SparseEdges, edge_weight)
            else:
                out = model(new_x, new_SparseEdges, edge_weight)  # all data+ aug
        elif args.net.startswith(('Mag', 'Sig', 'Qua')):
            new_x_cpu = new_x.cpu()
            newX_img = torch.FloatTensor(new_x_cpu).to(device)
            newX_real = torch.FloatTensor(new_x_cpu).to(device)
            if args.net.startswith('Mag'):
                out = model(newX_real, newX_img, new_edge_index, args.q, edge_weight)
            elif args.net.startswith('Sig'):
                NewSigedge_index, Newnorm_real, Newnorm_imag = laplacian.process_magnetic_laplacian(edge_index=new_edge_index, gcn=gcn, net_flow=args.netflow, x_real=newX_real, edge_weight=edge_weight,
                                                                                                    normalization='sym', return_lambda_max=False)
                out = model(newX_real, newX_img, Newnorm_real, Newnorm_imag, NewSigedge_index)  # TODO revise!
            elif args.net.startswith('Qua'):  # TODO might change
                NX_img_i = torch.FloatTensor(new_x_cpu).to(device)
                NX_img_j = torch.FloatTensor(new_x_cpu).to(device)
                NX_img_k = torch.FloatTensor(new_x_cpu).to(device)
                NewQuaedge_index, Nnorm_real, Nnorm_img_i, Nnorm_img_j, Nnorm_img_k = process_quaternion_laplacian(
                    edge_index=new_edge_index, x_real=newX_real, edge_weight=edge_weight,
                    normalization='sym', return_lambda_max=False)

                out = model(newX_real, NX_img_i, NX_img_j, NX_img_k,Nnorm_img_i, Nnorm_img_j, Nnorm_img_k, Nnorm_real,NewQuaedge_index)


        else:
            out = model(new_x, new_edge_index)  # all data + aug

        prev_out = (out[:data_x.size(0)]).detach().clone()
        add_num = out.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)

        optimizer.zero_grad()
        criterion(out[new_train_mask], new_y_train).backward()

    with torch.no_grad():
        model.eval()
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])  # train_edge_mask????
        # out = model(data_x, edges)
        if args.net.startswith('Sym') or args.net.startswith('addSym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges, data_y.size(-1), data.edge_weight)  # all original data, no augmented data
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
            # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges, data_y.size(-1), data.edge_weight)  # all original data,
            # no augmented data
            # out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)

        elif args.net.startswith('DiG'):
            # must keep this, don't know why, but will be error without it----to analysis it later
            edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)
            edge_index1 = edge_index1.to(device)
            edge_weights1 = edge_weights1.to(device)
            if args.net[-2:] == 'ib' or args.net[-2:] == 'ub':
                if args.net[-2:] == 'ib':
                    edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index2, edge_weights2 = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                edge_index2 = edge_index2.to(device)
                edge_weights2 = edge_weights2.to(device)
                SparseEdges = (edge_index1, edge_index2)
                edge_weight = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            elif args.net[-2:] == 'i3' or args.net[-2:] == 'u3':
                if args.net[-2:] == 'i3':
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1, )+ edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            elif args.net[-2:] == 'i4' or args.net[-2:] == 'u4':
                if args.net[-2:] == 'i4':
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1,) + edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            else:
                SparseEdges = edge_index1
                edge_weight = edge_weights1
            del edge_index1, edge_weights1
            if args.net[3:].startswith('Sym'):
                data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges, data_y.size(-1), data.edge_weight)
                out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        elif args.net.startswith('Qua'):  # TODO might change
            out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
        else:
            out = model(data_x, edges)
        val_loss = F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss, epoch)

    return val_loss, new_edge_index, new_x, new_y, new_y_train

def train_keepAug(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight,X_real, X_img, edge_Qin_in_tensor, edge_Qin_out_tensor, Sigedge_index, norm_real, norm_imag,
                  new_edge_index, new_x, new_y, new_y_train, Quaedge_index):
    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    try:
        model.train()
    except:
        pass

    optimizer.zero_grad()
    if args.AugDirect == 0:
        if args.net.startswith('Sym')  or args.net.startswith('addSym'):
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
            # out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)
        elif args.net.startswith('DiG'):
            if args.net[3:].startswith('Sym'):
                # out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, new_SparseEdges, edge_weight)
                out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight,SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)      # (1,5,183)
            # out = out.permute(2, 1, 0).squeeze()        # (183,5)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        else:
            out = model(data_x, edges)
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])
        criterion(out[data_train_mask], data_y[data_train_mask]).backward()
        # print('Aug', args.AugDirect, ',edges', edges.shape[1], ',x', data_x.shape[0])
    else:
        if args.net.startswith('Sym') or args.net.startswith('addSym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(new_edge_index, new_y.size(-1), data.edge_weight)  # all edge and all y, not only train
            # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(new_edge_index, new_y.size(-1), data.edge_weight)  # all edge and all
            # y, not only train
            # out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)  # all edges(aug+all edges)
            out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight)  # all edges(aug+all edges)
        elif args.net.startswith('DiG'):
            edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(), new_y.size(-1), new_x.dtype)
            edge_index1 = edge_index1.to(device)
            edge_weights1 = edge_weights1.to(device)
            if args.net[-2:] == 'ib' or args.net[-2:] == 'ub':
                if args.net[-2:] == 'ib':
                    edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index2, edge_weights2 = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                edge_index2 = edge_index2.to(device)
                edge_weights2 = edge_weights2.to(device)
                new_SparseEdges = (edge_index1, edge_index2)
                edge_weight = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            elif args.net[-2:] == 'i3' or args.net[-2:] == 'u3':
                if args.net[-2:] == 'i3':
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple  = get_third_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1,) + edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            elif args.net[-2:] == 'i4' or args.net[-2:] == 'u4':
                if args.net[-2:] == 'i4':
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1,) + edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            else:
                new_SparseEdges = edge_index1
                edge_weight = edge_weights1
            del edge_index1, edge_weights1
            if args.net[3:].startswith('Sym'):
                data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(new_edge_index, new_y.size(-1), data.edge_weight)
                # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges, data_y.size(-1), data.edge_weight)
                out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, new_SparseEdges, edge_weight)
            else:
                out = model(new_x, new_SparseEdges, edge_weight)  # all data+ aug
        elif args.net.startswith('Mag'):
            new_x_cpu = new_x.cpu()
            newX_img = torch.FloatTensor(new_x_cpu).to(device)
            newX_real = torch.FloatTensor(new_x_cpu).to(device)
            # out = model(newX_real, newX_img, edges, args.q, edge_weight).permute(2, 1, 0).squeeze()
            out = model(newX_real, newX_img, new_edge_index, args.q, edge_weight)
        elif args.net.startswith('Sig'):
            new_x_cpu = new_x.cpu()
            newX_img = torch.FloatTensor(new_x_cpu).to(device)
            newX_real = torch.FloatTensor(new_x_cpu).to(device)
            NewSigedge_index, Newnorm_real,  Newnorm_imag = laplacian.process_magnetic_laplacian(edge_index=new_edge_index, gcn=gcn, net_flow=args.netflow, x_real=newX_real, edge_weight=edge_weight,
                                                                                    normalization='sym', return_lambda_max=False)
            out = model(newX_real, newX_img, Newnorm_real, Newnorm_imag, NewSigedge_index)     # TODO revise!
        else:
            out = model(new_x, new_edge_index)  # all data + aug

        prev_out = (out[:data_x.size(0)]).detach().clone()
        add_num = out.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)

        optimizer.zero_grad()
        criterion(out[new_train_mask], new_y_train).backward()

    with torch.no_grad():
        model.eval()
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])  # train_edge_mask????
        # out = model(data_x, edges)
        if args.net.startswith('Sym') or args.net.startswith('addSym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges, data_y.size(-1), data.edge_weight)  # all original data, no augmented data
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
            # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges, data_y.size(-1), data.edge_weight)  # all original data,
            # # no augmented data
            # out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)

        elif args.net.startswith('DiG'):
            # must keep this, don't know why, but will be error without it----to analysis it later
            edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)
            edge_index1 = edge_index1.to(device)
            edge_weights1 = edge_weights1.to(device)
            if args.net[-2:] == 'ib' or args.net[-2:] == 'ub':
                if args.net[-2:] == 'ib':
                    edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index2, edge_weights2 = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                edge_index2 = edge_index2.to(device)
                edge_weights2 = edge_weights2.to(device)
                SparseEdges = (edge_index1, edge_index2)
                edge_weight = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            elif args.net[-2:] == 'i3' or args.net[-2:] == 'u3':
                if args.net[-2:] == 'i3':
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_third_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1,) + edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            elif args.net[-2:] == 'i4' or args.net[-2:] == 'u4':
                if args.net[-2:] == 'i4':
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
                else:
                    edge_index_tuple, edge_weights_tuple = get_4th_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
                SparseEdges = (edge_index1,) + edge_index_tuple
                edge_weight = (edge_weights1,) + edge_weights_tuple
                del edge_index_tuple, edge_weights_tuple
            else:
                SparseEdges = edge_index1
                edge_weight = edge_weights1
            del edge_index1, edge_weights1
            if args.net[3:].startswith('Sym'):
                data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges, data_y.size(-1), data.edge_weight)
                out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        else:
            out = model(data_x, edges)
        val_loss= F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss, epoch)

    return val_loss, new_edge_index, new_x, new_y, new_y_train

@torch.no_grad()
def test():
    model.eval()
    if args.net.startswith('Sym') or args.net.startswith('addSym'):
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges, data_y.size(-1), data.edge_weight)
        logits = model(data_x, edges[:, train_edge_mask], edge_in, in_weight, edge_out, out_weight)
    elif args.net.startswith('DiG'):
        if args.net[3:].startswith('Sym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges, data_y.size(-1), data.edge_weight)
            logits = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
        else:
            logits = model(data_x, SparseEdges, edge_weight)
    elif args.net.startswith('Mag'):
        # logits = model(X_real, X_img, edges, args.q, edge_weight).permute(2, 1, 0).squeeze()
        logits = model(X_real, X_img, edges, args.q, edge_weight)
    elif args.net.startswith('Sig'):  # TODO might change
        logits = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    elif args.net.startswith('Qua'):  # TODO might change
        logits = model(X_real, X_img_i, X_img_j, X_img_k, norm_imag_i, norm_imag_j, norm_imag_k, norm_real, Quaedge_index)
    else:
        logits = model(data_x, edges[:, train_edge_mask])
    accs, baccs, f1s = [], [], []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data_y[mask].cpu().numpy()
        acc = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    return accs, baccs, f1s


def test_UGCL():
    model.eval()
    # if args.net.startswith('Sym') or args.net.startswith('addSym'):
    #     logits = model(data_x, edges[:, train_edge_mask], edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)
    # elif args.net.startswith('DiG'):
    #     logits = model(data_x, SparseEdges, edge_weight)
    # elif args.net.startswith('Mag'):
    #     logits = model(X_real, X_img, edges, args.q, edge_weight).permute(2, 1, 0).squeeze()
    # elif args.net.startswith('Sig'):  # TODO might change
    #     logits = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    # else:
    logits = model(data_x, edges[:, train_edge_mask])
    accs, baccs, f1s = [], [], []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data_y[mask].cpu().numpy()
        acc = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    return accs, baccs, f1s


args = parse_args()
seed = args.seed
cuda_device = args.GPUdevice
if torch.cuda.is_available():
    print("cuda Device Index:", cuda_device)
    device = torch.device("cuda:%d" % cuda_device)
else:
    print("cuda is not available, using CPU.")
    device = torch.device("cpu")
if args.IsDirectedData and args.Direct_dataset.split('/')[0].startswith('dgl'):
    device = torch.device("cpu")
    print("dgl, using CPU.")
if args.CPU:
    device = torch.device("cpu")
    print("args.CPU true, using CPU.")
log_directory, log_file_name_with_timestamp = log_file(args)
print(args)
with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
    print(args, file=log_file)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

edge_in = None
in_weight = None
edge_out = None
out_weight = None
SparseEdges = None
edge_weight = None
X_img = None
X_real = None
edge_Qin_in_tensor = None
edge_Qin_out_tensor = None
Sigedge_index = None
norm_real = None
norm_imag = None
norm_imag_i =None
norm_imag_j = None
norm_imag_k = None
Quaedge_index = None
X_img_i = None
X_img_j = None
X_img_k = None

gcn = True

pos_edges = None
neg_edges = None

macro_F1 = []

criterion = CrossEntropy().to(device)

data, data_x, data_y, edges, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = load_dataset(args, device)
# if args.Direct_dataset.startswith('tel'):   # TODO delele it
#     print("x is all 1")
#     data_x.fill_(1)
if data_x.shape[0] > 5000:
    args.largeData = True
elif data_x.shape[0] < 1000:
    args.largeData = False
# args.largeData = True  # For temporary TODO delete it before commit
if args.net[-2:] not in ['ib', 'ub', 'i3', 'u3', 'i4', 'u4']:
    args.largeData = False
n_cls = data_y.max().item() + 1
if args.net.startswith('DiG'):
    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)  # consumiing for large graph
    edge_index1 = edge_index1.to(device)
    edge_weights1 = edge_weights1.to(device)
    if args.net[-2:] == 'ib' or args.net[-2:] == 'ub':
        if args.net[-2:] == 'ib':
            edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
        else:
            edge_index2, edge_weights2 = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
        edge_index2 = edge_index2.to(device)
        edge_weights2 = edge_weights2.to(device)
        SparseEdges = (edge_index1, edge_index2)
        edge_weight = (edge_weights1, edge_weights2)
        del edge_index2, edge_weights2
    elif args.net[-2:] == 'i3' or args.net[-2:] == 'u3':
        if args.net[-2:] == 'i3':
            edge_index_tuple, edge_weights_tuple = get_third_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
        else:
            edge_index_tuple, edge_weights_tuple = get_third_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
        SparseEdges = (edge_index1,)+ edge_index_tuple
        edge_weight = (edge_weights1,)+ edge_weights_tuple
        del edge_index_tuple, edge_weights_tuple
    elif args.net[-2:] == 'i4' or args.net[-2:] == 'u4':
        if args.net[-2:] == 'i4':
            edge_index_tuple, edge_weights_tuple = get_4th_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
        else:
            edge_index_tuple, edge_weights_tuple = get_4th_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype)
        SparseEdges = (edge_index1,) + edge_index_tuple
        edge_weight = (edge_weights1,) + edge_weights_tuple
        del edge_index_tuple, edge_weights_tuple
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1
    if args.net[3:].startswith('Sym'):
        # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges.long(), data_y.size(-1), data.edge_weight)
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(), data_y.size(-1), data.edge_weight)

elif args.net.startswith('Sym') or args.net.startswith('addSym'):
    data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(),data_y.size(-1),data.edge_weight)
    # data.edge_index, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor = F_in_out_Qin(edges.long(), data_y.size(-1), data.edge_weight)
elif args.net.startswith(('Mag', 'Sig', 'Qua')):
    data_x_cpu = data_x.cpu()
    X_img_i = torch.FloatTensor(data_x_cpu).to(device)
    X_img_j = torch.FloatTensor(data_x_cpu).to(device)
    X_img_k = torch.FloatTensor(data_x_cpu).to(device)
    X_img = torch.FloatTensor(data_x_cpu).to(device)
    X_real = torch.FloatTensor(data_x_cpu).to(device)
    if args.net.startswith('Sig'):
        Sigedge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edges, gcn=gcn, net_flow=args.netflow, x_real=X_real, edge_weight=edge_weight,
                                                                                   normalization='sym', return_lambda_max=False)
    elif args.net.startswith('Qua'):
        Quaedge_index, norm_real, norm_imag_i, norm_imag_j, norm_imag_k = process_quaternion_laplacian(edge_index=edges, x_real=X_real, edge_weight=edge_weight,
                                                                                                    normalization='sym', return_lambda_max=False)

else:
    pass

try:
    splits = data_train_maskOrigin.shape[1]
    print("splits", splits)
    if len(data_test_maskOrigin.shape) == 1:
        data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
except:
    splits = 1
if data_x.shape[0] > 2500 and splits > 5:     #
    splits = 5

try:
    start_time = time.time()
    with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
        # for split in range(splits - 1, -1, -1):
        for split in range(splits):
            model = CreatModel(args, num_features, n_cls, data_x, device).to(device)
            if split==0:
                print(model, file=log_file)
                print(model)
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
            if hasattr(model, 'coefs'):     # parameter without weight_decay will typically change faster
                optimizer = torch.optim.Adam(
                    [dict(params=model.reg_params, lr=args.lr, weight_decay=5e-4), dict(params=model.non_reg_params, lr=args.lr, weight_decay=0),
                     dict(params=model.coefs, lr=args.coeflr * args.lr, weight_decay=args.wd4coef), ],
                )
            elif hasattr(model, 'reg_params'):
                optimizer = torch.optim.Adam(
                [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)

            if splits == 1:
                data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),data_val_maskOrigin.clone(),data_test_maskOrigin.clone())
            else:
                try:
                    data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin[:, split].clone(),
                                                                      data_val_maskOrigin[:, split].clone(),
                                                                      data_test_maskOrigin[:, split].clone())
                except IndexError:
                    print("testIndex ,", data_test_mask.shape, data_train_mask.shape, data_val_mask.shape)
                    data_train_mask, data_val_mask = (
                        data_train_maskOrigin[:, split].clone(), data_val_maskOrigin[:, split].clone())
                    try:
                        data_test_mask = data_test_maskOrigin[:, 1].clone()
                    except:
                        data_test_mask = data_test_maskOrigin.clone()

            stats = data_y[data_train_mask]  # this is selected y. only train nodes of y
            n_data = []  # num of train in each class
            for i in range(n_cls):
                data_num = (stats == i).sum()
                n_data.append(int(data_num.item()))
            idx_info = get_idx_info(data_y, n_cls, data_train_mask, device)  # torch: all train nodes for each class
            # class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
            #     make_longtailed_data_remove(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
            if args.MakeImbalance:
                print("make imbalanced")
                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                    make_longtailed_data_remove(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
            else:
                print("not make imbalanced")
                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                    keep_all_data(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask)

            train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
            val_idx = data_val_mask.nonzero().squeeze()  # get the index of training data
            test_idx = data_test_mask.nonzero().squeeze()  # get the index of training data
            labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
            train_idx_list = train_idx.cpu().tolist()
            local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
            global2local = dict([val, key] for key, val in local2global.items())
            idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
            idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                              idx_info_list]  # train nodes position inside train

            if args.gdc == 'ppr':
                neighbor_dist_list = get_PPR_adj(data_x, edges[:, train_edge_mask], alpha=0.05, k=128, eps=None)
            elif args.gdc == 'hk':
                neighbor_dist_list = get_heat_adj(data_x, edges[:, train_edge_mask], t=5.0, k=None, eps=0.0001)
            elif args.gdc == 'none':
                neighbor_dist_list = get_ins_neighbor_dist(data_y.size(0), data.edge_index[:, train_edge_mask], data_train_mask, device)
            neighbor_dist_list = neighbor_dist_list.to(device)

            best_val_acc_f1 = 0
            best_val_f1 = 0
            best_test_f1 = 0
            saliency, prev_out = None, None
            test_acc, test_bacc, test_f1 = 0.0, 0.0, 0.0
            CountNotImproved = 0
            end_epoch = 0
            # for epoch in tqdm.tqdm(range(args.epoch)):
            goodAug=False
            for epoch in range(args.epoch):
                # if epoch>0:     # for test  TODO delete it
                #     raise NotImplementedError("1 epoch done right, test passed!")
                if args.net.startswith('UGCL'):

                    # val_loss = train_UGCL(pos_edges, neg_edges, size, train_idx, val_idx)
                    val_loss = train_UGCL(pos_edges, neg_edges, size, train_idx, val_idx)

                    z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, test_idx)
                    z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, test_idx)
                    out_test = model.prediction(z1, z2)
                    accs, baccs, f1s = test_UGCL()
                else:
                    if goodAug is False or args.AugDirect==100:
                        val_loss, new_edge_index, new_x, new_y, new_y_train = train(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real,norm_imag,
                                      X_img_i, X_img_j, X_img_k,norm_imag_i, norm_imag_j, norm_imag_k, Quaedge_index)
                    elif goodAug is True and args.AugDirect==100:
                        val_loss, new_edge_index, new_x, new_y, new_y_train = train_keepAug(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real,norm_imag,
                                      X_img_i, X_img_j, X_img_k,norm_imag_i, norm_imag_j, norm_imag_k, Quaedge_index)
                    accs, baccs, f1s = test()
                train_acc, val_acc, tmp_test_acc = accs
                train_f1, val_f1, tmp_test_f1 = f1s
                val_acc_f1 = (val_acc + val_f1) / 2.
                # print('train_acc:', train_acc,'val_acc:', val_acc, 'test_acc:', accs[2])
                # if val_acc_f1 > best_val_acc_f1:

                # if val_f1 > best_val_f1:
                if tmp_test_f1 > best_test_f1:
                    if epoch> 5:
                        goodAug = True
                    # best_val_acc_f1 = val_acc_f1
                    # best_val_f1 = val_f1
                    best_test_f1 = tmp_test_f1
                    test_acc = accs[2]
                    test_bacc = baccs[2]
                    test_f1 = f1s[2]
                    # print('hello')
                    CountNotImproved =0
                    print('test_f1 CountNotImproved reset to 0 in epoch', epoch)
                else:
                    goodAug = False
                    CountNotImproved += 1
                end_time = time.time()
                print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100))
                print(end_time - start_time, file=log_file)
                print(end_time - start_time)
                print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100),file=log_file)
                end_epoch = epoch
                if CountNotImproved > args.NotImproved:
                    print("No improved for consecutive {:3d} epochs, break.".format(args.NotImproved))
                    break
            if args.IsDirectedData:
                dataset_to_print = args.Direct_dataset + str(args.to_undirected)
            else:
                dataset_to_print = args.undirect_dataset + str(args.to_undirected)
            if args.MakeImbalance:
                net_to_print=args.net+'_Imbal'
            else:
                net_to_print = args.net + '_Bal'
            if args.largeData:
                net_to_print = net_to_print +'_batchSize' + str(args.batch_size)
            else:
                net_to_print = net_to_print +'_NoBatch'
            print(net_to_print, args.layer, dataset_to_print, "Aug", str(args.AugDirect), 'EndEpoch', str(end_epoch), 'lr', args.lr)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100))
            macro_F1.append(test_f1*100)

            print(end_time - start_time, file=log_file)
            print(end_time - start_time)
            print(net_to_print, args.layer, dataset_to_print, "Aug", str(args.AugDirect), 'EndEpoch', str(end_epoch), 'lr', args.lr, file=log_file)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100), file=log_file)
        last_time = time.time()
        print("Total time: ", last_time-start_time)
        print("Total time: ", last_time-start_time, file=log_file)
        if len(macro_F1)>1:
            average = statistics.mean(macro_F1)
            std_dev = statistics.stdev(macro_F1)
            print(net_to_print, args.layer,str(args.to_undirected), "Aug", str(args.AugDirect), 'Macro F1: ', f"{average:.3f}±{std_dev:.2f}")
            print(net_to_print, args.layer,str(args.to_undirected), "Aug", str(args.AugDirect), 'Macro F1: ', f"{average:.3f}±{std_dev:.2f}", file=log_file)

except KeyboardInterrupt:
    # If interrupted, the signal handler will be triggered

    # Ensure end_time is recorded
    end_time = time.time()
    calculate_time()
    log_results()
    sys.exit(0)
