import os
import os.path as osp
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.graphgym import optim

from args import parse_args
from data_utils import get_dataset, get_idx_info, make_longtailed_data_remove, get_step_split, load_directedData
from gens import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, \
    sampling_idx_individual_dst, neighbor_sampling_BiEdge, neighbor_sampling_BiEdge_bidegree, \
    neighbor_sampling_bidegree, neighbor_sampling_bidegreeOrigin, neighbor_sampling_bidegree_variant1, \
    neighbor_sampling_bidegree_variant2, neighbor_sampling_reverse, neighbor_sampling_bidegree_variant2_1, \
    neighbor_sampling_bidegree_variant2_0, neighbor_sampling_bidegree_variant2_1_, neighbor_sampling_bidegree_biTrainmask
from data_model import CreatModel, load_dataset, log_file
from utils import CrossEntropy, F1Scheduler
from sklearn.metrics import balanced_accuracy_score, f1_score
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist

import warnings
warnings.filterwarnings("ignore")

def train(train_idx):
    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    model.train()

    optimizer.zero_grad()
    if args.AugDirect == 0:
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])
        out = model(data_x, edges)
        criterion(out[data_train_mask], data_y[data_train_mask]).backward()
    else:
        if epoch > args.warmup:
            # identifying source samples
            prev_out_local = prev_out[train_idx]
            sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, args.tau, args.max, args.no_mask)

            if args.AugDirect == 1:
                # new_edge_index = neighbor_sampling(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling(data_x.size(0), edges, sampling_src_idx,neighbor_dist_list)
            elif args.AugDirect == -1:
                # new_edge_index = neighbor_sampling_reverse(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_reverse(data_x.size(0), edges, sampling_src_idx,neighbor_dist_list)

            elif args.AugDirect == 2:
                # new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges[:, train_edge_mask],
                #                                           sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges,sampling_src_idx,neighbor_dist_list)
            elif args.AugDirect == 4:
                # new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges[:, train_edge_mask],,sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges,sampling_src_idx,neighbor_dist_list)
            elif args.AugDirect == 20:
                # type 1
                # new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx,neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges,sampling_src_idx,neighbor_dist_list)  # has two types

            elif args.AugDirect == 21:
                # new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 22:
                # new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 23:
                # new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges[:, train_edge_mask],sampling_src_idx, neighbor_dist_list)
                new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)

            elif args.AugDirect == 231:
                new_edge_index = neighbor_sampling_bidegree_variant2_1(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 2311:
                new_edge_index = neighbor_sampling_bidegree_variant2_1_(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)

            elif args.AugDirect == 230:
                new_edge_index = neighbor_sampling_bidegree_variant2_0(data_x.size(0), edges,sampling_src_idx, neighbor_dist_list)


            else:
                raise NotImplementedError

            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = duplicate_neighbor(data_x.size(0), edges[:,train_edge_mask], sampling_src_idx)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

        # type 1
        out = model(new_x, new_edge_index)
        prev_out = (out[:data_x.size(0)]).detach().clone()
        add_num = out.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)
        sampling_src_idx = sampling_src_idx.to(torch.long).to(data_y.device)   # Ben for GPU error
        _new_y = data_y[sampling_src_idx].clone()
        new_y = torch.cat((data_y[data_train_mask], _new_y),dim=0)
        criterion(out[new_train_mask], new_y).backward()


    with torch.no_grad():
        model.eval()
        # type 1
        # out = model(data_x, edges[:,train_edge_mask])  # train_edge_mask????
        out = model(data_x, edges)
        val_loss= F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss, epoch)

    return val_loss

@torch.no_grad()
def test():
    model.eval()
    logits = model(data_x, edges[:,train_edge_mask])
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
    print("CUDA Device Index:", cuda_device)
    device = torch.device("cuda:%d" % cuda_device)
else:
    print("CUDA is not available, using CPU.")
    device = torch.device("cpu")
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

data, data_x, data_y, edges, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = load_dataset(args, device)

n_cls = data_y.max().item() + 1

model = CreatModel(args, num_features, n_cls, data_x, device)
model = model.to(device)
criterion = CrossEntropy().to(device)

try:
    splits = data_train_maskOrigin.shape[1]
    print("splits", splits)
    if len(data_test_maskOrigin.shape) == 1:
        data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
except IndexError:
    splits = 1



for split in range(splits):
    if args.net in ['GAT', 'GCN', 'SAGE']:
        optimizer = torch.optim.Adam(
            [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200,
                                                           verbose=True)
    # optimizer.
    # optimizer.param_groups[0]['lr'] = args.lr
    # optimizer.param_groups[1]['lr'] = args.lr
    if splits == 1:
        data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),
                                                          data_val_maskOrigin.clone(),
                                                          data_test_maskOrigin.clone())
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
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
        make_longtailed_data_remove(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), device)

    train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
    labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
    train_idx_list = train_idx.cpu().tolist()
    local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
    idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                      idx_info_list]  # train nodes position inside train

    if args.gdc=='ppr':
        neighbor_dist_list = get_PPR_adj(data_x, edges[:,train_edge_mask], alpha=0.05, k=128, eps=None)
    elif args.gdc=='hk':
        neighbor_dist_list = get_heat_adj(data_x, edges[:,train_edge_mask], t=5.0, k=None, eps=0.0001)
    elif args.gdc=='none':
        neighbor_dist_list = get_ins_neighbor_dist(data_y.size(0), data.edge_index[:,train_edge_mask], data_train_mask, device)
    neighbor_dist_list = neighbor_dist_list.to(device)

    best_val_acc_f1 = 0
    best_val_f1 = 0
    best_test_f1 =0
    saliency, prev_out = None, None
    test_acc, test_bacc, test_f1 = 0.0, 0.0, 0.0
    CountNotImproved = 0
    end_epoch =0
    # for epoch in tqdm.tqdm(range(args.epoch)):
    for epoch in range(args.epoch):
        val_loss = train(train_idx)
        accs, baccs, f1s = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, val_f1, tmp_test_f1 = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        # if val_acc_f1 > best_val_acc_f1:
        # if val_f1 > best_val_f1:
        if tmp_test_f1 > best_test_f1:
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
            CountNotImproved += 1

        print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100))
        end_epoch = epoch
        if CountNotImproved> 800:
            break
    if args.IsDirectedData:
        dataset_to_print = args.Direct_dataset
    else:
        dataset_to_print = args.undirect_dataset

    with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
        print(args.net, dataset_to_print, args.imb_ratio, "Aug", str(args.AugDirect), 'EndEpoch', str(end_epoch),'lr',args.lr)
        print('SHAsplit{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc*100, test_bacc*100, test_f1*100))

        print(args.net, dataset_to_print, args.imb_ratio, "Aug", str(args.AugDirect), 'EndEpoch', str(end_epoch), 'lr',args.lr, file=log_file)
        print('SHAsplit{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100,test_f1 * 100), file=log_file)

