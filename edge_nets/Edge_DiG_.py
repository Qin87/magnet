import copy

import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse
import torch.nn.functional as F
from torch_geometric_signed_directed.data import load_directed_real_data
import random
import pickle as pk

# internal files
from nets.DiGCN import *
from nets.geometric_baselines import *

try:
    from .edge_data import in_out_degree, get_appr_directed_adj, get_second_directed_adj
except:
    from edge_data import in_out_degree, get_appr_directed_adj, get_second_directed_adj
try:
    from .save_settings import write_log
except:
    from save_settings import write_log
try:
    from .edge_data_new import link_class_split_new, link_class_split_new_1split
except:
    from edge_data_new import link_class_split_new, link_class_split_new_1split

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="link prediction baseline--Digraph(NeurIPS2020)")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.05,0.15",
                        help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=str, default='three_class_digraph', help='Task: three_class_digraph,  direction, existence, ...')

    parser.add_argument('--method_name', type=str, default='DiG', help='method name')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--num_class_link', type=int, default=3,
                        help='number of classes for link direction prediction(2 or 3).')

    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=64, help='num of filters')
    # parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    # parser.add_argument('-dgrees', '-d', action='store_true', help='if use in degree+outdegree as feature')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('--randomseed', type=int, default=0, help='if set random seed in training')

    return parser.parse_args()


def acc(pred, label):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc


def main(args):
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) is False:
        os.makedirs(log_path)

    dataset_name = args.dataset.split('/')
    if len(dataset_name) == 1:
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[0])
    else:
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[1])

    edge_index = data.edge_index

    size = torch.max(edge_index).item() + 1
    data.num_nodes = size
    # datasets = link_class_split_new_1split(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], task=args.task)
    datasets = link_class_split_new(data, task=args.task)[0]

    # if args.task == 'existence':
    results = np.zeros((10, 4))
    # else:
    # results = np.zeros((10, 4, 5))
    # for i in range(10):
    log_str_full = ''
    edges = datasets['graph']

    ########################################
    # initialize model and load dataset
    ########################################
    x = in_out_degree(edges, size, datasets['weights']).to(device)
    edge_weight = datasets['weights']

    # get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None)
    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), size, x.dtype, edge_weight)
    edge_index1 = edge_index1.to(device)
    edge_weights1 = edge_weights1.to(device)
    if args.method_name[-2:] == 'ib':
        edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), size, x.dtype, edge_weight=edge_weight)
        edge_index2 = edge_index2.to(device)
        edge_weights2 = edge_weights2.to(device)
        edges = (edge_index1, edge_index2)
        edge_weight = (edge_weights1, edge_weights2)
        del edge_index2, edge_weights2
    else:
        edges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1

    ########################################
    # initialize model and load dataset
    ########################################
    if not args.method_name[-2:] == 'ib':
        model = DiGCNet(x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)
    else:
        model = DiGCNet_IB(x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)
    # model = nn.DataParallel(graphmodel)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    y_train = datasets['train']['label']
    y_val = datasets['val']['label']
    y_test = datasets['test']['label']
    y_train = y_train.long().to(device)
    y_val = y_val.long().to(device)
    y_test = y_test.long().to(device)

    train_index = datasets['train']['edges'].to(device)
    val_index = datasets['val']['edges'].to(device)
    test_index = datasets['test']['edges'].to(device)
    #################################
    # Train/Validation/Test
    #################################
    best_test_err = 1000000.0
    early_stopping = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        ####################
        # Train
        ####################
        train_loss, train_acc = 0.0, 0.0
        model.train()
        out = model(x, edges, train_index, edge_weight)

        train_loss = F.nll_loss(out, y_train)
        pred_label = out.max(dim=1)[1]
        train_acc = acc(pred_label, y_train)

        opt.zero_grad()
        train_loss.backward()
        opt.step()
        outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)

        ####################
        # Validation
        ####################
        train_loss, train_acc = 0.0, 0.0
        model.eval()
        out = model(x, edges, val_index, edge_weight)

        test_loss = F.nll_loss(out, y_val)
        pred_label = out.max(dim=1)[1]
        test_acc = acc(pred_label, y_val)

        outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)
        duration = "--- %.4f seconds ---" % (time.time() - start_time)
        log_str = ("%d / %d epoch" % (epoch, args.epochs)) + outstrtrain + outstrval + duration
        # print(log_str)
        log_str_full += log_str + '\n'
        ####################
        # Save weights
        ####################
        save_perform = test_loss.detach().item()
        # save_perform = test_acc.detach().item()        # Qin
        if save_perform <= best_test_err:
            early_stopping = 0
            best_test_err = save_perform
            # torch.save(model.state_dict(), log_path + '/model'  + str(epoch)+'.t7')     # loss smallest
            torch.save(model.state_dict(), log_path + '/model' + '.t7')  # loss smallest
        else:
            early_stopping += 1
        if early_stopping > 500:
            # torch.save(model.state_dict(), log_path + '/model_Qin'+ str(epoch) + '.t7')     #
            break

    write_log(vars(args), log_path)
    torch.save(model.state_dict(), log_path + '/model_latest' + '.t7')
    # if args.task == 'existence':
    ####################
    # Testing
    ####################
    model.load_state_dict(torch.load(log_path + '/model' + '.t7'))
    model.eval()
    out = model(x, edges, val_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    val_acc = acc(pred_label, y_val)
    out = model(x, edges, test_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    test_acc = acc(pred_label, y_test)

    model.load_state_dict(torch.load(log_path + '/model_latest' + '.t7'))
    model.eval()
    out = model(x, edges, val_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    val_acc_latest = acc(pred_label, y_val)

    out = model(x, edges, test_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    test_acc_latest = acc(pred_label, y_test)
    ####################
    # Save testing results
    ####################
    log_str = ('val_acc: {val_acc:.4f}, ' + 'test_acc: {test_acc:.4f}, ')
    log_str1 = log_str.format(val_acc=val_acc, test_acc=test_acc)
    log_str_full += log_str1
    log_str = ('val_acc_latest: {val_acc_latest:.4f}, ' + 'test_acc_latest: {test_acc_latest:.4f}, ')
    log_str2 = log_str.format(val_acc_latest=val_acc_latest, test_acc_latest=test_acc_latest)
    log_str_full += log_str2 + '\n'
    print(log_str1 + log_str2)
    results = [val_acc, test_acc, val_acc_latest, test_acc_latest]

    with open(log_path + '/log' + '.csv', 'w') as file:
        file.write(log_str_full)
        file.write('\n')
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(int(args.num_filter)) + 'alpha' + str(int(100 * args.alpha)) + 'task_' + args.task + '_noisy' + str(args.noisy)
    args.save_name = save_name
    args.log_path = os.path.join(args.log_path, args.method_name, args.dataset)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays', args.log_path, args.dataset + '/')

    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')

    results = main(args)
    np.save(dir_name + save_name, results)


def edge_prediction(args, data, sampling_src_idx, neighbor_dist_list):
    device = sampling_src_idx.device
    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(int(args.num_filter)) + 'alpha' + str(int(100 * args.alpha)) + 'task_' + args.task
    args.save_name = save_name

    edge_index = data.edge_index
    old_size = torch.max(edge_index).item() + 1
    size = data.x.size(0)
    data.num_nodes = size
    # data_x = data.x[:old_size]
    new_data_x = torch.arange(old_size,size).to(device)

    tgt = edge_index[1]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)
    src = edge_index[0]
    src_degree = scatter_add(torch.ones_like(src), src)
    max_tgt_degree = tgt_degree.max().item() + 1
    max_src_degree = src_degree.max().item() + 1
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx].to(device)
    top_neigh = torch.multinomial(mixed_neighbor_dist + 1e-12, np.min(max_tgt_degree+max_src_degree)).to(device)
    # n = int(y_train.size(0) *10)  # You can adjust this as needed, without int, it's float
    x_values = new_data_x
    y_values = top_neigh
    x_values = x_values.unsqueeze(1).repeat(1, y_values.size(1))
    x_values = x_values.view(-1, 1).to(device)
    y_values = y_values.view(-1, 1).to(device)
    # tensor_reshaped = tensor.view(-1,
    test_index = torch.cat((x_values, y_values), dim=1).to(device)
    # tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    # new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) is False:
        os.makedirs(log_path)


    datasets = link_class_split_new_1split(data, task=args.task)

    results = np.zeros((10, 4))
    log_str_full = ''
    edges = datasets['graph']

    ########################################
    # initialize model and load dataset
    ########################################
    new_x = in_out_degree(edges, size, datasets['weights']).to(device)
    old_x = new_x[: old_size]
    edge_weight = datasets['weights'].to(device)

    # get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None)
    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), old_size, old_x.dtype, edge_weight)
    edge_index1 = edge_index1.to(device)
    edge_weights1 = edge_weights1.to(device)
    if args.method_name[-2:] == 'ib':
        edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), old_size, old_x.dtype, edge_weight=edge_weight)
        edge_index2 = edge_index2.to(device)
        edge_weights2 = edge_weights2.to(device)
        edges = (edge_index1, edge_index2)
        edge_weight = (edge_weights1, edge_weights2)
        del edge_index2, edge_weights2
    else:
        edges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1

    ########################################
    # initialize model and load dataset
    ########################################
    if not args.method_name[-2:] == 'ib':
        model = DiGCNet(new_x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)
    else:
        model = DiGCNet_IB(new_x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    y_train = datasets['train']['label']
    y_val = datasets['train']['label']

    y_train = y_train.long().to(device)
    y_val = y_val.long().to(device)

    train_index = datasets['train']['edges'].to(device)
    val_index = datasets['train']['edges'].to(device)
    #################################
    # Train/Validation/Test
    #################################
    best_test_err = 1000000.0
    early_stopping = 0
    for epoch in range(args.epochs):
        start_time = time.time()

        ####################
        # Train
        ####################
        train_loss, train_acc = 0.0, 0.0
        model.train()
        out = model(old_x, edges, train_index, edge_weight)

        train_loss = F.nll_loss(out, y_train)
        pred_label = out.max(dim=1)[1]
        train_acc = acc(pred_label, y_train)

        opt.zero_grad()
        train_loss.backward()
        opt.step()
        outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)

        ####################
        # Validation
        ####################
        train_loss, train_acc = 0.0, 0.0
        model.eval()
        out = model(new_x, edges, val_index, edge_weight)

        test_loss = F.nll_loss(out, y_val)
        pred_label = out.max(dim=1)[1]
        test_acc = acc(pred_label, y_val)

        outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)
        duration = "--- %.4f seconds ---" % (time.time() - start_time)
        log_str = ("%d / %d epoch" % (epoch, args.epochs)) + outstrtrain + outstrval + duration
        # print(log_str)
        log_str_full += log_str + '\n'
        ####################
        # Save weights
        ####################
        save_perform = test_loss.detach().item()
        if save_perform <= best_test_err:
            early_stopping = 0
            best_test_err = save_perform
            torch.save(model.state_dict(), log_path + '/model' + '.t7')
        else:
            early_stopping += 1
        if early_stopping > 500:
            print(log_str)
            break

    write_log(vars(args), log_path)
    torch.save(model.state_dict(), log_path + '/model_latest' + '.t7')


    ####################
    # Testing
    ####################
    model.load_state_dict(torch.load(log_path + '/model' + '.t7'))
    model.eval()
    # model_val_loss_smallest = model.clone()
    model_val_loss_smallest = copy.deepcopy(model)
    out = model(new_x, edges, val_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    val_acc = acc(pred_label, y_val)

    model.load_state_dict(torch.load(log_path + '/model_latest' + '.t7'))
    model.eval()
    model_latest = model.clone()
    out = model(new_x, edges, val_index, edge_weight)
    pred_label = out.max(dim=1)[1]
    val_acc_latest = acc(pred_label, y_val)
    ####################
    # Save validation results
    ####################
    log_str = 'val_acc: {val_acc:.4f}, '
    log_str1 = log_str.format(val_acc=val_acc)
    log_str_full += log_str1
    log_str = 'val_acc_latest: {val_acc_latest:.4f}, '
    log_str2 = log_str.format(val_acc_latest=val_acc_latest)
    log_str_full += log_str2 + '\n'
    print(log_str1 + log_str2)

    if val_acc_latest > val_acc:
        print('using the latest model as val_acc_latest > val_acc')
        edge_pred = model(new_x, edges, test_index, edge_weight)
    else:
        print('using the val_loss_smallest model as val_acc_latest < val_acc')
        model.load_state_dict(torch.load(log_path + '/model' + '.t7'))
        model.eval()
        # edge_pred = model(new_x, edges, test_index, edge_weight)
        edge_pred = model_val_loss_smallest(new_x, edges, test_index, edge_weight)
    pred_label = edge_pred.max(dim=1)[1]
    _new_edge_index = generate_Edge(test_index, pred_label)
    new_edge_index = torch.cat([data.edge_index, _new_edge_index], dim=1)
    # new_edge_index = torch.stack([data.edge_index, _new_edge_index])

    with open(log_path + '/log' + '.csv', 'w') as file:
        file.write(log_str_full)
        file.write('\n')
    torch.cuda.empty_cache()
    return new_edge_index


def generate_Edge(edge_tensor, pred_label):
    '''
    get edge_index according to predition
    '''
    edge_indices = []
    for i, direction in enumerate(pred_label):
        edge = edge_tensor[i]
        if direction == 0:
            edge_indices.append(edge)
        elif direction == 1:
            # Reverse the edge and append
            edge_indices.append(torch.flip(edge, dims=(0,)))
        elif direction == 2:
            pass
    edge_index = torch.stack(edge_indices, dim=1)
    return edge_index
