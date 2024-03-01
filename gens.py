import math
import sys

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch

def test_directed(edge_index):
    set_edges = set()
    bi_direct = 0
    for i in range(edge_index.shape[1]):

        edge_inv = frozenset([edge_index[1][i].item(), edge_index[0][i].item()])
        edge = frozenset([edge_index[0][i].item(), edge_index[1][i].item()])
        if edge_inv in set_edges:
            bi_direct += 1
        set_edges.add(edge)
    print("Num_bidirect_edges: {}, total_num_edges: {}".format(bi_direct, edge_index.shape[1]))
    if bi_direct * 2 == edge_index.shape[1]:
        return False
    return True

def sampling_idx_individual_dst(class_num_list, idx_info, device):
    """
    to get the source nodes
    :param class_num_list:
    :param idx_info: # all train nodes for each class
    :param device:
    :return:
    """
    # print("Warmup: ", class_num_list, np.shape(class_num_list))  #  [14, 0, 1, 45, 4] (5,)
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # print(sampling_list)    # tensor([307., 334., 183.,   0., 268., 326., 338.])
    new_class_num_list = torch.Tensor(class_num_list).to(device)
    # print("new : ", new_class_num_list)   # tensor([ 34.,   7., 158., 341.,  73.,  15.,   3.])

    # Compute # of source nodes
    # print("samp: ", sampling_list)   # tensor([34., 48., 48.,  0., 45.])

    sampling_dst_idx = []
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    for cls_idx, samp_num in zip(idx_info, sampling_list):
        samp_num = int(samp_num.item())
        if samp_num <= 0:
            continue
        if len(cls_idx) <=0:
            continue
        # Sampling indices for dst
        # print("What::", samp_num, len(cls_idx))   # What:: 45 0
        dst_idx_local = cls_idx[torch.randint(len(cls_idx), (samp_num,))]
        sampling_dst_idx.append(dst_idx_local)
    # Concatenate the sampled indices
    sampling_dst_idx = torch.cat(sampling_dst_idx)
    # Sample indices for src
    src_idx = torch.multinomial(prob, sampling_dst_idx.shape[0], True)
    sampling_src_idx = temp_idx_info[src_idx]
    # print("\nChatGPT samp_src_idx", sampling_src_idx.shape)

    # sampling_src_idx = [cls_idx[torch.randint(len(cls_idx), (int(samp_num.item()),))]
    #                     for cls_idx, samp_num in zip(idx_info, sampling_list)]   # this code err when sam_num, or cls_idx is 0
    # sampling_src_idx = torch.cat(sampling_src_idx)  # this means 7 combines into one tensor
    # print("\noriginal samp_src_idx", sampling_src_idx.shape)
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()   # why use this as prob?
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    # the first is ascending ordered new tensor, the second is the original index
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]
    # print(sampling_src_idx, sampling_dst_idx)

    return sampling_src_idx, sampling_dst_idx

@torch.no_grad()
# def sampling_idx_individual_dst(class_num_list, idx_info, device):
#     # Selecting src & dst nodes
#     max_num, n_cls = max(class_num_list), len(class_num_list)
#     sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
#     new_class_num_list = torch.Tensor(class_num_list).to(device)
#
#     # Compute # of source nodes
#     sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
#                         for cls_idx, samp_num in zip(idx_info, sampling_list)]
#     sampling_src_idx = torch.cat(sampling_src_idx)
#
#     # Generate corresponding destination nodes
#     prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
#     prob = prob.repeat_interleave(new_class_num_list.long())
#     temp_idx_info = torch.cat(idx_info)
#     dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
#     sampling_dst_idx = temp_idx_info[dst_idx]
#
#     # Sorting src idx with corresponding dst idx
#     sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
#     sampling_dst_idx = sampling_dst_idx[sorted_idx]
#
#     return sampling_src_idx, sampling_dst_idx

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    sampling_src_idx = sampling_src_idx.to(torch.long).to(x.device)   # Ben
    sampling_dst_idx = sampling_dst_idx.to(torch.long).to(x.device)  # Ben
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim=0)
    return new_x

@torch.no_grad()
# def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
#     device = edge_index.device
#
#     # Assign node index for augmented nodes
#     src, tgt = edge_index[0], edge_index[1]
#     src, sort_idx = torch.sort(src)
#     tgt = tgt[sort_idx]
#     degree = scatter_add(torch.ones_like(src), src)
#     new_src =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
#     temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)
#
#     # Duplicate the edges of source nodes
#     node_mask = torch.zeros(total_node, dtype=torch.bool)
#     unique_src = torch.unique(sampling_src_idx)
#     node_mask[unique_src] = True
#     src_mask = node_mask[src]
#     edge_mask = tgt[src_mask]
#     b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
#     edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
#     if len(temp[temp!=0]) != edge_dense.shape[0]:
#         cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
#         cut_temp = temp[temp!=0][:-cut_num]
#     else:
#         cut_temp = temp[temp!=0]
#     edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
#     new_tgt = edge_dense[edge_dense!= -1]
#     inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
#     new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
#
#     return new_edge_index

def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    """

    :param total_node: num of total node
    :param edge_index: two dimensional torch, one is the first node, the other is the second node
    :param sampling_src_idx: anchor nodes
    :return: new dataset with augmented edges
    """
    # print(total_node, edge_index.shape, len(sampling_src_idx))
    device = edge_index.device
    # print("edge_index shape : ", np.shape(edge_index))  # torch.Size([2, 7798])
    # print("edge_index shape : ", edge_index)
    # tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],[ 633, 1862, 2582,  ...,  598, 1473, 2706]])

    # Assign node index for augmented nodes
    src, tgt = edge_index[0], edge_index[1]
    # print("original src is ", src[:10])
    src, sort_idx = torch.sort(src)  #
    # print("original tgt is ", tgt[:10])
    # print("new src is ", src[:10])
    tgt = tgt[sort_idx]
    # print(src.shape, edge_index.shape)
    src_degree = scatter_add(torch.ones_like(src), src)  # torch.Size([51]) torch.Size([2, 51])
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben
    for i in [src_degree, tgt_degree]:
        if i.shape[0] < total_node:
            num_zeros = total_node - len(i)
            zeros_to_add = torch.zeros(num_zeros, dtype=i.dtype)
            i.resize_(total_node)  # Resize the tensor in-place
            i[-num_zeros:] = zeros_to_add  # Add zeros to the end
    src_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node).repeat_interleave(
        src_degree[sampling_src_idx])
    tgt_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node).repeat_interleave(
        tgt_degree[sampling_src_idx])
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)  # src_degree of anchor nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool).to(device)
    unique_src = torch.unique(sampling_src_idx).to(device)
    node_mask[unique_src] = True  # get the torch where anchor nodes position are True

    src_mask = node_mask[src]  # select node in src, src_mask is bool. is Anchor node or not
    tgt_mask = node_mask[tgt]
    edge_mask = tgt[src_mask]  # anchor nodes' edge
    Newedge_mask = src[tgt_mask]
    # print("src_mask is", src_mask[:20], np.shape(src_mask))  # torch.Size([7798])
    # print("tgt is ", tgt[:20], np.shape(tgt))  # torch.Size([7798])
    # print("edge_mask is ", edge_mask, edge_mask.shape)  # torch.Size([14])
    # print("new edge_mask is ", Newedge_mask, Newedge_mask.shape)  # torch.Size([9])

    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(src_degree[unique_src])  # anchor nodes' src_degree
    tgt_b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(
        tgt_degree[unique_src])  # anchor nodes' src_degree

    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    # tgtedge_dense, _ = to_dense_batch(edge_mask, tgt_b_idx, fill_value=-1)
    # print("edge_mask is ", edge_mask[:10], "\nb_idx is ", b_idx[:10])
    # print(edge_dense[:10], np.shape(edge_dense))    # torch.Size([241, 155])  torch.Size([240, 29])
    # tensor([[723, 2614, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #          -1, -1, -1, -1, -1],
    # every line is a anchor node's edge, num of lines is num of anchor nodes.

    # edge_denseBen, _ = to_dense_batch(edge_mask, unique_src, fill_value=-1)
    # print(edge_denseBen[:2], np.shape(edge_dense))    # torch.Size([241, 155])

    # print(np.shape(temp), len(temp[temp != 0]), edge_dense.shape)  # torch.Size([1704]) 240 torch.Size([240, 29])
    if len(temp[temp != 0]) != edge_dense.shape[0]:  # edge_dense.shape[0] is the num of anchor nodes
        cut_num = len(temp[temp != 0]) - edge_dense.shape[0]
        cut_temp = temp[temp != 0][:-cut_num]
    else:
        cut_temp = temp[temp != 0]

    # if len(temp[temp != 0]) != tgtedge_dense.shape[0]:  # edge_dense.shape[0] is the num of anchor nodes
    #     tgtcut_num = len(temp[temp != 0]) - tgtedge_dense.shape[0]
    #     tgtcut_temp = temp[temp != 0][:-cut_num]
    # else:
    #     tgtcut_temp = temp[temp != 0]

    # print("Hi1, ", edge_dense[:10], np.shape(edge_dense))  # torch.Size([240, 29])
    edge_dense = edge_dense.repeat_interleave(cut_temp, dim=0).to(device)
    src_new_tgt = edge_dense[edge_dense != -1]
    inv_edge_index_src = torch.stack([src_new_tgt, src_new_src], dim=0).to(device)

    new_edge_index = torch.cat([edge_index, inv_edge_index_src], dim=1)  # original 7798 edges,

    return new_edge_index

@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = sampling_src_idx.long()
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    tgt = edge_index[1]
    degree = scatter_add(torch.ones_like(tgt), tgt)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)   # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_src = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_src = new_src.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


def neighbor_samplingDirection(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """

    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = sampling_src_idx.long()
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    tgt = edge_index[1]
    tat_degree = scatter_add(torch.ones_like(tgt), tgt)
    if len(tat_degree) < total_node:
        degree = torch.cat([tat_degree, tat_degree.new_zeros(total_node-len(tat_degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tat_degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)   # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_src = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_src = new_src.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_reverse(total_node, edge_index, sampling_src_idx,
                      neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = sampling_src_idx.long()
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    tgt = edge_index[1]
    degree = scatter_add(torch.ones_like(tgt), tgt)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node - len(degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(
        torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)  # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    new_src = new_src.repeat_interleave(aug_degree)
    # inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
    inv_edge_index = torch.stack([new_src, new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_BiEdge(total_node, edge_index, sampling_src_idx,
                      neighbor_dist_list, train_node_mask=None):
    """
    add inverse edges based on GraphSHA
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)

    # Find the nearest nodes and mix target pool
    # mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx.long()]

    # Compute degree
    tgt = edge_index[1]
    degree = scatter_add(torch.ones_like(tgt), tgt)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node - len(degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(
        torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)  # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    new_src = new_src.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
    inv_edge_index_inverse = torch.stack([new_src, new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index, inv_edge_index_inverse], dim=1)

    return new_edge_index

def neighbor_sampling_bidegreeOrigin(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    # sampling_src_idx = sampling_src_idx.clone().to(device)
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree (count in the end of edge)
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree   ( count in the source of edge)
    # print(tgt_degree)

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    # tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    # src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.

    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(device).type(torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]

    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    # tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    # src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_biTrainmask(total_node, edge_index, sampling_src_idx,neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        tgt_train_node_mask = torch.ones_like(tgt_degree,dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
        src_train_node_mask = torch.ones_like(src_degree,dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    if torch.all(tgt_train_node_mask == src_train_node_mask):
        print("train node mask is the same")        # train node mask is the same
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[tgt_train_node_mask]), tgt_degree[tgt_train_node_mask]).to(device).type(torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[src_train_node_mask]), src_degree[src_train_node_mask]).to(device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    # tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    # src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_variant1(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    # tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    # src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_variant2(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    # src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    # tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index], dim=1)

    return new_edge_index


def neighbor_sampling_bidegree_variant2_0(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    # src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    # tgt_inv_edge_index = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index], dim=1)

    return new_edge_index
def neighbor_sampling_bidegree_variant2_1(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    degree = scatter_add(torch.ones_like(tgt), tgt)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_tgt = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_src = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_src = new_src.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_tgt, new_src], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index
def neighbor_sampling_bidegree_variant2_1(args, total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree
    are_equal = torch.allclose(tgt_degree, src_degree)
    if not args.IsDirectedData:
        assert are_equal, "Strange: tgt_degree not equal to src_degree!"

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_max_degree = src_degree.max().item() + 1
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, src_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_variant2_1_(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    # if len(tgt_degree) < total_node:
    #     tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(src_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_max_degree = src_degree.max().item() + 1
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    src_inv_edge_index = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, src_inv_edge_index], dim=1)

    return new_edge_index



def neighbor_sampling_BiEdge_bidegree(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)  # Adjust dtype as needed
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # print(neighbor_dist_list)

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,
                                          dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    # print("tgt degree\n", tgt_degree, "\nsrc degree\n", src_degree, src_degree.shape)
    # if torch.equal(tgt_degree, src_degree):
    #     print("the same")
    # else:
    #     print("different")    # here
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    tgt_inv_edge_index_inv = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # Ben change direction
    src_inv_edge_index_inv = torch.stack([src_new_src, src_new_tgt], dim=0)
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index, src_inv_edge_index, tgt_inv_edge_index_inv, src_inv_edge_index_inv], dim=1)

    return new_edge_index

def neighbor_sampling_BiEdge_bidegree_Direction(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in src and tgt.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)  # Adjust dtype as needed
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute tgt_degree
    tgt = edge_index[1]
    src = edge_index[0]
    tgt_degree = scatter_add(torch.ones_like(tgt), tgt)  # Ben only tgt tgt_degree
    src_degree = scatter_add(torch.ones_like(src), src)  # Ben only tgt tgt_degree

    if len(tgt_degree) < total_node:
        tgt_degree = torch.cat([tgt_degree, tgt_degree.new_zeros(total_node - len(tgt_degree))], dim=0)
    if len(src_degree) < total_node:
        src_degree = torch.cat([src_degree, src_degree.new_zeros(total_node - len(src_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(tgt_degree,dtype=torch.bool)  # the same shape as the tgt_degree tensor, and all elements in the mask are set to True.
    # print("tgt degree\n", tgt_degree, "\nsrc degree\n", src_degree, src_degree.shape)
    # if torch.equal(tgt_degree, src_degree):
    #     print("the same")       # undirected graph is here
    # else:
    #     print("different")    # directed graph is here
    tgt_degree_dist = scatter_add(torch.ones_like(tgt_degree[train_node_mask]), tgt_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    src_degree_dist = scatter_add(torch.ones_like(src_degree[train_node_mask]), src_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample tgt_degree for augmented nodes
    tgt_prob = tgt_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    src_prob = src_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    tgt_aug_degree = torch.multinomial(tgt_prob, 1).to(device).squeeze(dim=1)  # (m)
    src_aug_degree = torch.multinomial(src_prob, 1).to(device).squeeze(dim=1)  # (m)
    tgt_max_degree = tgt_degree.max().item() + 1
    src_max_degree = src_degree.max().item() + 1
    tgt_aug_degree = torch.min(tgt_aug_degree, tgt_degree[sampling_src_idx])
    src_aug_degree = torch.min(src_aug_degree, src_degree[sampling_src_idx])

    # Sample neighbors
    tgt_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, tgt_max_degree)
    src_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, src_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', tgt_new_tgt)
    tgt_tgt_index = torch.arange(tgt_max_degree).unsqueeze(dim=0).to(device)
    src_tgt_index = torch.arange(src_max_degree).unsqueeze(dim=0).to(device)
    tgt_new_tgt = tgt_new_tgt[(tgt_tgt_index - tgt_aug_degree.unsqueeze(dim=1) < 0)]        # has anchor nodes in it
    src_new_src = src_new_tgt[(src_tgt_index - src_aug_degree.unsqueeze(dim=1) < 0)]    # has anchor nodes in it
    tgt_new_src = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    src_new_tgt = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    tgt_new_src = tgt_new_src.repeat_interleave(tgt_aug_degree)
    src_new_tgt = src_new_tgt.repeat_interleave(src_aug_degree)
    # tgt_inv_edge_index = torch.stack([tgt_new_tgt, tgt_new_src], dim=0)
    # src_inv_edge_index = torch.stack([src_new_tgt, src_new_src], dim=0)
    tgt_inv_edge_index_inv = torch.stack([tgt_new_src, tgt_new_tgt], dim=0)  # anchor nodes(tgt_new_tgt) in edge[1]
    src_inv_edge_index_inv = torch.stack([src_new_src, src_new_tgt], dim=0)  # anchor nodes(src_new_src) in edge[0]
    new_edge_index = torch.cat([edge_index, tgt_inv_edge_index_inv, src_inv_edge_index_inv], dim=1)

    return new_edge_index

@torch.no_grad()

def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False,
                         no_mask=False):
    """
    sampling for subsequent epochs
    :param class_num_list:
    :param prev_out_local: predicted label of train nodes
    :param idx_info_local:  # train nodes position inside train
    :param train_idx:
    :param tau:
    :param max_flag:
    :param no_mask:whether to mask the self class in sampling neighbor classes. default is mask
    :return:src_idx_all, dst_idx_all
    """
    device = prev_out_local.device
    max_num, n_cls = max(class_num_list), len(class_num_list)
    if not max_flag:  # mean
        max_num = math.ceil(sum(class_num_list) / n_cls)  # determined by args
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # print("After warm", sampling_list)    # tensor([ 21.6667,  -2.3333,  -8.3333, -11.3333,  -3.3333,   3.6667])
    prev_out_local = F.softmax(prev_out_local / tau, dim=1)
    # softmax is to transform a vector of real numbers into a probability distribution.
    prev_out_local = prev_out_local.cpu()     # Ben try
    # train_idx = train_idx.to(device)
    # if torch.is_tensor(idx_info_local) and len(idx_info_local) == 1:
    #     idx_info_local = idx_info_local.item()  # Extract the Python scalar from the one-element tensor
    # idx_info_local = torch.tensor(idx_info_local).to(device)

    src_idx_all = []
    dst_idx_all = []
    # print("sampling_list: ", sampling_list)
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0:
            continue
        # first sampling
        if idx_info_local[cls_idx].numel() == 0:
            continue
        # print(cls_idx)  # 1
        # print(idx_info_local[cls_idx])   # tensor([])
        prob = 1 - prev_out_local[idx_info_local[cls_idx].long()][:, cls_idx].squeeze()
        # print(prob)   # tensor(0.7912)
        if prob.shape == torch.Size([]):
            prob = torch.tensor([prob])
        src_idx_local = torch.multinomial(prob + 1e-12, num,replacement=True)  # the harder the sample, the more likely to be sampled
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]]  # each minor class has src_idx

        # second sampling
        conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]]
        if not no_mask:
            conf_src[:, cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist()

        # third sampling
        neighbor = [prev_out_local[idx_info_local[cls].long()][:, cls_idx] for cls in neighbor_cls if idx_info_local[cls].numel() != 0]
        dst_idx = []
        new_src_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0]
            # dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])    # index 32 is out of bounds for dimension 0 with size 0
            # Check if idx_info_local[neighbor_cls[i]] has non-zero size along dimension 0
            if idx_info_local[neighbor_cls[i]].numel() != 0:
                # Check if dst_idx_local is within valid bounds
                if 0 <= dst_idx_local < idx_info_local[neighbor_cls[i]].numel():
                    dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
                    new_src_idx.append(src_idx[i])
                else:
                    # Handle the case where dst_idx_local is out of bounds
                    # You can print a warning or take appropriate action
                    pass
                    # print(f"Warning: dst_idx_local {dst_idx_local} is out of bounds.")

            else:
                # Handle the case where idx_info_local[neighbor_cls[i]] has size 0
                # You can print a warning or take appropriate action
                # print(f"Warning: idx_info_local[neighbor_cls[i]] has size 0.")   # happened
                pass
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)
        # new_src_idx = torch.tensor(dst_idx).to(src_idx.device)
        new_src_idx = torch.tensor(new_src_idx, device=src_idx.device).clone().detach()

        src_idx_all.append(new_src_idx)
        dst_idx_all.append(dst_idx)

    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)

    return src_idx_all, dst_idx_all
