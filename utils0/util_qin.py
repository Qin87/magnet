import torch
import torch.sparse as sparse
import numpy as np


def edge_index_to_sparse(edge_index, size):
    """Convert edge_index to sparse tensor"""
    values = torch.ones(edge_index.shape[1])
    sparse_tensor = torch.sparse_coo_tensor(edge_index, values, (size, size))
    return sparse_tensor


def matrix_power_analysis(edge_index, size, k_max=5):
    """
    Calculate A^k and its sparsity for k from 1 to k_max

    Parameters:
    edge_index: torch.Tensor of shape (2, E) where E is number of edges
    size: int, size of the square matrix
    k_max: int, maximum power to calculate

    Returns:
    list of (sparse tensor, float) tuples containing A^k and its density
    """
    # Convert edge_index to sparse matrix
    A = edge_index_to_sparse(edge_index, size)

    results = []
    current_power = A

    for k in range(1, k_max + 1):
        # Calculate density
        nnz = current_power.coalesce().indices().shape[1]
        total_elements = size * size
        density = (nnz / total_elements) * 100

        results.append((current_power, density))

        print(f"\nPower {k}:")
        print(f"Non-zero elements: {nnz}")
        print(f"Density: {density:.4f}%")

        # Calculate next power
        if k < k_max:
            current_power = torch.sparse.mm(current_power, A)

    return results


# Example usage:
def analyze_edge_index(edge_index, size, k_max=5):
    results = matrix_power_analysis(edge_index, size, k_max)

    print(f"Analysis of matrix powers:")
    # for k, (matrix, density) in enumerate(results, 1):
    #     print(f"\nPower {k}:")
    #     print(f"Non-zero elements: {matrix.coalesce().indices().shape[1]}")
    #     print(f"Density: {density:.4f}%")

    return results


def remove_bidirectional_edges(edge_index):
    # Convert to set of tuples for easier comparison
    edges = set(map(tuple, edge_index.t().tolist()))
    edges_to_keep = set()

    # Iterate through edges and remove bidirectional pairs
    for src, dst in edges:
        # If the reverse edge exists and we haven't processed it yet
        if (dst, src) in edges and (dst, src) not in edges_to_keep:
            # Keep only one direction (we'll keep the smaller source node)
            if src < dst:
                edges_to_keep.add((src, dst))
            else:
                edges_to_keep.add((dst, src))
        # If it's a one-way edge, keep it
        elif (dst, src) not in edges:
            edges_to_keep.add((src, dst))

    # Convert back to tensor
    new_edge_index = torch.tensor(list(edges_to_keep)).t()
    return new_edge_index


def get_k_hop_edges(edge_index, size, k):
    """
    Get edge_index representation for A^k, representing k-hop connections.

    Parameters:
    edge_index: torch.Tensor of shape (2, E) where E is number of edges
    size: int, size of the square matrix
    k: int, power of the adjacency matrix

    Returns:
    torch.Tensor: edge_index representing A^k
    """
    # Convert edge_index to sparse matrix
    values = torch.ones(edge_index.shape[1])
    A = torch.sparse_coo_tensor(edge_index, values, (size, size))

    # Calculate A^k
    current_power = A
    for _ in range(k - 1):
        current_power = torch.sparse.mm(current_power, A)

    # Convert back to edge_index format
    k_hop_edge_index = current_power.coalesce().indices()

    return k_hop_edge_index