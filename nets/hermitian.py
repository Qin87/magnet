import numpy as np
import torch
from numpy import linalg as LA
from scipy.sparse import coo_matrix
# from torch import triu
from scipy.sparse import csr_matrix, triu

'''
def hermitian_decomp(A, q = 0.25):
    # this function is only tested based on the numpy array
    # should be updated if a sparse matrix is required

    A_upper = np.triu(A)
    A_lower = np.triu(A.T)

    #get A_s
    A_s = -np.ones_like(A)
    avg_mask = np.logical_and((A_upper > 0), (A_lower > 0))

    # u,v and v,u both exist position
    pos = (avg_mask == True)
    A_s[pos] = 0.5*(A_upper[pos] + A_lower[pos])

    # only one of u,v and v,u exists
    pos = (avg_mask == False)
    A_s[pos] = A_upper[pos] + A_lower[pos]
    A_s = np.triu(A_s) + np.triu(A_s).T

    # phase
    theta = 2*np.pi*q*((A_upper - A_lower) + (A_lower - A_upper).T)

    # degree
    D_s = np.diag(np.sum(A_s, axis = 1))

    # eigendecomposition
    L = D_s - A_s*np.exp(1j*theta)
    w, v = LA.eig(L) # column of v is the right eigenvector

    return L, w, v
'''


###########################################
####### Dense implementation ##############
###########################################
def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian[0] += np.eye(N, dtype=np.float32)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] += A
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] += 2 * np.dot(A, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]

    return multi_order_laplacian


def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
    A = 1.0 * np.array(A)
    if gcn_appr:
        A += 1.0 * np.eye(A.shape[0])

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.sum(np.array(A_sym), axis=0)
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = np.diag(d)
        A_sym = np.dot(np.dot(D, A_sym), D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array
        if norm:
            D = np.diag([1.0] * len(d))
        else:
            d = np.sum(np.array(A_sym), axis=0)  # diag of degree array
            D = np.diag(d)
        L = D - np.exp(Theta) * A_sym
    '''
    else:
        #transition matrix
        d_out = np.sum(np.array(A), axis = 1)
        d_out[d_out==0] = -1
        d_out = 1.0/d_out
        d_out[d_out<0] = 0
        D = np.diag(d_out)
        L = np.eye(len(d_out)) - np.dot(D, A)
    '''
    w, v = None, None
    if norm:
        if max_eigen == None:
            w, v = LA.eigh(L)
            L = (2.0 / np.amax(np.abs(w))) * L - np.diag([1.0] * len(A))
        else:
            L = (2.0 / max_eigen) * L - np.diag([1.0] * len(A))
            w = None
            v = None

    return L, w, v


def hermitian_decomp(As, q=0.25, norm=False, laplacian=True, max_eigen=None, gcn_appr=False):
    ls, ws, vs = [], [], []
    if len(As.shape) > 2:
        for i, A in enumerate(As):
            l, w, v = decomp(A, q, norm, laplacian, max_eigen, gcn_appr)
            vs.append(v)
            ws.append(w)
            ls.append(l)
    else:
        ls, ws, vs = decomp(As, q, norm, laplacian, max_eigen, gcn_appr)
    return np.array(ls), np.array(ws), np.array(vs)


###########################################
####### Sparse implementation #############
###########################################
def cheb_poly_sparse(A, K):
    '''
    T0(x) = 1, T1(x) = x, and Tk(x) = 2xTk−1(x) + Tk−2(x)
    Args:
        A:
        K:

    Returns: Chebyshev polynomial

    '''
    K += 1
    N = A.shape[0]  # [N, N]
    # multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append(coo_matrix((np.ones(N), (np.arange(N), np.arange(N))),shape=(N, N), dtype=np.float32))
    # creates a square sparse matrix with ones along the diagonal and zeros elsewhere.
    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append(2.0 * A.dot(multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2])  # from the third element
                #

    return multi_order_laplacian


def hermitian_decomp_sparse(row, col, size, q=0.25, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''

    Args:
        row: src node
        col: tgt node
        size: num of all nodes
        q:
        norm: normalization
        laplacian:
        max_eigen:
        gcn_appr: add self-loop
        edge_weight:

    Returns:signed directed magnetic Laplacian with the Hermitian adjacency matrix

    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
        # creates a sparse matrix A where the non-zero elements are located at the coordinates specified by row and col, and each non-zero element has a value of 1.
    else:
        A = coo_matrix((edge_weight.detach().numpy(), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency(A_sym has 0.5)

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]
        # max_eigen = d.max()
        # print("max_eigen set by Qin")
        # d has 0.5,2.5,(so it's not out-degree, it's bi_edge+*0.5*uni_edge)
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array # Each element of Theta will represent the phase angle value associated with the difference between corresponding elements of A and its transpose.
        Theta.data = np.exp(Theta.data)     # computes the element-wise exponential of the phase angle values stored in the matrix Theta  # Accessing the .data attribute of a sparse matrix in SciPy returns the non-zero elements of the sparse matrix
        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            d_1d = np.array(d).flatten()
            D = coo_matrix((d_1d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,

    if norm:      # Qin delete this block,no use(experiments show keeping is better, in deeper layers)
        L = (2.0 / max_eigen) * L - diag

    return L
def QinDirect_hermitian_decomp_sparse3(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    third version of MagQin, set Qin_Theta=q*I
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    # if edge_weight is None:
    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
        # creates a sparse matrix A where the non-zero elements are located at the coordinates specified by row and col, and each non-zero element has a value of 1.
    # else:
    #     A = coo_matrix((edge_weight.detach().numpy(), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)


    if QinDirect:
        # Qin_Theta= q*(A + A.T)
        Qin_Theta= q*diag
        diff = A - A.T
        diff = triu(diff)
        diff = diff + diff.T
        diff[diff == 0] = 1
        Qin_Theta = Qin_Theta.multiply(diff)

        if norm:
            D = diag
        else:       # without norm is worse
            d = np.sum(A_sym, axis=0)  # diag of degree array
            d_1d = np.array(d).flatten()
            diagonal_indices = np.arange(size)

            D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
        QinL = D - Qin_Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,

    if norm:
        QinL = (2.0 / max_eigen) * QinL - diag

    return QinL

def QinDirect_hermitian_decomp_sparse4(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    4th version of MagQin, normalize theta_Qin*A_sym
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    # if norm:
    #     d = np.array(A_sym.sum(axis=0))[0]  # Qin note: use A is the out_degree
    #     # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
    #     d[d == 0] = 1
    #     d = np.power(d, -0.5)
    #     D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #     A_sym = D.dot(A_sym).dot(D)


    if QinDirect:
        # Qin_Theta0= q*(A + A.T)
        # Qin_Theta0= A + A.T
        # Qin_Theta= q*diag
        diff = A - A.T
        diff = triu(diff)   # extract the upper triangle of differ
        diff = diff + diff.T    # diff is symmetric now
        diff = q*diff
        diff[diff == 0] = 1
        # Qin_Theta = Qin_Theta0.multiply(diff)

        if norm:
            D = diag
        else:       # without norm is worse
            d = np.sum(A_sym, axis=0)  # diag of degree array
            d_1d = np.array(d).flatten()
            diagonal_indices = np.arange(size)

            D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
        # QinL = D - Qin_Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,
        QinL = D - diff.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,
        # QinL = D - Qin_Theta  # Qin


    # if norm:
    #     QinL = (2.0 / max_eigen) * QinL - diag

    return QinL

def QinDirect_hermitian_decomp_sparse5(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    5th version of MagQin, change A_sym
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency
    A_sym[A_sym == 0.5] = q

    diff = A - A.T
    diff = triu(diff)   # extract the upper triangle of differ
    diff = diff + diff.T    # diff is symmetric now
    diff[diff == 0] = 1
    A_sym = A_sym.multiply(diff)

    if norm:
        A_sym_abs = np.abs(A_sym)
        d = np.array(A_sym_abs.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

        D = diag
    else:       # without norm is worse
        d = np.sum(A_sym, axis=0)  # diag of degree array
        d_1d = np.array(d).flatten()
        diagonal_indices = np.arange(size)

        D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
    QinL = D - A_sym  # element-wise  # L= D − H= D− A.P,

    return QinL

def QinDirect_hermitian_decomp_sparse6(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    6th version of MagQin, 5th edition but no normalization
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency
    A_sym[A_sym == 0.5] = q

    diff = A - A.T
    diff = triu(diff)   # extract the upper triangle of differ
    diff = diff + diff.T    # diff is symmetric now
    diff[diff == 0] = 1
    A_sym = A_sym.multiply(diff)

    norm = False

    if norm:
        A_sym_abs = np.abs(A_sym)
        d = np.array(A_sym_abs.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

        D = diag
    else:       # without norm is worse
        A_sym_abs = np.abs(A_sym)
        d = np.sum(A_sym_abs, axis=0)  # diag of degree array
        # d = np.sum(A_sym, axis=0)  # diag of degree array
        d_1d = np.array(d).flatten()
        diagonal_indices = np.arange(size)

        D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
    QinL = D - A_sym  # element-wise  # L= D − H= D− A.P,

    return QinL

def QinDirect_hermitian_decomp_sparse7(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    8th version of MagQin, avoid float comparison
    7th version:  5th edition but no normalization+directed edge add degree 1,not q
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency
    # tolerance = 1e-5
    A_sym_tensor = torch.tensor(A_sym.toarray())

    # Define tolerance for approximate comparison
    tolerance = 1e-5  # Adjust the tolerance based on your requirements

    # Replace elements close to 0.5 with the value of q
    mask = torch.isclose(A_sym_tensor, torch.tensor(0.5), atol=tolerance)
    A_sym[mask] = q
    count_true = torch.sum(mask).item()

    # print("Number of elements satisfying the condition:", count_true)
    diff = A - A.T      # 0, (-1, 1) , 0
    diff = triu(diff)   # extract the upper triangle of differ
    diff = diff + diff.T    # diff is symmetric now
    diff[diff == 0] = 1     # 1, (-1, 1) , 1
    A_sym = A_sym.multiply(diff)   # 0, (-q, q), 1

    if norm:
        A_one = A_sym
        A_one[A_one != 0] = 1
        d = np.array(A_one.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

        D = diag
    else:       # without norm is worse and fluctuate much
        A_one = A_sym
        A_one[A_one != 0] = 1
        d = np.array(A_one.sum(axis=0))[0]
        d_1d = np.array(d).flatten()
        diagonal_indices = np.arange(size)

        D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
    QinL = D - A_sym  # element-wise  # L= D − H= D− A.P,

    return QinL
def QinDirect_hermitian_decomp_sparse(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    8th version of MagQin, avoid float comparison; essentially, A is (0,1,1,1)=undirected
    7th version:  5th edition but no normalization+directed edge add degree 1,not q
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym_origin = 0.5 * (A + A.T)  # symmetrized adjacency
    A_sym = A_sym_origin.copy()
    A_sym_2 = 2 * A_sym  # symmetrized adjacency_ to avoid float
    # tolerance = 1e-5
    A_sym_tensor = torch.tensor(A_sym_2.toarray())
    mask = (A_sym_tensor == 1)

    A_sym[mask] = q

    # print("Number of elements satisfying the condition:", count_true)
    diff = A - A.T      # 0, (-1, 1) , 0
    diff = triu(diff)   # extract the upper triangle of differ
    diff = diff + diff.T    # diff is symmetric now
    diff[diff == 0] = 1     # 1, (-1, 1) , 1
    A_sym = A_sym.multiply(diff)   # 0, (-q, q), 1

    if norm:
        A_one = A_sym.copy()        # A_one = A_sym Qin attention! not copy!  Now I know all my MagQin is MagQinabs(1)(q not 0) or MagQin0
        A_one[A_one != 0] = 0.5
        d = np.array(A_one.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # d = np.array(A_sym_origin.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

        D = diag
    else:  # without norm is worse and fluctuate much
        A_one = A_sym.copy()
        A_one[A_one != 0] = 1
        d = np.array(A_one.sum(axis=0))[0]
        d_1d = np.array(d).flatten()
        diagonal_indices = np.arange(size)

        D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
    QinL = D - A_sym  # element-wise  # L= D − H= D− A.P,
    # print('A_sym', QinL)

    return QinL

def QinDirect_hermitian_decomp_sparse0(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    first version of MagQin, set q=0.5, so two layer is bad, 1 layer is ok
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    # if edge_weight is None:
    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
        # creates a sparse matrix A where the non-zero elements are located at the coordinates specified by row and col, and each non-zero element has a value of 1.
    # else:
    #     A = coo_matrix((edge_weight.detach().numpy(), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # Qin note: use A is the out_degree
        # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)


    if QinDirect:
        Qin_Theta= q*(A + A.T)   # q=0.5
        # Qin_Theta= q*diag
        diff = A - A.T
        diff = triu(diff)
        diff = diff + diff.T
        diff[diff == 0] = 1
        Qin_Theta = Qin_Theta.multiply(diff)

        if norm:
            D = diag
        else:       # without norm is worse
            d = np.sum(A_sym, axis=0)  # diag of degree array
            d_1d = np.array(d).flatten()
            diagonal_indices = np.arange(size)

            D = coo_matrix((d_1d, (diagonal_indices, diagonal_indices)), shape=(size, size), dtype=np.float32)
        QinL = D - Qin_Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,


    if norm:
        # L = (2.0 / max_eigen) * L - diag
        QinL = (2.0 / max_eigen) * QinL - diag

    return QinL

def QinDirect_hermitian_decomp_sparse2(row, col, size, q=0.25, norm=True,  QinDirect=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''
    version2: change norm of d
    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    # if edge_weight is None:
    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
        # creates a sparse matrix A where the non-zero elements are located at the coordinates specified by row and col, and each non-zero element has a value of 1.
    # else:
    #     A = coo_matrix((edge_weight.detach().numpy(), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    # if norm:
    #     d = np.array(A_sym.sum(axis=0))[0]  # Qin note: use A is the out_degree
    #     # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
    #     d[d == 0] = 1
    #     d = np.power(d, -0.5)
    #     D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    #     A_sym = D.dot(A_sym).dot(D)

    if QinDirect:
        Qin_Theta= 0.5*(A + A.T)
        diff = A - A.T
        diff = triu(diff)
        diff = diff + diff.T
        diff[diff == 0] = 1
        Qin_Theta = Qin_Theta.multiply(diff)

        if norm:
            d = np.array(np.abs(Qin_Theta).sum(axis=0))[0]
            # d = np.array(Qin_Theta.sum(axis=0))[0]  # Qin note: use A is the out_degree
            # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
            d[d == 0] = 1
            d = np.power(d, -0.5)
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
            Qin_Theta = D.dot(Qin_Theta).dot(D)
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        QinL = D - Qin_Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,


    if norm:
        # L = (2.0 / max_eigen) * L - diag
        QinL = (2.0 / max_eigen) * QinL - diag

    return QinL