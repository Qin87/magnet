import numpy as np


A = np.zeros((9, 9))
A[0][1] = 1  # A12=1
A[2][1] = 1  # A32=1
A[3][0] = 1  # A32=1
A[3][2] = 1  # A32=1
A[4][0] = 1  # A32=1
A[5][2] = 1  # A32=1
A[5][2] = 1  # A32=1
A[6][7] = 1  # A32=1
A[6][8] = 1  # A32=1
A[7][3] = 1  # A32=1
A[7][4] = 1  # A32=1
A[8][5] = 1  # A32=1

# Print the adjacency matrix A
print("Adjacency Matrix A:")
print(A)

row_sums = np.sum(A, axis=1)
D = np.diag(np.where(row_sums == 0, 1, row_sums))

# Compute the inverse of D
D_inv = np.linalg.inv(D)

# Compute D^{-1}A
True_P1 = np.dot(D_inv, A)
print("\nTrue_P1:")
print(True_P1)

True_AA_T = np.dot(True_P1, True_P1.T)
print("\nTrue_AA_T:")
print(True_AA_T)

True_A_TA = np.dot(True_P1.T, True_P1)
print("\nTrue_A_TA:")
print(True_A_TA)

True_P2 = np.logical_and(True_AA_T, True_A_TA).astype(int)
print("\nTrue_P2:")
print(True_P2)

True_P2_Qin = np.logical_or(True_AA_T, True_AA_T).astype(int)
print("\nTrue_AA_T:")
print(True_AA_T)

True_P1_2 =np.dot(True_P1, True_P1)
True_P1_T2=np.dot(True_P1.T, True_P1.T)
True_A2AT2= np.dot(True_P1_2, True_P1_T2)
True_AT2A2= np.dot(True_P1_T2, True_P1_2)
# print(np.dot(P2, P2.T))
# print(np.dot(P2.T, P2))

True_P3=np.logical_and(True_A2AT2, True_AT2A2).astype(int)
print("\nTrue_P3:")
print(True_P3)

True_P3_Qin=np.logical_or(True_A2AT2, True_AT2A2).astype(int)
print("\nTrue_P3_Qin:")
print(True_P3_Qin)
############################




AA_T = np.dot(A, A.T)
print("\nAA_T:")
print(AA_T)

A_TA = np.dot(A.T, A)
print("\nA_TA:")
print(A_TA)

P2 = np.logical_and(AA_T, A_TA).astype(int)
print("\nP2:")
print(P2)

P2_Qin = np.logical_or(AA_T, A_TA).astype(int)
print("\nP2_Qin:")
print(P2_Qin)

A2 =np.dot(A, A)
AT2=np.dot(A.T, A.T)
A2AT2= np.dot(A2, AT2)
AT2A2= np.dot(AT2, A2)
# print(np.dot(P2, P2.T))
# print(np.dot(P2.T, P2))

P3=np.logical_and(A2AT2, AT2A2).astype(int)
print("\nP3:")
print(P3)

P3_Qin=np.logical_or(A2AT2, AT2A2).astype(int)
print("\nP3_Qin:")
print(P3_Qin)

# # Define A matrix
# A = np.array([[3, 1], [2, 4]])
#
# # Compute B = DA
# B = np.dot(D, A)
#
# print(A,B,D)
#
# eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
#
# print("Eigenvalues of A:", eigenvalues_A)
# print("Eigenvectors of A:", eigenvectors_A)
# # Compute eigenvalues and eigenvectors of B
# eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
#
# print("Eigenvalues of B:", eigenvalues_B)
# print("Eigenvectors of B:", eigenvectors_B)

# # import numpy as np
# #
# # # Example matrix A
# # A = np.array([[1, 2, 3],
# #               [4, 5, 6],
# #               [7, 8, 9]])
# #
# # # Calculate the difference between A and its transpose in the upper triangle
# # upper_triangle_diff = A - A.T
# #
# # # Create an upper triangle boolean matrix with ones above the main diagonal
# # upper_triangle_mask = np.triu(np.ones_like(upper_triangle_diff), k=1)
# #
# # # Multiply the upper triangle mask with the difference matrix
# # diff = upper_triangle_mask * upper_triangle_diff
# #
# # print(diff)
# #
# # import numpy as np
# #
# # # Example matrix A
# # A = np.array([[1, 2, 3],
# #               [4, 5, 6],
# #               [7, 8, 9]])
# #
# # # Extract the upper triangle of matrix A
# # upper_triangle = np.triu(A)
# #
# # print(upper_triangle)
#
#
# from scipy.sparse import csr_matrix, triu
#
# # Example csr_matrix diff
# diff = csr_matrix([[1, 2, 3],
#                    [4, 0, 5],
#                    [0, 0, 6]])
#
# # Extract the upper triangle of the csr_matrix diff
# upper_triangle = triu(diff)
#
# print(upper_triangle)