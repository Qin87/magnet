import numpy as np


def matrix_power(A):
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Calculate A^4 using matrix multiplication
    result = np.linalg.matrix_power(A, 5)
    # Alternatively, you could do: result = A @ A @ A @ A

    return result


# Example usage
# Create a sample 5x5 matrix
# A = np.array([
#     [0,1,0,0,0],
#     [1,0,1,0,0],
#     [0,1,0,0,1],
#     [0,0,0,0,1],
#     [0,0,1,1,0]
# ])

# A = np.array([
#     [0,1,1,1],
#     [0,0,1,0],
#     [0,0,0,0],
#     [0,0,0,0]
# ])

A = np.array([
    [0,1,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [1,0,0,0]
])

# Calculate A^4
result = matrix_power(A)
print("A^4 =")
print(result)