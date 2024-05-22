from sympy import I, Matrix, eye

# Redefine the matrices using sympy to handle complex numbers
provided_matrices_sympy = [
    Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
    Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
    Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
    Matrix([[0, 0, -I, 0], [0, 0, 0, -I], [I, 0, 0, 0], [0, I, 0, 0]]),
    Matrix([[0, -I, 0, 0], [I, 0, 0, 0], [0, 0, 0, -I], [0, 0, I, 0]]),
    Matrix([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]),
    Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, -I, 0, 0], [I, 0, 0, 0]]),
    Matrix([[0, 0, -I, 0], [0, 0, 0, I], [I, 0, 0, 0], [0, -I, 0, 0]]),
    Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]),
    Matrix([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]),
    Matrix([[0, 0, 0, -I], [0, 0, -I, 0], [0, I, 0, 0], [I, 0, 0, 0]]),
    Matrix([[0, -I, 0, 0], [I, 0, 0, 0], [0, 0, 0, I], [0, 0, -I, 0]])
]


# Helper function to check if a matrix is unitary
def is_unitary(matrix):
    return matrix.H * matrix == eye(4) and matrix * matrix.H == eye(4)

# Checking the unitarity of each matrix
unitary_checks = [is_unitary(matrix) for matrix in provided_matrices_sympy]

print(unitary_checks)


# # Check the unitarity of each provided matrix
# unitarity_checks = [m.is_unitary() for m in provided_matrices_sympy]
# unitarity_checks
