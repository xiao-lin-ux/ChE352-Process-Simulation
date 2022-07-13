# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:44:34 2022

@author: lunac
"""
import numpy as np
from numpy import linalg as LA

# Perform the matrix calculations listed in part A on the given two matrices and one vector
def matrix_calculation(mat_1, mat_2, vec_1):
    # product = vec_1_T @ vec_1
    # print(product)
    ans = (vec_1.T @ vec_1) + vec_1.T @ mat_1 @ mat_2 @ vec_1
    print(f'A) {ans}')
    return 0

# Get the determinant of a given matrix
def get_determinant(mat):
    det = LA.det(mat)
    print(f'B) {det:.6}')
    return det

# Get the rank of a given matrix
def get_rank(mat):
    rank = LA.matrix_rank(mat)
    print(f'C) {rank}')
    return rank

# Solve for x in Ax = b, with given matrix A and vector b
def get_sol(mat, vec):
    x = LA.solve(mat, vec)
    # print(np.allclose(mat @ x, vec))
    print(f'D) {x = }')
    return x

# Get the eigenvalue of the given matrix
def get_eigen_val(mat):
    eigen_val = LA.eigvalsh(mat.T @ mat)  # Function eigvalsh() can be implemented since A is a symmetric matrix
    print(f'E) {eigen_val}')
    return eigen_val

# Get the 2_norm of the given matrix
def get_norm(mat):
    norm = LA.norm(mat, 2)
    print(f'F) {norm:.3}')
    return norm

# Get the norm_inf of the given combination of matrix and vector
def get_norm_inf(mat, vec):
    norm_inf = LA.norm(vec.T @ mat.T, np.inf)  # Use np.inf to indicate the order of the norm
    print(f'G) {norm_inf:.4}')
    return norm_inf

# Solve for one set of E and psi that satisfies A*psi = E *psi
def solve_Hamiltonian(mat):
    E, psi = LA.eigh(mat)  # Function eigh() can be implemented since A is a symmetric matrix
    print(f'H) E = {E[0]:.3}\n Psi = {psi[:,0]}')
    '''
    print(E)
    print(psi)
    for i in range(0,5):
        print(np.allclose(A @ psi[:, i], E[i] * psi[:, i]))
    '''
    return E, psi

# Define two matrices, A and B, and one vector, b
A = np.array([
    [9,   1.5, 0,   2.5, 0.5],
    [1.5, 10,  1.5, 0.5, 2  ],
    [0,   1.5, 11,  0,   2  ],
    [2.5, 0.5, 0,   8,   1  ],
    [0.5, 2,   2,   1,   5  ],
])

B = np.array([
    [8,  -1,  0, 0, 0],
    [0,  5,   2, 0, 0],
    [0,  0,   9, 1, 0],
    [0,  0.5, 0, 7, 0],
    [-1, 0,   0, 1, 6],
])

b = (np.array([[1, 2, 3, 4, 5]])).T

# Reference: Set array floating point precision
# https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
np.set_printoptions(precision=4)

# Run all the listed functions
matrix_calculation(A, B, b)
get_determinant(A)
get_rank(B)
get_sol(B, b)
get_eigen_val(A)
get_norm(B)
get_norm_inf(A, b)
solve_Hamiltonian(A)

'''
# The code below are for testing purposes
c = np.array([1, 2, 3, 4, 5])
c_T = c.T
print(b)
print(c)
print(c_T)
'''