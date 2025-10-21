import sympy as sp
import numpy as np
from scipy.linalg import null_space
from strucMatrices import *


def find_positive_in_nullspace(N, normalize_bound=1.0, tol=1e-9):
    """
    Find c so that b = N @ c has b_i >= eps > 0 (strictly positive numerically)
    Returns (b, c, t) where t is the achieved minimum entry of b.
    If no strictly positive b exists, returns (None, None, t_opt) with t_opt <= 0.
    """
    n, k = N.shape
    if k == 0:
        return None, None, 0.0

    # variables are [c_0..c_{k-1}, t]  length k+1
    # maximize t  ->  minimize -t
    f = np.zeros(k + 1)
    f[-1] = -1.0  # minimize -t  => maximize t

    # constraints: N c - t * 1 >= 0  ->  -N c + t * 1 <= 0
    A_ub = np.hstack([-N, np.ones((n, 1))])   # shape (n, k+1)
    b_ub = np.zeros(n)

    # bounds: for c_j use [-normalize_bound, normalize_bound], for t use (None, None)
    bounds = [(-normalize_bound, normalize_bound) for _ in range(k)] + [(None, None)]

    res = linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        # LP failed to converge; return what we have
        return None, None, None

    sol = res.x
    c = sol[:k]
    t_opt = sol[-1]
    b = N.dot(c)

    # small numerical tolerance: require t_opt > tol and min(b) > -tol
    if t_opt > tol and np.min(b) > -tol:
        return b, c, t_opt
    else:
        # No strictly positive vector found under normalization; return the best found
        return None, None, t_opt

S = test
A = S()
# S = StrucMatrix(S=A)
m, n = A.shape

# Build M with one row per off-diagonal pair (p < q)
rows = []
for p in range(m):
    for q in range(p+1, m):
        rows.append(A[p, :] * A[q, :])
M = np.vstack(rows) if len(rows) > 0 else np.zeros((0, n))
print("structure:", A)
print(A)
print("M shape:", M.shape)
print("Off-Diagonals:", M)
# print(np.linalg.matrix_rank(M))
# Compute null space of M
ns = null_space(M)
print("validity of structure:", S.biasForceSpace)
print(S.rankCondition)
print("validity of off-diagonal condition:", ns)

if ns.size == 0:
    print("Only trivial b=0 exists.")
else:
    print("Nullspace shape:", ns.shape)
    # choose a basis vector (first column) and scale for readability
    b_raw = ns[:, 0]
    # b_raw = find_positive_in_nullspace(ns)[0]
    print(b_raw)
    # scale to integer example: find scalar s so s*b_raw is close to integers
    # here we know theoretical solution proportional to [-8, 3], so we can scale
    s = 1.0/np.min(b_raw) if np.min(b_raw) != 0 else 1.0
    b = b_raw * s
    print("b (up to scale):", b)
    # Show the matrix A B A^T to verify off-diagonals are (close to) zero
    B = np.diag(b)
    ABA = A.dot(B).dot(A.T)
    print("ABA^T =\n", ABA)
    # print("Off-diagonal entries:", ABA - np.diag(np.diag(ABA)))

