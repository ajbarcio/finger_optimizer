import sympy as sp
import numpy as np
from scipy.linalg import null_space
from strucMatrices import *
from combinatorics import generate_centered_qutsm
from combinatorics import identify_strict_central, identify_strict_sign_central
np.set_printoptions(formatter={'float': '{:.5f}'.format})

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

for i in [2,3,4,5,6,7,8,9]:
    print("-----------------------------------------------------------------------------")
    print(i)
    # Choose structure matrix
    # S = centeredType1
    # S = Optimus
    # S = test2
    # S = canonB
    # S = S()
    # S = StrucMatrix(S=S[:-1,:-1])
    S = np.array(np.hstack([-2*np.eye(i)+np.ones((i,i)),np.atleast_2d(-np.ones(i)).T]))
    # print(S)
    S = StrucMatrix(S=S)
    # S = LED
    # S = diagonal
    # S = quasiHollow
    # S = generate_centered_qutsm([S()])
    # S = S[0]
    # # print(S)
    # print(identify_sign_central(S))
    print("Validly controllable?", identify_strict_central(S))
    print("Inherently controllable?", identify_strict_sign_central(S))

    # print(identify_SC()))
    # print(identify_SC(S()))
    # # S = S()[:-1,:-1]
    # S = StrucMatrix(S=S)
    m = S.numJoints
    n = S.numTendons
    Sm = S() # Extract structure as matrix


    # Construct M s.t. M * diag(K_A) = (S * K_A * S^T)_p,q s.t. p != q (all non-diagonal elements)
    # K_A is by def a diagonal matrix, so
    # M * diag(K_A) = 0 => K_J diagonal (all off-diagonal elements equal to 0)

    # Build M with one row per off-diagonal pair (p < q)
    rows = []
    # for each row  of K_J:
    for i in range(m):
        # and each element in that row above the diagonal
        for j in range(i+1, m):
            # there is a row in M that is the dot product of two rows in S
            rows.append(Sm[i, :] * Sm[j, :])
    # And we want M * K_J = 0, so we need null space of M
    M = np.vstack(rows)
    print("structure:", Sm)
    print(Sm)
    print("M shape:", M.shape)
    print("Off-Diagonals:", M)
    # Compute null space of M
    ns = null_space(M)
    print("validity of structure:", S.biasForceSpace)
    print(S.rankCondition)
    print("validity of off-diagonal condition:", ns)
    print("off-diagonal producing matrix", M)
    print(np.linalg.matrix_rank(M))

    if ns.size == 0:
        print("Only trivial b=0 exists.")
    else:
        print("Nullspace shape:", ns.shape)
        # For a 3x4 structure matrix, we expect only one basis vector, but I did testing on 2x3 matrices that have a 2-dimensional stiffness null space
        # stiffness_dir = ns[:, 0]
        stiffness_dir = find_positive_in_nullspace(ns)[0] # wanted to make sure the stiffnesses were all positive.
        if stiffness_dir is None:
            stiffness_dir = ns[:, 0]
        # For a 1-dimensional null space this ends up being redundant; uniformly-signed null spaces come out as pos.
        print(stiffness_dir)
        # scale to unit minimum: this just generally results in rounder numbers
        scale = 1.0/np.min(stiffness_dir) if np.min(stiffness_dir) != 0 else 1.0
        # scale = 1.0
        stiffnesses = stiffness_dir * scale
        print("tendon-space stiffnesses (to scale):", stiffnesses)
        # Confirm results
        K_A = np.diag(stiffnesses)
        K_J = Sm.dot(K_A).dot(Sm.T)
        print("K_J =\n", K_J)

