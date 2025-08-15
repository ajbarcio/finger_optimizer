import numpy as np
from scipy.optimize import linprog
from combinatorics import *
from strucMatrices import *
import warnings

warnings.filterwarnings("ignore")

def maximize_minimum_force(A, T, nonneg=True):

    A = np.asarray(A)
    T = np.asarray(T).flatten()
    r, m = A.shape

    nvar = m + 2
    idx_u = m
    idx_l = m + 1

    # objective: maximize l
    c = np.zeros(nvar)
    c[idx_u] = 0
    c[idx_l] = -1.0

    # Equality: A F = T  ->  A_eq x = b_eq
    A_eq = np.zeros((r, nvar))
    A_eq[:, :m] = A
    b_eq = T

    # Inequalities A_ub x <= b_ub
    A_ub = []
    b_ub = []

    # 1) F_i <= u  ->  F_i - u <= 0
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = 1.0
        row[idx_u] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    #u <= 1
    for i in range(m):
        row = np.zeros(nvar)
        row[idx_u] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    # 2) F_i >= l  ->  l - F_i <= 0  ->  -F_i + l <= 0
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = -1.0
        row[idx_l] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    if nonneg:
        for i in range(m):
            row = np.zeros(nvar)
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # also we can restrict l >= 0 (otherwise trivial)
        bounds = [(0, None)] * m + [(0, None), (0, None)]
    else:
        bounds = [(None, None)] * m + [(None, None), (None, None)]

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return False, None, None, None, res

    x = res.x
    F_opt = x[:m]
    u_opt = x[idx_u]
    l_opt = x[idx_l]
    return True, F_opt, u_opt, l_opt, res

def balance_forces_via_lp(A, T, nonneg=True):
    """
    Solve min_u-l subject to A F = T, l <= F_i <= u, and F_i >= 0 (optional).
    Returns (success, F_opt, u_opt, l_opt, result_object)
    """
    A = np.asarray(A)
    T = np.asarray(T).flatten()
    r, m = A.shape

    # Variables order: [F_0 ... F_{m-1}, u, l]  => total m+2 variables
    nvar = m + 2
    idx_u = m
    idx_l = m + 1

    # Objective: minimize u - l
    c = np.zeros(nvar)
    c[idx_u] = 1.0
    c[idx_l] = -1.0

    # Equality: A F = T  ->  A_eq x = b_eq
    A_eq = np.zeros((r, nvar))
    A_eq[:, :m] = A
    b_eq = T

    # Inequalities A_ub x <= b_ub
    A_ub = []
    b_ub = []

    # 1) F_i <= u  ->  F_i - u <= 0
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = 1.0
        row[idx_u] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # 2) F_i >= l  ->  l - F_i <= 0  ->  -F_i + l <= 0
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = -1.0
        row[idx_l] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # 3) optionally F_i >= 0  ->  -F_i <= 0
    if nonneg:
        for i in range(m):
            row = np.zeros(nvar)
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # also we can restrict l >= 0 (otherwise trivial)
        bounds = [(0, None)] * m + [(0, None), (0, None)]
    else:
        bounds = [(None, None)] * m + [(None, None), (None, None)]

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return False, None, None, None, res

    x = res.x
    F_opt = x[:m]
    u_opt = x[idx_u]
    l_opt = x[idx_l]
    return True, F_opt, u_opt, l_opt, res

def minimize_u_over_l_lp(A, T, tol_s=1e-9):
    A = np.atleast_2d(A)
    T = np.asarray(T).flatten()
    r, m = A.shape

    nvar = m + 2
    idx_alpha = m
    idx_s = m + 1

    c = np.zeros(nvar)
    c[idx_alpha] = 1.0

    A_eq = np.zeros((r, nvar))
    A_eq[:, :m] = A
    A_eq[:, idx_s] = -T
    b_eq = np.zeros(r)

    A_ub = []
    b_ub = []
    for i in range(m):
        # -G_i <= -1  (G_i >= 1)
        row = np.zeros(nvar)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-1.0)

    for i in range(m):
        # G_i - alpha <= 0
        row = np.zeros(nvar)
        row[i] = 1.0
        row[idx_alpha] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(1.0, None)] * m + [(0.0, None), (tol_s, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return False, None, None, None, res

    x = res.x
    G = x[:m]
    alpha = x[idx_alpha]
    s = x[idx_s]

    F_opt = G / s
    u_opt = alpha / s
    l_opt = 1.0 / s

    return True, F_opt, u_opt, l_opt, res

def minimize_u_over_l_lp_with_Fmax(A, T, F_max=None, tol_s=1e-9):
    """
    Solve min u/l subject to
        A F = T,
        l <= F_i <= u,
        F_i >= 0, l > 0,
    via scaled LP with G = s F, s = 1/l.

    Adds an optional per-component or scalar upper bound F_i <= F_max.

    Inputs
    ------
    A : (r, m) matrix
    T : (r,) vector
    F_max : None or scalar or (m,) array. If None, no upper bound is enforced.
    tol_s : small positive lower bound for s (to avoid s = 0)

    Returns
    -------
    success (bool), F_opt (m,), u_opt (scalar), l_opt (scalar), res (OptimizeResult)
    """
    A = np.atleast_2d(A)
    T = np.asarray(T).flatten()
    r, m = A.shape

    # Variables ordering: [G_0 ... G_{m-1}, alpha, s]  -> total m+2 variables
    nvar = m + 2
    idx_alpha = m
    idx_s = m + 1

    # objective: minimize alpha
    c = np.zeros(nvar)
    c[idx_alpha] = 1.0

    # Equality constraints: A G - s T = 0  (shape r x nvar)
    A_eq = np.zeros((r, nvar))
    A_eq[:, :m] = A
    A_eq[:, idx_s] = -T
    b_eq = np.zeros(r)

    # Inequalities A_ub x <= b_ub
    A_ub_list = []
    b_ub_list = []

    # 1) G_i >= 1  ->  -G_i <= -1
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = -1.0
        A_ub_list.append(row)
        b_ub_list.append(-1.0)

    # 2) G_i <= alpha -> G_i - alpha <= 0
    for i in range(m):
        row = np.zeros(nvar)
        row[i] = 1.0
        row[idx_alpha] = -1.0
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    # 3) Optional: upper bound F_i <= F_max  =>  G_i - s * F_max <= 0
    if F_max is not None:
        F_max_arr = np.asarray(F_max)
        if F_max_arr.size == 1:
            F_max_arr = np.full(m, float(F_max_arr))
        elif F_max_arr.size != m:
            raise ValueError("F_max must be scalar or length-m array")
        for i in range(m):
            row = np.zeros(nvar)
            row[i] = 1.0            # coeff for G_i
            row[idx_s] = -F_max_arr[i]  # coeff for s (negative because move to LHS)
            A_ub_list.append(row)
            b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    # Bounds: G_i in [1, +inf), alpha >= 0, s >= tol_s
    bounds = [(1.0, None)] * m + [(0.0, None), (tol_s, None)]

    # Call solver
    res = linprog(c,
                  A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return False, None, None, None, res

    x = res.x
    G = x[:m]
    alpha = x[idx_alpha]
    s = x[idx_s]

    # Recover original variables
    F_opt = G / s
    u_opt = alpha / s
    l_opt = 1.0 / s

    return True, F_opt, u_opt, l_opt, res

trials = [canonA, canonB, quasiHollow]

for S_base in trials:

    S1 = S_base.S
    # print(S1)
    S  = generate_centered_qutsm([canonB.S])
    S = S[0]
    # print(S)
    # print(np.linalg.norm(S))
    # T = np.array([0.5,0,0])
    S = StrucMatrix(S=S)
    # print(S.biasCondition())
    T_index = np.argmax([np.linalg.norm(boundaryGrasp) for boundaryGrasp in S.boundaryGrasps])
    T = [0.25,0,0]
    print(T)
    rand_wins = 0
    cent_wins = 0
    for i in np.linspace(0,2*np.pi,10):
        for j in np.linspace(0,2*np.pi,10):
            for k in np.linspace(0,2*np.pi,10):
                pass
                # R = np.array([
                #     [np.cos(j)*np.cos(k), -np.cos(j)*np.sin(k), np.sin(j)],
                #     [np.cos(i)*np.sin(k) + np.sin(i)*np.sin(j)*np.cos(k),
                #     np.cos(i)*np.cos(k) - np.sin(i)*np.sin(j)*np.sin(k),
                #     -np.sin(i)*np.cos(j)],
                #     [np.sin(i)*np.sin(k) - np.cos(i)*np.sin(j)*np.cos(k),
                #     np.sin(i)*np.cos(k) + np.cos(i)*np.sin(j)*np.sin(k),
                #     np.cos(i)*np.cos(j)]
                # ])

                # Tin = R @ T
                # _, _, _, l_opt_rand, _ = maximize_minimum_force(S1, Tin, nonneg=True)
                # _, _, _, l_opt_cent, _ = maximize_minimum_force(S.S, Tin, nonneg=True)

                # if l_opt_cent > l_opt_rand:
                #     cent_wins +=1
                #     print(i,j,k,'cent',end="\r")
                # elif l_opt_cent < l_opt_rand:
                #     rand_wins +=1
                #     print(i,j,k,'rand',end="\r")
                # else:
                #     print(i,j,k,'tie',end="\r")
    for tindex in range(len(S_base.boundaryGrasps)):
        Tin1 = S_base.boundaryGrasps[tindex]
        Tin2 = S.boundaryGrasps[tindex]
        _, _, _, l_opt_rand, _ = maximize_minimum_force(S1, Tin1, nonneg=True)
        _, _, _, l_opt_cent, _ = maximize_minimum_force(S.S, Tin2, nonneg=True)

        if l_opt_cent > l_opt_rand:
            cent_wins +=1
            print(i,j,k,'cent',end="\r")
        elif l_opt_cent < l_opt_rand:
            rand_wins +=1
            print(i,j,k,'rand',end="\r")
        else:
            print(i,j,k,'tie',end="\r")
    print("")
    print("name:", S_base.name)
    print(S_base.biasCondition())
    print(rand_wins/cent_wins)
    print("")

# Fp = np.linalg.pinv(S.S) @ T
# print(Fp)
# n = S.biasForceSpace
# print(n)
# print("")
# # # Example data
# # # Fp = np.array([2.0, -1.0, 0.5])
# # # n  = np.array([0.5, -0.2, 1.0])

# ok, F_opt, u_opt, l_opt, res = maximize_minimum_force(S.S, T, nonneg=True)

# if ok:
#     print("F_opt:", F_opt)
#     print("u:", u_opt, "l:", l_opt, "u/l:", u_opt / l_opt)
#     print("range u-l:", u_opt - l_opt)
# else:
#     print("LP failed:", res.message)

# if ok:
#     print("F_opt:", F_opt)
#     print("range u-l:", u_opt - l_opt)
#     print("u, l:", u_opt, l_opt)
# else:
#     print("LP failed:", res.message)
# m = len(Fp)

# #lambda, u, l, F
# c = np.array([0.0, 1.0, -1.0])

# # Inequality constraints: l <= Fi <= u  AND  Fi >= 0
# # Fi = Fp[i] + lambda * n[i]

# A_ub = []
# b_ub = []

# for i in range(m):
#     # Fi <= u: Fp[i] + lambda*n[i] - u <= 0
#     row = np.zeros(3)
#     row[0] = n[i]      # coeff of lambda
#     row[1] = -1        # coeff of u
#     row[2] = 0
#     b = -Fp[i]
#     A_ub.append(row)
#     b_ub.append(b)

#     # Fi >= l: -(Fp[i] + lambda*n[i]) + l <= 0
#     row = np.zeros(3)
#     row[0] = -n[i]
#     row[1] = 0
#     row[2] = 1
#     b = Fp[i]
#     A_ub.append(row)
#     b_ub.append(b)

#     # Fi >= 0: -(Fp[i] + lambda*n[i]) <= 0
#     row = np.zeros(3)
#     row[0] = -n[i]
#     row[1] = 0
#     row[2] = 0
#     b = Fp[i]
#     A_ub.append(row)
#     b_ub.append(b)

# A_ub = np.array(A_ub)
# b_ub = np.array(b_ub)

# # No bounds on lambda, but u and l must be >= 0
# bounds = [(None, None), (0, 1), (0, None)]

# res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
# print(res.success)
# lam_opt, u_opt, l_opt = res.x
# F_opt = Fp + lam_opt * n.T
# print("λ* =", lam_opt)
# # print("Min infinity norm =", t_opt)
# print("Force range:", u_opt - l_opt)
# print("optimal F:", F_opt)