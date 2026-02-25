import numpy as np
import cvxpy
from scipy.optimize import milp, LinearConstraint, Bounds

module = 0.5

# Form factor constraints:
#      da = smallest possible pitch diameter (of sun gear)
#      dc = largest possible pitch diameter (of ring gear)

OD = 51.2
ID = 10.525*2

OD = OD-1.5*2 # take off OD for material
ID = ID+1.5*2 # add ID for material

# add dedendum
OD = OD-1.25*module*2
ID = ID+1.25*module*2

ds = ID # mm
dr = OD # mm
dp = (dr-ds)/2

# def fit_3k_planetary(d, m):
#     d = np.asarray(d, dtype=float)
#     d = np.concatenate([d, ])
#     n = len(d)

#     # Objective: minimize sum of absolute deviations |m*z - d|
#     # Introduce auxiliary variables s_i ≥ 0 for absolute values
#     # Variables order: [z_0...z_{n-1}, s_0...s_{n-1}]
#     c = np.concatenate([np.zeros(n), np.ones(n)])  # minimize sum(s_i)

#     # Constraints:
#     #   m*z_i - d_i ≤ s_i
#     #  -(m*z_i - d_i) ≤ s_i   -> linear form for absolute value
#     A_ineq = np.block([
#         [ np.eye(n)*m, -np.eye(n)],   #  m*z - s ≤ d
#         [-np.eye(n)*m, -np.eye(n)]    # -m*z - s ≤ -d
#     ])
#     b_ineq = np.concatenate([d, -d])
#     # print(A_ineq)

#     #  m*z_3 < d_3
#     #  m*z_1 > d_1 -> -m*z_1 < -d_1 (assuming d, z positive, which they shall be)
#     A_ineq2 = np.zeros((2,6))
#     A_ineq2[0,0] = -m
#     A_ineq2[1,2] = m
#     b_ineq2 = np.array([-d[0],d[2]])

#     # Equality constraint: z2 = (z3 - z1)/2  -> 2*z2 - z3 + z1 = 0
#     A_eq = np.zeros((1, n*2))
#     A_eq[0, [0, 1, 2]] = [1, 2, -1]
#     b_eq = np.zeros(1)

#     constraints = [
#         LinearConstraint(A_ineq, -np.inf, b_ineq),
#         LinearConstraint(A_eq, b_eq, b_eq),
#         LinearConstraint(A_ineq2, -np.inf, b_ineq2)
#     ]

#     # Bounds: z free (integer), s ≥ 0
#     lb = np.concatenate([-np.inf*np.ones(n), np.zeros(n)])
#     ub = np.inf*np.ones(2*n)
#     bounds = Bounds(lb, ub)

#     integrality = np.concatenate([np.ones(n), np.zeros(n)])  # z integer, s continuous

#     res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
#     if res.success:
#         z = res.x[:n].round().astype(int)
#         return z, res
#     else:
#         return None, res


def fit_single_stage_planetary(d, m):
    d = np.asarray(d, dtype=float)
    n = len(d)

    # Objective: minimize sum of absolute deviations |m*z - d|
    # Introduce auxiliary variables s_i ≥ 0 for absolute values
    # Variables order: [z_0...z_{n-1}, s_0...s_{n-1}]
    c = np.concatenate([np.zeros(n), np.ones(n)])  # minimize sum(s_i)

    # Constraints:
    #   m*z_i - d_i ≤ s_i
    #  -(m*z_i - d_i) ≤ s_i   -> linear form for absolute value
    A_ineq = np.block([
        [ np.eye(n)*m, -np.eye(n)],   #  m*z - s ≤ d
        [-np.eye(n)*m, -np.eye(n)]    # -m*z - s ≤ -d
    ])
    b_ineq = np.concatenate([d, -d])
    # print(A_ineq)

    #  m*z_3 < d_3
    #  m*z_1 > d_1 -> -m*z_1 < -d_1 (assuming d, z positive, which they shall be)
    A_ineq2 = np.zeros((2,6))
    A_ineq2[0,0] = -m
    A_ineq2[1,2] = m
    b_ineq2 = np.array([-d[0],d[2]])

    # Equality constraint: z2 = (z3 - z1)/2  -> 2*z2 - z3 + z1 = 0
    A_eq = np.zeros((1, n*2))
    A_eq[0, [0, 1, 2]] = [1, 2, -1]
    b_eq = np.zeros(1)

    constraints = [
        LinearConstraint(A_ineq, -np.inf, b_ineq),
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_ineq2, -np.inf, b_ineq2)
    ]

    # Bounds: z free (integer), s ≥ 0
    lb = np.concatenate([-np.inf*np.ones(n), np.zeros(n)])
    ub = np.inf*np.ones(2*n)
    bounds = Bounds(lb, ub)

    integrality = np.concatenate([np.ones(n), np.zeros(n)])  # z integer, s continuous

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    if res.success:
        z = res.x[:n].round().astype(int)
        return z, res
    else:
        return None, res

M = module*np.eye(3)

d_des = np.array([ds, dp, dr])
z, res = fit_single_stage_planetary(d_des, module)
d = M.dot(z)

print(z, d, d_des)

print("possible ratios")
a_ratio = z[2]/z[0]+1
b_ratio = z[0]/z[2]+1
c_ratio = z[2]/z[0]
print("fix ring    ", a_ratio)
print("fix sun    ", b_ratio)
print("fix planet ", c_ratio)

delta = 1

tk_denominator = 1-((z[2]*z[1]-z[2]*delta)/
                    (z[2]*z[1]-z[1]*delta))
print(tk_denominator)
aa_ratio = a_ratio**2/(tk_denominator)
ab_ratio = a_ratio*b_ratio/(tk_denominator)
bb_ratio = b_ratio**2/(tk_denominator)

print("Fix two rings   ", aa_ratio)
print("fix ring and sun", ab_ratio)
print("Fix two suns    ", bb_ratio)

z = np.concatenate([z, [z[1]-delta], [z[2]-delta]])
print(z)

print("fix sun, then fix two rings", aa_ratio*b_ratio)
print("always fix sun             ", bb_ratio*b_ratio)