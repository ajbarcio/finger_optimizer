from itertools import product
import numpy as np
from scipy.optimize import minimize

# Matrix definition
M = np.array([
    [ 1, -1,  1, -1],
    [ 0, -1,  1, -1],
    [ 0,  0,  1, -1]
], dtype=int)

positions = [(i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) if M[i, j] != 0]
signs = np.array([M[i, j] for (i, j) in positions])
n_nonzero = len(positions)
n_cols = M.shape[1]

def feasible_for_partition(partition):
    """
    Check if there exists v_g > 0, y_j > 0 such that:
        sum_{j} sign_ij * v[group(i,j)] * y_j = 0 for all rows i
    """
    groups = list(set(partition))
    k = len(groups)
    
    # Map group index to 0..k-1
    group_map = {g: idx for idx, g in enumerate(groups)}
    part_mapped = [group_map[p] for p in partition]
    
    def objective(vars):
        # vars = v[0:k] + y[0:n_cols], all positive
        v = vars[:k]
        y = vars[k:]
        eqs = []
        for i in range(M.shape[0]):
            total = 0.0
            for idx, (r, c) in enumerate(positions):
                if r == i:
                    total += signs[idx] * v[part_mapped[idx]] * y[c]
            eqs.append(total)
        return np.sum(np.square(eqs))  # squared error of constraints
    
    # Positivity constraints
    bounds = [(1e-6, None)] * (k + n_cols)
    
    # Random restarts to avoid local minima
    for _ in range(10):
        x0 = np.random.rand(k + n_cols) + 0.5
        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < 1e-10:
            return True
    return False

# Brute force over k
min_k = None
best_part = None

for k in range(1, n_nonzero + 1):
    # Generate partitions as assignments of 0..k-1
    for part in product(range(k), repeat=n_nonzero):
        if max(part) != k-1:
            continue  # ensure all k groups are used
        if feasible_for_partition(part):
            min_k = k
            best_part = part
            break
    if min_k is not None:
        break

print(min_k, best_part)

