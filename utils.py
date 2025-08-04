from scipy.optimize import linprog
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from scipy.spatial import ConvexHull, QhullError
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import itertools

def generate_matrices_from_pattern(D, value_set={-1, 0, 1}):
    D = np.array(D)
    mask = D != 0  # where values are allowed to vary
    positions = np.argwhere(mask)
    num_vars = positions.shape[0]

    for values in itertools.product(value_set, repeat=num_vars):
        result = D.copy().astype(int)
        for (i, j), val in zip(positions, values):
            result[i, j] = val
        yield result

def canonical_form(mat):
    mat = mat.copy()

    # Step 1: Normalize each row's sign (first non-zero entry should be positive)
    for i in range(mat.shape[0]):
        row = mat[i]
        for val in row:
            if val != 0:
                if val < 0:
                    mat[i] *= -1
                break

    # Step 2: Sort columns lexicographically
    col_order = np.lexsort(mat[::-1])
    mat_sorted = mat[:, col_order]

    # Step 3: Convert to a hashable structure
    return tuple(map(tuple, mat_sorted))

def normalize_row_signs(mat):
    mat2 = mat.copy()
    for i in range(mat2.shape[0]):
        row = mat2[i]
        for val in row:
            if val != 0:
                if val < 0:
                    mat2[i] = -row
                break
    return mat2

def is_upper_triangular_by_leading_zeros(mat):
    for i, row in enumerate(mat):
        if not np.all(row[:i] == 0):
            return False
        if i < len(row) and np.any(row[i:] != 0) and row[i] == 0:
            # first nonzero after i zeros must occur at position i
            return False
    return True

def canonical_form(mat):
    mat = normalize_row_signs(mat)
    n_cols = mat.shape[1]

    best = None
    for perm in itertools.permutations(range(n_cols)):
        permuted = mat[:, perm]
        if is_upper_triangular_by_leading_zeros(permuted):
            candidate = tuple(map(tuple, permuted))
            if best is None or candidate < best:
                best = candidate
    return best

def remove_isomorphic(matrices):
    seen = set()
    unique = []

    for mat in matrices:
        canon = canonical_form(mat)
        if canon not in seen:
            seen.add(canon)
            unique.append(np.array(canon))

    return unique

def hsv_to_rgb(h, s, v):
   # Assume h, s, v are all given on [0,1]
   c = v*s
   x = c*(1-abs(h/(1/6) % 2 - 1))
   m = v - c
   if h<1/6:
      return [c,x,0]
   elif h<2/6:
      return [x,c,0]
   elif h<3/6:
      return [0,c,x]
   elif h<4/6:
      return [0,x,c]
   elif h<5/6:
      return [x,0,c]
   elif h<6/6:
      return [c,0,x]

def get_existing_3d_axes():
    for num in plt.get_fignums():
        fig = plt.figure(num)
        for ax in fig.get_axes():
            if isinstance(ax, Axes3D):
                return ax
    return None

def get_existing_axes():
    # Get the current figure manager, if any figures exist
    if not plt.get_fignums():
        return None  # No figures exist at all

    # Loop through existing figures to find one with axes
    for num in plt.get_fignums():
        fig = plt.figure(num)
        axes = fig.get_axes()
        if axes:
            return axes[0]  # Return first existing Axes
    return None  # No axes found

def nullity(basis):
  return null_space(basis).shape[1]

def intersects_negative_orthant(basis): #basis is a d x N matrix where the ith row is v_i
  # print(basis)
  # print(basis.shape)
  basis = np.atleast_2d(basis)
  (d, N) = basis.shape
  M = basis.T

  if matrix_rank(basis) == N: #if basis spans R^N
    return True

  A = M #negative sign for >=
  # print(A)
  b = np.zeros(N)
  for i in range(N):
    c = M[i,:] #negative sign to maximize
    # print("c", c)
    opt = linprog(c, A, b)
    # print(opt.status)
    # print(opt.fun)
    # print(opt.x)
    if opt.status == 2: #infeasible
      return False
    elif opt.status == 3: #unbounded
      continue
    elif opt.fun >= 0: #optimal value is not positive
      return False

  return True

def intersects_positive_orthant(basis): #basis is a d x N matrix where the ith row is v_i
  # print(basis)
  # print(basis.shape)
  basis = np.atleast_2d(basis)
  (d, N) = basis.shape
  M = basis.T

  if matrix_rank(basis) == N: #if basis spans R^N
    return True

  A = -M #negative sign for >=
  # print(A)
  b = np.zeros(N)
  for i in range(N):
    c = -M[i,:] #negative sign to maximize
    # print("c", c)
    opt = linprog(c, A, b)
    # print(opt.status)
    # print(opt.fun)
    # print(opt.x)
    if opt.status == 2: #infeasible
      return False
    elif opt.status == 3: #unbounded
      continue
    elif opt.fun <= 0: #optimal value is not positive
      return False

  return True

def special_minkowski(points):
  width = len(points)+2
  boundaryPoints = []
  for i in range(2**(len(points))):
    coeffs = [int(digit) for digit in list(f"{i:0{width}b}")[2:]]
    partialSum = [0]*len(points[0])
    for j in range(len(points)):
      partialSum = [a+b for a, b, in zip(partialSum, [coeffs[j] * component for component in points[j]])]
    if i:
      boundaryPoints.append(partialSum)
  if not [0]*len(points[0]) in boundaryPoints:
    boundaryPoints.append([0]*len(points[0]))
  boundaryPoints = np.array(boundaryPoints)
  try:
    hull = ConvexHull(boundaryPoints)
  except QhullError:
    hull = ConvexHull(boundaryPoints, qhull_options='QJ')
  return hull, boundaryPoints

def in_hull(hull, x):
    points = hull.points
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def in_hull2(hull, x):
   tol = 1e-8
   return np.all(hull.equations @ np.append(x,1) <= tol)

if __name__ == '__main__':
  basis = np.array([[1, 1]]) #1d subspace of R^2
  print(intersects_positive_orthant(basis)) #True

  basis = np.array([[1, -1]])
  print(intersects_positive_orthant(basis)) #False

  basis = np.array([[2, 3, -1], [1, 0, 2]]) #2d subspace of R^3
  print(intersects_positive_orthant(basis)) #True

  basis = np.array([[2, 3, -1], [1, 0, -2]])
  print(intersects_positive_orthant(basis)) #False

  basis = np.array([[1, 1, 1, 1]])
  print(intersects_positive_orthant(basis)) #True

  basis = np.array([[-1, -1, -1, -1]])
  print(intersects_negative_orthant(basis)) #True

  basis = np.array([[1, 1, 1, 1]])
  print(intersects_negative_orthant(basis)) #False

  basis = np.array([[0         ],
                    [0         ],
                    [0.70710678],
                    [0.70710678]])
  print(intersects_positive_orthant(basis.T)) #False