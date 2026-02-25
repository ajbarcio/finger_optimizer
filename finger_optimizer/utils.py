from scipy.optimize import linprog
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from scipy.spatial import ConvexHull, QhullError, HalfspaceIntersection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.differentiate import jacobian
from itertools import combinations
from math import comb
from numpy import ndarray

colors = [
    'xkcd:electric blue',  # #0652ff – deep glowing blue
    'xkcd:neon green',     # #0cff0c – retina-searing green
    'xkcd:hot pink',       # #ff028d – vibrant magenta-pink
    'xkcd:bright yellow',  # #fffd01 – classic highlighter yellow
    'xkcd:neon purple',    # #bc13fe – super-saturated violet
    'xkcd:bright orange',  # #ff5b00 – bold warm orange
    'xkcd:cyan',           # #00ffff – icy electric blue-green
    'xkcd:magenta',        # #c20078 – deeper than hot pink
    'xkcd:bright red',     # #ff000d – warning-light red
    'xkcd:bright turquoise' # #0ffef9 – glowing aqua
]
import itertools

def hArray(M: ndarray, strn):
   return strn+" "+str(M).replace('\n', '\n'+' '*(len(strn)+1))

def identify_strict_sign_central(S: ndarray):
    success = True
    m = S.shape[0]
    n = S.shape[1]
    Ds = signings_of_order(m)
    for i in range(len(Ds)):
        check = Ds[i] @ S
        valid = False
        for j in range(n):
            if np.all(check[:,j] >=0) & np.any(check[:,j] != 0):
                valid = True
        success &= valid
    return success

def identify_sign_central(S: ndarray):
    success = True
    m = S.shape[0]
    n = S.shape[1]
    Ds = strict_signings_of_order(m)
    for i in range(len(Ds)):
        check = Ds[i] @ S
        # print(check)
        # print(np.any(np.all(check >= 0, axis=0)))
        success &= (np.any(np.all(check >= 0, axis=0)))
    return success

def identify_strict_central(S: ndarray, boundsOverride=False):

    """
    Dual problem of the minimization for Theorem 2.1 presented by
    Brunaldi and Dahl in Strict Sign-Central Matrices
    """
    # A = S()
    n = S.shape[1]
    e = np.ones(n)

    c = np.zeros(n)                 # objective: minimize 0
    # c = np.ones(n)                  # objective: minimize 1-norm of w
    # c = -np.ones(n)                 # objective: maximize 1-norm of w
    A_eq = S                        # equality: A w = -A e
    b_eq = -S @ e
    if boundsOverride:
        bounds = [(None, None)] * n        # allow for the return of any value
    else:
        bounds = [(0, None)] * n        # w >= 0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.x is not None:
        if any(res.x[res.x<0]):
            res.success = False
    # if res.success:
    # print(res.x)
    return res.success, res.x


def worker_chunk(D, signs, positions, chunk_start, chunk_end,
                 counter, counter_lock, return_dict, worker_id):

    checkpoint_file = f"worker_{worker_id}_checkpoint.npy"
    results_file = f"worker_{worker_id}_results.npy"
    # try:
    #     last_idx, processed_so_far = np.load(checkpoint_file)
    #     start_idx = max(chunk_start, last_idx + 1)
    #     seen = set(np.load(results_file, allow_pickle=True))
    # except FileNotFoundError:
    start_idx = chunk_start
    processed_so_far = 0
    seen = set()

    num_vars = len(positions)
    base = len(signs)
    with counter_lock:
       counter.value = processed_so_far

    row_map = {}
    col_map = {}
    for idx, (r, c) in enumerate(positions):
        row_map.setdefault(r, []).append(idx)
        col_map.setdefault(c, []).append(idx)

    for idx in range(start_idx, chunk_end):
        values = []
        x = idx
        for _ in range(num_vars):
            x, r = divmod(x, base)
            values.append(signs[r])
        values.reverse()

        # skip certain already-useless matrices
        # skip = False
        # for i in row_map:
        #     if sum(values[idx] != 0 for idx in row_map[i]) <= 1:
        #        skip = True
        #        break
        # if not skip:
        #     for j in col_map:
        #         if all(values[idx] == 0 for idx in col_map[j]):
        #             skip = True
        #             break
        # if skip:
        #     continue

        result = D.copy().astype(int)
        for (i, j), val in zip(positions, values):
            result[i, j] = val

        canon = canonical_form_general(result)
        seen.add(canon)

        if idx % 1000 == 0:
            with counter_lock:
                counter.value += 1000
            np.save(results_file, np.array(list(seen), dtype=object))
            np.save(checkpoint_file, np.array([idx, len(seen)], dtype=np.int64))

    remainder = (chunk_end - chunk_start) % 1000
    if remainder:
        with counter_lock:
            counter.value += remainder

    # store results
    return_dict[worker_id] = seen

def signings_of_order(m, strict=False):
   if strict:
    vals = [1,-1]
   else:
    vals = [0,1,-1]
   n = len(vals)
   signings = np.zeros([n**m, m, m])
   combs = list(itertools.product(vals, repeat=m))
   for i in range(n**m):
      signings[i,:,:] = np.diag(combs[i])
   if not strict:
      signings = np.delete(signings, 0, axis=0)
   return signings

def strict_signings_of_order(m):
   return signings_of_order(m, strict=True)

def clean_array(arr, tol=1e-8):
    arr_np = np.asarray(arr, dtype=float)  # ensures vectorized ops
    arr_np[np.isclose(arr_np, 0, atol=tol)] = 0

    # return same type as input
    if isinstance(arr, np.ndarray):
        return arr_np
    elif isinstance(arr, tuple):
        return tuple(arr_np.tolist())
    elif isinstance(arr, list):
        return arr_np.tolist()
    else:
        return arr_np

def rot(q,l):
    return np.array([[np.cos(q), -np.sin(q), l],
                     [np.sin(q),  np.cos(q), 0],
                     [0,                 0,  1]])

# def f_for_jac(Q, L):
#     # Q = np.asarray(Q).reshape(-1)
#     return trans(Q, L) @ np.array([0, 0, 1])

def ee_func(x, l):
    x = np.asarray(x)
    # If jacobian passes shape (m, k) with k==1, reduce to (m,)
    if x.ndim > 1:
        # collapse trailing axes -> shape (m, k)
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] == 1:
            # single point: use 1D vector
            qvec = x[:, 0]
            out = (trans(qvec, l) @ np.array([0, 0, 1]))
        else:
            # batch of k points: compute each column
            k = x.shape[1]
            out = np.empty((3, k))
            for col in range(k):
                out[:, col] = trans(x[:, col], l) @ np.array([0, 0, 1])
    else:
        print("wrong way")
        # already 1-D
        out = trans(x, l) @ np.array([0, 0, 1])
    if out.ndim==1:
       out = np.atleast_2d(out).T
    return out

def ee_func(x, l):
    x = np.asarray(x)
    # If jacobian passes shape (m, k) with k==1, reduce to (m,)
    if x.ndim > 1:
        # collapse trailing axes -> shape (m, k)
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] == 1:
            # single point: use 1D vector
            qvec = x[:, 0]
            out = (trans(qvec, l) @ np.array([0, 0, 1]))
        else:
            # batch of k points: compute each column
            k = x.shape[1]
            out = np.empty((3, k))
            for col in range(k):
                out[:, col] = trans(x[:, col], l) @ np.array([0, 0, 1])
    else:
        out = trans(x, l) @ np.array([0, 0, 1])
    if out.ndim==1:
       out = np.atleast_2d(out).T
    return out

def trans(Q, L):
    # print('trans call')
    if Q.ndim == 2:
        Q = Q[:,0]
    trans = np.eye(3)
    # print(Q)
    # print(L)
    for i in range(len(Q)+1):
        # print(Q[i] if 0<=i<len(Q) else 0)
        # print(L[i-1] if 0<=i-1<len(Q) else 0)
        trans = trans @ rot(Q[i] if 0<=i<len(Q) else 0, L[i-1] if 0<=i-1<len(Q) else 0)
    # print(trans)
    return(trans)

def jac(Q, L):
    if len(Q) == 1:
        if not len(L) == 1:
           print("EXPECT ISSUES, SIZE MISMATCH")
    # print(Q,L)
    # x = trans(Q, L) @ np.array([0,0,1])
    def w_ee_func(Q):
       return ee_func(Q,L)
    J = jacobian(w_ee_func, Q)
    return J.df

def volume_centroid(points):
    hull = ConvexHull(points)
    total_volume = 0
    weighted_centroid = np.zeros(3)
    origin = np.mean(points[hull.vertices], axis=0)

    for simplex in hull.simplices:
        # Each simplex is a triangle (face); use it to form a tetrahedron with the origin
        p0 = origin
        p1, p2, p3 = points[simplex]

        tetra = np.array([p0, p1, p2, p3])
        centroid = np.mean(tetra, axis=0)

        # Volume of tetrahedron
        v = np.abs(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6

        weighted_centroid += centroid * v
        total_volume += v

    return weighted_centroid / total_volume

class DegenerateHull():
   def __init__(self) -> None:
      self.volume = 0

def intersection_with_orthant(hull: ConvexHull, orthant):
  orthant = orthant-1
  intersection_with_orthant.signs = [(-1)**int(bit) for bit in format(orthant, f'03b')]
  orthVectors = np.vstack([np.diag(intersection_with_orthant.signs), [0,0,0]])*1000 # make it arbitrarily large to fit any torque capability
  Orthant = ConvexHull(orthVectors)
  # print(Orthant.equations)
  halfspace = (np.vstack([Orthant.equations, hull.equations]))
  try:
    for point in hull.points:
      if in_hull3(Orthant, point):
        centroid  = volume_centroid(hull.points)
        direction = centroid - point
        direction = direction/np.linalg.norm(direction)
        epsilon   = 1e-3
        intersection_with_orthant.interiorPoint = point+epsilon*direction
        break
      else:
        intersection_with_orthant.interiorPoint = np.diag([1e-6]*len(intersection_with_orthant.signs)) @ intersection_with_orthant.signs
    # print('calculating intersection')
    hs = HalfspaceIntersection(halfspace, intersection_with_orthant.interiorPoint)
    # print(hs)
    vertices = hs.intersections
    intersection = ConvexHull(vertices)
    # print('worked')
  except QhullError:
    # print('excepted')
    intersection = DegenerateHull()
  return intersection

def generate_structure_matrix_variants(A):
    m, n = A.shape
    assert n >= 2, "Need at least 2 columns to permute the last two."

    # Generate all combinations of row sign flips (2^m)
    sign_flips = list(itertools.product([1, -1], repeat=m))

    # Permute only the last two columns
    base_cols = list(range(n - 2))  # columns before the last two
    last_two = [n - 2, n - 1]
    col_perms = [base_cols + list(p) for p in itertools.permutations(last_two)]

    variants = []

    for signs in sign_flips:
        flipped = A * np.array(signs)[:, None]  # flip each row
        for perm in col_perms:
            permuted = flipped[:, perm]
            variants.append(permuted)

    return variants

# WE ARE DOING CHAT SHIT HERE

def normalize_row_signs_stream(mat):
    out = mat.copy()
    for i in range(out.shape[0]):
        r = tuple(out[i])
        rn = tuple(-out[i])
        if rn < r:
            out[i] = -out[i]
    return out
   
def canonical_form_stream(mat):
#    mat = np.array(mat)
   mat = normalize_row_signs_stream(mat)
   m, n = mat.shape

   best = None

   def search(partial_perm, remaining):
        nonlocal best
        k = len(partial_perm)

        if k == n:
            permuted = mat[:, partial_perm]
            # global sign rule
            if (permuted < 0).sum() > (permuted > 0).sum():
                permuted = -permuted
            tup = tuple(map(tuple, permuted))
            if best is None or tup < best:
                best = tup
            return

        for c in sorted(remaining):
            newp = partial_perm + (c,)
            partial_mat = mat[:, newp]

            if best is not None:
                bp = tuple(r[:len(newp)] for r in best)
                cp = tuple(tuple(r) for r in partial_mat)
                if cp > bp:
                    continue

            search(newp, remaining - {c})

   search((), set(range(mat.shape[1])))
   return best

# WE ARE DONE DOING CHAT SHIT

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

def fill_matrix(D, positions, values):
    """
    Construct a matrix from a template D, where 'positions' are coordinates
    that get replaced with the entries from 'values'.
    """
    mat = D.copy().astype(int)
    for (i, j), val in zip(positions, values):
        mat[i, j] = val
    return mat

def generator_index_to_values(idx, base, num_vars, value_list):
    vals = []
    for _ in range(num_vars):
        vals.append(value_list[idx % base])
        idx //= base
    return tuple(vals[::-1])

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
    if best is None:
       print("WHAAAAAAAAAAAAAAAAAAAAAAAAAA")
    return best

def lexical_column_order(mat):
  m = mat.shape[0]
  keys = tuple(mat[i,:] for i in range(m-1,-1,-1))
  perm = np.lexsort(keys)
  return perm

def canonical_column_order(mat):
   m, n = mat.shape
   zero_counts = np.sum(mat == 0, axis=0)

   # Find where the first zero appears (if none, use -1 so they sort last)
   first_zero_idx = np.full(n, -1)
   for j in range(n):
       zeros = np.where(mat[:, j] == 0)[0]
       if zeros.size > 0:
           first_zero_idx[j] = zeros[0]

  #  keys = tuple(mat[i,:] for i in range(m-1,-1,-1))
   keys = tuple(mat[i, :] for i in range(m-1, -1, -1))
  #  keys =
   perm = np.lexsort(keys + (first_zero_idx, -zero_counts))
   return perm

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

# def canonical_form_general(mat):
#    m, n = mat.shape
#    best = None
#    for sign_pattern in itertools.product([-1,1], repeat=m):
#       signed_mat = np.diag(sign_pattern) @ mat

#       perm = lexical_column_order(mat)
#       current = signed_mat[:, perm]
#       hashable = tuple(map(tuple, current))
#       if best is None or hashable < best:
#          best = hashable
#    return best

def canonical_form_general(mat):
    m, n = mat.shape
    canon_unique = None

    # try all row sign combinations
    for row_sign_pattern in itertools.product([-1, 1], repeat=m):
        signed = mat * np.array(row_sign_pattern)[:, None]
        for perm in itertools.permutations(range(n)):
            signed_and_permuted = signed[:,perm]
            if np.sum(signed_and_permuted < 0) > np.sum(signed_and_permuted > 0):
               signed_and_permuted = -signed_and_permuted
            hashable = tuple(map(tuple, signed_and_permuted))
            if canon_unique is None or hashable < canon_unique:
                canon_unique = hashable

    return canon_unique

def remove_isomorphic(matrices):
    seen = set()
    unique = []
    i = 1
    for mat in matrices:
        # print(mat)
        print(f'checking for duplicates of matrix {i:3d} of {len(matrices)}, found {len(unique)}', end='\r')
        canon = canonical_form_general(mat)
        # print(canon.__class__)
        if canon not in seen:
            seen.add(canon)
            unique.append(np.array(canon))
        i+=1
    print("")
    return unique

def triangularize_and_orient(canon_tuple):
    mat = np.array(canon_tuple)
    m, n = mat.shape
    zero_counts = np.sum(mat == 0, axis=0)
    first_zero_idx = np.full(mat.shape[1], mat.shape[0])
    for j in range(mat.shape[1]):
        zeros = np.where(mat[:, j] == 0)[0]
        if zeros.size > 0:
            first_zero_idx[j] = zeros[0]
    perm = np.lexsort((-zero_counts, first_zero_idx))
    permuted = mat[:,perm]
    mostPositves = -1
    bestSigned = None
    for row_sign_pattern in itertools.product([-1, 1], repeat=m):
        signed = permuted * np.array(row_sign_pattern)[:, None]
        numPositives = np.sum(signed > 0)
        if numPositives > mostPositves:
           mostPositves = numPositives
           bestSigned = signed
    bestVersion = bestSigned[:,lexical_column_order(bestSigned)]
    zero_counts = np.sum(mat == 0, axis=0)
    first_zero_idx = np.full(mat.shape[1], mat.shape[0])
    for j in range(mat.shape[1]):
        zeros = np.where(mat[:, j] == 0)[0]
        if zeros.size > 0:
            first_zero_idx[j] = zeros[0]
    perm = np.lexsort((-zero_counts, first_zero_idx))
    bestVersion = bestVersion[:,perm]
    # bestVersion = bestSigned
    return bestVersion

def remove_isomorphic_QUTSM(matrices):
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
#   if not [0]*len(points[0]) in boundaryPoints:
#     boundaryPoints.append([0]*len(points[0]))
  boundaryPoints = np.array(boundaryPoints)
  try:
    hull = ConvexHull(boundaryPoints)
  except QhullError:
    hull = ConvexHull(boundaryPoints, qhull_options='QJ')
  return hull, boundaryPoints

def special_minkowski_with_mins(points, minCoeffs):
    
  # this is apparently the amount of digits in the binary value 2**len(points)
  width = len(points)+2
  # vertexes for convex hull
  boundaryPoints = []
  # for each possible combination of vectors (2^num points, any binary combination of single force vectors)
#   print(points)
#   print("---------------------")
  for i in range(2**(len(points))):
    # turn index into binary list
    coeffs = [int(digit) for digit in list(f"{i:0{width}b}")[2:]]
    # initalize zero vector
    partialSum = [0]*len(points[0])
    # add each vector which you should add
    for j in range(len(points)):
      # add full value of each 'on' column
      partialSum = [a+b for a, b, in zip(partialSum, [coeffs[j] * component for component in points[j]])]
      # add min value of each 'off' column
      partialSum = [a+b for a, b, in zip(partialSum, [[~c+2 for c in coeffs][j] * component * minCoeffs[j] for component in points[j]])]
    # add the final vector to the boundary points
    boundaryPoints.append(partialSum)
  # honestly not sure why I insist on having the zero vector in here
#   if not [0]*len(points[0]) in boundaryPoints:
#     boundaryPoints.append([0]*len(points[0]))
  # stick it in a numpy array
  boundaryPoints = np.array(boundaryPoints)
#   print(boundaryPoints)
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

def in_hull3(hull, x):
  tol = -1e-3
  return np.all(hull.equations @ np.append(x,1) <= tol)

def find_axis_extent_lp(vertices, axis_direction):
    m, n = vertices.shape
    d = axis_direction / np.linalg.norm(axis_direction)  # unit vector

    # Variables: [lambda_1,...,lambda_m, t]
    # Objective: maximize t (or minimize t)
    c_max = np.zeros(m + 1)
    c_max[-1] = -1  # maximize t <=> minimize -t

    c_min = np.zeros(m + 1)
    c_min[-1] = 1   # minimize t

    # Equality constraints:
    # sum_i lambda_i * x_i - t * d = 0
    # sum_i lambda_i = 1
    A_eq = np.zeros((n + 1, m + 1))

    # For coordinate constraints
    # sum_i lambda_i * x_i_j - t * d_j = 0  for j in [0,n-1]
    A_eq[0:n, 0:m] = vertices.T
    A_eq[0:n, m] = -d

    # sum_i lambda_i = 1
    A_eq[n, 0:m] = 1
    A_eq[n, m] = 0

    b_eq = np.zeros(n + 1)
    b_eq[n] = 1

    # Bounds for lambda_i: [0,1], for t: unbounded
    bounds = [(0, None)] * m + [(None, None)]

    # Solve for max t
    res_max = linprog(c=c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    # Solve for min t
    res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not (res_max.success and res_min.success):
        raise ValueError("Linear programming failed to find a solution.")

    t_max = res_max.x[-1]
    t_min = res_min.x[-1]

    # Intersection segment on axis:
    p_min = t_min * d
    p_max = t_max * d
    return p_min, p_max

if __name__ == '__main__':
   print(signings_of_order(3)[0])
   print(strict_signings_of_order(3))
  # basis = np.array([[1, 1]]) #1d subspace of R^2
  # print(intersects_positive_orthant(basis)) #True

  # basis = np.array([[1, -1]])
  # print(intersects_positive_orthant(basis)) #False

  # basis = np.array([[2, 3, -1], [1, 0, 2]]) #2d subspace of R^3
  # print(intersects_positive_orthant(basis)) #True

  # basis = np.array([[2, 3, -1], [1, 0, -2]])
  # print(intersects_positive_orthant(basis)) #False

  # basis = np.array([[1, 1, 1, 1]])
  # print(intersects_positive_orthant(basis)) #True

  # basis = np.array([[-1, -1, -1, -1]])
  # print(intersects_negative_orthant(basis)) #True

  # basis = np.array([[1, 1, 1, 1]])
  # print(intersects_negative_orthant(basis)) #False

  # basis = np.array([[0         ],
  #                   [0         ],
  #                   [0.70710678],
  #                   [0.70710678]])
  # print(intersects_positive_orthant(basis.T)) #False

  # M = np.array([
  #     [1, 2, 3],
  #     [-4, 5, -6]
  # ])

  # variants = generate_structure_matrix_variants(M)
  # print(f"{len(variants)} total variants generated")

  # for i, variant in enumerate(variants):
  #     print(f"\nVariant {i + 1}:\n{variant}")

  # vecs = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[0,0,1],[0,0,-1],])
  # hull = ConvexHull(vecs)
  # for i in range(8):
  #   intersection = intersection_with_orthant(hull, i+1)

  # vecs = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[0,0,1],[0,0,-1],])
  # hull = ConvexHull(vecs)
  # fig = plt.figure(f"figure")
  # # if orthant==0:
  # ax = fig.add_subplot(111, projection="3d")
  # # else:
  # #   ax = plt.gca()
  # color = colors[0 % len(colors)]
  # ax.scatter(*vecs.T, alpha=0.78, color=color)
  # ax.scatter(0.25,0.25,0.25)
  # for simplex in hull.simplices:
  #   triangle = vecs[simplex]
  #   ax.add_collection3d(Poly3DCollection([triangle], alpha=0.4, color=color))
  # for vec in vecs:
  #     ax.quiver(0,0,0,vec[0],vec[1],vec[2], color=color)
  # plt.tight_layout()
  # xlim = ax.get_xlim()
  # ylim = ax.get_ylim()
  # zlim = ax.get_zlim()
  # ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
  # ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
  # ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)

  # for i in range(1):
  #   intersection = intersection_with_orthant(hull, i+1)
  #   fig = plt.figure(f"intersection")
  #   if i==0:
  #     ax = fig.add_subplot(111, projection="3d")
  #   else:
  #     ax = plt.gca()
  #   color = colors[i % len(colors)]
  #   ax.scatter(*intersection.points.T, alpha=0.78, color=color)
  #   ax.scatter(0.25,0.25,0.25)
  #   for simplex in intersection.simplices:
  #     triangle = intersection.points[simplex]
  #     ax.add_collection3d(Poly3DCollection([triangle], alpha=0.4, color=color))
  #   for vec in intersection.points:
  #       ax.quiver(0,0,0,vec[0],vec[1],vec[2], color=color)
  #   plt.tight_layout()
  #   xlim = ax.get_xlim()
  #   ylim = ax.get_ylim()
  #   zlim = ax.get_zlim()
  #   ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
  #   ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
  #   ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
  # plt.show()