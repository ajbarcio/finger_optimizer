import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

torque_vectors = np.array([
    [0.5, -1.2, 3.0],
    [2.3, 0.0, -0.7],
    [-1.1, 4.5, 1.0],
    [3.0, 2.0, 0.5],
    [0.0, -3.0, 4.0]
])

# Find index of max/min along each axis
idx_max_x = np.argmax(torque_vectors[:, 0])
idx_min_x = np.argmin(torque_vectors[:, 0])

idx_max_y = np.argmax(torque_vectors[:, 1])
idx_min_y = np.argmin(torque_vectors[:, 1])

idx_max_z = np.argmax(torque_vectors[:, 2])
idx_min_z = np.argmin(torque_vectors[:, 2])

# Get the points at those indices
point_max_x = torque_vectors[idx_max_x]
point_min_x = torque_vectors[idx_min_x]

point_max_y = torque_vectors[idx_max_y]
point_min_y = torque_vectors[idx_min_y]

point_max_z = torque_vectors[idx_max_z]
point_min_z = torque_vectors[idx_min_z]

print("Point with max x:", point_max_x)
print("Point with min x:", point_min_x)

print("Point with max y:", point_max_y)
print("Point with min y:", point_min_y)

print("Point with max z:", point_max_z)
print("Point with min z:", point_min_z)