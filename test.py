import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from utils import transform_ConvexHull, hull_containment_vector

class dumb:
    pass

dumb.shit = 'piss'

dumbass = dumb()
dumbbss = dumb()

dumbbss.shit = 'waow'

print(dumbass.shit)
print(dumbbss.shit)

hullA = ConvexHull(((0,1), (1,0), (0,-1), (-1,0)))
hullB = transform_ConvexHull(hullA, np.diag([0.5,0.5,1]))

print(hullA, hullB)
print(hullA.points)
print(hullB.points)

print(hull_containment_vector(hullA, hullB))

points = np.array([
    [-0.1496, -0.1496, -0.1496],
    [ 0.1496,  0.1496,  0.    ],
    [ 0.    ,  0.    , -0.1496],
    [ 0.1496,  0.    ,  0.1496],
    [ 0.    , -0.1496,  0.    ],
    [ 0.2992,  0.1496,  0.1496],
    [ 0.1496,  0.    ,  0.    ],
    [ 0.    ,  0.1496,  0.1496],
    [-0.1496,  0.    ,  0.    ],
    [ 0.1496,  0.2992,  0.1496],
    [ 0.    ,  0.1496,  0.    ],
    [ 0.1496,  0.1496,  0.2992],
    [ 0.    ,  0.    ,  0.1496],
    [ 0.2992,  0.2992,  0.2992],
    [ 0.1496,  0.1496,  0.1496],
    [ 0.    ,  0.    ,  0.    ]
])

hull = ConvexHull(points)

# Set up LP to minimize x[0] with y = 0, z = 0
A = hull.equations[:, :-1]
b = -hull.equations[:, -1]
c = np.array([1, 0, 0])
A_eq = np.array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
b_eq = np.array([0, 0, 0])

res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, method='highs')

print("Success:", res.success)
print("x minimizing point along x-axis:", res.x)
print("Minimum x along axis:", res.x[0])