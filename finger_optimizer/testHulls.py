import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

from .utils import *
from .strucMatrices import *


D = np.array([[-1, -1, -1,  1],
              [ 0, -1,  1, -1],
              [ 0,  0,  1, -1]])
R = np.ones_like(D)
S = obj(R=R, D=D, name='test')
for orthant in range(8):
    # print(orthant.equations)
    print(intersection_with_orthant(S.torqueDomainVolume()[0], orthant+1).volume)
    print(intersection_with_orthant.interiorPoint)
    print(intersection_with_orthant.signs)
S.plotCapability(showBool=True)

