import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import intersects_positive_orthant, special_minkowski, in_hull

class StrucMatrix():
    def __init__(self, R, D, name='Placeholder') -> None:
        self.S = R*D
        self.validity                     = self.isValid()
        self.domain,  self.boundaryGrasps = self.torqueDomainVolume()
        self.name = name

    def __call__(self):
        return self.S

    def isValid(self, suppress=True):
        numJoints = self.S.shape[0]
        rankCondition = np.linalg.matrix_rank(self.S)>=numJoints
        if not rankCondition and not suppress:
            warnings.warn(f"WARNING: structure matrix failed rank condition (null space condition not checked) \nrank            : {np.linalg.matrix_rank(self.S)} \nnumber of Joints: {numJoints}")
            return False
        nullSpace = sp.linalg.null_space(self.S)
        # Condition the nullSpace output well for future checking
        for i in range(nullSpace.shape[0]):
            for j in range(nullSpace.shape[1]):
                if np.isclose(nullSpace[i,j], 0):
                    nullSpace[i,j] = 0
        self.nullSpace = nullSpace
        # Check to make sure that there exists an all-positive vector in the null space
        if np.shape(self.nullSpace)[-1]>1:
            nullSpaceCondition = intersects_positive_orthant(nullSpace.T)
        else:
            nullSpaceCondition = all([i > 0 for i in self.nullSpace])
        if not nullSpaceCondition and not suppress:
            warnings.warn("WARNING: structure matrix failed null space condition (rank condition passed)")
            return False
        return True
        # return (np.linalg.matrix_rank(S)>=numJoints) and (all([x>0 for x in sp.linalg.null_space(S)]))

    def torqueDomainVolume(self):
        numJoints = self.S.shape[0]
        numTendons = self.S.shape[1]
        # S is a structure matrix
        singleForceVectors = list(np.transpose(self.S))
        domain, boundaryGrasps = special_minkowski(singleForceVectors)
        return domain, boundaryGrasps

    def contains(self, point):
        return in_hull(self.domain, point)

    def plotCapability(self, showBool=False):
        self.fig = plt.figure(f"Structure Matrix {self.name}")
        self.ax = self.fig.add_subplot(111, projection="3d")
        numJoints = self.S.shape[0]
        if numJoints == 3:
            singleForceVectors = list(np.transpose(self.S))
            self.ax.scatter(*[0,0,0], color="red")
            self.ax.scatter(*self.boundaryGrasps.T, color='xkcd:blue', alpha=0.78)
            for grasp in singleForceVectors:
                self.ax.quiver(0,0,0,grasp[0],grasp[1],grasp[2],color='xkcd:blue')
            for simplex in self.domain.simplices:
                triangle = self.boundaryGrasps[simplex]
                self.ax.add_collection3d(Poly3DCollection([triangle], color='xkcd:blue', alpha=0.4))
            plt.tight_layout()
            # Axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim()
            # Draw "axes" through origin
            self.ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
            self.ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
            self.ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
            if showBool:
                plt.show()
        else:
            warnings.warn("Cannot plot anything other than 3d grasps at this time")

    def plotGrasp(self, grasp, showBool=False):
        if self.contains(grasp):
            color='xkcd:green'
        else:
            color='xkcd:red'
        self.ax.scatter(*grasp, color=color, alpha=1)

# Centered type 1
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]]) 
R = np.array([[1/3,1/3,1/3,3/3],
              [0,1/2,1/2,2/2],
              [0,0,1,1]]) 
centeredType1 = StrucMatrix(R,D,name='centered1')

# Centered type 2
D = np.array([[1,-1,1,-1],
              [0,-1,1,-1],
              [0,0,1,-1]])
R = np.array([[0.5,0.5,0.5,0.5],
              [0,0.5,1,0.5],
              [0,0,1,1]])      
centeredType2 = StrucMatrix(R,D,name='centered2')

# Centered type 3
R = np.array([[1,0.5,1,0.5],
              [0, 1 ,0.5,0.5],
              [0, 0 ,1 ,1]])
D = np.array([[1,1,-1,-1],
              [0,1,-1,-1],
              [0,0,-1,1]])
centeredType3 = StrucMatrix(R,D,name='centered3')

# AMBROSE MATRIX
R = np.array([[.2188,.2188,.2188,.2188],
              [0,.1719,.1719,.1719],
              [0,0,.1484,.1484]])
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
# I'm not saying Dr. Ambrose's design is naiive, this is just a naiive implementation of that design in my system
naiiveAmbrose = StrucMatrix(R,D,name='Ambrose')

# Hollow Design
r = .1496
R = np.array([[r,r,r,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[0,1,1,-1],
              [1,0,1,-1],
              [1,1,0,-1]])
quasiHollow = StrucMatrix(R,D,name='Hollow')