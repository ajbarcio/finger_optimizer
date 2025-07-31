import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import intersects_positive_orthant, special_minkowski, in_hull, get_existing_axes, get_existing_3d_axes

colors = [
    'xkcd:neon green',     # #0cff0c – retina-searing green
    'xkcd:electric blue',  # #0652ff – deep glowing blue
    'xkcd:hot pink',       # #ff028d – vibrant magenta-pink
    'xkcd:bright yellow',  # #fffd01 – classic highlighter yellow
    'xkcd:neon purple',    # #bc13fe – super-saturated violet
    'xkcd:bright orange',  # #ff5b00 – bold warm orange
    'xkcd:cyan',           # #00ffff – icy electric blue-green
    'xkcd:magenta',        # #c20078 – deeper than hot pink
    'xkcd:bright red',     # #ff000d – warning-light red
    'xkcd:bright turquoise' # #0ffef9 – glowing aqua
]
class Constraint():
    def __init__(self, function, args):
        self.function = function
        self.args = args
    def __call__(self, instance, *args, **kwds):
        return self.function(instance, *self.args)

class StrucMatrix():

    plot_count = 0

    def __init__(self, R=None, D=None, S=None, constraints=[], name='Placeholder') -> None:
        if R is not None and D is not None:
            self.R = R
            self.D = D
            self.S = R*D
        if S is not None:
            self.S = S
            self.D = np.sign(S)
            self.R = np.absolute(S)
        self.constraints = constraints
        # print("constraints passed to object,", self.constraints)
        # print("checked validity on init")
        self.domain,  self.boundaryGrasps = self.torqueDomainVolume()
        self.validity                     = self.isValid()
        self.name = name

    def __call__(self):
        return self.S

    def flatten_r_matrix(self):
        r = self.R.flatten()
        r = np.array([x for x in r if x != 0])
        return r

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        self.validity = self.isValid()

    def isValid(self, suppress=True):
        # print('going to check constraint')
        for constraint in self.constraints:
            if not constraint(self):
                return False
            # print('checked constraint', constraint)

        numJoints = self.S.shape[0]
        self.rankCondition = np.linalg.matrix_rank(self.S)>=numJoints

        if not self.rankCondition:
            if not suppress:
                warnings.warn(f"WARNING: structure matrix failed rank condition (null space condition not checked) \nrank            : {np.linalg.matrix_rank(self.S)} \nnumber of Joints: {numJoints}")
            return False
        nullSpace = sp.linalg.null_space(self.S)
        # Condition the nullSpace output well for future checking
        for i in range(nullSpace.shape[0]):
            for j in range(nullSpace.shape[1]):
                if np.isclose(nullSpace[i,j], 0):
                    nullSpace[i,j] = 0
        self.biasForceSpace = nullSpace
        # Check to make sure that there exists an all-positive vector in the null space
        if np.shape(self.biasForceSpace)[-1]>1:
            # print("intersecting positive orthant")
            self.nullSpaceCondition = intersects_positive_orthant(nullSpace.T)
            for row in self.biasForceSpace:
                # print(row)
                if all([x==0 for x in row]):
                    self.nullSpaceCondition =  False
        else:
            self.nullSpaceCondition = all([i > 0 for i in self.biasForceSpace])
        if not self.nullSpaceCondition:
            if not suppress:
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

    def pulleyVariation(self):
        # print(self.flatten_r_matrix())
        r = self.flatten_r_matrix()
        r = r/np.max(r)
        variation = np.std(r)
        return variation

    def maxGrip(self):
        maxStrength = 0
        # print(self.boundaryGrasps)
        for grasp in self.boundaryGrasps:
            if all([x>0 for x in grasp]) and np.linalg.norm(grasp)>maxStrength:
                # print(grasp)
                maxStrength = np.linalg.norm(grasp)
        return maxStrength

    def contains(self, point):
        # print(f"see if contains {point}")
        # print(in_hull(self.domain, point))
        return in_hull(self.domain, point)

    def plotCapability(self, showBool=False):
        StrucMatrix.plot_count += 1
        self.fig = plt.figure(f"Structure Matrix {self.name}")
        existing_ax = get_existing_3d_axes()
        if existing_ax is None:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = plt.gca()
        numJoints = self.S.shape[0]
        if numJoints == 3:
            singleForceVectors = list(np.transpose(self.S))
            self.ax.scatter(*[0,0,0], color="red")
            self.ax.scatter(*self.boundaryGrasps.T, color=colors[StrucMatrix.plot_count % len(colors)], alpha=0.78)
            for grasp in singleForceVectors:
                self.ax.quiver(0,0,0,grasp[0],grasp[1],grasp[2],color=colors[StrucMatrix.plot_count % len(colors)])
                # print(singleForceVectors)
            for simplex in self.domain.simplices:
                triangle = self.boundaryGrasps[simplex]
                self.ax.add_collection3d(Poly3DCollection([triangle], color=colors[StrucMatrix.plot_count % len(colors)], alpha=0.4))
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

# test
r = 1
R = np.array([[r,r,r,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[0,1,1,-1],
              [1,0,1,-1],
              [1,1,0,-1]])
test = StrucMatrix(R,D, name='Test')