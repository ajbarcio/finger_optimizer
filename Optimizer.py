import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import intersects_positive_orthant, special_minkowski, in_hull
import warnings

def test3dofn1():
    R = np.array([[1,0.5,1,0.5],
                  [0, 1 ,0.5,0.5],
                  [0, 0 ,1 ,1]])
    D = np.array([[1,1,-1,-1],
                  [0,1,-1,-1],
                  [0,0,-1,1]])
    S = StrucMatrix(R, D)
    print(S())
    print(S.validity)
    print(S.nullSpace)
    S.plotCapability()
    S.plotGrasp([0.5,0,0])
    S.plotGrasp([2,0,0])

    print(S.contains([0.5,0,0]))
    print(S.contains([2,0,0]))

    plt.show()

def optimize_standard(numJoints, numTendons):
    pass

class StrucMatrix():
    def __init__(self, R, D) -> None:
        self.S = R*D
        self.validity                     = self.isValid()
        self.domain,  self.boundaryGrasps = self.torqueDomainVolume()

        self.fig = plt.figure("Structure Matrix")
        self.ax = self.fig.add_subplot(111, projection="3d")

    def __call__(self):
        return self.S

    def isValid(self):
        numJoints = self.S.shape[0]
        rankCondition = np.linalg.matrix_rank(self.S)>=numJoints
        if not rankCondition:
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
        nullSpaceCondition = intersects_positive_orthant(nullSpace.T)
        if not nullSpaceCondition:
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

class Finger():
    def __init__(self, lengths=None, numJoints=None, numTendons=None, S=None, grasps=None) -> None:
        if S is None:
            R = np.array([[1,0.5,1,0.5],
                        [0, 1 ,0.5,0.5],
                        [0, 0 ,1 ,1]])
            D = np.array([[1,1,-1,-1],
                        [0,1,-1,-1],
                        [0,0,-1,1]])
            self.S = StrucMatrix(R,D)
        else:
            self.S = S

        if grasps is not None:
            self.grasps = grasps
        else:
            self.grasps = []

def main():
    test3dofn1()

if __name__ == "__main__":
    main()