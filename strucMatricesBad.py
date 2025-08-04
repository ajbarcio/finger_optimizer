import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog

from utils import intersects_positive_orthant, special_minkowski, in_hull, get_existing_axes, get_existing_3d_axes, in_hull2, intersects_negative_orthant
from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult


def r_from_vector(r_vec, D):
    R = D*D
    R = np.array(R, dtype=float)
    indices = np.array(np.nonzero(R))
    for i in range(len(r_vec)):
        R[indices[0,i],indices[1,i]] = r_vec[i]
        # print(r_vec[i], R[indices[0,i],indices[1,i]])
    return R

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
    figures = {}  # Dict to track figures by name
    figures_with_axes = set()

    def __init__(self, R=None, D=None, S=None, F=None, constraints=[], name='Placeholder') -> None:
        if R is not None and D is not None:
            self.R = R
            self.D = D
            self.S = R*D
        if S is not None:
            self.S = S
            self.D = np.sign(S)
            self.R = np.absolute(S)
        if F is None:
            self.F = np.ones(self.S.shape[1])
        else:
            self.F = F
        self.constraints = constraints
        # print("constraints passed to object,", self.constraints)
        # print("checked validity on init")
        self.domain,  self.boundaryGrasps = self.torqueDomainVolume()
        self.validity                     = self.isValid()
        self.numTendons = self.S.shape[1]
        self.name = name

    def __call__(self):
        return self.S

    def reinit(self, R=None, D=None, S=None):
        if R is not None and D is not None:
            self.R = R
            self.D = D
            self.S = R*D
        if S is not None:
            self.S = S
            self.D = np.sign(S)
            self.R = np.absolute(S)
        self.F = self.F
        self.constraints = self.constraints
        self.domain,  self.boundaryGrasps = self.torqueDomainVolume()
        self.validity                     = self.isValid()

    def flatten_r_matrix(self):
        r = self.R*(self.D != 0).astype(int)
        r = r.flatten()
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

        self.numJoints = self.S.shape[0]
        self.rankCondition = np.linalg.matrix_rank(self.S)>=self.numJoints

        if not self.rankCondition:
            if not suppress:
                warnings.warn(f"WARNING: structure matrix failed rank condition (null space condition not checked) \nrank            : {np.linalg.matrix_rank(self.S)} \nnumber of Joints: {self.numJoints}")
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
        r = np.unique(r)
        variation = np.std(r)
        return variation

    def maxExtn(self):
        maxStrength = 0
        for grasp in self.boundaryGrasps:
            strength = np.linalg.norm(grasp)
            if intersects_negative_orthant(grasp) and strength>maxStrength:
                    # print(grasp)
                    maxStrength = strength
        return maxStrength

    def maxGrip(self):
        maxStrength = 0
        # print(self.boundaryGrasps)
        for grasp in self.boundaryGrasps:
            strength = np.linalg.norm(grasp)
            if intersects_positive_orthant(grasp) and strength>maxStrength:
                    # print(grasp)
                    maxStrength = strength
        return maxStrength

    def contains(self, point):
        # print(f"see if contains {point}")
        # print(in_hull(self.domain, point))
        check1 = in_hull(self.domain, point)
        check2 = in_hull2(self.domain, point)
        if check1==check2:
            return check1
        else:
            warnings.warn(f'the two checking methods disagree, linprog says {check1}, geometry says {check2}')
            return check1

    def plotCapability(self, showBool=False, colorOverride=None):
        StrucMatrix.plot_count += 1
        if colorOverride == None:
            color = colors[StrucMatrix.plot_count % len(colors)]
        else:
            color = colorOverride
        # Handle figure reuse by name
        if self.name in StrucMatrix.figures:
            fig, ax = StrucMatrix.figures[self.name]
        else:
            fig = plt.figure(f"Structure Matrix: {self.name}")
            ax = fig.add_subplot(111, projection="3d")
            StrucMatrix.figures[self.name] = (fig, ax)

        self.fig = fig
        self.ax = ax
        figID = id(self.fig)
        numJoints = self.S.shape[0]
        if numJoints == 3:
            singleForceVectors = list(np.transpose(self.S))
            self.ax.scatter(*[0,0,0], color="black")
            self.ax.scatter(*self.boundaryGrasps.T, color=color, alpha=0.78)
            for grasp in singleForceVectors:
                self.ax.quiver(0,0,0,grasp[0],grasp[1],grasp[2],color=color)
                # print(singleForceVectors)
            for simplex in self.domain.simplices:
                triangle = self.boundaryGrasps[simplex]
                self.ax.add_collection3d(Poly3DCollection([triangle], color=color, alpha=0.4))
            # if StrucMatrix.plot_count == 1: plt.tight_layout()
            # Axis limits
            # Draw "axes" through origin
            if figID not in StrucMatrix.figures_with_axes:
                plt.tight_layout()
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                zlim = self.ax.get_zlim()
                self.ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
                self.ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
                self.ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
                StrucMatrix.figures_with_axes.add(figID)
            if showBool:
                plt.show()
        else:
            warnings.warn("Cannot plot anything other than 3d grasps at this time")

    def plotGrasp(self, grasp, showBool=False):
        if self.contains(grasp):
            color='xkcd:green'
        else:
            color='xkcd:red'
        # Handle figure reuse by name
        if self.name in StrucMatrix.figures:
            fig, ax = StrucMatrix.figures[self.name]
        else:
            fig = plt.figure(f"Structure Matrix: {self.name}")
            ax = fig.add_subplot(111, projection="3d")
            StrucMatrix.figures[self.name] = (fig, ax)

        self.fig = fig
        self.ax = ax
        self.ax.scatter(*grasp, color=color, alpha=1)
        if showBool:
            plt.show()

    def jointCapability(self, j):
        ''' at joint j '''
        c = self.S[j,:]
        # find maximum extension (negative) torque
        maxExt = linprog(c, A_ub=np.identity(self.numTendons), b_ub=self.F)
        # find maximum flexion  (positive) torque
        maxFlx = linprog(-c, A_ub=np.identity(self.numTendons), b_ub=self.F)
        return maxExt.fun, -maxFlx.fun

    def independentJointCapabilities(self):
        capabilities = np.array([[0,0],
                                 [0,0],
                                 [0,0]], dtype=float)
        for j in range(self.numJoints):
            c = self.S[j,:]
            # A_eq = self.S[j+1:,:]
            A_eq = np.vstack((self.S[:j, :], self.S[j+1:, :]))
            b_eq = [0,0]
            # print(A_eq)
            # print(b_eq)
            # find maximum extension (negative) torque
            maxExt = linprog(c, A_ub=np.identity(self.numTendons), b_ub=self.F, A_eq=A_eq, b_eq=b_eq)
            # print(maxExt.success)
            maxFlx = linprog(-c, A_ub=np.identity(self.numTendons), b_ub=self.F, A_eq=A_eq, b_eq=b_eq)
            # print(maxExt.fun, -maxFlx.fun)
            capabilities[j] = [maxExt.fun, -maxFlx.fun]
            # print(capabilities)
        # print(capabilities)
        return capabilities

    def independentJointCapability(self, j):
        c = self.S[j,:]
        # A_eq = self.S[j+1:,:]
        A_eq = np.vstack((self.S[:j, :], self.S[j+1:, :]))
        b_eq = [0,0]
        # find maximum extension (negative) torque
        maxExt = linprog(c, A_ub=np.identity(self.numTendons), b_ub=self.F, A_eq=A_eq, b_eq=b_eq)
        # print(maxExt.success)
        maxFlx = linprog(-c, A_ub=np.identity(self.numTendons), b_ub=self.F, A_eq=A_eq, b_eq=b_eq)
        # print(maxExt.fun, -maxFlx.fun)
        capabilities = [maxExt.fun, -maxFlx.fun]
        return capabilities

    def optimizer(self):
        def maxGrip(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            return -self.maxGrip()
        # def balanceConstraint(rvec):
        #     R = r_from_vector(rvec, self.D)
        #     self.reinit(R=R, D=self.D)
        #     capabilities = self.independentJointCapabilities()
        #     error = 0
        #     errors = []
        #     for pair in capabilities:
        #         ratio = abs(pair[-1]/pair[0])
        #         print(pair, ratio)
        #         errors.append(ratio-2)
        #     print('---')
        #     error = np.linalg.norm(errors)
        #     print(error)    
        #     print("---")
        #     return error
        iteration = 0
        def overallbalance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = [self.maxExtn(), self.maxGrip()]
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 2
            except ZeroDivisionError:
                return -2
        def j1balance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapability(0)
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 2
            except ZeroDivisionError:
                return -2
        def j2balance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapability(1)
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 2
            except ZeroDivisionError:
                return -2
        def j3balance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapability(2)
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 2
            except ZeroDivisionError:
                return -2
        def validity(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            return int(not self.isValid())
        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            print(intermediate_result.x)
            best_x = intermediate_result.x.copy()
            self.plotCapability()
        rvecInit = self.flatten_r_matrix()
        print(rvecInit)
        constraints = [{'type': 'eq',
                       'fun':   j1balance},
                       {'type': 'eq',
                       'fun':   j3balance},
                       {'type': 'eq',
                       'fun':   j2balance},
                       {'type': 'eq',
                        'fun':  validity}
                      ]
        j1BalanceObject = NonlinearConstraint(j1balance, -.05, .05)
        j2BalanceObject = NonlinearConstraint(j2balance, -.05, .05)
        j3BalanceObject = NonlinearConstraint(j3balance, -.05, .05)
        normBalanceObject = NonlinearConstraint(overallbalance, -.05, .05)
        validityObject    = NonlinearConstraint(validity, -0.5, 0.5)
        constraintsObjects = [j1BalanceObject,j2BalanceObject,j3BalanceObject]
        constraintsObjects = [normBalanceObject, validityObject]
        try:
            E = minimize(maxGrip, rvecInit, method='trust-constr', constraints=constraintsObjects, callback=plotCallback, bounds=[(0,1)]*len(rvecInit))
        except KeyboardInterrupt:
            return best_x, -maxGrip(best_x)
        print(E.success, E.message)
        return E.x, -E.fun


class InsufficientRanges(Exception):
    def __init__(self, message='ranges must match number of variable pulleys'):
        super().__init__(message)

class VariableStrucMatrix():
    def __init__(self, R=None, D=None, S=None, ranges=[], constraints=[], name='Placeholder'):
        if not np.sum(np.isnan(D)) == len(ranges):
            raise InsufficientRanges()
        self.variablePulleys = [list(x) for x in np.where(np.isnan(D if D is not None else S))]
        
        super().__init__(R, D, S, constraints, name)

    def torquDomainVolume(self):
        pass



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
R = np.array([[r,r,r*1.5,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[0,1,1,-1],
              [1,0,1,-1],
              [1,1,0,-1]])
quasiHollow = StrucMatrix(R,D,name='Hollow')

# Diagonal Design
r = 1
R = np.array([[r,r,r,r/3],
              [r,r,r,r/3],
              [r,r,r,r/3]])
D = np.array([[1,0,0,-1],
              [0,1,0,-1],
              [0,0,1,-1]])
diagonal = StrucMatrix(R,D,name='Hollow')

# test
r = 1
R = np.array([[r,r,r,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[1,1,1,-1],
              [1,1,1,-1],
              [1,1,1,-1]])
test = StrucMatrix(R,D, name='Test')