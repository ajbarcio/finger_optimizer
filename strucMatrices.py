import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
from scipy.linalg import null_space

from utils import intersects_positive_orthant, special_minkowski, in_hull, get_existing_axes, get_existing_3d_axes, in_hull2, intersects_negative_orthant, intersection_with_orthant
from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult, dual_annealing, differential_evolution
from types import SimpleNamespace

cmap = plt.cm.cool

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
        try:
            if self.nullSpaceCondition or self.rankCondition:
                self.halfValidity = True
            else:
                self.halfValidity = False
        except AttributeError:
            if self.rankCondition:
                self.halfValidity = True
            else:
                self.halfValidity = False
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

        self.numJoints = self.S.shape[0]
        self.rankCondition = np.linalg.matrix_rank(self.S)>=self.numJoints
        # print(self.rankCondition)
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
        self.singleForceVectors = list(np.transpose(self.S @ np.diag(self.F)))
        domain, boundaryGrasps = special_minkowski(self.singleForceVectors)
        return domain, boundaryGrasps

    def pulleyVariation(self):
        # print(self.flatten_r_matrix())
        r = self.flatten_r_matrix()
        r = r/np.max(r)
        r = np.unique(r)
        variation = np.std(r)
        return variation

    def biasCondition(self):
        if np.min(self.biasForceSpace)<=0:
            return 1000000000
        else:
            return np.max(abs(self.biasForceSpace))/np.min(abs(self.biasForceSpace))

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
            # if intersects_positive_orthant(grasp) and strength>maxStrength:
            if strength>maxStrength:
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

    def contains_by(self, rvec, point):
        R = r_from_vector(rvec, self.D)
        self.reinit(R=R, D=self.D)
        return -np.max(self.domain.equations @ np.append(point, 1))

    def plotCapability(self, showBool=False, colorOverride=None, transOverride=None, obj=None):

        if obj is None:
            obj = type(self)
            print("object:", obj)
        else:
            print("object given as:", obj)

        color = colorOverride if colorOverride is not None else colors[obj.plot_count % len(colors)]

        # Pick transparency
        alpha = transOverride if transOverride is not None else 0.4
        # Handle figure reuse by name
        if self.name in obj.figures:
            fig, ax = obj.figures[self.name]
        else:
            fig = plt.figure(f"Structure Matrix: {self.name}")
            ax = fig.add_subplot(111, projection="3d")
            obj.figures[self.name] = (fig, ax)

        self.fig = fig
        self.ax = ax
        figID = id(self.fig)
        numJoints = self.S.shape[0]
        if numJoints == 3:
            singleForceVectors = list(np.transpose(self.S))
            # self.ax.scatter(*[0,0,0], color="black")
            self.ax.scatter(*self.boundaryGrasps.T, color=color, alpha=alpha)
            # for grasp in singleForceVectors:
            #     self.ax.quiver(0,0,0,grasp[0],grasp[1],grasp[2],color="black")
                # print(grasp)
            for simplex in self.domain.simplices:
                triangle = self.boundaryGrasps[simplex]
                self.ax.add_collection3d(Poly3DCollection([triangle], color=color, alpha=alpha))
            # if StrucMatrix.plot_count == 1: plt.tight_layout()
            # Axis limits
            # Draw "axes" through origin
            if figID not in obj.figures_with_axes:
                plt.tight_layout()
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                zlim = self.ax.get_zlim()
                self.ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
                self.ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
                self.ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
                ax.set_xlabel('τ₁')
                ax.set_ylabel('τ₂')
                ax.set_zlabel('τ₃')
                # ax.set_title('Torque Components τ₁, τ₂, τ₃')
                ax.set_title(f'{self.name} Capability Polytope')
                # ax.view_init(elev=30, azim=45)
                ax.grid(True)
                plt.tight_layout()

                obj.figures_with_axes.add(figID)
            if showBool:
                plt.show()
        else:
            warnings.warn("Cannot plot anything other than 3d grasps at this time")
        obj.plot_count += 1

    def plotGrasp(self, grasp, showBool=False, obj=None):
        
        if obj is None:
            obj = type(self)
            print("object:", obj)
        else:
            print("object given as:", obj)
        
        if self.contains(grasp):
            color='xkcd:green'
        else:
            color='xkcd:red'
        # Handle figure reuse by name
        if self.name in obj.figures:
            fig, ax = obj.figures[self.name]
        else:
            fig = plt.figure(f"Structure Matrix: {self.name}")
            ax = fig.add_subplot(111, projection="3d")
            obj.figures[self.name] = (fig, ax)

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
            # print(self.S)
            return -self.maxGrip()
        def maxJoint(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapabilities()
            flexCapability = np.linalg.norm(capability[:,-1])
            return -flexCapability

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
                return capability[-1]/abs(capability[0]) - 1.5
            except ZeroDivisionError:
                return -2
        def j2balance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapability(1)
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 1.5
            except ZeroDivisionError:
                return -2
        def j3balance(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapability(2)
            # print(capability)
            try:
                return capability[-1]/abs(capability[0]) - 1.5
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
            print(intermediate_result)
            # best_x = intermediate_result
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
        # constraintsObjects = [normBalanceObject, validityObject]
        try:
            # success = False
            # while not success:
                E = minimize(maxGrip, rvecInit, method='trust-constr', constraints=constraintsObjects, callback=plotCallback, bounds=[(0,1)]*len(rvecInit))
                # if E.success:
                #     break
                # else:
                #     rvecInit = E.x
        except KeyboardInterrupt:
            return best_x, -maxGrip(best_x)
        print(E.success, E.message)
        return E.x, -E.fun

    def optimizer2(self):
        method='trust-constr'

        def maxGrip(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            # print(self.S)
            return -self.maxGrip()
        def maxJoint(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            capability = self.independentJointCapabilities()
            flexCapability = np.linalg.norm(capability[:,-1])
            return -flexCapability

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            self.plotCapability()

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = maxJoint

        conditionConstraint = NonlinearConstraint(condition, 10, 10)
        constraintsObjects = [conditionConstraint]
        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraintsObjects, callback=plotCallback, bounds=[(0,1)]*len(rvecInit))
        except KeyboardInterrupt:
            return best_x, objective(best_x)
        print(E.success, E.message)
        return E.x, -E.fun

    def optimizer3(self):
        method='trust-constr'

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def radiusCondition(rvec):
            return np.max(abs(rvec))/np.min(abs(rvec))

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            self.plotCapability()

        def validity(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            return int(not self.isValid())

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = condition

        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraintsObjects = []
        # For each grasp constraint passed to the StrucMatrix instance,
        for constraint in self.constraints:
            # If the type is inequality, we just need a positive value (indicates inclusion)
            if constraint.type == 'ineq':
                constraintsObjects.append(NonlinearConstraint(constraint, 0, np.inf))
            # If the type is equality, we want a value near zero (indicates that the point is on the boundary)
            elif constraint.type == 'eq':
                constraintsObjects.append(NonlinearConstraint(constraint, -0.001, 0.001))
                print(f"enforcing equality on point: {constraint.args}")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        if not any([constraint.type=='eq' for constraint in self.constraints]):
            constraintsObjects.append(NonlinearConstraint(slackness, -0.001, 0.001))
            print("enforcing slackness")
        # We cannot allow degenerate forms (no radius value may approach 0)
        constraintsObjects.append(NonlinearConstraint(radiusCondition, 1, 8))
        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraintsObjects, callback=plotCallback, bounds=[(0,1)]*len(rvecInit),
                         options={'gtol': 1e-4, 'xtol': 1e-4})
        except KeyboardInterrupt:
            return best_x, objective(best_x)
        print(E.success, E.message)
        return E.x, E.fun

    def optimizer4(self):
        method='trust-constr'

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            # self.plotCapability()

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = condition

        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraintsObjects = []
        # For each grasp constraint passed to the StrucMatrix instance,
        for constraint in self.constraints:
            # If the type is inequality, we just need a positive value (indicates inclusion)
            if constraint.type == 'ineq':
                constraintsObjects.append(NonlinearConstraint(constraint, 0, np.inf))
            # If the type is equality, we want a value near zero (indicates that the point is on the boundary)
            elif constraint.type == 'eq':
                constraintsObjects.append(NonlinearConstraint(constraint, -0.001, 0.001))
                print(f"enforcing equality on point: {constraint.args}")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        if not any([constraint.type=='eq' for constraint in self.constraints]):
            constraintsObjects.append(NonlinearConstraint(slackness, -0.001, 0.001))
            print("enforcing slackness")
        # This is a dimension-aware optimizer, so bounds are set for radii in inches
        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraintsObjects, callback=plotCallback, bounds=[(0,1)]*len(rvecInit),
                         options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 1000})
        except KeyboardInterrupt:
            self.optSuccess = str(str(E.success)+str(E.message))
            return best_x, objective(best_x)
        # print(E.success, E.message)
        self.optSuccess = str(str(E.success)+str(E.message))
        return E.x, E.fun

    def optimizer5(self, bounds):
        method='trust-constr'

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            # self.plotCapability()

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = condition

        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraintsObjects = []
        # For each grasp constraint passed to the StrucMatrix instance,
        print(f" we are receiving {len(self.constraints)} constraints in the optimizer")
        for constraint in self.constraints:
            # If the type is inequality, we just need a positive value (indicates inclusion)
            if constraint.type == 'ineq':
                constraintsObjects.append(NonlinearConstraint(constraint, 0, np.inf))
            # If the type is equality, we want a value near zero (indicates that the point is on the boundary)
            elif constraint.type == 'eq':
                constraintsObjects.append(NonlinearConstraint(constraint, -0.001, 0.001))
                print(f"enforcing equality on point: {constraint.args}")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        if not any([constraint.type=='eq' for constraint in self.constraints]):
            constraintsObjects.append(NonlinearConstraint(slackness, -0.001, 0.001))
            print("enforcing slackness")
        # This is a dimension-aware optimizer, so bounds are set for radii in inches
        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraintsObjects, callback=plotCallback, bounds=[bounds]*len(rvecInit),
                         options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 1000, 'initial_constr_penalty': 2})
            self.optSuccess = str(str(E.success)+str(E.message))
            return E.x, E.fun
        except KeyboardInterrupt:
            self.optSuccess = str(str(E.success)+str(E.message))
            return best_x, objective(best_x)
        # print(E.success, E.message)

    def optimizer6(self, bounds):
        method='trust-constr'

        def quad1Vol(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            return intersection_with_orthant(self.torqueDomainVolume()[0], 1).volume

        def maxGrip(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            # print(self.S)
            return -self.maxGrip()

        def radiusCondition(rvec):
            return np.max(abs(rvec))/np.min(abs(rvec))

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            # self.plotCapability()

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = maxGrip

        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraintsObjects = []
        # For each grasp constraint passed to the StrucMatrix instance,
        print(f" we are receiving {len(self.constraints)} constraints in the optimizer")
        for constraint in self.constraints:
            # If the type is inequality, we just need a positive value (indicates inclusion)
            if constraint.type == 'ineq':
                constraintsObjects.append(NonlinearConstraint(constraint, 0, np.inf))
            # If the type is equality, we want a value near zero (indicates that the point is on the boundary)
            elif constraint.type == 'eq':
                constraintsObjects.append(NonlinearConstraint(constraint, -0.001, 0.001))
                print(f"enforcing equality on point: {constraint.args}")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        # if not any([constraint.type=='eq' for constraint in self.constraints]):
        #     constraintsObjects.append(NonlinearConstraint(slackness, -0.001, 0.001))
        #     print("enforcing slackness")
        # This is a dimension-aware optimizer, so bounds are set for radii in inches
        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraintsObjects, callback=plotCallback, bounds=[bounds]*len(rvecInit),
                         options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 1000, 'initial_constr_penalty': 2})
            self.optSuccess = str(str(E.success)+str(E.message))
            return E.x, E.fun
        except KeyboardInterrupt:
            self.optSuccess = str(str(E.success)+str(E.message))
            return best_x, objective(best_x)
        # print(E.success, E.message)

    def globalOptimizer(self):

        def condition(rvec):
            R = r_from_vector(rvec, self.D)
            self.reinit(R=R, D=self.D)
            condition = self.biasCondition()
            return condition

        def reportOut(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            print(intermediate_result.x, intermediate_result.fun)
            print(null_space(self.S))
            best_x = intermediate_result.x.copy()
            # self.plotCapability()

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        bounds=[(0.1,0.5)]*len(rvecInit)
        # print('starting with:', rvecInit)
        objective = condition
        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraintsObjects = []
        # For each grasp constraint passed to the StrucMatrix instance,
        for constraint in self.constraints:
            # If the type is inequality, we just need a positive value (indicates inclusion)
            if constraint.type == 'ineq':
                constraintsObjects.append(NonlinearConstraint(constraint, 0, np.inf))
            # If the type is equality, we want a value near zero (indicates that the point is on the boundary)
            elif constraint.type == 'eq':
                constraintsObjects.append(NonlinearConstraint(constraint, -0.001, 0.001))
                print(f"enforcing equality on point: {constraint.args}")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        if not any([constraint.type=='eq' for constraint in self.constraints]):
            constraintsObjects.append(NonlinearConstraint(slackness, -0.001, 0.001))
            print("enforcing slackness")
        # This is a dimension-aware optimizer, so bounds are set for radii in inches
        try:
            E = differential_evolution(objective,
                                       bounds,
                                       maxiter=1000,
                                       constraints=constraintsObjects,
                                       callback=reportOut)
            self.optSuccess = str(str(E.success)+str(E.message))
            return E.x, E.fun
        except KeyboardInterrupt:
            self.optSuccess = str(str(E.success)+str(E.message))
            return best_x, objective(best_x)

class InsufficientRanges(Exception):
    def __init__(self, message='ranges must match number of variable pulleys'):
        super().__init__(message)

class VariableStrucMatrix():

    plot_count = 0
    figures = {}  # Dict to track figures by name
    figures_with_axes = set()

    class linear_effort_change_joint():
        def __init__(self, min, max, idx) -> None:
            self.min = min
            self.max = max
            self.idx = idx
        def __call__(self, theta, *args, **kwds):
            return (self.max-self.min)/(np.pi/2)*theta

    class convergent_circles_joint():
        """
        A class to define a type of tendon routing where each link in the joint
            has an arc, the two of which converge to a single circle at 90
            degrees of deflection. Max and Min are passed as the distance from
            the pivot to the tendon at 90 degrees and 0 degrees respectively,
            and the distance at any angle theta is calculated based on geometry.
        """
        def __init__(self, min, max, idx):
            self.min = min
            self.max = max
            self.idx = idx
            # distance from center of convergent circle to joint
            self.c = (self.max-self.min)/(1-np.sqrt(2)/2)
            # radius of convergent circle
            self.r = self.c-self.max

            self.min = self(0)
            self.max = self(np.pi/2)

        def __call__(self, theta):
            return self.c*np.cos((np.pi/2-theta)/2)-self.r

    class convergent_circles_joint_with_limit():
        """
        Similar to convergent circles, but with a convex arc around 
        """
        def __init__(self, min, max, idx, minOverwrite=None):
            self.min = min
            self.max = max

            self.idx = idx
            if minOverwrite is None:
                minOverwrite = 0

            # distance from center of convergent circle to joint
            self.c = (self.max-minOverwrite)/(1-np.sqrt(2)/2)
            # radius of convergent circle
            self.r = self.c-self.max
            # Check to confirm
            self.max = self(np.pi/2)

        def __call__(self, theta):
            val = self.c*np.cos((np.pi/2-theta)/2)-self.r
            if val <= self.min:
                return self.min
            else:
                return val

    def __init__(self, R, D, F=None, ranges=None, types=None, constraints=None, name='Placeholder'):

        # Avoid mutable defaults
        if ranges is None:
            ranges = []
        if types is None:
            types = []
        if constraints is None:
            constraints = []

        # Direction array
        self.D = D
        # Radius array
        self.R = R

        self.numJoints  = self.D.shape[0]
        self.numTendons = self.D.shape[1]

        self.numFixed    = self.numJoints*self.numTendons - np.sum(np.isnan(R))
        self.numVariable = np.sum(np.isnan(R))

        # default for ranges
        if len(ranges)==0:
            # average radius value, one each per variable tendon
            ranges=[np.average(R)]*np.sum(np.isnan(R))
            self.numVDoF = len(ranges)
        # Allow a single range to be passed to all variable tendons
        elif len(ranges)==1:
            ranges = ranges*np.xum(np.isnan(R))
            self.numVDoF = 1 # assume this means only one range
        # At this point, if a ranges has been passed of insufficient length, bitch about it
        if not np.sum(np.isnan(R)) == len(ranges):
            raise InsufficientRanges()

        # store S matrix indices of variable tendons
        self.variableTendons = list(zip(np.array(np.where(np.isnan(R)))[0,:],np.array(np.where(np.isnan(R)))[1,:]))
        # list of effort functions according to type
        self.effortFunctions = []
        # default to linear if none specified
        if types==[]:
            types = [self.linear_effort_change_joint]*len(self.variableTendons)
            # create function and add to list for each variable tendon
            for i, idx in enumerate(self.variableTendons):
                setattr(self, f'j{idx[0]}t{idx[1]}r', types[i](ranges[i][0],ranges[i][1],idx))
                self.effortFunctions.append(getattr(self, f'j{idx[0]}t{idx[1]}r'))
        else:
            for i, idx in enumerate(self.variableTendons):
                setattr(self, f'j{idx[0]}t{idx[1]}r', types[i](ranges[i][0],ranges[i][1],idx))
                self.effortFunctions.append(getattr(self, f'j{idx[0]}t{idx[1]}r'))

        # Force-based scaling vector
        if F is None:
            self.F = np.ones(self.D.shape[1])
        else:
            self.F = F

        self.constraints = constraints
        self.name = name

        self.controllability = self.Controllability(self)

        self.extS = self.S([0]*self.numJoints)

    def __call__(self, THETA, *args, **kwds):
        # return the structure matrix at a certain pose
        return self.S(THETA)

    def S(self, THETA):
        R = self.R.copy()
        # For each variable range tendon
        for function in (self.effortFunctions):
            # Update the corresponding effort radius
            R[function.idx] = function(THETA[function.idx[0]])
        # Make sure you did all of them (this really only works the first time)
        if np.sum(np.isnan(R)):
            raise InsufficientRanges()
        # Create and return structure matrix
        S = self.D*R
        return S

    def maxGrip(self, THETA):
        maxStrength = 0
        _, boundaryGrasps = self.torqueDomainVolume(THETA)
        # print(self.boundaryGrasps)
        for grasp in boundaryGrasps:
            strength = np.linalg.norm(grasp)
            # if intersects_positive_orthant(grasp) and strength>maxStrength:
            if strength>maxStrength:
                    # print(grasp)
                    maxStrength = strength
        return maxStrength

    def torqueDomainVolume(self, THETA):
        S = self.S(THETA)
        singleForceVectors = list(np.transpose(S @ np.diag(self.F)))
        domain, boundaryGrasps = special_minkowski(singleForceVectors)
        return domain, boundaryGrasps

    def plotGrasp(self, THETA, grasp, showBool=False):
        Smat = self.S(THETA)
        S = StrucMatrix(S=Smat, F=self.F, name=self.name)
        S.plotGrasp(grasp, showBool=showBool, obj=type(self))

    def plotCapability(self, THETA, showBool=False, colorOverride=None):
        Smat = self.S(THETA)
        S = StrucMatrix(S=Smat, F=self.F, name=self.name)
        S.plotCapability(showBool = showBool, colorOverride=colorOverride, obj=type(self))

    def plotCapabilityAcrossAllGrasps(self, resl=10, showBool=False):
        # np.linspace
        axes = [np.linspace(0*i,np.pi/2,resl) for i in range(self.numJoints)]
        grid = np.meshgrid(*axes, indexing='ij')
        THETAS = np.stack([g.ravel() for g in grid], axis=-1)
        valids = 0
        for THETA in THETAS:
            Smat = self.S(THETA)
            S = StrucMatrix(S=Smat, F=self.F, name=self.name)
            if S.validity:
                valids += 1
                color=[0,1,0]
                a = 1
            else:
                color=[1,0,0]
                a = 0.01
            # [THETA[0]/(np.pi/2),THETA[1]/(np.pi/2),THETA[2]/(np.pi/2)]
            S.plotCapability(showBool = False, colorOverride=color, obj=type(self), transOverride=a)
            print(f"attempting to plot {self.name} for {', '.join(f'{x:.3f}' for x in THETA)}", end = "\r")
        print("")
        print(f"{valids/(resl**self.numJoints):.3%} controllable")
        if showBool:
            plt.show()
        return valids/(resl**self.numJoints)

    def plotCapabilityAcrossPowerGrasps(self, resl=100, showBool=False):
        # np.linspace
        valids = 0
        for theta in np.linspace(0,np.pi/2,resl):
            Smat = self.S([theta]*self.numJoints)
            S = StrucMatrix(S=Smat, F=self.F, name=self.name)
            S.plotCapability(showBool = False, colorOverride=cmap(theta/(np.pi/2)), obj=type(self))
            print(f"attempting to plot {self.name} for {[', '.join(f'{x:.3f}' for x in [theta,theta,theta])]}", end = "\r")
            if S.validity:
                valids += 1
        print("")
        print(f"{valids/(resl):.3%} controllable")
        if showBool:
            plt.show()
        return valids/(resl)

    def contains(self, THETA, grip):
        domain, _ = self.torqueDomainVolume(THETA)
        check1 = in_hull(domain, grip)
        check2 = in_hull2(domain, grip)
        if check1==check2:
            return check1
        else:
            warnings.warn(f'the two checking methods disagree, linprog says {check1}, geometry says {check2}')
            return check1

    def grip_from_tensions(self, THETA, T):
        # pass # TODO: PLOT GRASPS FROM TENDON TENSIONS
        Taus = self.S(THETA).dot(T)
        return Taus

    def add_grasp(self, THETA, grip, type='ineq'):
        constraint = GraspConstraintWrapper(self.contains_by, type, THETA, grip)
        self.constraints.append(constraint)

    def contains_by(self, rvec, THETA, grip):
        self.reinit(rvec)
        domain, _ = self.torqueDomainVolume(THETA)
        return -np.max(domain.equations @ np.append(grip, 1))

    def flatten_r_matrix(self):
        """
        flatten the R matrix into fixed radii, then ranges (maintains row-first order of both fixed and variable effort radii)
        """
        r = self.R[np.isfinite(self.R) & (self.R != 0)].flatten().tolist()
        for function in self.effortFunctions:
            r.append(float(function.min))
            r.append(float(function.max))
        return np.array(r)

    def reinit(self, rvec):
        """
        properly update the variable structure matrix object based on a design vection [r, [ranges]], where r is a row-first ordered list of fixed efforts, and [ranges] is a row-first ordered list of min, max, min, max for each range
        """
        fixed = rvec[:self.numFixed]
        variable = rvec[self.numFixed:]
        # update ranges
        for i, function in enumerate(self.effortFunctions):
            function.min = variable[2*i]
            function.max = variable[2*i+1]
        # update fixed radii
        indices = np.argwhere(np.isfinite(self.R) & (self.R != 0)).T
        for i in range(len(fixed)):
            self.R[indices[0,i],indices[1,i]] = fixed[i]
        self.extS = self.S([0]*self.numJoints)

    def ffNorm(self, order):
        rvec = self.flatten_r_matrix()
        limitingRadii = np.concatenate([rvec[:self.numFixed], rvec[self.numFixed+1::2]])
        return(np.linalg.norm(limitingRadii, order))

    def optimizer(self):
        method='trust-constr'

        def radiusLimit(rvec):
            self.reinit(rvec)
            return self.ffNorm(np.inf)

        def plotCallback(intermediate_result: OptimizeResult):
            # iteration += 1
            global best_x
            if (not method=='TNC') and (not method=='SLSQP') and (not method=='COBYLA'):
                print(intermediate_result.x)
                best_x = intermediate_result.x.copy()
            else:
                print(intermediate_result)
                best_x = intermediate_result
            # self.plotCapability()

        def slackness(rvec):
            return min(abs(np.array([self.contains_by(rvec, point) for point in [constraint.args for constraint in self.constraints]])))

        rvecInit = self.flatten_r_matrix()
        print('starting with:', rvecInit)
        objective = radiusLimit

        # validityConstraint = NonlinearConstraint(validity, -.5, 0.5)
        # Apply the appropriate grasp constraints to the optimizer
        constraints = [con.constraint for con in self.constraints]
        # For each grasp constraint passed to the StrucMatrix instance,
        print(f"Receiving {len(self.constraints)} constraints in the optimizer")
        # If we did not include an equality constraint, we need to make sure that at least one of the constraints is active
        if not any([constraint.type=='eq' for constraint in self.constraints]):
            constraints.append(NonlinearConstraint(slackness, -0.001, 0.001))
            print("enforcing slackness")

        try:
            E = minimize(objective, rvecInit, method=method, constraints=constraints, callback=plotCallback,
                         options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 1000, 'initial_constr_penalty': 2})
            self.optSuccess = str(str(E.success)+str(E.message))
            return E.x, E.fun
        except KeyboardInterrupt:
            self.optSuccess = str(str(E.success)+str(E.message))
            return best_x, objective(best_x)
        # print(E.success, E.message)
    # Used to track controllability features across different poses

    class Controllability:
        def __init__(self, parent):
            self.parent = parent
            self.rankCriterion = None
            self.nullSpaceCriterion = None
            self.biasForceSpace = None
            self.biasForceCondition = None

        def __call__(self, THETA, suppress=True):
            returnFlag = 1
            S = self.parent.S(THETA)
            self.rankCriterion = np.linalg.matrix_rank(S, tol=1e-6)>=self.parent.numJoints
            # print(self.rankCondition)
            if not self.rankCriterion:
                if not suppress:
                    warnings.warn(f"WARNING: structure matrix {self.parent.name} failed rank condition (null space condition not checked) \nrank            : {np.linalg.matrix_rank(S)} \nnumber of Joints: {self.parent.numJoints}")
                # return False
                returnFlag = 0
            nullSpace = sp.linalg.null_space(S)
            # Condition the nullSpace output well for future checking
            nullSpace[np.isclose(nullSpace, 0)] = 0
            self.biasForceSpace = nullSpace
            # Record condition of the null space
            if min(abs(self.biasForceSpace)) == 0:
                self.biasForceCondition = np.inf
            else:
                self.biasForceCondition = np.max((self.biasForceSpace))/np.min((self.biasForceSpace))
            # Check to make sure that there exists an all-positive vector in the null space
            if np.shape(self.biasForceSpace)[-1]>1:
                # print("intersecting positive orthant")
                self.nullSpaceCriterion = intersects_positive_orthant(nullSpace.T)
                for row in self.biasForceSpace:
                    # print(row)
                    if all([x==0 for x in row]):
                        self.nullSpaceCriterion =  False
            else:
                # simpler and (more reliable?) condition for valid null space assuming n+1 (1-D null space)
                self.nullSpaceCriterion = all([i > 0 for i in self.biasForceSpace])
            if not self.nullSpaceCriterion:
                if not suppress:
                    warnings.warn(f"WARNING: structure matrix {self.parent.name} failed null space condition (rank condition passed)")
                returnFlag = 0

            return returnFlag

class GraspConstraintWrapper():
    def __init__(self, function, type, *args) -> None:
        self.function = function
        self.type = type
        self.args = args
        if type=='ineq':
            self.constraint = NonlinearConstraint(self, lb=0, ub=np.inf)
        elif type=='eq':
            self.constraint = NonlinearConstraint(self, lb=-.001, ub=.001)
    def __call__(self, rvec):
        return self.function(rvec, *self.args)

# Centered type 1
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
R = np.array([[1/3,1/3,1/3,3/3],
              [0,1/2,1/2,2/2],
              [0,0,1,1]])
centeredType1 = StrucMatrix(R,D,name='centered1')

# Balanced type 1
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
R = np.array([[1,1,1,0.875],
              [0,1,1,0.375],
              [0,0,1,0.125]])
balancedType1 = StrucMatrix(R,D,name='balanced1')

# Individual type 1
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
R = np.array([[ 0.971,  0.033,  0.151, 0.568],
              [ 0.   ,  0.968,  0.141, 0.531],
              [ 0.   ,  0.   ,  0.953, 0.42 ]])/0.971
# print(R)
individualType1 = StrucMatrix(R,D,name='individual1')

# Centered type 2
D = np.array([[1,-1,1,-1],
              [0,-1,1,-1],
              [0,0,1,-1]])
R = np.array([[0.5,0.5,0.5,0.5],
              [0,  0.5,1,  0.5],
              [0,  0,  1,  1]])
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
# R = np.array([[.2188,.2188,.2188,.2188],
#               [0,.1719,.1719,.1719],
#               [0,0,.1484,.1484]])
r = 1
R = np.array([[r,r,r,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
# I'm not saying Dr. Ambrose's design is naiive, this is just a naiive implementation of that design in my system
naiiveAmbrose = StrucMatrix(R,D,name='Ambrose')

# Hollow Design
r = 1
R = np.array([[r,r,r,r],
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
diagonal = StrucMatrix(R,D,name='Diagonal')

# test
r = 1
R = np.array([[r,r,r,r],
              [r,r,r,r],
              [r,r,r,r]])
D = np.array([[-1, 1,1,-1],
              [ 1,-1,1,-1],
              [ 1, 1,-1,-1]])
test = StrucMatrix(R,D, name='Test')

# free result
r = 1
R = np.array([[r,r,r,0],
              [r,0.1*r,r,r],
              [r,r,0,r]])
D = np.array([[-1, 1,1,0],
              [ 1, 1,1,-1],
              [ 1, 1,0,-1]])
resultant = StrucMatrix(R,D, name='Resultant')

r = 1
R = np.array([[r,r,r,0],
              [r,0,r,r],
              [r,r,0,r]])
D = np.array([[-1, 1,1,0],
              [ 1, 0,1,-1],
              [ 1, 1,0,-1]])
resultant2 = StrucMatrix(R,D, name='Resultant2')

# canon A
R = np.ones([3,4])
D = np.array([[-1,1,-1,1],
              [0,-1,1,1],
              [0, 0,-1,1]])
canonA = StrucMatrix(R,D,name='Canon A')

# canon B
R = np.ones([3,4])
D = -np.array([[1,-1,-1,1],
              [0, 1,-1,-1],
              [0, 0, 1,-1]])
canonB = StrucMatrix(R,D,name='Canon B')


D = np.array([[1,1,1,-1],
              [1,1,1,-1],
              [1,1,1,-1]])
r = .1625
R = np.array([[np.nan,r     ,r     ,r],
              [r     ,np.nan,r     ,r],
              [r     ,r     ,np.nan,r]])

r_1 = .261281
r_2 = .190271
r_3 = .307475

c_1 = .42378
c_2 = .35277
c_3 = .46997

skinnyLegend = VariableStrucMatrix(R, D, ranges=[(c_1*np.sqrt(2)/2-r_1,c_1-r_1),
                                                 (c_2*np.sqrt(2)/2-r_2,c_2-r_2),
                                                 (c_3*np.sqrt(2)/2-r_3,c_3-r_3),],
                                         types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                         F = np.array([50]*4),
                                         name='Skinny Legend')

D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])

R = np.array([[.203125,.203125,.203125,.171875],
              [0      ,np.nan ,np.nan ,.125   ],
              [0      ,0      ,np.nan ,.101103]])
c1 = .9541575
c2 = .9505297
r = .5625
dimensionalAmbrose = VariableStrucMatrix(R, D, ranges=[(c1*np.sqrt(2)/2-r,c1-r)]*2+
                                                      [(c2*np.sqrt(2)/2-r,c2-r)],
                                               types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                               F = np.array([50]*4),
                                               name='The Ambrose')