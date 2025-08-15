import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
from scipy.linalg import null_space

from utils import intersects_positive_orthant, special_minkowski, in_hull, get_existing_axes, get_existing_3d_axes, in_hull2, intersects_negative_orthant, intersection_with_orthant
from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult, dual_annealing, differential_evolution


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
        # print('going to check validity')
        # for constraint in self.constraints:
        #     if not constraint(self):
        #         return False
            # print('checked constraint', constraint)

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

    def plotCapability(self, showBool=False, colorOverride=None):

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
                self.ax.quiver(0,0,0,grasp[0],grasp[1],grasp[2],color="black")
                # print(grasp)
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
                ax.set_xlabel('τ₁')
                ax.set_ylabel('τ₂')
                ax.set_zlabel('τ₃')
                # ax.set_title('Torque Components τ₁, τ₂, τ₃')
                ax.set_title(f'{self.name} Capability Polytope')
                # ax.view_init(elev=30, azim=45)
                ax.grid(True)
                plt.tight_layout()

                StrucMatrix.figures_with_axes.add(figID)
            if showBool:
                plt.show()
        else:
            warnings.warn("Cannot plot anything other than 3d grasps at this time")
        StrucMatrix.plot_count += 1

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
    def __init__(self, R=None, D=None, S=None, ranges=[], constraints=[], name='Placeholder'):
        if not np.sum(np.isnan(D)) == len(ranges):
            raise InsufficientRanges()
        self.variablePulleys = [list(x) for x in np.where(np.isnan(D if D is not None else S))]

        super().__init__(R, D, S, constraints, name)

    def torquDomainVolume(self):
        pass

class NonlinearConstraintContainer():
    def __init__(self, function, type, *args) -> None:
        self.function = function
        self.type = type
        self.args = args
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