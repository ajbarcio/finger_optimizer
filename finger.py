import numpy as np
import warnings
from strucMatrices import VariableStrucMatrix, StrucMatrix
from utils import trans, jac, clean_array, hArray
from scipy.optimize import nnls, lsq_linear, linprog

class StructureKineMismatch(Warning):
    def __init__(self, message='WARNING: number of link lengths does not match \
                                         number of joints'):
        super().__init__(message)

class Finger():

    # // a grip is a vector of numJoints torques

    def grasp(self, F, q, l=None, frame="World"):
        return Finger.Grasp(self, F, q, l, frame)

    class Grasp:
        """
        A grasp is a defined by reaction forces at each joint and joint angles (pose)
        """
        def __init__(self, finger, F, q, l=None, frame="World"):
            self.finger = finger
            self.l = l if l is not None else [0.5] * finger.numJoints
            self.F = F
            self.q = q
            self.frame=frame
            
    def __init__(self, structure: VariableStrucMatrix | StrucMatrix, lengths, tensionLimit=50):
        self.structure = structure
        self.lengths = lengths
        if self.structure.numJoints != len(self.lengths):
            warnings.warn(StructureKineMismatch())

        self.numJoints = self.structure.numJoints
        self.numTendons = self.structure.numTendons
        # print(self.numJoints, self.numTendons)
        self.tensionLimit = tensionLimit
    
    def get_jacobian_at_pose(self, THETA, lengths=None):
        # F = trans(THETA, self.lengths)
        if lengths==None:
            lengths=self.lengths
        J = jac(THETA, lengths)
        return J

    def tip_wrench_at_pose_to_grip(self, THETA, F, lengths=None, frame="world"):
        '''
        takes a tip wrench in the EE frame given a pose and generates a set of 
        joint torques to satisfy 
        '''
        if lengths==None:
            lengths=self.lengths
        if frame=="EE":
            q = np.asarray(THETA)
            T = trans(q, lengths)
            R = T[:3,:3]
            F = R @ F
        Taus = (self.get_jacobian_at_pose(THETA, lengths).T @ F).flatten()
        Taus = clean_array(Taus)
        return Taus

    def grasp_to_grip(self, grasp: Grasp):
        Taus = np.zeros(3)
        for i, f in reversed(list(enumerate(grasp.F))):
            lengths = self.lengths[:i+1]
            lengths[-1] = lengths[-1]*grasp.l[i]
            THETA = grasp.q[:i+1]
            taus = self.tip_wrench_at_pose_to_grip(THETA, grasp.F[i], lengths, frame=grasp.frame)
            # print(f"torques contributed from force at index {i}: {taus}")
            while len(taus) < self.numJoints:
                taus = np.append(taus, 0)
            Taus+=taus
        # print(f"resultant torques: {Taus}")
        return Taus

    def tensions_to_tip_wrench(self, THETA, T):
        if self.numTendons != len(T):
            warnings.warn(StructureKineMismatch(
                message=f'passed tension vector of length {len(T)}, \
                          expected {self.numTendons}'))
        # if 
        Taus = self.structure.grip_from_tensions(THETA, T)
        print(Taus)
        wrench = jac(THETA, self.lengths) @ Taus
        wrench = clean_array(wrench)
        return wrench
    
    def grip_to_tensions(self, THETA, Taus):
        
        A = self.structure(THETA)
        b = Taus
        best = np.inf
        bestRes = None
        for i in range(self.numTendons):
            # c = M[i,:] #negative sign to maximize
            c = np.zeros(self.numTendons)
            c[i] = 1
            # print("c", c)
            self.structure.controllability(THETA)
            minFactor = 1/self.structure.controllability.biasForceCondition*0.85
            opt = linprog(c, A_eq=A, b_eq=b, bounds=(self.tensionLimit*minFactor, None), method='interior-point')
            # print(opt.fun, opt.x, c)
            if (opt.fun < best):
                best = opt.fun
                bestRes = opt.x
        return bestRes
        # self.structure.controllability(THETA)
        # controllability = self.structure.controllability
        # S = self.structure(THETA)
        # res = lsq_linear(S, Taus, bounds=(0,self.tensionLimit))
        # T = res.x
        # rnorm = np.linalg.norm(S @ res.x - Taus)
        # print(rnorm)
        # print("solution:", T)
        # if 0 in clean_array(T):
        #     max_biases = []
        #     min_biases = []
        #     for i in range(len(T)):
        #         if controllability.biasForceSpace[i] > 0:
        #             min_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
        #             max_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
        #         elif controllability.biasForceSpace[i] < 0:
        #             min_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
        #             max_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
        #         else:
        #             min_bias = biasForce
        #             max_bias = self.tensionLimit
        #         max_biases.append(max_bias)
        #         min_biases.append(min_bias)
        #     if np.max(min_biases) < np.min(max_biases):
        #         biasScale = np.min(max_biases)
        #     else:
        #         biasScale = 0
        #     # biasScale = np.min(max_biases)
        #     print(f"used calculated scale {biasScale}")
        #     T = T + (controllability.biasForceSpace*biasScale).flatten()
        # else:
        #     print("blind scale")
        #     T = T + (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten()
        # # print("normalized bias:", controllability.biasForceSpace.flatten())
        # # print("scaled bias", (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten())
        # print("solution with bias:", T)
        # confirm = S @ T
        # # print(confirm)
        # # print(Taus)
        # confirm = clean_array(confirm)
        # Taus = clean_array(Taus)
        # if  np.allclose(confirm, Taus):
        #     return T, "exact", confirm
        # else:
        #     return T, "best-case", confirm
