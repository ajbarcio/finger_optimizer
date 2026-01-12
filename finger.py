import numpy as np
import warnings
from strucMatrices import VariableStrucMatrix, StrucMatrix
from utils import trans, jac, clean_array
from scipy.optimize import nnls, lsq_linear

class StructureKineMismatch(Warning):
    def __init__(self, message='WARNING: number of link lengths does not match \
                                         number of joints'):
        super().__init__(message)

class Finger():

    # // a grip is a vector of numJoints torques

    def grasp(self, F, q, l=None):
        return Finger.Grasp(self, F, q, l)

    class Grasp:
        """
        A grasp is a defined by reaction forces at each joint and joint angles (pose)
        """
        def __init__(self, finger, F, q, l=None):
            self.finger = finger
            self.l = l if l is not None else [0.5] * finger.numJoints
            self.F = F
            self.q = q
            

    def __init__(self, structure: VariableStrucMatrix | StrucMatrix, lengths, tensionLimit=50):
        self.structure = structure
        self.lengths = lengths
        if self.structure.numJoints != len(self.lengths):
            warnings.warn(StructureKineMismatch())

        self.numJoints = self.structure.numJoints
        self.numTendons = self.structure.numTendons

        self.tensionLimit = tensionLimit
    
    def get_jacobian_at_pose(self, THETA):
        # F = trans(THETA, self.lengths)
        J = jac(THETA,self.lengths)
        return J

    def tip_wrench_at_pose_to_grip(self, THETA, F):
        '''
        takes a tip wrench in the EE frame given a pose and generates a set of 
        joint torques to satisfy 
        '''
        Taus = (jac(THETA, self.lengths).T @ F).flatten()
        Taus = clean_array(Taus)
        return Taus

    def grasp_to_grip(self, grasp: Grasp):

        for i, f in reversed(list(enumerate(grasp.F))):
            lengths = self.lengths[:i+1]
            print(lengths)
            lengths[-1] = lengths[-1]*grasp.l[i]
            print(lengths)
            THETA = grasp.q[:i+1]
            print(THETA)
            print(lengths)
            tempJ = (jac(THETA, lengths))
            taus = clean_array(tempJ.T @ np.array([0,grasp.F[i],0]).flatten())
            # taus = self.tip_wrench_at_pose_to_grip(THETA, [0,grasp.F[i],0])
            print(f"torques contributed from force at index {i}: {taus}")


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
    
    def grasp_to_tensions(self, THETA, Taus, biasForce = 2):
        self.structure.controllability(THETA)
        controllability = self.structure.controllability
        S = self.structure(THETA)
        res = lsq_linear(S, Taus, bounds=(0,self.tensionLimit))
        T = res.x
        rnorm = np.linalg.norm(S @ res.x - Taus)
        print(rnorm)
        print("solution:", T)
        if 0 in clean_array(T):
            max_biases = []
            min_biases = []
            for i in range(len(T)):
                if controllability.biasForceSpace[i] > 0:
                    min_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
                    max_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
                elif controllability.biasForceSpace[i] < 0:
                    min_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
                    max_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
                else:
                    min_bias = biasForce
                    max_bias = self.tensionLimit
                max_biases.append(max_bias)
                min_biases.append(min_bias)
            if np.max(min_biases) < np.min(max_biases):
                biasScale = np.min(max_biases)
            else:
                biasScale = 0
            # biasScale = np.min(max_biases)
            print(f"used calculated scale {biasScale}")
            T = T + (controllability.biasForceSpace*biasScale).flatten()
        else:
            print("blind scale")
            T = T + (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten()
        # print("normalized bias:", controllability.biasForceSpace.flatten())
        # print("scaled bias", (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten())
        print("solution with bias:", T)
        confirm = S @ T
        # print(confirm)
        # print(Taus)
        confirm = clean_array(confirm)
        Taus = clean_array(Taus)
        if  np.allclose(confirm, Taus):
            return T, "exact", confirm
        else:
            return T, "best-case", confirm
