import numpy as np
import warnings
from strucMatrices import VariableStrucMatrix, StrucMatrix
from utils import trans, jac

class StructureKineMismatch(Warning):
    def __init__(self, message='WARNING: number of link lengths does not match \
                                         number of joints'):
        super().__init__(message)

class Finger():
    def __init__(self, structure: VariableStrucMatrix | StrucMatrix, lengths):
        self.structure = structure
        self.lengths = lengths
        if self.structure.numJoints != len(self.lengths):
            warnings.warn(StructureKineMismatch())

        self.numJoints = self.structure.numJoints
        self.numTendons = self.structure.numTendons
    
    def get_jacobian_at_pose(self, THETA):
        # F = trans(THETA, self.lengths)
        J = jac(THETA,self.lengths)
        return J

    def tip_wrench_to_grasp(self, THETA, F):
        '''
        takes a tip wrench in the EE frame given a pose and generates a set of 
        joint torques to satisfy 
        '''
        pass

    def tensions_to_tip_wrench(self, THETA, T):
        if self.numTendons != len(T):
            warnings.warn(StructureKineMismatch(
                message=f'passed tension vector of length {len(T)}, \
                          expected {self.numTendons}'))
        # if 
        taus = self.structure.grip_from_tensions(THETA, T)
        
