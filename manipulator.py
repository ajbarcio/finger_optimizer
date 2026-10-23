import numpy as np

from strucMatrices import StrucMatrix
from utils import *
from combinatorics import identify_strict_sign_central, identify_strict_central

def Manipulator(structure, lengths, dirs):
    def __init__(self):
        self.structure = structure
        self.lengths = lengths
        self.dirs = dirs

    def get_jacobian_at_pose(self):
        pass

D = np.hstack((np.eye(7), -np.ones((7,1))))
print(D)
print(f"Is the uniform case controllable? {"Yes" if identify_strict_central(D)[0] else "No"}")
print(f"Will all cases be controlalble? {"Yes" if identify_strict_sign_central(D) else "No"}")
print(f"Centrality: {null_space(D)}")

D1 = np.zeros((7,8))
for i in range(D1.shape[0]):
    for j in range(D1.shape[1]):
        if i==j: D1[i,j]=1
        if j==i+1: D1[i,j]=-1
print(D1)
print(f"Is the uniform case controllable? {"Yes" if identify_strict_central(D1)[0] else "No"}")
print(f"Will all cases be controlalble? {"Yes" if identify_strict_sign_central(D1) else "No"}")
print(f"Centrality: {null_space(D1)}")