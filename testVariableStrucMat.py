import numpy as np
from strucMatrices import VariableStrucMatrix

D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])

R = np.array([[np.nan,1,np.nan,-1],
              [0,np.nan,1,-1],
              [0,0,np.nan,-1]])
test = VariableStrucMatrix(R, D, ranges=[(0,1)]*4)
# print(np.array(test.variablePulleys))
# print(test.j0t0r(0))
print(test.extS)