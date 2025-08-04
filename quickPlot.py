from strucMatrices import *
from utils import *
import itertools

D1 = np.array([[1,1,1,-1],
               [0,1,1,-1],
               [0,0,1,-1]])
R = np.array([[1,1,1,1],
              [0,1,1,.5],
              [0,0,1,.25]])
structure = StrucMatrix(R,D1,name='structure')
print(structure())
print(structure.validity)
print(structure.biasForceSpace)
structure.plotCapability(showBool = True)