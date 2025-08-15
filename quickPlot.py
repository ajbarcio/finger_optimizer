from strucMatrices import *
from utils import *
import itertools

# S1 = quasiHollow
S = np.array([[ 0.439, -0.449,  0.189,  0.078],
              [ 0.   ,  0.131,  0.444, -0.434],
              [ 0.   ,  0.   ,  0.219, -0.08 ]])
S1 = StrucMatrix(S=S)
S1.name = "Example"
# S1.F = np.array([50,50,50,50])
# S1.reinit()
# structure1 = StrucMatrix(S=S1,name='structure')
# structure2 = StrucMatrix(S=S1,name='structure')
# structure3 = StrucMatrix(S=S1,name='structure')
# print(structure())
# print(structure.validity)
print(S1.biasForceSpace)
print(S1())
S1.plotCapability(showBool = True, colorOverride = 'xkcd:Blue')