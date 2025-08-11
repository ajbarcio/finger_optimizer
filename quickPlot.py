from strucMatrices import *
from utils import *
import itertools

S1 = canonA
S1.F = np.array([50,50,50,50])
S1.reinit()
# structure1 = StrucMatrix(S=S1,name='structure')
# structure2 = StrucMatrix(S=S1,name='structure')
# structure3 = StrucMatrix(S=S1,name='structure')
# print(structure())
# print(structure.validity)
print(S1.biasForceSpace)
print(S1())
S1.plotCapability(showBool = True)