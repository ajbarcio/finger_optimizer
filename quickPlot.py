from strucMatrices import *
from utils import *
import itertools

S1 = naiiveAmbrose
S1.F = np.array([50,50,50,50])
S1.reinit()
# structure1 = StrucMatrix(S=S1,name='structure')
# structure2 = StrucMatrix(S=S1,name='structure')
# structure3 = StrucMatrix(S=S1,name='structure')
# print(structure())
# print(structure.validity)
# print(structure.biasForceSpace)
print(S1())
S1.plotCapability(showBool = True)