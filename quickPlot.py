from strucMatrices import *
from utils import *
import itertools

# S1 = quasiHollow
# S = np.array([[ .1477, .1477,  .1477, -.1477],
#               [ 0.   ,  .1477, .1477, -.1477],
#               [ 0.   ,  0.   , .1477, -.1477 ]])
# S1 = StrucMatrix(S=S)
# S1.name = "Example"
# S1.F = np.array([50,50,50,50])

# F = np.array([50,50,50,2])
# torques = S1.S.dot(F)

D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])

R = np.array([[.203125,.203125,.203125,.171875],
              [0      ,np.nan ,np.nan ,.125   ],
              [0      ,0      ,np.nan ,.101103]])
c1 = .9541575
c2 = .9505297
r = .5625
dimensionalAmbrose = VariableStrucMatrix(R, D, ranges=[(c1*np.sqrt(2)/2-r,c1-r)]*2+
                                                      [(c2*np.sqrt(2)/2-r,c2-r)],
                                               types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                               F = np.array([50]*4),
                                               name='The Ambrose')
THETA = np.array([np.pi/2]*3)
F = np.array([50,50,50,2])
S = dimensionalAmbrose(THETA)
torques = S.dot(F)
print(torques)

# S1.reinit()
# structure1 = StrucMatrix(S=S1,name='structure')
# structure2 = StrucMatrix(S=S1,name='structure')
# structure3 = StrucMatrix(S=S1,name='structure')
# print(structure())
# print(structure.validity)
# print(S1.biasForceSpace)
# print(S1())
# S1.plotCapability(showBool = True, colorOverride = 'xkcd:Blue')