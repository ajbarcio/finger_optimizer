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
print(f"required ultimate torque at each joint: {torques}")

data = np.loadtxt('AvailableSprings.csv', delimiter=',')

stiffnesses = data[:,2]
springData  = data
commonSprings = None

# choose minimum stiffness spring that can achieve at least J_n torque and have room for preload
for torque in torques:
       
       # max torque is greater than required torque, needs at least 10 degrees of displacement before it reaches that torque, plus arbitrary form factor constraint
       feasibilityCriterion = (springData[:,1]>torque)
       preloadCriterion     = (torque-(springData[:,2]*10)>0)
       formFactorCriterion  = (90-(torque-(springData[:,2]*10))/(springData[:,2])>=0)
       
       feasibleSprings = springData[feasibilityCriterion & preloadCriterion & formFactorCriterion]
       # hashable list of springs
       feasibleSet = set(map(tuple, feasibleSprings))
       # keep track of springs that are feasible for all joints
       if commonSprings is None:
              commonSprings = feasibleSet
       else:
              commonSprings &= feasibleSet
# back into an array
if commonSprings:
    commonSprings = np.array(list(commonSprings))
else:
    commonSprings = np.empty((0, springData.shape[1]))

# print(comemonSprings)
# best spring is least stiff (flattest torque)
bestSpring = commonSprings[np.argmin(commonSprings[:,2])]

# calculate torque at contact
initTorques = np.array([torque-bestSpring[2]*10 for torque in torques])
print(f"torque at contact: {initTorques}")

print(f"spring with flattest torque: {bestSpring}")

# calculate preload angle for torque at contact
preloadAngls = np.array([torque/bestSpring[2] for torque in initTorques])
print(f"required preload angle at each joint in degrees: {preloadAngls}")

# S1.reinit()
# structure1 = StrucMatrix(S=S1,name='structure')
# structure2 = StrucMatrix(S=S1,name='structure')
# structure3 = StrucMatrix(S=S1,name='structure')
# print(structure())
# print(structure.validity)
# print(S1.biasForceSpace)
# print(S1())
# S1.plotCapability(showBool = True, colorOverride = 'xkcd:Blue')