from strucMatrices import *
from utils import *
from combinatorics import *
from variableOptimizer import createFingerFromVector
import itertools

# S = Optimus
# Optimus.plotCapability(showBool=True)

# D = np.array([[1,1,-1,-1,-1],
#               [-1,1,1,1,1],
#               [-1,1,1,1,0],
#               [-1,1,1,0,0]])
# R= np.absolute(D)
# fourdof = StrucMatrix(R=R, D=D)
# print(fourdof.S)
# print(fourdof.isValid())
# print(fourdof.biasForceSpace)
# print(null_space(D))
# S = inherent
# S.plotCapability(showBool=True, colorOverride='blue')

def decouplability_eval(theta, Fing):
    M = create_decoupling_matrix(Fing.structure([theta]*3))
    success = identify_strict_central(M)
    if success:
        return 0.0
    else:
        stiffness_dir = null_space(M)
        # scale to unit minimum: this just generally results in rounder numbers
        scale = 1.0/np.min(stiffness_dir) if np.min(stiffness_dir) != 0 else 1.0
        # scale = 1.0
        stiffnesses = stiffness_dir * scale
        K_A = np.diag(stiffnesses)
        K_J = Fing.structure().dot(K_A).dot(Fing.structure().T)
        decoupledness = np.linalg.norm(K_J-np.diag(np.diag(K_J)))
        return decoupledness

v = [0.40526359,
     0.32743508,
     0.46607661,
     0.41733753,
     0.49212554,
     0.4864513,
     0.4991715,
     0.48895549,
     0.49701881,
     0.42181808,
     0.42260185,
     0.3690979]
Finger = createFingerFromVector(v)
# evaluator = FingerEvaluator()
for theta in np.linspace(0,np.pi/2,50):
    M = create_decoupling_matrix(Finger.structure([theta]*3))
    print(identify_strict_sign_central(M))
    print(identify_sign_central(M))
    print(decouplability_eval(theta, Finger))



# D = np.array([[-1,1,1,1],
#               [0,-1,1,1],
#               [0,0,-1,1],])

# R = np.ones_like(D) # *np.random.random(D.shape)
# # print(R)
# test = StrucMatrix(R=R, D=D)
# # print(test())
# # print(test.biasCondition())
# # print((test.biasForceSpace))
# tests = [test()]

# # minFactor = 1/test.biasCondition()
# minFactor = 0.3


# print()
# print(minFactor, 1/minFactor)
# print()

# test.minFactor = minFactor

# domain, bgs = test.torqueDomainVolume(enforcePosTension=False)
# # print((bgs))
# print(len(bgs[domain.vertices]))
# domain, bgs = test.torqueDomainVolume(enforcePosTension=True)
# # print((bgs))
# print(len(bgs[domain.vertices]))

# _ = test.plotCapability(showBool=False, enforcePosTension=False)
# g = test.plotCapability(showBool=False, enforcePosTension=True)
# # print(g)

# E = np.eye(4) + ((np.ones([4,4])-np.eye(4))*minFactor)
# # print(E)



# test2 = StrucMatrix(S=g)
# # print(test2.biasCondition())
# # print(test2.biasForceSpace)
# domain, bgs = test2.torqueDomainVolume()
# # print((bgs))
# # print(bgs[domain.vertices])
# # test2.plotCapability()

# print(g)
# print(test2.singleForceVectors)
# print(test() @ E)

# plt.show()

# # p = generate_centered_qutsm(tests)[0]

# # print(p)
# # print(null_space(p))



# # for m in np.linspace(0,1,2):
# #     test2 =StrucMatrix(S=p, minFactor=m)
# #     result = test2.plotCapability(enforcePosTension=True)
#     print(result)
#     print(null_space(result))
# plt.show()

# ext = .16929
# flx = .2825

# R = np.array([[ext,flx,flx,flx],
#               [0,ext,flx,flx],
#               [0,0,ext,flx],])

# D = np.array([[-1,1,1,1],
#               [0,-1,1,1],
#               [0,0,-1,1],])
# S = StrucMatrix(R=R, D=D, minFactor=0.01, name="flexed")
# S.F = np.array([50,50,50,50])
# S.reinit()

# ext = 0.24
# flx = 0.125

# R2 = np.array([[ext,flx,flx,flx],
#               [0,ext,flx,flx],
#               [0,0,ext,flx],])

# S2 = StrucMatrix(R=R2, D=D, minFactor=0.01, name="extended")
# S2.F = np.array([50,50,50,50])
# S2.reinit()

# print(S.validity)
# print(S.biasForceSpace)
# S.plotCapability(showBool=False, enforcePosTension=True)
# S.plotCapability(showBool=False, enforcePosTension=False)

# print(S2.validity)
# print(S2.biasForceSpace)
# S2.plotCapability(showBool=False, enforcePosTension=True)
# S2.plotCapability(showBool=False, enforcePosTension=False)

# plt.show()
# _, pointsFull    = special_minkowski(S.singleForceVectors)
# _, pointsDerated = special_minkowski_with_mins(S.singleForceVectors)

# print(pointsFull)
# print(pointsDerated)

# friction = np.array([1,1,1])
# friction = np.eye()
# F = np.linalg.pinv(S()) @ friction
# print(F)    
# print(S.biasForceSpace)
# # S1 = quasiHollow
# # S = np.array([[ .1477, .1477,  .1477, -.1477],
# #               [ 0.   ,  .1477, .1477, -.1477],
# #               [ 0.   ,  0.   , .1477, -.1477 ]])
# # S1 = StrucMatrix(S=S)
# # S1.name = "Example"
# # S1.F = np.array([50,50,50,50])

# # F = np.array([50,50,50,2])
# # torques = S1.S.dot(F)

# D = np.array([[1,1,1,-1],
#               [0,1,1,-1],
#               [0,0,1,-1]])

# R = np.array([[.203125,.203125,.203125,.171875],
#               [0      ,np.nan ,np.nan ,.125   ],
#               [0      ,0      ,np.nan ,.101103]])
# c1 = .9541575
# c2 = .9505297
# r = .5625
# dimensionalAmbrose = VariableStrucMatrix(R, D, ranges=[(c1*np.sqrt(2)/2-r,c1-r)]*2+
#                                                       [(c2*np.sqrt(2)/2-r,c2-r)],
#                                                types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
#                                                F = np.array([50]*4),
#                                                name='The Ambrose')
# THETA = np.array([np.pi/2]*3)
# F = np.array([50,50,50,2])
# S = dimensionalAmbrose(THETA)
# torques = S.dot(F)
# print(f"required ultimate torque at each joint: {torques}")

# data = np.loadtxt('AvailableSprings.csv', delimiter=',')

# stiffnesses = data[:,2]
# springData  = data
# commonSprings = None

# # choose minimum stiffness spring that can achieve at least J_n torque and have room for preload
# for torque in torques:
       
#        # max torque is greater than required torque, needs at least 10 degrees of displacement before it reaches that torque, plus arbitrary form factor constraint
#        feasibilityCriterion = (springData[:,1]>torque)
#        preloadCriterion     = (torque-(springData[:,2]*10)>0)
#        formFactorCriterion  = (315-(torque-(springData[:,2]*10))/(springData[:,2])>=0)
       
#        feasibleSprings = springData[feasibilityCriterion & preloadCriterion & formFactorCriterion]
#        # hashable list of springs
#        feasibleSet = set(map(tuple, feasibleSprings))
#        # keep track of springs that are feasible for all joints
#        if commonSprings is None:
#               commonSprings = feasibleSet
#        else:
#               commonSprings &= feasibleSet
# # back into an array
# if commonSprings:
#     commonSprings = np.array(list(commonSprings))
# else:
#     commonSprings = np.empty((0, springData.shape[1]))

# # print(comemonSprings)
# # best spring is least stiff (flattest torque)
# bestSpring = commonSprings[np.argmin(commonSprings[:,2])]

# # calculate torque at contact
# initTorques = np.array([torque-bestSpring[2]*10 for torque in torques])
# print(f"torque at contact: {initTorques}")

# print(f"spring with flattest torque: {bestSpring}")

# # calculate preload angle for torque at contact
# preloadAngls = np.array([torque/bestSpring[2] for torque in initTorques])
# print(f"required preload angle at each joint in degrees: {preloadAngls}")

# # S1.reinit()
# # structure1 = StrucMatrix(S=S1,name='structure')
# # structure2 = StrucMatrix(S=S1,name='structure')
# # structure3 = StrucMatrix(S=S1,name='structure')
# # print(structure())
# # print(structure.validity)
# # print(S1.biasForceSpace)
# # print(S1())
# # S1.plotCapability(showBool = True, colorOverride = 'xkcd:Blue')