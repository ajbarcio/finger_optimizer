from strucMatrices import *
from finger import *
from utils import *
from combinatorics import *
from variableOptimizer import createFingerFromVector
import itertools

# F = np.array([0,5,0])
lengths = [1.4,1.4,1.2]

# D = np.array([[-1,1,1,1],
#               [0,-1,1,1],
#               [0,0,-1,1]])

# R = np.array([[np.nan,np.nan,np.nan,np.nan],
#               [0,     np.nan,np.nan,np.nan],
#               [0,     0     ,np.nan,np.nan]])

# # [(min, max, minim), (), etc...]
# flexure_extents = [(0.06,0.35,0.2),(0.06,0.35,0.2),(0.06,0.35,0.2)]
# # [(min, max), (), etc...]
# extensure_extents = [(.25, .367),(.25, .364),(.25, .364)]

# PaperStructure = VariableStrucMatrix(R, D, ranges = [extensure_extents[0]]+[flexure_extents[0]]*3
#                                               +[extensure_extents[1]]+[flexure_extents[1]]*2
#                                               +[extensure_extents[2]]+[flexure_extents[2]],
#                                        types = [VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*3
#                                               +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*2
#                                               +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit],
#                                            F = np.array([50]*4),
#                                       name="Prototype")

fs2 = [(.138, .413, .191),
      (.134, .405, .338),
      (.120, .385, .340)]

es2 = [(0.217, .306),
      (0.153, .261),
      (0.155, .2625)]
# ps = [(.625/2*0.65,.625/2,0.4),(.625/2*0.65,.4,0.4)]

# print(fs)

testFingerFuckMe = VariableStrucMatrix(R, D, ranges = [es2[0]]+[fs2[0]]*3
                                                   +[es2[1]]+[fs2[1]]*2
                                                   +[es2[2]]+[fs2[2]],
                                           types  = [VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*3
                                                   +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*2
                                                   +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit],
                                                F = np.array([50]*5),
                                        minFactor = 0.1,
                                             name = "TFFM")

quickFinger = Finger(testFingerFuckMe, lengths)
print(quickFinger.structure.name)

qs = np.linspace(0,np.pi/2,1000)
tvecs = []
tvecs2 = []
overall_transmission_ratios = []
j0t1_effort = []
j1t2_effort = []
j2t3_effort = []

for q in qs:
    j0t1_effort.append(quickFinger.structure.j0t1r(q))
    j1t2_effort.append(quickFinger.structure.j1t2r(q))
    j2t3_effort.append(quickFinger.structure.j2t3r(q))
    # tensions  = PaperFinger.grip_to_tensions([q]*PaperFinger.numJoints,
    #                                             PaperFinger.grasp_to_grip(PaperFinger.grasp(
    #                                                                                         [F]*PaperFinger.numJoints,
    #                                                                                         [q]*PaperFinger.numJoints,
    #                                                                                         frame="EE")))
    # tensions2 = PaperFinger.grip_to_tensions([q]*PaperFinger.numJoints,
    #                                             PaperFinger.tip_wrench_at_pose_to_grip([q]*PaperFinger.numJoints,
    #                                                                                     -F*0.1,
    #                                                                                     frame="EE"))

    # condition = PaperFinger.structure.controllability([q]*PaperFinger.numJoints)
    # transmission_ratio = PaperStructure.get_magnitude([q]*PaperFinger.numJoints)

    # tvecs.append(tensions)
    # tvecs2.append(tensions2)
    # overall_transmission_ratios.append(transmission_ratio)

# print(quickFinger.structure.j0t1r.angleThreshold*180/np.pi)
# print(quickFinger.structure.j1t2r.angleThreshold*180/np.pi)
print("--")
print(f"c_flex: {quickFinger.structure.j1t2r.c}")
print(f"r_flex: {quickFinger.structure.j1t2r.r}")
print(f"parameters: {quickFinger.structure.j1t2r.min}, {quickFinger.structure.j1t2r.minOverwrite}, {quickFinger.structure.j1t2r.max}")
print("--")
# print(f"c_ext: {testFinger.j1t1r.c}")
# print(f"r_ext: {testFinger.j1t1r.r}")
print("--")
# print(quickFinger.structure.j0t1r.angleThreshold)
print(quickFinger.structure.j0t1r.r, quickFinger.structure.j0t1r.c, quickFinger.structure.j0t1r.minOverwrite)
print(quickFinger.structure.j1t2r.r, quickFinger.structure.j1t2r.c, quickFinger.structure.j1t2r.minOverwrite)
print(quickFinger.structure.j2t3r.r, quickFinger.structure.j2t3r.c, quickFinger.structure.j2t3r.minOverwrite)


plt.plot(qs, j0t1_effort)
plt.plot(qs, j1t2_effort)
plt.plot(qs, j2t3_effort)

# plt.figure("Flexion Grasp Tensions")
# plt.plot(qs, tvecs)
# plt.title("Flexion Grasp Tensions")
# plt.figure("magnitudes")
# plt.plot(qs, np.array(overall_transmission_ratios)/np.min(overall_transmission_ratios))
# plt.title("magnitudes")


# plt.figure("magnitudes2")
# plt.plot(qs*180/np.pi, np.array(overall_transmission_ratios)*16387.1, lw=3, color='black')
# plt.title("OTV of Prototype Finger over Uniform Grasps")
# q_vector = "θ\u20D7"
# degree = "\u00b0"
# plt.xlabel(f"{q_vector} ({degree})")
# plt.ylabel(f"OTV (mm\u00b3)")
# plt.xticks([0, 30, 60, 90])
plt.show()
# PaperFinger.structure.plotCapability([0]*PaperFinger.numJoints, enforcePosTension=False, metric=True)
# PaperFinger.structure.plotCapability([np.pi/2]*PaperFinger.numJoints, enforcePosTension=False, metric=True)
# S = overall_transmission_ratios
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

# def decouplability_eval(theta, Fing):
#     M = create_decoupling_matrix(Fing.structure([theta]*3))
#     success = identify_strict_central(M)
#     if success:
#         return 0.0
#     else:
#         stiffness_dir = null_space(M)
#         # scale to unit minimum: this just generally results in rounder numbers
#         scale = 1.0/np.min(stiffness_dir) if np.min(stiffness_dir) != 0 else 1.0
#         # scale = 1.0
#         stiffnesses = stiffness_dir * scale
#         K_A = np.diag(stiffnesses)
#         K_J = Fing.structure().dot(K_A).dot(Fing.structure().T)
#         decoupledness = np.linalg.norm(K_J-np.diag(np.diag(K_J)))
#         return decoupledness

# v = [0.40526359,
#      0.32743508,
#      0.46607661,
#      0.41733753,
#      0.49212554,
#      0.4864513,
#      0.4991715,
#      0.48895549,
#      0.49701881,
#      0.42181808,
#      0.42260185,
#      0.3690979]
# Finger = createFingerFromVector(v)
# # evaluator = FingerEvaluator()
# for theta in np.linspace(0,np.pi/2,50):
#     M = create_decoupling_matrix(Finger.structure([theta]*3))
#     print(identify_strict_sign_central(M))
#     print(identify_sign_central(M))
#     print(decouplability_eval(theta, Finger))

# R = np.array([[np.nan,np.nan,np.nan,np.nan],
#               [0,     np.nan,np.nan,np.nan],
#               [0,     0     ,np.nan,np.nan]])

# # [(min, max, minim), (), etc...]
# flexure_extents = [(0.0,0.35,0.2),(0.0,0.35,0.2),(0.0,0.35,0.2)]
# # [(min, max), (), etc...]
# extensure_extents = [(.25, .367),(.25, .367),(.25, .367)]

# VaraibleArbitrary = VariableStrucMatrix(R, D, ranges = [extensure_extents[0]]+[flexure_extents[0]]*3
#                                               +[extensure_extents[1]]+[flexure_extents[1]]*2
#                                               +[extensure_extents[2]]+[flexure_extents[2]],
#                                        types = [VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*3
#                                               +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*2
#                                               +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit],
#                                            F = np.array([50]*4),
#                                       name="Arbitrary Variable")

# print(VaraibleArbitrary.S([0]*3))
# print(VaraibleArbitrary.S([np.pi/2]*3))

# VaraibleArbitrary.plotCapability([0]*3)
# VaraibleArbitrary.plotCapability([np.pi/2]*3)

# qs = np.linspace(0,np.pi/2,75)
# tvecs = []
# tvecs2 = []
# conditions = []

# resultFinger = Finger(VaraibleArbitrary, lengths = [1.4,1.4,1.2])
# F = np.array([0,5,0])
# for q in qs:
#     tensions  = resultFinger.grip_to_tensions([q]*resultFinger.numJoints,
#                                                 resultFinger.grasp_to_grip(resultFinger.grasp(
#                                                                                             [F]*resultFinger.numJoints,
#                                                                                             [q]*resultFinger.numJoints,
#                                                                                             frame="EE")))
#     tensions2 = resultFinger.grip_to_tensions([q]*resultFinger.numJoints,
#                                                 resultFinger.tip_wrench_at_pose_to_grip([q]*resultFinger.numJoints,
#                                                                                         -F*0.1,
#                                                                                         frame="EE"))

#     condition = resultFinger.structure.controllability([q]*resultFinger.numJoints)

#     tvecs.append(tensions)
#     tvecs2.append(tensions2)
#     conditions.append(condition)
# plt.figure()
# plt.plot(qs, tvecs)
# plt.figure()
# plt.plot(qs, tvecs2)
# plt.figure()
# plt.plot(qs, conditions)

# plt.show()

# # D = np.array([[-1,1,1,1],
# #               [0,-1,1,1],
# #               [0,0,-1,1],])

# # R = np.ones_like(D) # *np.random.random(D.shape)
# # # print(R)
# # test = StrucMatrix(R=R, D=D)
# # # print(test())
# # # print(test.biasCondition())
# # # print((test.biasForceSpace))
# # tests = [test()]

# # # minFactor = 1/test.biasCondition()
# # minFactor = 0.3


# # print()
# # print(minFactor, 1/minFactor)
# # print()

# # test.minFactor = minFactor

# # domain, bgs = test.torqueDomainVolume(enforcePosTension=False)
# # # print((bgs))
# # print(len(bgs[domain.vertices]))
# # domain, bgs = test.torqueDomainVolume(enforcePosTension=True)
# # # print((bgs))
# # print(len(bgs[domain.vertices]))

# # _ = test.plotCapability(showBool=False, enforcePosTension=False)
# # g = test.plotCapability(showBool=False, enforcePosTension=True)
# # # print(g)

# # E = np.eye(4) + ((np.ones([4,4])-np.eye(4))*minFactor)
# # # print(E)



# # test2 = StrucMatrix(S=g)
# # # print(test2.biasCondition())
# # # print(test2.biasForceSpace)
# # domain, bgs = test2.torqueDomainVolume()
# # # print((bgs))
# # # print(bgs[domain.vertices])
# # # test2.plotCapability()

# # print(g)
# # print(test2.singleForceVectors)
# # print(test() @ E)

# # plt.show()

# # # p = generate_centered_qutsm(tests)[0]

# # # print(p)
# # # print(null_space(p))



# # # for m in np.linspace(0,1,2):
# # #     test2 =StrucMatrix(S=p, minFactor=m)
# # #     result = test2.plotCapability(enforcePosTension=True)
# #     print(result)
# #     print(null_space(result))
# # plt.show()

# # ext = .16929
# # flx = .2825

# # R = np.array([[ext,flx,flx,flx],
# #               [0,ext,flx,flx],
# #               [0,0,ext,flx],])

# # D = np.array([[-1,1,1,1],
# #               [0,-1,1,1],
# #               [0,0,-1,1],])
# # S = StrucMatrix(R=R, D=D, minFactor=0.01, name="flexed")
# # S.F = np.array([50,50,50,50])
# # S.reinit()

# # ext = 0.24
# # flx = 0.125

# # R2 = np.array([[ext,flx,flx,flx],
# #               [0,ext,flx,flx],
# #               [0,0,ext,flx],])

# # S2 = StrucMatrix(R=R2, D=D, minFactor=0.01, name="extended")
# # S2.F = np.array([50,50,50,50])
# # S2.reinit()

# # print(S.validity)
# # print(S.biasForceSpace)
# # S.plotCapability(showBool=False, enforcePosTension=True)
# # S.plotCapability(showBool=False, enforcePosTension=False)

# # print(S2.validity)
# # print(S2.biasForceSpace)
# # S2.plotCapability(showBool=False, enforcePosTension=True)
# # S2.plotCapability(showBool=False, enforcePosTension=False)

# # plt.show()
# # _, pointsFull    = special_minkowski(S.singleForceVectors)
# # _, pointsDerated = special_minkowski_with_mins(S.singleForceVectors)

# # print(pointsFull)
# # print(pointsDerated)

# # friction = np.array([1,1,1])
# # friction = np.eye()
# # F = np.linalg.pinv(S()) @ friction
# # print(F)
# # print(S.biasForceSpace)
# # # S1 = quasiHollow
# # # S = np.array([[ .1477, .1477,  .1477, -.1477],
# # #               [ 0.   ,  .1477, .1477, -.1477],
# # #               [ 0.   ,  0.   , .1477, -.1477 ]])
# # # S1 = StrucMatrix(S=S)
# # # S1.name = "Example"
# # # S1.F = np.array([50,50,50,50])

# # # F = np.array([50,50,50,2])
# # # torques = S1.S.dot(F)

# # D = np.array([[1,1,1,-1],
# #               [0,1,1,-1],
# #               [0,0,1,-1]])

# # R = np.array([[.203125,.203125,.203125,.171875],
# #               [0      ,np.nan ,np.nan ,.125   ],
# #               [0      ,0      ,np.nan ,.101103]])
# # c1 = .9541575
# # c2 = .9505297
# # r = .5625
# # dimensionalAmbrose = VariableStrucMatrix(R, D, ranges=[(c1*np.sqrt(2)/2-r,c1-r)]*2+
# #                                                       [(c2*np.sqrt(2)/2-r,c2-r)],
# #                                                types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
# #                                                F = np.array([50]*4),
# #                                                name='The Ambrose')
# # THETA = np.array([np.pi/2]*3)
# # F = np.array([50,50,50,2])
# # S = dimensionalAmbrose(THETA)
# # torques = S.dot(F)
# # print(f"required ultimate torque at each joint: {torques}")

# # data = np.loadtxt('AvailableSprings.csv', delimiter=',')

# # stiffnesses = data[:,2]
# # springData  = data
# # commonSprings = None

# # # choose minimum stiffness spring that can achieve at least J_n torque and have room for preload
# # for torque in torques:

# #        # max torque is greater than required torque, needs at least 10 degrees of displacement before it reaches that torque, plus arbitrary form factor constraint
# #        feasibilityCriterion = (springData[:,1]>torque)
# #        preloadCriterion     = (torque-(springData[:,2]*10)>0)
# #        formFactorCriterion  = (315-(torque-(springData[:,2]*10))/(springData[:,2])>=0)

# #        feasibleSprings = springData[feasibilityCriterion & preloadCriterion & formFactorCriterion]
# #        # hashable list of springs
# #        feasibleSet = set(map(tuple, feasibleSprings))
# #        # keep track of springs that are feasible for all joints
# #        if commonSprings is None:
# #               commonSprings = feasibleSet
# #        else:
# #               commonSprings &= feasibleSet
# # # back into an array
# # if commonSprings:
# #     commonSprings = np.array(list(commonSprings))
# # else:
# #     commonSprings = np.empty((0, springData.shape[1]))

# # # print(comemonSprings)
# # # best spring is least stiff (flattest torque)
# # bestSpring = commonSprings[np.argmin(commonSprings[:,2])]

# # # calculate torque at contact
# # initTorques = np.array([torque-bestSpring[2]*10 for torque in torques])
# # print(f"torque at contact: {initTorques}")

# # print(f"spring with flattest torque: {bestSpring}")

# # # calculate preload angle for torque at contact
# # preloadAngls = np.array([torque/bestSpring[2] for torque in initTorques])
# # print(f"required preload angle at each joint in degrees: {preloadAngls}")

# # # S1.reinit()
# # # structure1 = StrucMatrix(S=S1,name='structure')
# # # structure2 = StrucMatrix(S=S1,name='structure')
# # # structure3 = StrucMatrix(S=S1,name='structure')
# # # print(structure())
# # # print(structure.validity)
# # # print(S1.biasForceSpace)
# # # print(S1())
# # # S1.plotCapability(showBool = True, colorOverride = 'xkcd:Blue')