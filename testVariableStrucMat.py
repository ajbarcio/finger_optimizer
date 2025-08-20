import numpy as np
from matplotlib import pyplot as plt
from strucMatrices import VariableStrucMatrix


D = np.array([[1,1,1,-1],
              [1,1,1,-1],
              [1,1,1,-1]])

R = np.array([[np.nan,1     ,1     ,1],
              [1     ,np.nan,1     ,1],
              [1     ,1     ,np.nan,1]])
skinnyLegend = VariableStrucMatrix(R, D, ranges=[(0,1)]*np.sum(np.isnan(R)), name='Skinny Legend')

D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])

R = np.array([[np.nan,np.nan,np.nan,1],
              [0     ,np.nan,np.nan,1],
              [0     ,0     ,np.nan,1]])
theAmbrose = VariableStrucMatrix(R, D, ranges=[(.1,1)]*np.sum(np.isnan(R)), name='The Ambrose')

print(skinnyLegend)
print(theAmbrose)

print(skinnyLegend.constraints)
print(theAmbrose.constraints)
# add a grasp at full extension:
skinnyLegend.add_grasp([0,0,0], [1,0.5,0.25], type='ineq')
theAmbrose.add_grasp([0,0,0], [1,0.5,0.25], type='ineq')
# add a grasp at full flexionf:
skinnyLegend.add_grasp([np.pi/2,np.pi/2,np.pi/2], [1,0.5,0.25], type='ineq')
theAmbrose.add_grasp([np.pi/2,np.pi/2,np.pi/2], [1,0.5,0.25], type='ineq')

rtestSL = skinnyLegend.flatten_r_matrix()
# rtestTA = theAmbrose.flatten_r_matrix()

for constraintContainer in skinnyLegend.constraints:
    result = constraintContainer.constraint.fun(rtestSL)
    print("Constraint value at x_test:", result)
    print("Lower bound:", constraintContainer.constraint.lb, "Upper bound:", constraintContainer.constraint.ub)

# Test that the NonlinearConstraints work properly


# print(theAmbrose.S([0]*theAmbrose.numJoints))
# print(theAmbrose.S([np.pi/2]*theAmbrose.numJoints))

# # # print(np.array(test.variablePulleys))
# # # print(test.j0t0r(0))
# # print(skinnyLegend.extS)
# # # print(skinnyLegend([np.pi/2,np.pi/2,np.pi/2]))
# # print(skinnyLegend.flatten_r_matrix())
# # print(skinnyLegend.reinit(skinnyLegend.flatten_r_matrix()/2))

# # print(skinnyLegend.extS)

# print(f"overall valid at THETA={[0]*skinnyLegend.numJoints}? {skinnyLegend.controllability([0]*skinnyLegend.numJoints)}")
# print(f"rank valid? {skinnyLegend.controllability.rankCondition}")
# print(f"null space valid? {skinnyLegend.controllability.nullSpaceCondition}")
# print(f"bias force direction: {skinnyLegend.controllability.biasForceSpace.T}")
# print(f"overall valid at THETA={[np.pi/2]*skinnyLegend.numJoints}? {skinnyLegend.controllability([np.pi/2]*skinnyLegend.numJoints)}")
# print(f"rank valid? {skinnyLegend.controllability.rankCondition}")
# print(f"null space valid? {skinnyLegend.controllability.nullSpaceCondition}")
# print(f"bias force direction: {skinnyLegend.controllability.biasForceSpace.T}")

# skinnyLegend.plotCapability([0]*skinnyLegend.numJoints)
# skinnyLegend.plotCapability([np.pi/2]*skinnyLegend.numJoints)
# # skinnyLegend.plotCapabilityAcrossAllGrasps()
# theAmbrose.plotCapability([0]*skinnyLegend.numJoints)
# theAmbrose.plotCapability([np.pi/2]*skinnyLegend.numJoints)
# # theAmbrose.plotCapabilityAcrossAllGrasps()

# # resls = np.logspace(0,3,5)
# # resls = np.round(resls).astype(int)
# # print(resls)
# # pSkinnys = []
# # pAmbroses = []
# # for resl in resls:
# #     print(resl)
# #     percentSkinny = skinnyLegend.plotCapabilityAcrossAllGrasps(resl=resl, showBool=False)
# #     percentAmbrose = theAmbrose.plotCapabilityAcrossAllGrasps(resl=resl, showBool=False)
# #     pSkinnys.append(percentSkinny)
# #     pAmbroses.append(percentAmbrose)

# # plt.close('all')
# # plt.plot(resls, pSkinnys)
# # plt.plot(resls, pAmbroses)

# plt.show()