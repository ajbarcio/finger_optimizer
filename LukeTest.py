import numpy as np
from finger import *
from strucMatrices import *
from numpy import pi
from matplotlib import pyplot as plt
from utils import hArray, ee_func

np.set_printoptions(precision=4, suppress=True)

testF = [0,20,0]
testFinger = Finger(inherentFixedLuke,[42.074,27.613,20.615])
print("THIS SECTION OF PRINT STATEMENTS WORKS ON A FIXED FINGER")
q = 0

S = testFinger.structure()

print(f"Joint lengths: {testFinger.lengths}, total length: {np.sum(testFinger.lengths)}")
print(hArray(S, "Structure:"))
# print("structure", S)
print(f"this structure matrix has a relative scale of {testFinger.structure.magnitude}")
print(testFinger.structure.S.T @ testFinger.structure.S)
print(np.sqrt(np.linalg.det(testFinger.structure.S.T @ testFinger.structure.S)))


print("Validity:",testFinger.structure.nullSpaceCondition, testFinger.structure.rankCondition)
print(hArray(testFinger.structure.biasForceSpace, "Bias Force Direction:"))

grip = testFinger.tip_wrench_at_pose_to_grip([q]*testFinger.numJoints, testF, frame="EE")

print(hArray(testFinger.get_jacobian_at_pose([q]*testFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF} at tip of finger:"))


minFactor = 1/testFinger.structure.biasCondition()*0.1
print(f"Enforcing minimum tension of {minFactor} based on Null Space Condition of {testFinger.structure.biasCondition()} (10% of max allowable value)")
tens = testFinger.grip_to_tensions([q]*testFinger.numJoints, grip)
print(hArray(tens, f"best case tensions for F={testF} at tip of finger:"))

# print("------------------------------")
# F = [0,5,0]
# q = 0
# print(testFinger.structure())
# print("grip", testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE")))
# print(hArray(testFinger.grip_to_tensions([q]*testFinger.numJoints, testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))), "Best Case Tensions"))

# print(VariableStrucMatrix.plot_count)
# print(VariableStrucMatrix.figures)
# print(VariableStrucMatrix.figures_with_axes)
# testFinger.structure.plotCapability([q]*3, colorOverride='xkcd:blue')
# print(VariableStrucMatrix.plot_count)
# print(VariableStrucMatrix.figures)
# print(VariableStrucMatrix.figures_with_axes)
# testFinger.structure.plotGrasp([q]*3, grip)
# print(VariableStrucMatrix.plot_count)
# print(VariableStrucMatrix.figures)
# print(VariableStrucMatrix.figures_with_axes)

# print("Joint torques:", grip)

# tensions, msg, closest = testFinger.grasp_to_tensions([q]*3, grip)

# print("Tendon Tensions:", tensions, msg, closest)

# testFinger.structure.plotGrasp([q]*3, closest)

# plt.show()