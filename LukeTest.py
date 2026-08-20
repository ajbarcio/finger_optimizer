import numpy as np
from finger import *
from strucMatrices import *
from numpy import pi
from matplotlib import pyplot as plt
from utils import hArray, ee_func

np.set_printoptions(precision=4, suppress=True)

testF = [0,20,0]
quickFinger = Finger(inherentFixedLuke,[42.074,27.613,20.615])
print("THIS SECTION OF PRINT STATEMENTS WORKS ON A FIXED FINGER")
q = 0

S = quickFinger.structure()

print(f"Joint lengths: {quickFinger.lengths}, total length: {np.sum(quickFinger.lengths)}")
print(hArray(S, "Structure:"))
# print("structure", S)
print(f"this structure matrix has a relative scale of {quickFinger.structure.magnitude}")
print(quickFinger.structure.S.T @ quickFinger.structure.S)
print(np.sqrt(np.linalg.det(quickFinger.structure.S.T @ quickFinger.structure.S)))


print("Validity:",quickFinger.structure.nullSpaceCondition, quickFinger.structure.rankCondition)
print(hArray(quickFinger.structure.biasForceSpace, "Bias Force Direction:"))

grip = quickFinger.tip_wrench_at_pose_to_grip([q]*quickFinger.numJoints, testF, frame="EE")

print(hArray(quickFinger.get_jacobian_at_pose([q]*quickFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF} at tip of finger:"))


minFactor = 1/quickFinger.structure.biasCondition()*0.1
print(f"Enforcing minimum tension of {minFactor} based on Null Space Condition of {quickFinger.structure.biasCondition()} (10% of max allowable value)")
tens = quickFinger.grip_to_tensions([q]*quickFinger.numJoints, grip)
print(hArray(tens, f"best case tensions for F={testF} at tip of finger:"))

print(inherentFixedLuke.S)
print(f"magnitude: {quickFinger.structure.get_magnitude()}")
# inherentFixedLuke.R *= 2
inherentFixedLuke.reinit()
print(inherentFixedLuke.S)
print(f"magnitude: {quickFinger.structure.get_magnitude()}")


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