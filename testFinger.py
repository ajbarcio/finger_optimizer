import numpy as np
from finger import *
from strucMatrices import *
from numpy import pi
from matplotlib import pyplot as plt
from utils import hArray, ee_func

np.set_printoptions(precision=4, suppress=True)

quickFinger = Finger(secondaryDev, [1.4,1.4,1.2])
print("THIS SECTION OF PRINT STATEMENTS WORKS ON THE VARIABLE FINGER")

q = 0
testF = [0,5,0]

S = quickFinger.structure([q]*3)
quickFinger.structure.controllability([q]*3)
controllability = quickFinger.structure.controllability
# plt.show()

# print('\t' + str(a).replace('\n', '\n\t'))
print(hArray(S, "Structure:"))
# print("structure", S)
print("Validity:",controllability.nullSpaceCriterion, controllability.rankCriterion)
print(hArray(controllability.biasForceSpace, "Bias Force Direction:"))

grip = quickFinger.tip_wrench_at_pose_to_grip([q]*quickFinger.numJoints, testF)

print(hArray(quickFinger.get_jacobian_at_pose([q]*quickFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF}:"))

quickFinger = Finger(secondaryDev, [1.4,1.4,0.6])
grip = quickFinger.tip_wrench_at_pose_to_grip([q]*quickFinger.numJoints, testF)

print(hArray(quickFinger.get_jacobian_at_pose([q]*quickFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF}:"))

quickFinger = Finger(secondaryDev, [1.4,1.4,1.2])
testGrasp = quickFinger.grasp([testF]*quickFinger.numJoints, [q]*quickFinger.numJoints)

print("starting grip test")
if hasattr(ee_func, "_called3"):
    del ee_func._called3
if hasattr(ee_func, "_called2"):
    del ee_func._called2

print(quickFinger.structure([q]*quickFinger.numJoints))

grip = quickFinger.grasp_to_grip(testGrasp)
print(hArray(grip, f"resulting torques for uniform normal grasp:"))
testGrasp.frame = "EE"
grip = quickFinger.grasp_to_grip(testGrasp)
print(hArray(grip, f"resulting torques for uniform normal grasp (EE Frame):"))
tens = quickFinger.grip_to_tensions([q]*quickFinger.numJoints, grip)
print(hArray(tens, f"best case tensions for uniform normal grasp (EE Frame):"))

print("------------------------------")
F = [0,5,0]
q = 0
print(quickFinger.structure([q]*quickFinger.numJoints))
print("grip", quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE")))
print(hArray(quickFinger.grip_to_tensions([q]*quickFinger.numJoints, quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE"))), "Best Case Tensions"))

F = [0,5,0]
q = np.pi/4
print(quickFinger.structure([q]*quickFinger.numJoints))
print("grip", quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE")))
print(hArray(quickFinger.grip_to_tensions([q]*quickFinger.numJoints, quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE"))), "Best Case Tensions"))

F = [0,5,0]
q = np.pi/2
print(quickFinger.structure([q]*quickFinger.numJoints))
print("grip", quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE")))
print(hArray(quickFinger.grip_to_tensions([q]*quickFinger.numJoints, quickFinger.grasp_to_grip(quickFinger.grasp([F]*quickFinger.numJoints, [q]*quickFinger.numJoints, frame="EE"))), "Best Case Tensions"))
print("------------------------------")

print("MAGNITUDE INCREASES -------------------------")
print(quickFinger.structure.get_magnitude([0]*quickFinger.numJoints))
print(quickFinger.structure.get_magnitude([np.pi/2]*quickFinger.numJoints))

qs = np.linspace(0,np.pi/2,75)
tvecs = []
tvecs2 = []
scales = []
for q in qs:

    grip = quickFinger.tip_wrench_at_pose_to_grip([q]*quickFinger.numJoints, testF, frame="EE")
    # grip = testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))

    tensions  = quickFinger.grip_to_tensions([q]*quickFinger.numJoints,  grip)
    tensions2 = quickFinger.grip_to_tensions([q]*quickFinger.numJoints, -grip*0.25)
    
    tvecs.append(tensions)
    tvecs2.append(tensions2)

    scales.append(quickFinger.structure.get_magnitude([q]*quickFinger.numJoints))
plt.plot(qs, tvecs)
plt.figure()
plt.plot(qs, tvecs2)
plt.figure()
plt.plot(qs, scales)
plt.show()

quickFinger = Finger(inherentFixed,[1.4,1.4,1.2])
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