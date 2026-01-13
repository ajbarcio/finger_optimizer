import numpy as np
from finger import *
from strucMatrices import *
from numpy import pi
from matplotlib import pyplot as plt
from utils import hArray, ee_func

np.set_printoptions(precision=4, suppress=True)

testFinger = Finger(secondaryDev, [1.4,1.4,1.2])

q = 0
testF = [0,3,0]

S = testFinger.structure([q]*3)
testFinger.structure.controllability([q]*3)
controllability = testFinger.structure.controllability
# plt.show()

# print('\t' + str(a).replace('\n', '\n\t'))
print(hArray(S, "Structure:"))
# print("structure", S)
print("Validity:",controllability.nullSpaceCriterion, controllability.rankCriterion)
print(hArray(controllability.biasForceSpace, "Bias Force Direction:"))

grip = testFinger.tip_wrench_at_pose_to_grip([q]*testFinger.numJoints, testF)

print(hArray(testFinger.get_jacobian_at_pose([q]*testFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF}:"))

testFinger = Finger(secondaryDev, [1.4,1.4,0.6])
grip = testFinger.tip_wrench_at_pose_to_grip([q]*testFinger.numJoints, testF)

print(hArray(testFinger.get_jacobian_at_pose([q]*testFinger.numJoints), "J:"))
print(hArray(grip, f"resulting torques for F={testF}:"))

testFinger = Finger(secondaryDev, [1.4,1.4,1.2])
testGrasp = testFinger.grasp([testF]*testFinger.numJoints, [q]*testFinger.numJoints)

print("starting grip test")
if hasattr(ee_func, "_called3"):
    del ee_func._called3
if hasattr(ee_func, "_called2"):
    del ee_func._called2

print(testFinger.structure([q]*testFinger.numJoints))

grip = testFinger.grasp_to_grip(testGrasp)
print(hArray(grip, f"resulting torques for uniform normal grasp:"))
testGrasp.frame = "EE"
grip = testFinger.grasp_to_grip(testGrasp)
print(hArray(grip, f"resulting torques for uniform normal grasp (EE Frame):"))
tens = testFinger.grip_to_tensions([q]*testFinger.numJoints, grip)
print(hArray(tens, f"best case tensions for uniform normal grasp (EE Frame):"))

print("------------------------------")
F = [0,5,0]
q = 0
print(testFinger.structure([q]*testFinger.numJoints))
print("grip", testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE")))
print(hArray(testFinger.grip_to_tensions([q]*testFinger.numJoints, testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))), "Best Case Tensions"))

F = [0,5,0]
q = np.pi/4
print(testFinger.structure([q]*testFinger.numJoints))
print("grip", testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE")))
print(hArray(testFinger.grip_to_tensions([q]*testFinger.numJoints, testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))), "Best Case Tensions"))

F = [0,5,0]
q = np.pi/2
print(testFinger.structure([q]*testFinger.numJoints))
print("grip", testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE")))
print(hArray(testFinger.grip_to_tensions([q]*testFinger.numJoints, testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))), "Best Case Tensions"))
print("------------------------------")

qs = np.linspace(0,np.pi/2,75)
tvecs = []
tvecs2 = []
for q in qs:
    tensions  = testFinger.grip_to_tensions([q]*testFinger.numJoints,  testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE")))
    tensions2 = testFinger.grip_to_tensions([q]*testFinger.numJoints, -testFinger.grasp_to_grip(testFinger.grasp([F]*testFinger.numJoints, [q]*testFinger.numJoints, frame="EE"))*0.25)
    
    tvecs.append(tensions)
    tvecs2.append(tensions2)
plt.plot(qs, tvecs)
plt.figure()
plt.plot(qs, tvecs2)
plt.show()

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