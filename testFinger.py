import numpy as np
from finger import *
from strucMatrices import *
from numpy import pi
from matplotlib import pyplot as plt

testFinger = Finger(skinnyLegend, [1.375,1.4375,1.23])

q = 0

S = testFinger.structure([q]*3)
testFinger.structure.controllability([q]*3)
controllability = testFinger.structure.controllability
# plt.show()

print("structure", S)
print("Validity:",controllability.nullSpaceCriterion, controllability.rankCriterion)
print("Bias Force Direction:", controllability.biasForceSpace)

grip = testFinger.tip_wrench_at_pose_to_grip([q]*3, [0,6.15,0])

print(VariableStrucMatrix.plot_count)
print(VariableStrucMatrix.figures)
print(VariableStrucMatrix.figures_with_axes)
testFinger.structure.plotCapability([q]*3, colorOverride='xkcd:blue')
print(VariableStrucMatrix.plot_count)
print(VariableStrucMatrix.figures)
print(VariableStrucMatrix.figures_with_axes)
testFinger.structure.plotGrasp([q]*3, grip)
print(VariableStrucMatrix.plot_count)
print(VariableStrucMatrix.figures)
print(VariableStrucMatrix.figures_with_axes)

print("Joint torques:", grip)

tensions, msg, closest = testFinger.grasp_to_tensions([q]*3, grip)

print("Tendon Tensions:", tensions, msg, closest)

testFinger.structure.plotGrasp([q]*3, closest)

plt.show()