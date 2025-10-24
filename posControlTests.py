from strucMatrices import *
from utils import *
from scipy.optimize import nnls


np.set_printoptions(5, suppress=True)
# r = 1
# R = np.array([[r,r,r,r],
#               [r,r,r,r],
#               [r,r,r,r]])
# D = np.array([[1,1,1,-1],
#               [0,1,1,-1],
#               [0,0,1,-1]])
amplitude_biases = np.array([1,1,1,1])
# testStructure = StrucMatrix(R,D,name='test')
testStructure = testbedFinger
# testStructure = canonB
rangeSpace = testStructure.positionRange()
print("basis for pos range of structure:")
print( rangeSpace)
print("basis for null space")
print(testStructure.biasForceSpace)
# Then create some coefficients representing the relative magnitudes
# of some trajectory at any time (uniform)
desiredTrajCoeffs = testStructure().T.dot(np.array([1,1,1]).T)
# Perturb for allowances (trying to make certain tendons slack,
# or make others take up more slack)
desiredTrajCoeffs = np.diag(amplitude_biases).dot(desiredTrajCoeffs)
print("relative magnitudes of tendon displacement:", desiredTrajCoeffs)
# calcualte a solution for the desired trajectory in terms of the bases
coeffs, res, rank, s = np.linalg.lstsq(rangeSpace, desiredTrajCoeffs, rcond=None)
print("coefficient for each basis vector:", coeffs)
print("Fully satisfies?", np.isclose(res[0], 0) if np.isclose(res[0], 0) else res[0])

print("--------------------------------------------------------------")
# print(np.linalg.pinv(testStructure()).dot(testStructure()).dot(desiredTrajCoeffs))
# print(np.linalg.pinv(testStructure()).dot(testStructure()).dot(testStructure().T))
print("S*.S (projector)")
print(np.linalg.pinv(testStructure()).dot(testStructure()))
print(testStructure().T.dot(np.linalg.pinv(testStructure().T)))
print("S.S* (identity?)")
print((testStructure()).dot(np.linalg.pinv(testStructure())))


k_p = 1
l_t = np.array([.333,.666,1,-1])*0.05
# l_m = l_t*0.80
l_m = np.array([.333]*4)
# l_m = np.array([-0.25, -0.5,  -1,    1])
l_d = np.linalg.pinv(testStructure()).dot(testStructure()).dot(l_m+k_p*(l_t-l_m))
print(l_t-l_m)
print(l_m+k_p*(l_t-l_m))
print(l_t, l_m, l_d)

# sign flips become unstable

# print(np.linalg.cond(np.eye(4)+np.linalg.pinv(testStructure()).dot(testStructure())))
# print(np.linalg.inv(np.eye(4)+np.linalg.pinv(testStructure()).dot(testStructure())))

# print("--------------------------------------------------------------")
# print(testStructure.biasForceSpace)
# print(np.linalg.pinv(testStructure()))
# print(np.linalg.pinv(testStructure()).dot(testStructure()))
# print(testStructure().dot(np.linalg.pinv(testStructure())))
# print(sp.linalg.null_space(np.linalg.pinv(testStructure())))