import numpy as np
import scipy as sp
import warnings
import itertools

from matplotlib import pyplot as plt
from strucMatrices import Constraint, StrucMatrix, r_from_vector, centeredType1, centeredType2, centeredType3, naiiveAmbrose, quasiHollow, diagonal, test, balancedType1, individualType1
from utils import nullity, hsv_to_rgb
from numpy.linalg import matrix_rank as rank
from scipy.optimize import minimize

import time

def finger3Space():
    R = np.array([[1,1,1,1],
                  [1,1,1,1],
                  [1,1,1,1]])
    D = np.array([[1,1,1,-1],
                  [0,1,1,-1],
                  [0,0,1,-1]])
    R2 = np.array([[0.5,0.5,0.5,0.5],
                   [1,0.5,0.5,0.5],
                   [1,1,0.5,0.5]])
    S1 = R*D
    S2 = R2*D
    S2 = S2/np.max(S2)
    print(S1, S2)
    print(np.linalg.norm(S1)-np.linalg.norm(S2))
    print(np.linalg.norm(S1-S2))

def test3dofn1():
    # S = centeredType1
    # S = centeredType2
    # S = centeredType3
    # S = naiiveAmbrose
    # S = centeredType2
    S = StrucMatrix(S = np.array([[1,-0.5,1  ,-0.5],
                                  [0,-0.5,0.5,-0.5],
                                  [0,0   ,1  ,-0.5]]))
    print(S())
    S.isValid(suppress=False)
    print(S.validity)
    print(S.biasForceSpace)
    S.plotCapability()
    print(S.independentJointCapabilities())
    # print(S.contains([0,0,0]))
    # print(S.contains([.1,0,0]))
    # print(S.contains([.1496,0,0]))
    # print(S.contains([-.1496,0,0]))
    # S.plotGrasp([0.5,0,0])
    # S.plotGrasp([2,0,0])

    # print(S.contains([0.5,0,0]))
    # print(S.contains([2,0,0]))

    plt.show()

def testVariable():
    S1 = centeredType1.D
    S2 = centeredType2.D
    S3 = centeredType3.D
    r = .1496
    D = np.array([[1,1,1,-1],
                  [1,1,1,-1],
                  [1,1,1,-1]])
    rs = np.linspace(0,r,6)
    differences = []
    colors = []
    for r1 in rs:
        for r2 in rs:
            for r3 in rs:
                print(r1, r2, r3)
                R = np.array([[r1,r,r,r],
                              [r,r2,r,r],
                              [r,r,r3,r]])
                S = StrucMatrix(R, D)
                Sn = S()/np.max(S())
                # color = hsv_to_rgb(r1/r,r2/r,r3/r)
                color = [r1/r,r2/r,r3/r]
                colors.append(color)
                S.plotCapability(colorOverride=color)
                differences.append([np.linalg.norm(Sn)-np.linalg.norm(S1),
                                    np.linalg.norm(Sn)-np.linalg.norm(S2),
                                    np.linalg.norm(Sn)-np.linalg.norm(S3)])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    differences = np.array(differences)
    ax.scatter(*differences.T, c=colors)
    plt.show()

def testFeasibility():
    S = test
    # print(S.boundaryGrasps)
    matrixRank = rank(S.D)
    print(matrixRank)
    print(S.validity)
    print(S.biasForceSpace)
    capabilitySpace = S.boundaryGrasps.T
    dimensionality = rank(capabilitySpace)
    print(dimensionality)
    S.plotCapability(showBool=True)

def optimize_filter(D, res0, constraints=[], useZero=False, suppress=False):
    overallBest = 0
    bestS = None
    leastVariation = 1000
    valids = []
    equivalents = []
    
    min = 1.0/res0 if not useZero else 0
    vars = len(D[np.nonzero(D)])
    vals = [float(x) for x in list(np.linspace(min,1,res0+1 if useZero else res0))]
    rngs = [vals]*vars
    # vcts = list(itertools.product(*rngs))
    total = (res0+1 if useZero else res0)**vars
    i = 0
    print('about to iterate')
    for vct in itertools.product(*rngs):
        try:
            vct = list(vct)
            R = r_from_vector(vct, D)
            S = StrucMatrix(R, D, constraints=constraints)
            if S.validity:
                valids.append(S)
                strength = S.maxGrip()
                variation = S.pulleyVariation()
                # print(S.pulleyVariation())
                # print(strength)
                if not suppress: print("found valid")
                if strength>=overallBest:
                    overallBest = strength
                    bestS = S.S
                    if not suppress: print("found new best")
                if strength==overallBest:
                    if variation<leastVariation:
                        bestS = S.S
                    equivalents.append(S)
                    if not suppress: print("found equivalent")
            if (not i % 100) or (not i % total):
                print(f"{i/total*100:4.1f}% done {i}/{total}", end='\r')
        except KeyboardInterrupt:
            return overallBest, bestS, valids, equivalents
        i+=1

    if bestS is None or len(valids)==0:
        warnings.warn(f"No valid solutions found for discrete fingers of {res0} radii")
        return 0, 0, [], []
    return overallBest, bestS, valids, equivalents

def testOptimizer():
    # test3dofn1()
    # testFeasibility()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
#     [[ 0.944 -0.47   0.844 -0.878]
#  [ 0.    -0.243  1.078 -0.85 ]
#  [ 0.     0.     0.974 -0.974]]
    S = centeredType2
    # S = StrucMatrix(S=np.array([[ 0.944, -0.47 ,  0.844, -0.878],
    #                             [ 0.   , -0.243,  1.078, -0.85 ],
    #                             [ 0.   ,  0.   ,  0.974, -0.974]]))
    necessaryGrasps = np.array([[1,0,0],[-.25,0,0],[0,-.25,0],[0,.25,0],[0,0,0.5],[0,0,-0.5]])
    constraints = []
    # for grasp in necessaryGrasps:
    #     constraints.append(Constraint(StrucMatrix.contains, [grasp]))
    # constraint = Constraint(StrucMatrix.contains, [np.array([1,0,0])])
    # constraint = Constraint(StrucMatrix.contains, [np.array([-.25,0,0])])
    D = S.D
    print('calling optimizer')
    # bestGrip, bestS, valids, equivalents = optimize_filter(D, 5, useZero=False, suppress=True)
    bestR, bestGrip = S.optimizer()
    print(bestGrip)
    # print(f"there are {len(valids)} valid structures")
    # print(f"there are {len(equivalents)} equivalent structures")
    bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, name='best')
    print(np.array2string(bestS(), precision=3, suppress_small=True))

    bestS.plotCapability(colorOverride='xkcd:blue')
    print(bestS.independentJointCapabilities())
    # print(bestS.jointCapability(0), bestS.jointCapability(1), bestS.jointCapability(2))
    print(bestS.validity)
    # for equivalent in equivalents:
    #     equivalent.plotCapability()
    # for grasp in necessaryGrasps:
    #     bestS.plotGrasp(grasp)
    plt.show()

def testJointCapability():
    S = centeredType1
    capabilities = S.independentJointCapabilities()
    axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
    S.plotCapability()
    for j in range(S.numJoints):
        axis = axes[j]
        S.plotGrasp(axis*capabilities[j,0])
        S.plotGrasp(axis*capabilities[j,1])
    fullStrengths = [S.maxExtn(), S.maxGrip()]
    print(fullStrengths)
    plt.show()
    
    # S.plotGrasp([joint0Capability[1],0,0])
    # S.plotCapability(True)

def main():
    # testVariable()
    testOptimizer()
    # finger3Space()
    # testJointCapability()
    # test3dofn1()

if __name__ == "__main__":
    main()