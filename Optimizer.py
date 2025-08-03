import numpy as np
import scipy as sp
import warnings
import itertools
from matplotlib import pyplot as plt
from strucMatrices import Constraint, StrucMatrix, centeredType1, centeredType2, centeredType3, naiiveAmbrose, quasiHollow, test
from utils import nullity, hsv_to_rgb
from numpy.linalg import matrix_rank as rank

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
    S = naiiveAmbrose
    # S = quasiHollow
    print(S())
    S.isValid(suppress=False)
    print(S.validity)
    print(S.biasForceSpace)
    S.plotCapability()
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

def r_from_vector(r_vec, D):
    R = D*D
    R = np.array(R, dtype=float)
    indices = np.array(np.nonzero(R))
    for i in range(len(r_vec)):
        R[indices[0,i],indices[1,i]] = r_vec[i]
        # print(r_vec[i], R[indices[0,i],indices[1,i]])
    return R

def testOptimizer():
    # test3dofn1()
    # testFeasibility()
    S = centeredType2
    necessaryGrasps = np.array([[1,0,0],[-.25,0,0],[0,-.25,0],[0,.25,0],[0,0,0.5],[0,0,-0.5]])
    constraints = []
    for grasp in necessaryGrasps:
        constraints.append(Constraint(StrucMatrix.contains, [grasp]))
    # constraint = Constraint(StrucMatrix.contains, [np.array([1,0,0])])
    # constraint = Constraint(StrucMatrix.contains, [np.array([-.25,0,0])])
    D = S.D
    print('calling optimizer')
    bestGrip, bestS, valids, equivalents = optimize_filter(D, 5, useZero=False, suppress=True)
    print(bestGrip)
    print(bestS)
    print(f"there are {len(valids)} valid structures")
    print(f"there are {len(equivalents)} equivalent structures")
    bestS = StrucMatrix(S=bestS)
    bestS.plotCapability()
    # for equivalent in equivalents:
    #     equivalent.plotCapability()
    # for grasp in necessaryGrasps:
    #     bestS.plotGrasp(grasp)
    plt.show()

def main():
    # testVariable()
    testOptimizer()
    # finger3Space()

if __name__ == "__main__":
    main()