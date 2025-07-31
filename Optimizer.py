import numpy as np
import scipy as sp
import warnings
from matplotlib import pyplot as plt
from strucMatrices import Constraint, StrucMatrix, centeredType1, centeredType2, centeredType3, naiiveAmbrose, quasiHollow, test
from utils import nullity
from numpy.linalg import matrix_rank as rank

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

def optimize_filter(D, res0, constraints=[]):
    overallBest = 0
    bestS = None
    valids = []
    # print("constraints passed to parent function,", constraints)
    for a in range(res0):
        for b in range(res0):
            for c in range(res0):
                for d in range(res0):
                    for e in range(res0):
                        for f in range(res0):
                            for g in range(res0):
                                for h in range(res0):
                                    for i in range(res0):
                                        r0 = [a,b,c,d,e,f,g,h,i]
                                        r0 = [x/res0+1/res0 for x in r0]
                                        R = r_from_vector(r0, D)
                                        S = StrucMatrix(R, D, constraints=constraints, name=str((a,b,c,d,e,f,g,h,i)))
                                        # print(S())
                                        if S.validity:
                                            valids.append(S)
                                            strength = S.maxGrip()
                                            # print(S.pulleyVariation())
                                            # print(strength)
                                            if strength>=overallBest:
                                                overallBest = strength
                                                bestS = S.S
    if bestS is None or len(valids)==0:
        warnings.warn(f"No valid solutions found for discrete fingers of {res0} radii")
        return 0, 0, []
    return overallBest, bestS, valids

def r_from_vector(r_vec, D):
    R = D*D
    R = np.array(R, dtype=float)
    indices = np.array(np.nonzero(R))
    for i in range(len(r_vec)):
        R[indices[0,i],indices[1,i]] = r_vec[i]
        # print(r_vec[i], R[indices[0,i],indices[1,i]])
    return R

def main():
    # test3dofn1()
    # testFeasibility()
    S = centeredType1
    necessaryGrasps = np.array([[1,0,0],[-.25,0,0],[0,-.25,0]])
    constraints = []
    for grasp in necessaryGrasps:
        constraints.append(Constraint(StrucMatrix.contains, [grasp]))
    # constraint = Constraint(StrucMatrix.contains, [np.array([1,0,0])])
    # constraint = Constraint(StrucMatrix.contains, [np.array([-.25,0,0])])
    D = S.D
    bestGrip, bestS, valids = optimize_filter(D, 2, constraints)
    print(bestGrip)
    print(bestS)
    print(f"there are {len(valids)} valid structures")
    bestS = StrucMatrix(S=bestS)
    bestS.plotCapability()
    for grasp in necessaryGrasps:
        bestS.plotGrasp(grasp)
    plt.show()
if __name__ == "__main__":
    main()