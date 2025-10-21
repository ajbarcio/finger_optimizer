import numpy as np
import scipy as sp
import warnings
import itertools
import time

from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank as rank
from scipy.optimize import minimize, NonlinearConstraint

from combinatorics import generate_canonical_well_posed_qutsm, generate_centered_qutsm, generate_rankValid_well_posed_qutsm, generate_valid_dimensional_qutsm
from strucMatrices import GraspConstraintWrapper, Constraint, StrucMatrix, r_from_vector, centeredType1, centeredType2, centeredType3, naiiveAmbrose, quasiHollow, diagonal, Optimus, balancedType1, individualType1, resultant, resultant2, canonA, canonB
from utils import nullity, hsv_to_rgb, intersection_with_orthant
from grasps import generateAllVertices, generateNecessaryVertices

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
    S = Optimus
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
    S = resultant2
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

def testOptimizer2():
    # You will get a lot of useless warnings so kill them
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Start with some initial R matrix (and D matrix)
    S = canonA
    # Run the optimizer
    bestR, bestGrip = S.optimizer2()
    # report out:
    print("best grip in InLbs:", bestGrip)
    bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, name='best')
    print('Optimal Structure:')
    print(np.array2string(bestS(), precision=3, suppress_small=True))

    print('Null Space:')
    print(bestS.biasForceSpace, bestS.biasCondition())

    print('Single-axis joint capabilities')
    print(bestS.independentJointCapabilities())
    #Ensure that the matrix is valid (this should never be false)
    print("controllable:", bestS.validity)
    bestS.plotCapability(colorOverride='xkcd:blue')
    plt.show()

def testOptimizer3():
    # You will get a lot of useless warnings so kill them
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Start with some initial R matrix (and D matrix)
    S = canonB
    necessaryGrasps = np.array([[0.5,0,0],[-0.25,0,0],[0,0.5,0],[0,-0.25,0],[0,0,0.5],[0,0,-0.25]])
    boundaryGrasps = np.array([[0.98,0.98,0.98]])
    graspConstraints = []
    for grasp in necessaryGrasps:
        graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
        S.add_constraint(graspConstraints[-1])
    for grasp in boundaryGrasps:
        graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'eq', grasp))
        S.add_constraint(graspConstraints[-1])
    # Run the optimizer
    bestR, bestCondition = S.optimizer3()
    # report out:
    print("Null Space Condition:", bestCondition)
    bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, name='best')
    print('Null Space:')
    print(bestS.biasForceSpace)
    print('Optimal Structure:')
    print(np.array2string(bestS(), precision=3, suppress_small=True))
    print('bestGrip:', S.maxGrip())

    print('Single-axis joint capabilities')
    print(bestS.independentJointCapabilities())
    #Ensure that the matrix is valid (this should never be false)
    print("controllable:", bestS.validity)
    bestS.plotCapability(colorOverride='xkcd:blue')
    for grasp in necessaryGrasps:
        bestS.plotGrasp(grasp)
    for constraint in graspConstraints:
        print(constraint(bestR))
    plt.show()

def OptimizeAllCanonical():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    startingPoints = generate_canonical_well_posed_qutsm()
    i = 0
    for startingPoint in startingPoints:
        S = StrucMatrix(S=startingPoint, name=f'CanonForm{i}')
        necessaryGrasps = np.array([[0.5,0,0],[-0.25,0,0],[0,0.5,0],[0,-0.25,0],[0,0,0.5],[0,0,-0.25], [.98,.98,.98]])
        graspConstraints = []
        for grasp in necessaryGrasps:
            graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            S.add_constraint(graspConstraints[-1])
        bestR, bestCondition = S.optimizer3()
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, name=f'Best of {i}')
        with open('allUpperTriangularOutput.S', 'a') as f:
            print("Null Space Condition:", bestCondition, file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('bestGrip:', S.maxGrip(), file=f)
            print('Single-axis joint capabilities', file=f)
            print(bestS.independentJointCapabilities(), file=f)
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in necessaryGrasps:
            bestS.plotGrasp(grasp)
        i+=1
    plt.show()

def dimensionalOptimizer():
    # startingPoint = naiiveAmbrose
    # S = StrucMatrix(S=startingPoint, name=f'CanonForm{i}')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    startingPoints = generate_canonical_well_posed_qutsm()
    i = 0
    for startingPoint in startingPoints:
        S = StrucMatrix(S=startingPoint, name=f'CanonForm{i}')
        S.R = S.R / 4
        S.F = np.array([50,50,50,50])
        S.reinit()
        #[11.38,27.00,15.16]
        necessaryGrasps = np.array([[2.5,0,0],[-2.5,0,0],[0,2.5,0],[0,-2.5,0],[0,0,2.5],[0,0,-2.5],[0,0,0]])
        graspConstraints = []
        for grasp in necessaryGrasps:
            graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            S.add_constraint(graspConstraints[-1])
        graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'eq', np.array([24.25,16.005,7.38])))
        # Run the optimizer
        bestR, bestCondition = S.optimizer4()
        print(S.optSuccess)
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, name=f'Best of {i}')
        bestS.F = np.array([50,50,50,50])
        bestS.reinit()
        with open('allUpperTriangularDimensional.S', 'a') as f:
            print(f'Matrix index: {i}', file=f)
            print(f'Optimizer success: {S.optSuccess}')
            print("Null Space Condition:", bestCondition, file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('bestGrip:', S.maxGrip(), file=f)
            print('Single-axis joint capabilities', file=f)
            print(bestS.independentJointCapabilities(), file=f)
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in necessaryGrasps:
            bestS.plotGrasp(grasp)
        i+=1
    plt.show()

def dimensionalOptimizer2():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Dimensional Optimizer", end='\n')
    # startingPoints = [canonB, centeredType2]

    bounds=(0.125,0.4)

    canonical = generate_rankValid_well_posed_qutsm()
    startingPoints = generate_valid_dimensional_qutsm(canonical, bounds)
    startingPoints = [startingPoints[k] for k in [5]]
    i = 0

    necessaryGrasps, _ = generateNecessaryVertices(np.array([1.375,1.4375,1.23]))
    allGrasps = generateAllVertices(np.array([1.375,1.4375,1.23]))
    # graspConstraints = []
    ref = np.array([20.2125,13.3375,6.15])
    # print(necessaryGrasps)
    for startingPoint in startingPoints:

        S = StrucMatrix(S=startingPoint, F = np.array([50,50,50,50]), name=f"dimensional valid {i}", constraints=[])
        # S = StrucMatrix(R=r_from_vector(rnew, S.D), D=S.D, F=np.array([50,50,50,50]),
        #                       name=f"dimensional balanced canonical {i}")
        #[11.38,27.00,15.16]
        # necessaryGrasps = np.array([[2.5,0,0],[-2.5,0,0],[0,2.5,0],[0,-2.5,0],[0,0,2.5],[0,0,-2.5],[0,0,0]])
        graspConstraints=[]
        print(f"this should be empty {S.constraints}")
        print(necessaryGrasps)
        for grasp in necessaryGrasps:
            # print(grasp)
            if all([np.isclose(grasp[m],ref[m],atol=1e-2) for m in range(len(grasp))]):
                graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'eq', grasp))
                print('properly added equality constraint')
            else:
                graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            # S.add_constraint(graspConstraints[-1])
        for constraint in graspConstraints:
            S.add_constraint(constraint)
        print(f"there should be {len(graspConstraints)} grasp constraints")
        print(f"we are passing {len(S.constraints)} at the start of the optimizer")
        # graspConstraints.append(NonlinearConstraintContainer(S.contains_by, 'eq', np.array([24.25,16.005,7.38])))
        # Run the optimizer
        bestR, bestCondition = S.optimizer5(bounds)
        print(f"optimizing {S.name}")
        print(S.optSuccess)
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, F=np.array([50,50,50,50]), name=f'Best of {S.name}')

        with open(f'dimensional_upper_triangular_6.12Lbf_0.125_0.4.S', 'a') as f:
            print(f'Matrix index: {i}', file=f)
            print(f'Optimizer success: {S.optSuccess}')
            print("Null Space Condition:", bestCondition, file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('bestGrip:', bestS.maxGrip(), file=f)
            print('Single-axis joint capabilities', file=f)
            print(bestS.independentJointCapabilities(), file=f)
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in allGrasps:
            bestS.plotGrasp(grasp)
        for grasp in necessaryGrasps:
            bestS.plotGrasp(grasp)
        plt.savefig(f'{bestS.name}_6.12Lbf_0.125_0.4.png')
        if np.max(bestS.R)>np.max(bounds):
            plt.figtext(0.5,-.1, f"only acheivable with {np.max(bestS.R)}")
        i+=1
    plt.show()

def dimensionalOptimizer3():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Dimensional Optimizer", end='\n')
    # startingPoints = [canonB, centeredType2]

    bounds=(0.125,0.4)

    canonical = generate_rankValid_well_posed_qutsm()
    startingPoints = generate_valid_dimensional_qutsm(canonical, bounds)
    # startingPoints = [startingPoints[k] for k in [0]]
    i = 0

    necessaryGrasps, _ = generateNecessaryVertices(np.array([1.375,1.4375,1.23]))
    allGrasps = generateAllVertices(np.array([1.375,1.4375,1.23]))
    # graspConstraints = []
    ref = np.array([20.2125,13.3375,6.15])
    # print(necessaryGrasps)
    for startingPoint in startingPoints:

        S = StrucMatrix(S=startingPoint, F = np.array([50,50,50,50]), name=f"dimensional valid {i}", constraints=[])
        # S = StrucMatrix(R=r_from_vector(rnew, S.D), D=S.D, F=np.array([50,50,50,50]),
        #                       name=f"dimensional balanced canonical {i}")
        #[11.38,27.00,15.16]
        # necessaryGrasps = np.array([[2.5,0,0],[-2.5,0,0],[0,2.5,0],[0,-2.5,0],[0,0,2.5],[0,0,-2.5],[0,0,0]])
        graspConstraints=[]
        print(f"this should be empty {S.constraints}")
        # print(necessaryGrasps)
        for grasp in necessaryGrasps:
            # print(grasp)
            # if all([np.isclose(grasp[m],ref[m],atol=1e-2) for m in range(len(grasp))]):
            #     graspConstraints.append(NonlinearConstraintContainer(S.contains_by, 'eq', grasp))
            #     print('properly added equality constraint')
            # else:
                graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            # S.add_constraint(graspConstraints[-1])
        for constraint in graspConstraints:
            S.add_constraint(constraint)
        print(f"there should be {len(graspConstraints)} grasp constraints")
        print(f"we are passing {len(S.constraints)} at the start of the optimizer")
        # graspConstraints.append(NonlinearConstraintContainer(S.contains_by, 'eq', np.array([24.25,16.005,7.38])))
        # Run the optimizer
        bestR, bestCondition = S.optimizer6(bounds)
        print(f"optimizing {S.name}")
        print(S.optSuccess)
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, F=np.array([50,50,50,50]), name=f'Best of {S.name}')

        with open(f'MAXGRIP_dimensional_upper_triangular_6.12Lbf_0.125_0.4.S2', 'a') as f:
            print(f'Matrix index: {i}', file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print(f'Optimizer success: {S.optSuccess}')
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
                print("", file=f)
            print("Null Space Condition:", bestS.biasCondition(), file=f)
            print("Quad1 Volume:", intersection_with_orthant(bestS.torqueDomainVolume()[0], 1).volume, file=f)
            print('bestGrip:', bestS.maxGrip(), file=f)
            # print('Single-axis joint capabilities', file=f)
            # print(bestS.independentJointCapabilities(), file=f)
            print("", file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in allGrasps:
            bestS.plotGrasp(grasp)
        for grasp in necessaryGrasps:
            bestS.plotGrasp(grasp)
        plt.savefig(f'{bestS.name}_6.12Lbf_0.125_0.4_NoSlack_MAXGRIP_2.png')
        if np.max(bestS.R)>np.max(bounds):
            plt.figtext(0.5,-.1, f"only acheivable with {np.max(bestS.R)}")
        i+=1
    plt.show()


def dimensionalOptimizerGlobal():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Dimensional Optimizer", end='\n')
    # startingPoints = [canonB, centeredType2]
    canonical = generate_rankValid_well_posed_qutsm()
    startingPoints = generate_centered_qutsm(canonical)
    startingPoints = [startingPoints[k] for k in [1,12,14]]
    i = 0
    for startingPoint in startingPoints:

        S = StrucMatrix(S=startingPoint, name=f"balanced canonical {i}")

        necessaryGrasps, _ = generateNecessaryVertices(np.array([1.375,1.4375,1.23]))
        allGrasps = generateAllVertices(np.array([1.375,1.4375,1.23]))
        graspConstraints = []
        ref = np.array([20.2125,13.3775,6.15])
        for grasp in necessaryGrasps:
            if all(np.isclose(grasp[m],ref[m]) for m in range(len(necessaryGrasps))):
                graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'eq', grasp))
            else:
                graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            S.add_constraint(graspConstraints[-1])
        # graspConstraints.append(NonlinearConstraintContainer(S.contains_by, 'eq', np.array([24.25,16.005,7.38])))
        # Run the optimizer
        bestR, bestCondition = S.globalOptimizer()
        print(S.optSuccess)
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, F=np.array([50,50,50,50]), name=f'Best of {S.name}')
        with open('PhysicalUpperTriangularDimensionalGlobal.S', 'w') as f:
            print("")
        with open('PhysicalUpperTriangularDimensionalGlobal.S', 'a') as f:
            print(f'Matrix index: {i}', file=f)
            print(f'Optimizer success: {S.optSuccess}')
            print("Null Space Condition:", bestCondition, file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('bestGrip:', bestS.maxGrip(), file=f)
            print('Single-axis joint capabilities', file=f)
            print(bestS.independentJointCapabilities(), file=f)
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in allGrasps:
            bestS.plotGrasp(grasp)
        for grasp in necessaryGrasps:
            bestS.plotGrasp(grasp)
        i+=1
    plt.show()


def quasiDimensionalOptimizer():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Quasi-Dimensional Optimizer", end='\n')
    # startingPoints = [canonB, centeredType2]
    canonical = generate_rankValid_well_posed_qutsm()
    startingPoints = generate_centered_qutsm(canonical)

    necessaryGrasps, _ = generateNecessaryVertices(np.array([1.375,1.4375,1.23]))
    necessaryGrasps = necessaryGrasps/np.max(necessaryGrasps)
    allGrasps = generateAllVertices(np.array([1.375,1.4375,1.23]))
    allGrasps = allGrasps/np.max(allGrasps)


    # print(allGrasps)
    i = 0
    for startingPoint in startingPoints:
        S = StrucMatrix(S=startingPoint, name=f"QUASIDIMENSIONAL balanced canonical {i}")
        #[11.38,27.00,15.16]
        # necessaryGrasps = np.array([[2.5,0,0],[-2.5,0,0],[0,2.5,0],[0,-2.5,0],[0,0,2.5],[0,0,-2.5],[0,0,0]])

        graspConstraints = []
        for grasp in necessaryGrasps:
            graspConstraints.append(GraspConstraintWrapper(S.contains_by, 'ineq', grasp))
            S.add_constraint(graspConstraints[-1])
        # graspConstraints.append(NonlinearConstraintContainer(S.contains_by, 'eq', np.array([24.25,16.005,7.38])))
        # Run the optimizer
        bestR, bestCondition = S.optimizer4()
        print(S.optSuccess)
        bestS = StrucMatrix(R = r_from_vector(bestR, S.D), D = S.D, F=np.array([1,1,1,1]), name=f'Best of {S.name}')

        with open('PhysicalUpperTriangularNonDimensional.S', 'a') as f:
            print(f'Matrix index: {i}', file=f)
            print(f'Optimizer success: {S.optSuccess}', file=f)
            print("Null Space Condition:", bestCondition, file=f)
            print('Null Space:', file=f)
            print(bestS.biasForceSpace, file=f)
            print('Optimal Structure:', file=f)
            print(np.array2string(bestS(), precision=3, suppress_small=True), file=f)
            print('bestGrip:', S.maxGrip(), file=f)
            print('Single-axis joint capabilities', file=f)
            print(bestS.independentJointCapabilities(), file=f)
            #Ensure that the matrix is valid (this should never be false)
            print("controllable:", bestS.validity, file=f)
            print("torque grasp constraints: (one of these should be near 0)", file=f)
            for constraint in graspConstraints:
                print(constraint(bestR), file=f)
        bestS.plotCapability(colorOverride='xkcd:blue')
        for grasp in allGrasps:
            bestS.plotGrasp(grasp)
        i+=1
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
    # testOptimizer()
    # finger3Space()
    # testJointCapability()
    # test3dofn1()
    # testOptimizer3()
    # OptimizeAllCanonical()
    # quasiDimensionalOptimizer()
    dimensionalOptimizer3()
    # dimensionalOptimizerGlobal()

if __name__ == "__main__":
    main()