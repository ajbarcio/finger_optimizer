from utils import *
from strucMatrices import *
import itertools

def generate_canonical_well_posed_qutsm():

# D = np.array([[1,1,1,1],
#               [1,1,1,1],
#               [1,1,1,1]])
    D = np.array([[1,1,1,1],
                [0,1,1,1],
                [0,0,1,1]])
    signs = [-1,1]
    halfValids = []
    fullyValids = []

    positions = np.argwhere(D != 0)
    total = len(signs) ** len(positions)

    for i, mat  in enumerate(generate_matrices_from_pattern(D, signs)):
        S = StrucMatrix(S=mat)
        if S.rankCondition:
            print(f'trying {i:6d} of {total}', end='\r')
            # # print(S())
            halfValids.append(S())
            if S.validity:
                fullyValids.append(S())
        else:
            print(f'trying {i:6d} of {total}', end='\r')
    print("                                 ", end='\r')

    fullValidUnique = remove_isomorphic(fullyValids)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    fullValidUnique = remove_isomorphic(fullValidUnique)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    unique = remove_isomorphic(halfValids)

    jointUniformValids = []
    for m in unique:
        R1 = np.diag([3,2,1])
        R2 = np.diag([1,2,3])
        newR1 = R1 @ m
        newR2 = R2 @ m
        S1 = StrucMatrix(S=newR1)
        S2 = StrucMatrix(S=newR2)
        if S1.validity or S2.validity:
            jointUniformValids.append(S1.D)

    extensionUniformValids = []
    for i, m in enumerate(unique):
        result = np.eye(3) @ m
        result[result < 0] = -0.5
        S = StrucMatrix(S=result)
        if S.validity:
            extensionUniformValids.append(S.D)

    jointAndExtensionUniformValids = []
    for i, m in enumerate(unique):
        R1 = np.diag([3,2,1])
        R2 = np.diag([1,2,3])
        newR1 = R1 @ m
        newR2 = R2 @ m
        result1 = np.array(newR1)
        result2 = np.array(newR2)
        result1[result1 < 0] = -4
        result2[result2 < 0] = -4
        S1 = StrucMatrix(S=result1)
        S2 = StrucMatrix(S=result2)
        if S1.validity or S2.validity:
            jointAndExtensionUniformValids.append(S1.D)

    allValids = fullValidUnique+jointUniformValids+extensionUniformValids+jointAndExtensionUniformValids
    allUniqueValids = remove_isomorphic(allValids)

    for i, m in enumerate(allUniqueValids):
        allVariants = generate_structure_matrix_variants(m)
        success = False
        graspConditionAcheived = False
        # print(i)
        for variant in allVariants:
            S = StrucMatrix(S=variant)
            graspCondition = False
            for grasp in S.boundaryGrasps:
                strength = np.linalg.norm(grasp)
                if intersects_positive_orthant(grasp) and strength>=S.maxGrip():
                    graspCondition = True
            # print(intersection_with_orthant(S.torqueDomainVolume()[0], 1).volume)
            # print([intersection_with_orthant(S.torqueDomainVolume()[0], i).volume for i in np.arange(1,9)])
            volumeCondition = all([intersection_with_orthant(S.torqueDomainVolume()[0], 1).volume>=intersection_with_orthant(S.torqueDomainVolume()[0], i).volume for i in np.arange(1,9)])
            # print(volumeCondition)
            if graspCondition and volumeCondition:
                # print(f"found ideal canonical for {i}")
                canonVariant = variant
                success = True
                break
            if graspCondition:
                # print(f"found max grasp in positive orthant for {i}")
                canonVariant = variant
                success = True
                graspConditionAcheived = True
            if volumeCondition:
                # print(f"found biggest volume in positive orthant for {i}")
                if graspConditionAcheived:
                    pass
                else:
                    success = True
                    canonVariant = variant
        if success:
            allUniqueValids[i] = canonVariant
        if not success:
            print(f"no variant for {i} with best grip in positive orthant OR most volume in positive orthant")
        else:
            pass

    # allUniqueValids = remove_isomorphic(allUniqueValids)

    return allUniqueValids

if __name__ == "__main__":
    allUniqueValids = generate_canonical_well_posed_qutsm()
    i=0
    for S in allUniqueValids:
        print(S)
        if any([np.array_equal(M, centeredType1.D) for M in generate_structure_matrix_variants(S)]):
            print(f'found type 1 at canon {i}:')
            print(centeredType1.D)
            print("-----------------")
        elif any([np.array_equal(M, centeredType3.D) for M in generate_structure_matrix_variants(S)]):
            print(f'found type 3 at canon {i}:')
            print(centeredType3.D)
            print("-----------------")
        elif any([np.array_equal(M, centeredType2.D) for M in generate_structure_matrix_variants(S)]):
            print(f'found type 2 at canon {i}:')
            print(centeredType2.D)
            print("-----------------")
        T = StrucMatrix(S=S, name=f'e{i}')
        T.plotCapability()
        i+=1
    m = centeredType2.D
    R1 = np.diag([3,2,1])
    R2 = np.diag([1,2,3])
    newR1 = R1 @ m
    newR2 = R2 @ m
    result1 = np.array(newR1)
    result2 = np.array(newR2)
    result1[result1 < 0] = -4
    result2[result2 < 0] = -4
    S1 = StrucMatrix(S=result1)
    S2 = StrucMatrix(S=result2)
    if S1.validity or S2.validity:
        print("yes")
    # plt.show()