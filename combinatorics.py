from utils import *
from strucMatrices import *
import itertools
from scipy.optimize import least_squares
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
obj=StrucMatrix # why on earth is this here, what did I do, what weird merge conflict created this

def identify_strict_sign_central(S: StrucMatrix):
    success = True
    struc = S()
    m = S.numJoints
    n = S.numTendons
    Ds = signings_of_order(m)
    for i in range(len(Ds)):
        check = Ds[i] @ struc
        valid = False
        for j in range(n):
            if np.all(check[:,j] >=0) & np.any(check[:,j] != 0):
                valid = True
        success &= valid
    return success

def identify_sign_central(S: StrucMatrix):
    success = True
    struc = S()
    Ds = strict_signings_of_order(S.numJoints)
    for i in range(len(Ds)):
        check = Ds[i] @ struc
        # print(check)
        # print(np.any(np.all(check >= 0, axis=0)))
        success &= (np.any(np.all(check >= 0, axis=0)))
    return success

def identify_strict_central(S: StrucMatrix):
    struc = S()
    c = [np.sum(struc[i,:]) for i in range(S.numJoints)]
    res = linprog(c)
    if res.status == 3:
        success = False
    else:
        # print(res.fun)
        success = True
    return success

def generate_all_unique(shape):
    # D = np.array([[1,1,1,1],
    #               [1,1,1,1],
    #               [1,1,1,1]])
    D = np.ones(shape)
    signs = [0,-1,1]

    positions = np.argwhere(D != 0)
    total = len(signs) ** len(positions)
    allSM = []

    for i, mat  in enumerate(generate_matrices_from_pattern(D, signs)):
        try:
            S = obj(S=mat)
            allSM.append(S.S)
            print(f'trying {i:3d} of {total}', end='\r')
        except KeyboardInterrupt:
            break
    print("                                 ", end='\r')

    uniqueSM = remove_isomorphic(allSM)
    length = len(uniqueSM)
    while True:
        uniqueSM = remove_isomorphic(uniqueSM)
        if length == len(uniqueSM):
            break
        else:
            length = len(uniqueSM)
    # print(f"there are {len(uniqueQUTSM)} unique qutsm")
    # for i in range(len(uniqueSM)):
    #     print(uniqueSM[i])

    return uniqueSM

def select_all_possible(S_):
    possibleStructures = []
    i=0
    for S in S_:
        i+=1
        # for each column:
        success = True
        for j in range(S.shape[1]):
            zero_found = False
            # for each value:
            for i in range(S.shape[0]):
                if S[i,j] == 0:
                    zero_found = True
                elif zero_found:
                    success = False
        if success:
            possibleStructures.append(S)

    return possibleStructures


def generate_valid_dimensional_qutsm(S_, bounds):
    def evenness(radii, D):
        R = r_from_vector(radii, D)
        S = D * R
        N = null_space(S)
        return (N-np.array([[0.5],[0.5],[0.5],[0.5]])).flatten()

    def valid_debug_callback(x):
        if any([y<0 for y in x]):
            print("INVALID")
            raise StopIteration

    validStructures = []
    i=0
    for S in S_:
        i+=1
        print(f"trying to even out {i}", end="\r")
        Struc = obj(S=S)
        if not Struc.validity:
            D = Struc.D
            mask = D != 0
            positions = np.argwhere(mask)
            nRadii = positions.shape[0]
            radii = np.ones(nRadii)*(np.max(bounds)+np.min(bounds))/2
            # result = least_squares(evenness, radii, bounds=bounds, args=[D], callback=valid_debug_callback)
            result = least_squares(evenness, radii, bounds=bounds, args=[D])
            evenRadii = result.x
            if not (any([np.isclose(r,0,atol=1e-4) for r in evenRadii])) and \
               not (any([np.min(bounds) < x > np.max(bounds) for x in evenRadii])):
                R = r_from_vector(evenRadii, D)
                SValid = obj(R, D)
                if SValid.validity:
                    validStructures.append(SValid.S)
            else:
                # print(f"\n No solution found for {i}", end='\n')
                pass
        else:
            validStructures.append(Struc.S*np.max(bounds)/np.max(Struc.S))
    print("                                   ", end="\r")
    # successes = len(evenStructures)
    # print(f"there were {successes} successes")
    return validStructures

def generate_centered_qutsm(S_):
    def evenness(radii, D):
        R = r_from_vector(radii, D)
        # print("R", R)
        S = D * R
        # print("S", S)
        N = null_space(S)
        # print(N)
        return (N-np.array([[0.5],[0.5],[0.5],[0.5]])).flatten()

    def valid_debug_callback(x):
        if any([y<0 for y in x]):
            print("INVALID")
            raise StopIteration

    evenStructures = []
    i=0
    print("starting for loop")
    for S in S_:
        i+=1
        print(f"trying to even out {i}") #, end="\r")
        Struc = obj(S=S)
        # if not Struc.validity:
        D = Struc.D
        # print(D)
        mask = D != 0
        positions = np.argwhere(mask)
        nRadii = positions.shape[0]
        radii = np.ones(nRadii)*0.5
        R = r_from_vector(radii, D)
        S = D*R
        N = null_space(S)
        if N.shape[1] > 1:
            radii = Struc.flatten_r_matrix()
        print(radii)
        result = least_squares(evenness, radii, bounds=(0,1), args=[D], callback=valid_debug_callback)
        evenRadii = result.x/np.max(result.x)
        if np.isclose(result.cost,0) or not (any([np.isclose(r,0,atol=1e-4) for r in evenRadii])):
            R = r_from_vector(evenRadii, D)
            evenStructures.append(obj(R=R, D=D).S)
        else:
            # print(f"\n No solution found for {i}", end='\n')
            pass
        # else:
        #     evenStructures.append(Struc.S)
    print("                                   ", end="\r")
    # successes = len(evenStructures)
    # print(f"there were {successes} successes")
    return evenStructures

def find_centerable_qutsm(S_):
    def evenness(radii, D):
        R = r_from_vector(radii, D)
        S = D * R
        N = null_space(S)
        return (N-np.array([[0.5],[0.5],[0.5],[0.5]])).flatten()

    def valid_debug_callback(x):
        if any([y<0 for y in x]):
            print("INVALID")
            raise StopIteration

    centerableStructures = []
    i=0
    for S in S_:
        i+=1
        print(f"trying to even out {i}", end="\r")
        Struc = obj(S=S)
        if not Struc.validity:
            D = Struc.D
            mask = D != 0
            positions = np.argwhere(mask)
            nRadii = positions.shape[0]
            radii = np.ones(nRadii)*0.5
            result = least_squares(evenness, radii, bounds=(0,1), args=[D], callback=valid_debug_callback)
            evenRadii = result.x/np.max(result.x)
            if result.success:
                R = r_from_vector(evenRadii, D)
                Snew = obj(R=R, D=D)
                # if Snew.validity:
                centerableStructures.append(obj(R=R, D=D).S)
        else:
            centerableStructures.append(Struc)
    print("                                   ", end="\r")
    # successes = len(evenStructures)
    # print(f"there were {successes} successes")
    return centerableStructures

def generate_uniformly_valid_qutsm():

    D = np.array([[1,1,1,1],
                [0,1,1,1],
                [0,0,1,1]])
    signs = [-1,1]
    halfValids = []
    fullyValids = []

    positions = np.argwhere(D != 0)
    total = len(signs) ** len(positions)

    for i, mat  in enumerate(generate_matrices_from_pattern(D, signs)):
        S = obj(S=mat)
        if S.rankCondition:
            print(f'trying {i:6d} of {total}', end='\r')
            # # print(S())
            halfValids.append(S())
            if S.validity:
                fullyValids.append(S())
        else:
            print(f'trying {i:6d} of {total}', end='\r')
    print("                                 ", end='\r')

    fullValidUnique = remove_isomorphic_QUTSM(fullyValids)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    fullValidUnique = remove_isomorphic_QUTSM(fullValidUnique)
    return fullValidUnique

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
        S = obj(S=mat)
        if S.rankCondition:
            print(f'trying {i:6d} of {total}', end='\r')
            # # print(S())
            halfValids.append(S())
            if S.validity:
                fullyValids.append(S())
        else:
            print(f'trying {i:6d} of {total}', end='\r')
    print("                                 ", end='\r')

    fullValidUnique = remove_isomorphic_QUTSM(fullyValids)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    fullValidUnique = remove_isomorphic_QUTSM(fullValidUnique)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    unique = remove_isomorphic_QUTSM(halfValids)

    jointUniformValids = []
    for m in unique:
        R1 = np.diag([3,2,1])
        R2 = np.diag([1,2,3])
        newR1 = R1 @ m
        newR2 = R2 @ m
        S1 = obj(S=newR1)
        S2 = obj(S=newR2)
        if S1.validity or S2.validity:
            jointUniformValids.append(S1.D)

    extensionUniformValids = []
    for i, m in enumerate(unique):
        result = np.eye(3) @ m
        result[result < 0] = -0.5
        S = obj(S=result)
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
        S1 = obj(S=result1)
        S2 = obj(S=result2)
        if S1.validity or S2.validity:
            jointAndExtensionUniformValids.append(S1.D)

    allValids = fullValidUnique+jointUniformValids+extensionUniformValids+jointAndExtensionUniformValids
    allUniqueValids = remove_isomorphic_QUTSM(allValids)

    for i, m in enumerate(allUniqueValids):
        print(f"well-posing {i} of {len(allUniqueValids)}", end="\r")
        allVariants = generate_structure_matrix_variants(m)
        success = False
        graspConditionAcheived = False
        # print(i)
        for variant in allVariants:
            S = obj(S=variant)
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
    print("                                                                                                                ", end="\r")

    # allUniqueValids = remove_isomorphic(allUniqueValids)

    return allUniqueValids

def generate_rankValid_qutsm():

    D = np.array([[1,1,1,1],
                [0,1,1,1],
                [0,0,1,1]])
    signs = [-1,1]
    halfValids = []
    fullyValids = []

    positions = np.argwhere(D != 0)
    total = len(signs) ** len(positions)

    for i, mat  in enumerate(generate_matrices_from_pattern(D, signs)):
        S = obj(S=mat)
        if S.rankCondition:
            print(f'trying {i:6d} of {total}', end='\r')
            # # print(S())
            halfValids.append(S())
            if S.validity:
                fullyValids.append(S())
        else:
            print(f'trying {i:6d} of {total}', end='\r')
    print("                                 ", end='\r')

    unique = remove_isomorphic_QUTSM(halfValids)
    allUniqueValids = remove_isomorphic_QUTSM(unique)

    return allUniqueValids

def generate_rankValid_well_posed_qutsm():

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
        S = obj(S=mat)
        if S.rankCondition:
            print(f'trying {i:6d} of {total}', end='\r')
            # # print(S())
            halfValids.append(S())
            if S.validity:
                fullyValids.append(S())
        else:
            print(f'trying {i:6d} of {total}', end='\r')
    print("                                 ", end='\r')

    fullValidUnique = remove_isomorphic_QUTSM(fullyValids)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    fullValidUnique = remove_isomorphic_QUTSM(fullValidUnique)
    # for i, m in enumerate(fullValidUnique):
    #     print(f"matrix {i}:")
    #     print(m, '\n')
    unique = remove_isomorphic_QUTSM(halfValids)

    allValids = unique+fullValidUnique
    allUniqueValids = remove_isomorphic_QUTSM(allValids)

    for i, m in enumerate(allUniqueValids):
        print(f"well-posing {i} of {len(allUniqueValids)}", end="\r")
        allVariants = generate_structure_matrix_variants(m)
        success = False
        graspConditionAcheived = False
        # print(i)
        for variant in allVariants:
            S = obj(S=variant)
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
    print("                                                                                                                ", end="\r")

    # allUniqueValids = remove_isomorphic(allUniqueValids)

    return allUniqueValids

def generate_all_unique_qutsm():

# D = np.array([[1,1,1,1],
#               [1,1,1,1],
#               [1,1,1,1]])
    D = np.array([[1,1,1,1],
                  [0,1,1,1],
                  [0,0,1,1]])
    signs = [-1,1]

    positions = np.argwhere(D != 0)
    total = len(signs) ** len(positions)
    allQUTSM = []

    for i, mat  in enumerate(generate_matrices_from_pattern(D, signs)):
        S = obj(S=mat)
        allQUTSM.append(S.S)
        print(f'trying {i:3d} of {total}', end='\r')
    print("                                 ", end='\r')

    uniqueQUTSM = remove_isomorphic_QUTSM(allQUTSM)
    uniqueQUTSM = remove_isomorphic_QUTSM(uniqueQUTSM)
    # uniqueQUTSM = remove_isomorphic_QUTSM(uniqueQUTSM)
    # print(f"there are {len(uniqueQUTSM)} unique qutsm")

    return uniqueQUTSM

if __name__ == "__main__":

    m = 3
    try:
        uniqueAll = np.load("allUnique3x4.npy", mmap_mode='r')
    except:
        uniqueAll = generate_all_unique([3,4])
        print(len(uniqueAll))
        np.save("allUnique3x4.npy", uniqueAll)
    dense = uniqueAll[np.count_nonzero(uniqueAll == 0, axis=(1,2)) == 0]
    # for i in dense:
    #     print(i)
    print(f'There are all {3**12} possible 3dof n+1 tendon routings:', )
    print(f'{len(uniqueAll)} of them are unique')
    uniqueRankValids = 0
    for i in uniqueAll:
        if matrix_rank(i)==3:
            uniqueRankValids+=1
    print(f'{uniqueRankValids} are likely to be controllable, if the correct radii are chosen')
    # print(len(dense))
    universal = []
    uniform = []
    j=0
    for i in uniqueAll:
        print(j, end="\r")
        if identify_strict_sign_central(StrucMatrix(S=i)):
            # print(i, "True")
            universal.append(i)
        if identify_strict_central(StrucMatrix(S=i)):
            uniform.append(i)
    print(f'{len(uniform)} are controllable if every radius is the same')
    print(f'{len(universal)} of these are ALWAYS controllable, no matter what radii you choose')
    alwaysControllableDecouplable = []
    controllableUniformlyDecouplable = []
    for Sm in universal:
        # create the decouplability matrix
        rows = []
        # for each row  of K_J:
        for i in range(m):
            # and each element in that row above the diagonal
            for j in range(i+1, m):
                # there is a row in M that is the dot product of two rows in S
                rows.append(Sm[i, :] * Sm[j, :])
        # And we want M * K_J = 0, so we need null space of M
        M = np.vstack(rows)
        # instead of manually checking the null space, check for
        # strict centrality and strict sign centrality
        StC = identify_strict_central(StrucMatrix(S=M))
        SSC = identify_strict_sign_central(StrucMatrix(S=M))
        if StC:
            controllableUniformlyDecouplable.append(Sm)
        if SSC:
            alwaysControllableDecouplable.append(Sm)
    print(f'of these inherently controllable SM, {len(controllableUniformlyDecouplable)} are also decouplable with uniform radii')
    print(f'and there are {len(alwaysControllableDecouplable)} which are both decoupleable and controllable regardless of radius choice')
    # allUniqueValids = generate_rankValid_well_posed_qutsm()
    uniqueQUTSM = generate_all_unique_qutsm()
    print()
    print(f"Considering practical issues of 'skipping joints',")
    print()
    constructible = select_all_possible(uniqueAll)
    # for i in constructible:
    #     print(i)
    print(f"only {len(constructible)} of the {len(uniqueAll)} unique matrices are actually possible to construct")
    constructibleRankValids = []
    for i in constructible:
        if matrix_rank(i)==3:
            constructibleRankValids.append(i)
    print(f'and of these, {len(constructibleRankValids)} are likely to be controllable, if the correct radii are chosen')
    universalPractical = []
    uniformPractical = []
    inherentlyDecouplable = []
    j=0
    for i in constructibleRankValids:
        print(j, end="\r")
        if identify_strict_sign_central(StrucMatrix(S=i)):
            # print(i, "True")
            universalPractical.append(i)
        if identify_strict_central(StrucMatrix(S=i)):
            uniformPractical.append(i)
            # create the decouplability matrix
        rows = []
        # for each row  of K_J:
        for k in range(m):
            # and each element in that row above the diagonal
            for l in range(k+1, m):
                # there is a row in M that is the dot product of two rows in S
                rows.append(i[k, :] * i[l, :])
        # And we want M * K_J = 0, so we need null space of M
        M = np.vstack(rows)
        SSC = identify_strict_sign_central(StrucMatrix(S=M))
        if SSC:
            inherentlyDecouplable.append(i)
    print(f'{len(uniformPractical)} of which are controllable for uniform radii and')
    print(f"{len(universalPractical)} of which are 'inherently' controllable:")
    # print(universalPractical)
    for i in universalPractical:
        print(i)
    print(f"There are {len(inherentlyDecouplable)} routings which are practical to produce and also inherently decouplable")
    print()
    print(f"Additionally, there are {len(uniqueQUTSM)} unique upper triangular matrices")
    allDimensionalValids = generate_valid_dimensional_qutsm(uniqueQUTSM, (0.125,0.4))
    print(f'{len(allDimensionalValids)} of which can be made controllable under certain practical constraints on radius size')
    universalQUTSM = []
    uniformQUTSM = []
    j=0
    for i in uniqueQUTSM:
        print(j, end="\r")
        if identify_strict_sign_central(StrucMatrix(S=i)):
            # print(i, "True")
            universalQUTSM.append(i)
        if identify_strict_central(StrucMatrix(S=i)):
            uniformQUTSM.append(i)
    print(f'{len(uniformQUTSM)} of which are controllable for uniform radii and')
    print(f"{len(universalQUTSM)} of which is 'inherently' controllable:")
    print(universalQUTSM)
    universalQUTSM = universalQUTSM[0]
    # create the decouplability matrix
    rows = []
    # for each row  of K_J:
    for i in range(m):
        # and each element in that row above the diagonal
        for j in range(i+1, m):
            # there is a row in M that is the dot product of two rows in S
            rows.append(universalQUTSM[i, :] * universalQUTSM[j, :])
    # And we want M * K_J = 0, so we need null space of M
    M = np.vstack(rows)
    # instead of manually checking the null space, check for
    # strict centrality and strict sign centrality
    StC = identify_strict_central(StrucMatrix(S=M))
    SSC = identify_strict_sign_central(StrucMatrix(S=M))
    if SSC:
        print('man you fucked up somewhere')
    if StC:
        print("This QUTSM which is inherently controllable is also decouplable in the uniform condition")
    else:
        print("for uniform radii, that QUTSM is not decouplable")
    print("")
    print(f"Using another perspective, there are {len(dense)} unique matrices which all terminate at the most distal joint")
    rankValids = []
    for i in dense:
        if matrix_rank(i)==3:
            rankValids.append(i)
    print(f"{len(rankValids)} of these are likely to be controllable with correct choice of radii")
    # print()
    universal = []
    uniform = []
    j=0
    for i in dense:
        print(j, end="\r")
        if identify_strict_sign_central(StrucMatrix(S=i)):
            # print(i, "True")
            universal.append(i)
        if identify_strict_central(StrucMatrix(S=i)):
            uniform.append(i)
    # for i in rankValids:
    #     print(i)
    print(f'and {len(uniform)} are controllable for uniform radius, while')
    print(f"{len(universal)} are controllable regardless of radius choice")
    # print("[[-1 -1  1  1] \
    #         [-1  1 -1  1] \
    #         [ 1 -1 -1  1]]")
    # uniformQUTSM = generate_uniformly_valid_qutsm()
    # print(f"there are {len(uniformQUTSM)} valid qutsm with uniform radii")
    # rankValidQUTSM = generate_rankValid_qutsm()
    # print(f"there are {len(rankValidQUTSM)} unique qutsm which fulfill the rank condition")

    # allCenteredValids = generate_centered_qutsm(uniqueQUTSM)
    # allCenterableValids = find_centerable_qutsm(uniqueQUTSM)
    # if np.array_equal(uniqueQUTSM, rankValidQUTSM):
    #     print("the unique qutsm and rank-valid qutsm are identical")

    # for i, valid in enumerate(allDimensionalValids):
    #     # valid = valid/np.max(valid)
    #     S = obj(S=valid)
    #     print(f"valid matrix {i}")
    #     print(np.array2string(valid, precision=3, suppress_small=True), end="\n")
    #     print(np.max(abs(S.flatten_r_matrix()))/np.min(abs(S.flatten_r_matrix())))
    #     print(S.biasCondition())
    # print(f"there are {len(allDimensionalValids)} valid qutsm within dimensional bounds")
    # print(f"there are {len(allCenterableValids)} centerable qutsm")
    # print(f"there are {len(allCenteredValids)} centered qutsm")
    # for i, valid in enumerate(allCenteredValids):
    #     S = obj(S=valid)
    #     print(f"centered matrix {i}")
    #     print(np.array2string(valid, precision=3, suppress_small=True), end="\n")
    #     print(np.max(abs(S.flatten_r_matrix()))/np.min(abs(S.flatten_r_matrix())))
    #     print(S.biasCondition())
        # print(null_space(valid))
    # i=0
    # for S in allUniqueValids:
    #     print(S)
    #     if any([np.array_equal(M, centeredType1.D) for M in generate_structure_matrix_variants(S)]):
    #         print(f'found type 1 at canon {i}:')
    #         print(centeredType1.D)
    #         print("-----------------")
    #     elif any([np.array_equal(M, centeredType3.D) for M in generate_structure_matrix_variants(S)]):
    #         print(f'found type 3 at canon {i}:')
    #         print(centeredType3.D)
    #         print("-----------------")
    #     elif any([np.array_equal(M, centeredType2.D) for M in generate_structure_matrix_variants(S)]):
    #         print(f'found type 2 at canon {i}:')
    #         print(centeredType2.D)
    #         print("-----------------")
    #     T = StrucMatrix(S=S, name=f'e{i}')
    #     T.plotCapability()
    #     i+=1
    # m = centeredType2.D
    # R1 = np.diag([3,2,1])
    # R2 = np.diag([1,2,3])
    # newR1 = R1 @ m
    # newR2 = R2 @ m
    # result1 = np.array(newR1)
    # result2 = np.array(newR2)
    # result1[result1 < 0] = -4
    # result2[result2 < 0] = -4
    # S1 = StrucMatrix(S=result1)
    # S2 = StrucMatrix(S=result2)
    # if S1.validity or S2.validity:
    #     print("yes")
    # plt.show()