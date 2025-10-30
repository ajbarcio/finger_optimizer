from utils import *
from strucMatrices import *
import itertools
from scipy.optimize import least_squares
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
obj=StrucMatrix # why on earth is this here, what did I do, what weird merge conflict created this

def create_decoupling_matrix(S):
    rows = []
    m, n = S.shape
    M = np.zeros((int(m*(m-1)/2),n))
    # for each row  of K_J:
    for i in range(m):
        # and each element in that row above the diagonal
        for j in range(i+1, m):
            # there is a row in M that is the dot product of two rows in S
            # rows.append(S[i, :] * S[j, :])
            M[i,:]=S[i, :] * S[j, :]
    return M

def test_functional_decoupling_matrix():
    S1 = inherent
    print(S1())
    M = create_decoupling_matrix(S1())
    print(M)
    print(identify_strict_central(obj(S=M), boundsOverride=True))
    S2 = balancedType1
    print(S2())
    M = create_decoupling_matrix(S2())
    print(M)
    print(identify_strict_central(obj(S=M), boundsOverride=True))

    decoupled = select_all_decouplable([S1(), S2(), LED(), diagonal()])
    # print(decoupled)
    for i in decoupled:
        print(i)


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

def identify_strict_central(S: StrucMatrix, boundsOverride=False):

    """
    Dual problem of the minimization for Theorem 2.1 presented by
    Brunaldi and Dahl in Strict Sign-Central Matrices
    """
    A = S()
    n = A.shape[1]
    e = np.ones(n)

    c = np.zeros(n)                 # objective: minimize 0
    # c = np.ones(n)                  # objective: minimize 1-norm of w
    # c = -np.ones(n)                 # objective: maximize 1-norm of w
    A_eq = A                        # equality: A w = -A e
    b_eq = -A @ e
    if boundsOverride:
        bounds = [(None, None)] * n        # allow for the return of any value
    else:
        bounds = [(0, None)] * n        # w >= 0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.x is not None:
        if any(res.x[res.x<0]):
            res.success = False
    # if res.success:
    # print(res.x)
    return res.success, res.x

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
            allSM.append(mat)
            print(f'generating {i:3d} of {total}', end='\r')
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
    for i, SM in enumerate(uniqueSM):
        uniqueSM[i] = triangularize_and_orient(SM)
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

def select_all_controllable(S_):
    controllableStructures = []
    i=0
    for S in S_:
        if matrix_rank(S)==3:
            if identify_strict_central(StrucMatrix(S=S))[0]:
                controllableStructures.append(S)

    return controllableStructures

def select_all_inherently_controllable(S_):
    if len(select_all_controllable(S_))==len(S_):
        inherentlyControllableStructures = []
        for S in S_:
            if identify_strict_sign_central(StrucMatrix(S=S)):
                inherentlyControllableStructures.append(S)
        return inherentlyControllableStructures
    else:
        print("please only submit uniformly controllable structures to this function. its not necessary but it enforces you being meticulous")

def select_all_centerable(S_):
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
 
def select_all_decouplable(S_):
    def decoupledness(radii, D):
        R = r_from_vector(radii, D)
        S = D * R
        M = create_decoupling_matrix(S)
        # N = null_space(S)
        valid, vector = identify_strict_central(obj(S=M), boundsOverride=True)
        return (vector).flatten()

    # def valid_debug_callback(x):
    #     if any([y<0 for y in x]):
    #         print("INVALID")
    #         raise StopIteration

    decouplableStructures = []
    i=0
    for S in S_:
        i+=1
        print(f"trying to decouple {i} of {len(S_)}", end="\r")
        Struc = obj(S=S)
        M = create_decoupling_matrix(S)
        if not identify_strict_central(StrucMatrix(S=M))[0]:
            D = Struc.D
            mask = D != 0
            positions = np.argwhere(mask)
            nRadii = positions.shape[0]
            radii = np.ones(nRadii)*0.5
            result = least_squares(decoupledness, radii, bounds=(0,1), args=[D])
            evenRadii = result.x/np.max(result.x)
            if result.success:
                R = r_from_vector(evenRadii, D)
                # if Snew.validity:
                decouplableStructures.append(obj(R=R, D=D).S)
        else:
            decouplableStructures.append(Struc.S)
    print("                                   ", end="\r")
    # successes = len(evenStructures)
    # print(f"there were {successes} successes")
    return decouplableStructures

# ^^ General Combinatorics
# --------------------------------------------------------------------------
# vv QUTSM-Specific (old)

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
        allQUTSM.append(mat)
        print(f'generating {i:3d} of {total}', end='\r')
    print("                                 ", end='\r')
    # print("Not going to remove isomorphics")
    uniqueQUTSM = remove_isomorphic(allQUTSM)
    length = len(uniqueQUTSM)
    while True:
        uniqueQUTSM = remove_isomorphic(uniqueQUTSM)
        if length == len(uniqueQUTSM):
            break
        else:
            length = len(uniqueQUTSM)
    for i, SM in enumerate(uniqueQUTSM):
        uniqueQUTSM[i] = triangularize_and_orient(SM)
    return uniqueQUTSM

# ^^ QUTSM-specific
# --------------------------------------------------------------------------
# vv Functional processes

def total_combinatoric_analysis():
    m = 3
    try:
        uniqueAll = np.load("allUnique3x4.npy", mmap_mode='r')
    except:
        uniqueAll = generate_all_unique([3,4])
        print(uniqueAll[0].shape)
        np.save("allUnique3x4.npy", uniqueAll)
    print(f'There are all {3**12} possible 3dof n+1 tendon routings:', )
    print(f'{len(uniqueAll)} of them are unique')
    uniqueRankValids = []
    for i in uniqueAll:
        if matrix_rank(i)==3:
            uniqueRankValids.append(i)
    print(f'{len(uniqueRankValids)} are likely to be controllable, if the correct radii are chosen')
    # print(len(dense))
    try:
        universal = np.load('universal.npy', mmap_mode='r')
        uniform = np.load('uniform.npy', mmap_mode='r')
        possiblyControllableDecouplable = np.load('possiblyControllableDecouplable.npy', mmap_mode='r')
        possiblyControllableAlwaysDecouplable = np.load('possiblyControllableAlwaysDecouplable.npy', mmap_mode='r')
    except:
        universal = []
        uniform = []
        possiblyControllableDecouplable = []
        possiblyControllableAlwaysDecouplable = []
        progress = 0
        for Sm in uniqueRankValids:
            # print(j, end="\r")
            if identify_strict_sign_central(StrucMatrix(S=Sm)):
                # print(i, "True")
                universal.append(Sm)
            if identify_strict_central(StrucMatrix(S=Sm))[0]:
                uniform.append(Sm)
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
            StC = identify_strict_central(StrucMatrix(S=M))[0]
            SSC = identify_strict_sign_central(StrucMatrix(S=M))
            if StC:
                possiblyControllableDecouplable.append(Sm)
            if SSC:
                possiblyControllableAlwaysDecouplable.append(Sm)
            progress+=1
            print(progress, end="\r")
        np.save('universal.npy', np.array(universal))
        np.save('uniform.npy', np.array(uniform))
        np.save('possiblyControllableDecouplable.npy', np.array(possiblyControllableDecouplable))
        np.save('possiblyControllableAlwaysDecouplable.npy', np.array(possiblyControllableAlwaysDecouplable))
    print(f'{len(uniform)} of the rank-valid matrices are controllable for uniform radii')
    print(f'{len(universal)} of these are inherently controllable')
    print(f'{len(possiblyControllableDecouplable)} of the rank-valid matrices are decouplable for uniform radii')
    print(f'{len(possiblyControllableAlwaysDecouplable)} are inherently decouplable (all of which are quasi-hollow)')
    for i in possiblyControllableAlwaysDecouplable:
        print(i)
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
        StC = identify_strict_central(StrucMatrix(S=M))[0]
        SSC = identify_strict_sign_central(StrucMatrix(S=M))
        if StC:
            controllableUniformlyDecouplable.append(Sm)
        if SSC:
            alwaysControllableDecouplable.append(Sm)
    print(f'of the inherently controllable SM, {len(controllableUniformlyDecouplable)} are also decouplable with uniform radii')
    print(f'and there are {len(alwaysControllableDecouplable)} which are both inherently controllable and inherently decoupleable')
    # allUniqueValids = generate_rankValid_well_posed_qutsm()
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
    uniformDecouplable = []
    j=0
    for i in constructibleRankValids:
        print(j, end="\r")
        if identify_strict_sign_central(StrucMatrix(S=i)):
            # print(i, "True")
            universalPractical.append(i)
        if identify_strict_central(StrucMatrix(S=i))[0]:
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
        StC = identify_strict_central(StrucMatrix(S=M))[0]
        if SSC:
            inherentlyDecouplable.append(i)
        if StC:
            uniformDecouplable.append(i)
    uniformPracticalDecouplable = []
    for i in uniformPractical:
        rows = []
        # for each row  of K_J:
        for k in range(m):
            # and each element in that row above the diagonal
            for l in range(k+1, m):
                # there is a row in M that is the dot product of two rows in S
                rows.append(i[k, :] * i[l, :])
        # And we want M * K_J = 0, so we need null space of M
        M = np.vstack(rows)
        StC = identify_strict_central(StrucMatrix(S=M))[0]
        if StC:
            uniformPracticalDecouplable.append(i)
    print(f'{len(uniformPractical)} of which are controllable for uniform radii:')
    for i in uniformPractical:
        print(i)
    print(f"and {len(universalPractical)} of which are 'inherently' controllable:")
    # print(universalPractical)
    for i in universalPractical:
        print(i)
    # print(f'{remove_isomorphic(universalPractical)}')
    print(f'There are {len(uniformDecouplable)} routings which are practical to produce and decouplable for uniform radii')
    print(f"There are {len(inherentlyDecouplable)} routings which are practical to produce and also inherently decouplable")
    print(f'of the {len(uniformPractical)} controllable for uniform radii, {len(uniformPracticalDecouplable)} are also decouplable for uniform radii:')
    for i in uniformPracticalDecouplable:
        print(i)
    
    print()

def qutsm_focus():
    print(f"there are {2**9} possible QUTSM (assuming that everything above the diagonal must be populated)")
    allUniqueQUTSM = generate_all_unique_qutsm()
    print(f"There are {len(allUniqueQUTSM)} QUTSM that are unique wrt isomorphisms")
    # for S in allUniqueQUTSM:
    #     print(S)
    uniformControllableQUTSM = select_all_controllable(allUniqueQUTSM)
    print(f"there are {len(uniformControllableQUTSM)} which are controllable for uniform radius:")
    for S in uniformControllableQUTSM:
        print(S)
    inherentlyControllableQUTSM = select_all_inherently_controllable(uniformControllableQUTSM)
    print(f"there is {len(inherentlyControllableQUTSM)} QUTSM which is inherently controllable:")
    for S in inherentlyControllableQUTSM:
        print(S)
        StrucMatrix(S=S).plotCapability()
    plt.show()

if __name__ == "__main__":

    # total_combinatoric_analysis()
    # qutsm_focus()
    test_functional_decoupling_matrix()