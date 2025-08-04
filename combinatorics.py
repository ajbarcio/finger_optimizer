from utils import *
from strucMatrices import *
import itertools
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
        # print(S())
        halfValids.append(S())
        if S.validity:
            fullyValids.append(S())
    else:
        print(f'trying {i:6d} of {total}', end='\r')
print("                                 ", end='\r')

fullValidUnique = remove_isomorphic(fullyValids)
print('uniform valid matrices')
for i, m in enumerate(fullValidUnique):
    print(f'Matrix {i}:\n{m}\n')

unique = remove_isomorphic(halfValids)
# print('rank-valid matrices')
# for i, m in enumerate(unique):
#     if np.array_equal(m,canonical_form(D1)):
#         print('THIS ONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print(f'Matrix {i}:\n{m}\n')

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

print('joint-uniform valid matrices')
for i, m in enumerate(jointUniformValids):
    print(f'Matrix {i}:\n{m}\n')

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

print('extension-and-joint-uniform valid matrices')
for i, m in enumerate(jointAndExtensionUniformValids):
    print(f'Matrix {i}:\n{m}\n')

extensionUniformValids = []
for i, m in enumerate(unique):
    result = R1 @ m
    result[result < 0] = -0.5
    S = StrucMatrix(S=result)
    if S.validity:
        extensionUniformValids.append(S.D)

print('extension-uniform valid matrices')
for i, m in enumerate(extensionUniformValids):
    print(f'Matrix {i}:\n{m}\n')