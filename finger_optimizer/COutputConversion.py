import numpy as np
import os

fb = '08b'

np.set_printoptions(formatter={"int": lambda x: f"{x:08b}"})

m = 5
n = 4

# get the path to the output of the c code
here  = os.getcwd()
relative_path = os.path.join(here, 'Combinatorics', 'unique4x5.out')

# print(relative_path)

raw = np.fromfile(relative_path, dtype=np.uint8)

assert raw.size % 5 == 0
p = raw.size // 5

unit = raw.reshape(p, 5)
print(unit.shape)
# print(f"{unit[0,0]:08b}")
print(unit[0])

bits = np.unpackbits(unit, axis=1, bitorder='big')
print(bits.shape)
print(bits[0,:16])
# quit()

pairs = bits.reshape(p, 20, 2)
print(pairs.shape)
print(pairs[0,:8,:])

values = (pairs[:, :, 0] << 1) + pairs[:, :, 1]  # shape (num_mats, 20)
# print((0 << 1) + 1)
print(values.shape)
print(values[0,:8])
# quit()
values-=1
print(values.shape)
print(values[0,:8])

matrices = values.reshape(p,m,n).transpose(0,2,1).view(np.int8)
# matrices = matrices.transpose
np.set_printoptions(formatter=None)
print(matrices.shape)
print(matrices[0])

np.save("4x5_unique.npy", matrices)
# print(pairs.shape)
# print(pairs[0])
# # quit()
# print(pairs.size)
# matrices = pairs.reshape(p, m, n).transpose(0,2,1)

# print(matrices[0])