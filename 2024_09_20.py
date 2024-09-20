# %%
import numpy as np
print(np.__version__)
# %%
a1 = np.array([1, 2, 3, 4, 5])
a2 = np.array([7, 8, 9, 10, 11, 12, 13, 14])

print(a1[2:5])
print(a2[3:6])
# %%
a1[2:5] = a2[3:6]
print(a1)
# %%
z1 = np.zeros(3)
print(z1)
# %%
z2 = np.zeros((3,) + (2, 2))
print(z2)
print(z2.shape)
# %%
a1 = np.arange(0, 10, 1)
print(a1)
print(a1[0:5])
# %%
_list = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]
a2 = np.array(_list)
print(a2)
print(a2.shape)
# %%
print(a2[2, 0])
print(a2[2, :])
print(a2[:, 1])
print(a2[:, 0:2])
print(a2[:, 2:4])
# %%
_list = [
    [[1, 2, 3, 4]],
    [[5, 6, 7, 8]],
    [[9, 10, 11, 12]]
]
a1 = np.array(_list)
print(a1)
print(a1.shape)

print(a1[0])
print(a1[0:])
print(a1[0, 0, 1])
# %%
print(a1[1, 0, 1])
print(a1[2, 0, 2])
# %%
a1 = np.array([3, 4])
a2 = np.array([5, 6])
print(a1 + a2)
print(a1 / a2)
print(a1 * a2)
# %%
a1 = np.zeros(10, dtype=np.uint8)
print(a1)

print(a1.dtype)
print(a1.shape)
# %%
a2 = np.arange(0, 10, 1, dtype=np.uint8)
print(a2.dtype)
print(a2.shape)
print(a2[3:7])
# %%
print(a2)
print(a2.reshape(5, 2))
print(a2.reshape(2, 5))
# %%
print(np.random.rand(10))
a3 = np.random.rand(10)
print(a3)

_a3 = a3*100
print(_a3)

__a3 = a3.astype(np.uint32)
print(__a3)
# %%
