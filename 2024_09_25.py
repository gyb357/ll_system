# %%
import numpy as np
# %%
_array_float = np.random.uniform(0, 100, 10)
print(_array_float)
# %%
_array_int = np.random.randint(0, 100, 10)
print(_array_int)
# %%
x1 = np.random.randint(0, 100, 10) # 0 ~ 99
print(x1)
# %%
print(np.sort(x1))
print(x1)
# %%
print(np.argsort(x1))
print(x1)
sort_x1 = np.argsort(x1)
# %%
print(x1[sort_x1[0]]) # min
print(x1[x1.argmin()]) # min
print(x1[x1.argmax()]) # max
# %%
x2 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
print(x1[x2])
# %%
x1 = np.array([
    [1, 9],
    [0, 8],
    [6, 3],
    [3, 4],
    [4, 1],
])
print(x1)
# %%
print(x1[:, 0])
print(x1[:, 1])
# %%
sort_indice_1 = x1[:, 0].argsort()
print(sort_indice_1)
print(x1[:, 0][sort_indice_1])

sort_indice_2 = x1[:, 1].argsort()
print(sort_indice_2)
print(x1[:, 1][sort_indice_2])
# %%
x1 = np.array([
    [[102, 131]],
    [[101, 237]],
    [[148, 11]],
    [[24, 189]],
    [[247, 19]]
])
print(x1)
# %%
print(x1[:, 0, 0])
print(x1[:, 0, 1])
print(x1[0, :, 0])

print(x1[0, 0, :])
# %%
sort_indice = x1[:, 0, 0].argsort()
print(sort_indice)
print(x1[:, 0, 0][sort_indice])
# %%
x1 = np.random.randint(0, 100, 10)
print(x1)
# %%
x2 = np.array([value for value in x1])
print(x2)
# %%
x3 = np.array([value for value in x1 if value % 2 == 0])
print(x3)
# %%
x1 = np.random.randint(0, 100, 10)
x2 = np.random.randint(0, 100, 10)
print(x1, x2)
# %%
x3 = np.array([value for value in zip(x1, x2)])
print(x3)
# %%
x4 = np.array([value for value in zip(x1, x2) if value[0] < 50])
print(x4)
# %%
x1 = np.random.randint(0, 100, 10)
print(x1)
# %%
print("sum: ", x1.sum())
print("avg: ", x1.mean())
print("std: ", x1.std())
print("var: ", x2.var())
# %%
x2 = np.array([
    np.random.randint(0, 100, 10),
    np.random.randint(0, 100, 10),
])
print(x2)
# %%
x1 = np.random.randint(0, 100, 16)
print(x1)
print(x1.shape)
# %%
_x1 = np.expand_dims(x1, axis=0)
print(_x1)
print(_x1.shape)
# %%
x2 = np.reshape(x1, [4, 4])
print(x2)
print(x2.shape)
# %%
x3 = np.reshape(x1, [-1, 8])
print(x3)
print(x3.shape)
# %%
x4 = np.reshape(x1, [2, -1])
print(x4)
print(x4.shape)
# %%
# %pip install torch
# %%
import torch
print(torch.__version__)
# %%
