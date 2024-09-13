# %%
a = lambda x: x**2
print(a(3))
# %%
sum = lambda x, y: x + y
print(sum(3, 4))
# %%
def f(n):
    return lambda x: x + n

g = f(3)
print(g(4))
# %%
a = [1, 2, 3]
b = [4, 5, 6]

print(list(map(lambda x, y: x + y, a, b)))
# %%
_list = [5, 8, 2, 6, 1, 9, 3, 7, 4]

new_list = list(map(lambda x: 0 if x < 3 else 1, _list))
print(new_list)
# %%
_list = [5, 8, 2, 6, 1, 9, 3, 7, 4]

new_list = list(filter(lambda x: x < 3, _list))
print(new_list)
# %%
_dict = {
    'a': 1,
    'b': 2,
    'c': 3
}
print(_dict['a'])
# %%
import sys
import numpy as np

print(sys.version)
print(np.__version__)
# %%
a = np.empty(0)
print(a)
# %%
a = np.append(a, 1)
print(a)
# %%
_list = [1, 2, 3, 4]

arry1 = np.array(_list)
print(arry1)
# %%
print(arry1.shape)

print(arry1[0:2])
# %%
print(type(arry1))

print(arry1.dtype)
# %%
arry1 = arry1.astype(np.float64)
print(arry1.dtype)
# %%
arry2 = np.array([1, 2, 3])
arry3 = np.array([4, 5, 6])

print(arry2 + arry3)
# %%
