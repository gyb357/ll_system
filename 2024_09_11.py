# %%
a = 10
# a?
# %%
b = 3.14
# b?
# %%
print(type(a))
# %%
c = True
# c?
# %%
d = (1, 2, 3)
# d?
# %%
e = [1, 2, 3]
# e?
# %%
for i in range(10):
    print(i)
# %%
for i in range(1, 11, 2):
    print(i)
# %%
for i in range(10, 0, -1):
    print(i)
# %%
_list = [3, 6, 9]
for i in [3, 6, 9]:
    print(i)
for i in range(3):
    print(_list[i])
# %%
_tuple = (1, 2, 3, 4, 5, 6)
print(_tuple)
# %%
print(_tuple[3])
# _tuple[3] = 0
# %%
_tuple = (v for v in range(10))
print(_tuple)
# %%
_list = [1, 2, 3]
print(_list)
print(_list[2])
# %%
_list[2] = 7
print(_list[2])
# %%
_list = [v for v in range(10)]
print(_list)
# %%
__list = [_list]
print(__list)
# %%
print(__list[0])
print(__list[0][2])
# %%
_list1 = [1, 3, 5, 7, 9]
_list2 = [2, 4, 6, 8, 10]
print(_list + _list2)
# %%
poly = [(x + y) for x, y in zip(_list1, _list2)]
print(poly)
# %%
_list = [1, 2, 3, 4, 5, 6]

a = _list[0]
b = _list[1]
print(a, b)
# %%
a, b, c, d, *_ = _list
print(c, d)
# %%
*a, = _list
print(a)
# %%
_, _, a, b, _, _ = _list
print(a, b)
# %%
_, _, a, b, *_ = _list
print(a, b)
# %%
_, _, *a, _, _ = _list
print(a)
# %%
_list = [1, 2, 3]
print(_list)

_list.append(77)
print(_list)

_list.insert(0, 88)
print(_list)

_list.insert(-1, 99)
print(_list)

_list.insert(0, 100)
print(_list)
# %%
_list.pop()
_list.pop(0)
# %%
_list = [4, 8, 6, 2, 1]
print(_list)

new_list = [i + 1 for i in _list]
print(new_list)
# %%
new_list.sort()
print(new_list)
# %%
tensor = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(tensor)

new_tensor = []
for i in tensor:
    new_tensor += i
print(new_tensor)

new_tensor2 = [v for t in tensor for v in t]
print(new_tensor2)
# %%
import random

random_list = [i for i in range(10)]
print(random_list)

random.shuffle(random_list)
print(random_list)