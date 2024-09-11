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
