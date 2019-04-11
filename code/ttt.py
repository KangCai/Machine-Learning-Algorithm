import numpy as np

a = np.array([1, 0, 0])
b = np.array([1, 0, 1])
print(a == 1)
print(b == 1)
k = (a == 1).astype(int) + (b == 1).astype(int) == 2
print(k)
print(np.where(k==True))
print(len(np.where(k==True)[0]))