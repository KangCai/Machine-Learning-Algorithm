import numpy as np

a = np.array([[2, 1, 2],[2, 1, 1]])
print(a + 5)
print(a / a.sum(axis=0))
print(a / a.sum(axis=0).reshape(1, 3))
