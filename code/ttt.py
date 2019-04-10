import numpy as np

a = [
    [1, 2],
    [3, 4],
]

b = np.hstack((a, np.ones((2, 1))))
print(b)