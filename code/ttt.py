# coding=utf-8

import numpy as np
a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
print(np.hstack((a[:, 0:3], a[:, 4:])))