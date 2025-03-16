import numpy as np

arr = np.array([1, 2, 3, 9, 9, 9, 1])
commen = np.argmax(np.bincount(arr))
print(commen)