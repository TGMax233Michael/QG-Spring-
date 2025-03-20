import numpy as np
from sklearn.datasets import make_regression

data = make_regression(n_samples=100, n_features=3, n_informative=3, random_state=42)

X, y = data[0], data[-1]

print(np.linalg.det(X.T.dot(X)))
print(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y))
