import numpy as np

def polynomial_features(X: np.ndarray, degree: int):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    m, n = X.shape[0], X.shape[1]
    
    for i in range(n):
        for j in range(2, degree + 1):
            X = np.column_stack((X, X[:, i] ** j))
            
    return X

def min_max_scaler(X: np.ndarray, min: int = 0, max: int = 1):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = X_std * (max - min) + min
    
    return X