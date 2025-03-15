import numpy as np

def min_max_scaler(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    features = X.shape[1]
    
    for i in range(features):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
        
    return X