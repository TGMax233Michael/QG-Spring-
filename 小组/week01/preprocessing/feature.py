import numpy as np

def polynomial_feature(X, degrees: int):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_features = X.shape[1]
    
    for i in range(n_features):
        for degree in range(1, degrees):
            X = np.column_stack((X, X[:, i] ** (degree+1)))
            
    return X