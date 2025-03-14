import numpy as np

def min_max_scaler(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    features = X.shape[1]
    
    for i in range(features):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
        
    return X

if __name__ == "__main__":
    x = np.array([[1000, 2, 0.02], 
                  [2000, 1, 0.001], 
                  [1900, 8, 0.98]])
    
    print(min_max_scaler(x))