import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, 
                     random_state: int = -1, test_size: float = 0.2):
    if random_state != -1:
        np.random.seed(random_state)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_samples = X.shape[0]  
    
    test_indices = np.random.choice(n_samples, int(n_samples * test_size), replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]