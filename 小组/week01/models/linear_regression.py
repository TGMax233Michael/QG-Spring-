import numpy as np
from enum import IntEnum, auto

class info_density(IntEnum):
    no = auto()
    few = auto()
    all = auto()
    
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

class MyLinearRegression:
    """Linear Model

        Attributes:
            iterations (int): Number of iterations. Default 1000
            learning_rate (float): Learning rate. Default 0.1
            print_info (info_density): Print fitting process information. Default info_density.no
            polynomial (bool): Enable polynomial fitting. Default False
            degree (int): Degree of polynomial fitting. Default None
            adagrad (bool): Enable Adagrad gradient algorithm. Default False
            epsilon (float): Epsilon parameter in Adagrad algorithm. Default 1e-8
            mini_batch (bool): Enable mini-batch gradient descent. Default False
            adam (bool): Enable Adam gradient algorithm. Default False
            beta1 (float): Beta1 parameter in Adam algorithm. Default 0.9
            beta2 (float): Beta2 parameter in Adam algorithm. Default 0.999
    """
    def __init__(self, 
                iterations=1000, 
                learning_rate=0.1, 
                print_info=info_density.no,
                polynomial=False,
                degree=2,
                adagrad=False,
                epsilon=1e-8,
                mini_batch=False,
                adam=False,
                beta1=0.9,
                beta2=0.999
                ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.polynomial = polynomial
        self.print_info = print_info
        self.degree = degree
        self.adagrad = adagrad
        self.epsilon = epsilon
        self.mini_batch = mini_batch
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight = None
        self.mse = None
        
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the data

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.polynomial:
            X = polynomial_features(X, self.degree)

        X = min_max_scaler(X)
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        n_samples, n_features = X.shape[0], X.shape[1]
        
        if self.adagrad:
            r = np.ones(n_features)
            
        if self.adam:
            m = np.zeros(shape=(n_features,))
            v = np.zeros(shape=(n_features,))
        
        # Initial weights
        weight = np.zeros(n_features)
        
        if self.print_info != info_density.no:
            print(f"Initial weights\n{weight}\n")
        
        for i in range(self.iterations):
            # Calculate gradient
            if self.mini_batch:                 # Use mini-batch gradient descent
                indices = np.random.choice(n_samples, max(n_samples//10, 32), replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
                y_batch_pred = X_batch.dot(weight)
                gradient = 2 / (n_samples//10) * X_batch.T.dot(y_batch_pred - y_batch)
            else:                               # Use batch gradient descent
                y_pred = X.dot(weight)
                gradient = 2 / n_samples * X.T.dot(y_pred - y)
            
            # Gradient clipping
            clip_threshold = 5.0
            if np.linalg.norm(gradient) > clip_threshold:
                gradient = (clip_threshold / np.linalg.norm(gradient)) * gradient
            
            # Update weights
            if self.adam:
                # Adam
                m = self.beta1 * m + (1 - self.beta1) * gradient
                v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
                m_hat = m / (1 - self.beta1 ** (i+1))
                v_hat = v / (1 - self.beta2 ** (i+1))
                weight -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            elif self.adagrad:
                # Adagrad
                r += gradient ** 2
                weight -= gradient * (self.learning_rate / (np.sqrt(r) + self.epsilon))
            else:
                # SGD
                weight -= gradient * self.learning_rate

            y_pred_new = X.dot(weight)
            mse = np.mean((y_pred_new - y) ** 2)

            # Print training information
            if self.print_info == info_density.few:
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.iterations} MSE {mse}\n Weight {weight}\n\n")
            elif self.print_info == info_density.all:
                print(f"Iteration {i+1}/{self.iterations} MSE {mse}\n Weight {weight}\n\n")
            
        self.weight = weight
        self.mse = mse
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.polynomial:
            X = polynomial_features(X, self.degree)
        
        X = min_max_scaler(X, 0, 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        return X.dot(self.weight)