import numpy as np

class LinearRegression:
    """
        Linear Regression
        Attributes:
            n_epoches: 训练轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.weights = None
        
    def _init_weights(self, n_features):
        self.weights = np.zeros(shape=(n_features))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in range(self.n_epoches):
            y_pred = X.dot(self.weights)
            loss = np.mean((y-y_pred) ** 2)
            gradient = -X.T.dot(y-y_pred) / n_samples
            self.weights -= gradient * self.learning_rate
            
            # print(loss)
            
    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
            
        return X.dot(self.weights)
        