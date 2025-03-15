import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
        Linear Regression
        Attributes:
            n_epoches: 训练轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, batch_size=0):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = None
        self.losses: dict = {}
        
    def _init_weights(self, n_features):
        self.weights = np.zeros(shape=(n_features))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in range(self.n_epoches):
            if self.batch_size <= 0:
                y_pred = X.dot(self.weights)
                loss = np.mean((y-y_pred) ** 2)
                gradient = -X.T.dot(y-y_pred) / n_samples
            else:
                batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = X_batch.dot(self.weights)
                loss = np.mean((y_batch-y_pred) ** 2)
                gradient = -X_batch.T.dot(y_batch-y_pred) / n_samples
            
            self.weights -= gradient * self.learning_rate
            self.losses[epoch+1] = loss
            # print(loss)
            
    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
            
        return X.dot(self.weights)
    
    def loss_plot(self):
        plt.figure(figsize=(8, 8), dpi=80)
        plt.plot(list(self.losses.keys()), list(self.losses.values()), label="MSE", color="Red")
        plt.axhline(y=min(list(self.losses.values())), label="MIN MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()
        