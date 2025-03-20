import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
        Linear Regression
        Attributes:
            n_epoches: 训练轮数
            learning_rate: 学习率
            batch_size: 批量大小 (等于0时表示不使用小批量梯度下降)
            solver: 求解器类型，"normal_equation" 表示最小二乘法，"gradient_descent" 表示梯度下降法，"auto" 表示自动选择
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, batch_size=0, solver="auto"):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.solver: str = solver
        self.weights = None
        
    def _init_weights(self, n_features):
        self.weights = np.zeros(shape=(n_features))
    
    # 梯度下降求解
    def _gradient_descent_fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in range(self.n_epoches):
            if self.batch_size <= 0:
                y_pred = X @ self.weights
                loss = np.mean((y-y_pred) ** 2)
                gradient = -X.T @ (y-y_pred) / n_samples
            else:
                batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = X_batch @ self.weights
                loss = np.mean((y_batch-y_pred) ** 2)
                gradient = -X_batch.T @ (y_batch-y_pred) / n_samples
            
            self.weights -= gradient * self.learning_rate
    
    # 最小二乘法求解权重
    def _normal_equation_fit(self, X: np.ndarray, y: np.ndarray):
        # Gram矩阵 = X.T @ T
        gram_matrix = X.T @ X
        
        # 判断 
        if np.linalg.det(gram_matrix) != 0:
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y 
        else:
            raise np.linalg.LinAlgError("Gram矩阵是奇异矩阵（不可逆）, 无法使用最小二乘法，请使用梯度下降求解器")

    # 拟合数据
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        n_samples, n_features = X.shape
        gram_matrix = X.T @ X
        conditions_number = np.linalg.cond(gram_matrix)
        
        if self.solver.lower() == "auto":
            if n_samples < 1e4 and conditions_number < 1e8:
                try:
                    self._normal_equation_fit(X, y)
                except np.linalg.LinAlgError:
                    self._gradient_descent_fit(X, y)
            else:
                self._gradient_descent_fit(X, y)
                
        elif self.solver.lower() == "normal_equation":
            self._normal_equation_fit(X, y)
        elif self.solver.lower() == "gradient_descent":
            self._gradient_descent_fit(X, y)
        else:
            raise KeyError("未知求解器")
    
    # 预测
    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
            
        return X @ self.weights

def sigmoid(X):
    return 1/(1+np.e ** (-X))
def softmax(X):
    return np.e**X / np.sum(np.e**X, axis=1, keepdims=True)

class BinaryLogisticRegression:
    """
        Binary Logistic Regression
        
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, batch_size=0):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = None
        
    def _init_weights(self, n_features):
        self.weights =  np.zeros(shape=(n_features))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in range(self.n_epoches):
            if self.batch_size <= 0:
                y_pred = sigmoid(X @ self.weights)
                loss = np.mean(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
                gradient = -1/n_samples * X.T @ (y-y_pred)
            else:
                batch_indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = sigmoid(X_batch @ self.weights)
                loss = np.mean(-y_batch*np.log(y_pred) - (1-y_batch)*np.log(1-y_pred))
                gradient = -1/n_samples * X_batch.T @ (y_batch-y_pred)
            
            self.weights -= gradient * self.learning_rate
            
            
    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        y_pred = sigmoid(X @ self.weights)
        # return np.where(y_pred > 0.5, 1, 0)
        return y_pred
    
class LogisticRegression_ova:
    """
        Logistic Regression ova
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
            batch_size: 批量大小
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, batch_size=0):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.classifier: dict[int, BinaryLogisticRegression] = {}
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        
        for c in self.classes:
            y_binary = (y==c).astype(int)
            clf = BinaryLogisticRegression(n_epoches=self.n_epoches, learning_rate=self.learning_rate, batch_size=self.batch_size)
            clf.fit(X, y_binary)
            self.classifier[c] = clf
    
    def predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        
        probabilities = np.zeros(shape=(n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            probabilities[:, i] = self.classifier[c].predict(X)
            
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        
        return predictions

class SoftmaxRegression:
    """
        Softmax Regression
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, batch_size=0):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.classes = None
        self.weights = None
        
    def __init_weights(self, n_features):
        self.weights = np.zeros(shape=(n_features, len(self.classes)))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        self.classes = np.arange(y.shape[1])  
        n_samples, n_features = X.shape
        self.__init_weights(n_features)
        
        for epoch in range(self.n_epoches):
            if self.batch_size <= 0:
                y_pred = X @ self.weights
                probs = softmax(y_pred)
                loss = np.mean(-np.sum(y * np.log(probs), axis=1))
                gradient = -1/n_samples * X.T @ (y - probs)
            else:
                batch_indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = X_batch @ self.weights
                probs = softmax(y_pred)
                loss = np.mean(-np.sum(y_batch * np.log(probs), axis=1))
                gradient = -1/n_samples * X_batch.T @ (y_batch - probs)
            
            self.weights -= gradient * self.learning_rate
            
    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.column_stack((np.ones(shape=(X.shape[0], 1)), X))
        
        return np.argmax(softmax(X @ self.weights), axis=1)

class LogisticRegression:
    """
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
            method: 包括两种方法 -> ["ova", "softmax]
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, method="ova", batch_size=0):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.method = method
        self.batch_size = batch_size
        self.models_dict = {"ova": LogisticRegression_ova,
                            "softmax": SoftmaxRegression}
        self.model = None
        self._init_model()
        
    def _init_model(self):
        try:
            if self.method.lower() not in self.models_dict.keys():
                raise KeyError
        except KeyError:
            print(f"Unknown Methods! Existed Methods: {self.models_dict.keys()}")
        self.model = self.models_dict[self.method.lower()](self.n_epoches, self.learning_rate, self.batch_size)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)