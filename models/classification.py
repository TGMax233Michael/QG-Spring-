import numpy as np
from tqdm import trange

def sigmoid(x):
    return 1/(1+np.e ** (-x))

class BinaryLogisticRegression:
    """
        Binary Logistic Regression
        
        Attributes:
            n_epoches: epoches threshold for model training
            learning_rate: learning rate for model to fit data and update weights
            
    """
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.weights = None
        
    def _init_weights(self, n_features):
        self.weights =  np.zeros(shape=(n_features))
    
    def _calc_gradient(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n_samples):
        return -1/n_samples * x.T.dot(y-y_pred)
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_samples, n_features = x.shape
        self._init_weights(n_features)
        
        for i in trange(self.n_epoches):
            y_pred = sigmoid(x.dot(self.weights))
            loss = -1/n_samples * (y*y_pred - np.log(1+np.e ** y_pred))
            gradient = self._calc_gradient(x, y, y_pred, n_samples)
            self.weights -= gradient * self.learning_rate
            
    def predict(self, x: np.ndarray):
        y_pred = sigmoid(x.dot(self.weights))
        return np.where(y_pred > 0.5, 1, 0)
    
    
class LogisticRegression:
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.classifier: dict[any, BinaryLogisticRegression] = {}
        self.classes = None
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        
        for c in self.classes:
            y_binary = (y==c).astype(int)
            clf = BinaryLogisticRegression(n_epoches=self.n_epoches, learning_rate=self.learning_rate)
            clf.fit(x, y_binary)
            self.classifier[c] = clf
    
    
    def predict(self, x: np.ndarray):
        n_samples = x.shape[0]
        
        probabilities = np.zeros(shape=(n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            probabilities[:, i] = self.classifier[c].predict(x)
            
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        
        return predictions
        
    