import numpy as np

# 计算基尼指数
def gini(y: np.ndarray):
    labels, counts = np.unique(y, return_counts=True)
    ratio = counts / len(y)
    return 1 - np.sum(ratio**2)

# 计算均方误差
def MSE(y: np.ndarray):
    return 1/len(y) * np.sum((y - np.mean(y))**2)

# 树节点
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class TreeClassifier:
    """
        分类决策树
        Attributes:
            max_depth: 最深层数
    """
    def __init__(self, max_depth=100):
        self.tree: DecisionTreeNode|None = None
        self.max_depth = max_depth
    
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        best_threshold = None
        best_feature = None
        base_gini = gini(y)
        best_gain = -np.inf
        
        for feature in range(n_features):
            sorted_uniques = np.sort(np.unique(X[:, feature]))
            if len(sorted_uniques) == 1:
                thresholds = sorted_uniques
            else:
                thresholds = (sorted_uniques[:-1] + sorted_uniques[1:]) / 2
                
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                weighted_gini = len(y_left)/n_samples * gini(y_left) + len(y_right)/n_samples * gini(y_right)
                gain = base_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, layer):
        if len(np.unique(y)) == 1:
            return DecisionTreeNode(label=y[0])
        if layer == self.max_depth:
            # 将出现最多的类别数作为标签
            return DecisionTreeNode(label=np.argmax(np.bincount(y)))
        
        feature, threshold = self._best_split(X, y)
        
        left_mask = X[:, feature] > threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], layer+1)
        right = self._build_tree(X[right_mask], y[right_mask], layer+1)
        
        return DecisionTreeNode(feature, threshold, left, right)
    
    def _travel_tree(self, X: np.ndarray, node: DecisionTreeNode, layer):
        if node is None:
            return None
        if node.label is not None:
            return np.full(X.shape[0], node.label)
        
        predictions = np.empty(X.shape[0], dtype=np.float64)
        left_mask = X[:, node.feature] > node.threshold
        right_mask = ~left_mask
        
        predictions[left_mask] = self._travel_tree(X[left_mask], node.left, layer+1)
        predictions[right_mask] = self._travel_tree(X[right_mask], node.right, layer+1)
        
        return predictions
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.tree = self._build_tree(X, y, 1)
        
    def predict(self, X: np.ndarray):
        node = self.tree
        return self._travel_tree(X, node, 0)
    
class TreeRegressor:
    def __init__(self):
        self.tree: DecisionTreeNode|None = None
    
    def _best_split(X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_featurs = X.shape
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        base_MSE = MSE(y)
        
        for feature in range(n_featurs):
            sorted_unique = np.sort(np.unique(y))
            if len(sorted_unique) == 1:
                thresholds = sorted_unique
            else:
                thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2
            
            for threshold in thresholds:
                left_mask = X[:, feature] < left_mask
                right_mask = ~left_mask
                
                weighted_MSE = len(y[left_mask])/len(y) * MSE(y[left_mask]) + len(y[right_mask])/len(y) * MSE(y[right_mask])
                gain = base_MSE - weighted_MSE
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    # 等待完成
    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        ...
        
    def _travel_tree(self, X: np.ndarray, node):
        ...
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...
        
    def predict(self, X: np.ndarray):
        ...
                
        
         