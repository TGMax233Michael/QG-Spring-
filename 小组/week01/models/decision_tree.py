import numpy as np

def gini(y: np.ndarray):
    labels, counts = np.unique(y, return_counts=True)
    ratio = counts / len(y)
    return 1 - np.sum(ratio**2)

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class TreeClassifier:
    def __init__(self):
        self.tree: DecisionTreeNode|None = None
    
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        best_threshold = None
        best_feature = None
        base_gini = gini(y)
        best_gain = -np.inf
        
        for feature in range(n_features):
            sorted_uniques = np.unique(X[:, feature])
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
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) == 1:
            return DecisionTreeNode(label=y[0])
        
        feature, threshold = self._best_split(X, y)
        
        left_mask = X[:, feature] > threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask])
        right = self._build_tree(X[right_mask], y[right_mask])
        
        return DecisionTreeNode(feature, threshold, left, right)
    
    def _travel_tree(self, X: np.ndarray, node: DecisionTreeNode):
        if node is None:
            return None
        if node.label is not None:
            return np.full(X.shape[0], node.label)
        
        predictions = np.empty(X.shape[0], dtype=np.float64)
        left_mask = X[:, node.feature] > node.threshold
        right_mask = ~left_mask
        
        predictions[left_mask] = self._travel_tree(X[left_mask], node.left)
        predictions[right_mask] = self._travel_tree(X[right_mask], node.right)
        
        return predictions
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.tree = self._build_tree(X, y)
        
    def predict(self, X: np.ndarray):
        node = self.tree
        return self._travel_tree(X, node)