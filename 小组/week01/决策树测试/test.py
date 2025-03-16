import numpy as np
from sklearn.datasets import load_iris
from model_selection import train_test_split
from metrics.classification import Accuracy

iris = load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, 42, 0.2)


# X = np.random.randint(1, 100, 100).reshape(20, 5)

# y = np.random.choice(4, 20)

# 计算基尼系数
def gini(y):
    labels, counts = np.unique(y, return_counts=True)
    ratio = counts / len(y)
    return 1 - np.sum(ratio ** 2)

def best_split(X, y):
    # 找出能使信息增益最大的特征
    n_samples, n_featurs = X.shape
    best_threshold = None
    best_feature = None
    base_gini = gini(y)
    best_gain = -np.inf
    
    for feature in range(n_featurs):
        # print(f"feature: {feature}")
        
        sorted_uniques = np.unique(np.sort(X[:, feature]))
        
        # 选取阈值
        thresholds = (sorted_uniques[:-1] + sorted_uniques[1:])/2 if len(sorted_uniques) > 1 else sorted_uniques
        
        for threshold in thresholds:
            left_mask = X[:, feature] > threshold
            right_mask = X[:, feature] < threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # print(f"y_left: {y_left}")
            # print(f"y_right: {y_right}")
            
            # 计算加权基尼系数
            weighted_gini = (len(y_left) / n_samples * gini(y_left)) + (len(y_right) / n_samples * gini(y_right))
            
            delta_gini = base_gini - weighted_gini
            # print(f"delta_gini: {delta_gini}")
            # print()
            
            if delta_gini > best_gain:
                best_gain = delta_gini
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
    
def build_tree(X: np.array, y: np.array):
    if len(np.unique(y)) == 1:
        return TreeNode(label=y[0])
    
    feature, threshold = best_split(X, y)
    
    left_mask = X[:, feature] < threshold
    right_mask = ~left_mask
    
    left = build_tree(X[left_mask], y[left_mask])
    right = build_tree(X[right_mask], y[right_mask])
    return TreeNode(feature=feature, threshold=threshold, left=left, right=right)
    
tree = build_tree(X_train, y_train)

def show_tree(node: TreeNode, depth=0):
    indent = " " * depth
    if node is None:
        return
    if node.label is not None:
        print(f"label: {node.label}")
        return
    
    print(f"feature: {node.feature} threshold: {node.threshold}")
    
    print(f"{indent}L ->")
    show_tree(node.left, depth+1)
    
    print(f"{indent}R ->")
    show_tree(node.right, depth+1)

        

def predict(X, node: TreeNode):
    if not node:
        return None
    if node.label is not None:
        return np.full(X.shape[0], node.label)
    
    left_mask = X[:, node.feature] < node.threshold
    right_mask = ~left_mask
    
    prediction = np.zeros(X.shape[0], int)
    prediction[left_mask] = predict(X[left_mask], node.left)
    prediction[right_mask] = predict(X[right_mask], node.right)
    
    return prediction

show_tree(tree)
y_train_pred = predict(X_train, tree)
y_test_pred = predict(X_test, tree)
print(f"训练集")
print(f"ACC {Accuracy(y_train, y_train_pred)}")
print(f"测试集")
print(f"ACC {Accuracy(y_test, y_test_pred)}")