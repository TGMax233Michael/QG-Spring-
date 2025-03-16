from models.decision_tree import TreeClassifier
from metrics.classification import Accuracy
from sklearn.datasets import make_classification, load_iris
from model_selection import train_test_split

if __name__ == "__main__":
    # data = load_iris()
    data = make_classification(n_samples=1000, n_classes=3, 
                               n_features=5, n_informative=4, 
                               n_redundant=1, n_repeated=0, 
                               random_state=42)
    
    # X, y = data["data"], data["target"]
    X, y = data[0], data[-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 42, 0.2)
    model = TreeClassifier(max_depth=20)
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("训练集")
    print(f"Accuracy: {Accuracy(y_train, y_train_pred)}")
    print("测试集")
    print(f"Accuracy: {Accuracy(y_test, y_test_pred)}")