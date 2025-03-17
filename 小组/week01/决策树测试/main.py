from models.decision_tree import TreeClassifier, TreeRegressor
from metrics.classification import Accuracy
from metrics.regression import MeanSqaureError, r2_score
from sklearn.datasets import make_classification, load_iris, make_regression
from model_selection import train_test_split

if __name__ == "__main__":
    # data = load_iris()
    # data = make_classification(n_samples=10000, n_classes=3, 
    #                            n_features=6, n_informative=4, 
    #                            n_redundant=1, n_repeated=1, 
    #                            random_state=100)
    
    # # X, y = data["data"], data["target"]
    # X, y = data[0], data[-1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, 42, 0.2)
    # model = TreeClassifier(max_depth=20, min_batch=10, gain_threshold=0.001)
    
    # model.fit(X_train, y_train)
    # y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)
    
    # print("训练集")
    # print(f"Accuracy: {Accuracy(y_train, y_train_pred)}")
    # print("测试集")
    # print(f"Accuracy: {Accuracy(y_test, y_test_pred)}")
    
    data = make_regression(n_samples=100, n_features=2, random_state=42)
    X, y = data[0], data[-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 42, 0.2)
    
    model = TreeRegressor()
    model.fit(X, y)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("训练集")
    print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    print(f"R2: {r2_score(y_train, y_train_pred)}")
    print("测试集")
    print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    print(f"R2: {r2_score(y_test, y_test_pred)}")