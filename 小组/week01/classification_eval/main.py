from models.classification import LogisticRegression
from model_selection import train_test_split
from preprocessing.feature import polynomial_feature
from metrics.classification import Accuracy
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

if __name__ == "__main__":
    iris = load_iris(as_frame=True)
    X, y = iris["data"], iris["target"]
    print(X.head(5))
    print(y.head(5))
    X, y = np.array(X), np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ohe = OneHotEncoder()
    
    model = LogisticRegression(method="ova", n_epoches=1000, learning_rate=0.05, batch_size=32)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("Ova")
    print(f"训练集")
    print(f"Accuracy: {Accuracy(y_train, y_train_pred)}")
    print(f"测试集")
    print(f"Accuracy: {Accuracy(y_test, y_test_pred)}")
    
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
    y_train_ohe = y_train_ohe.toarray()
    model = LogisticRegression(method="softmax", n_epoches=1000, learning_rate=0.05, batch_size=32)
    model.fit(X_train, y_train_ohe)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print()
    print("Softmax")
    print(f"训练集")
    print(f"Accuracy: {Accuracy(y_train, y_train_pred)}")
    print(f"测试集")
    print(f"Accuracy: {Accuracy(y_test, y_test_pred)}")