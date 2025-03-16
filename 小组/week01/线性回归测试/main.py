from models.linear_model import LinearRegression
from model_selection import train_test_split
from metrics.regression import r2_score, MeanSqaureError
from sklearn.datasets import make_regression
import numpy as np

if __name__ == "__main__":
    data = make_regression(n_samples=1000, n_features=4, n_informative=3, random_state=42)
    X, y = data[0], data[-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42 ,test_size=0.2)
    # print(X_train, y_train)
    
    model = LinearRegression(n_epoches=1000, learning_rate=0.005)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"训练集")
    print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    print(f"R2: {r2_score(y_train, y_train_pred)}")
    
    print(f"测试集")
    print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    print(f"R2: {r2_score(y_test, y_test_pred)}")
    