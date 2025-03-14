from models.linear_regression import LinearRegression
from model_selection import train_test_split
from preprocessing.feature import polynomial_feature
from preprocessing.scaler import min_max_scaler
from metrics.regression import MeanSqaureError, r2_score
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./Boston/BostonHousing.csv")
    data = data.dropna(axis=0)
    # print(data.head(10))
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    # print(X, y)
    
    X = min_max_scaler(X)
    X = polynomial_feature(X, degrees=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression(n_epoches=2500, learning_rate=0.02, batch_size=32)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"训练集")
    print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    print(f"R2: {r2_score(y_train, y_train_pred)}")
    
    print(f"测试集")
    print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    print(f"R2: {r2_score(y_test, y_test_pred)}")
    
    model.loss_plot()