from models.linear_regression import LinearRegression
from model_selection import train_test_split
from preprocessing.feature import polynomial_feature
from preprocessing.scaler import min_max_scaler
from metrics.regression import MeanSqaureError, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_relevance(X: pd.DataFrame, y: pd.DataFrame):
    plt.figure(figsize=(16, 10), dpi=100)
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.scatter(X.iloc[:, i], y)
        plt.xlabel(f"{X.columns[i]}")
        plt.ylabel(f"{y.name}")
        
    plt.figure(figsize=(16, 10), dpi=100)
    for i in range(9, 13):
        plt.subplot(2, 2, i-8)
        plt.scatter(X.iloc[:, i], y)
        plt.xlabel(f"{X.columns[i]}")
        plt.ylabel(f"{y.name}")
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("./Boston/BostonHousing.csv")
    data = data.dropna(axis=0)
    # print(data.head(10))
    # print(data.columns)
    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    print(X.columns)
    
    X = X.drop(columns=['chas', 'rad', 'tax', 'indus', 'b'])
    print(X.columns)
    
    
    # show_relevance(X, y)
    
    X, y = np.array(X), np.array(y)
 
    X = min_max_scaler(X)
    X = polynomial_feature(X, degrees=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression(n_epoches=6000, learning_rate=0.05, batch_size=len(X)//2)
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