from myMLModule.models.linear_model import LinearRegression
from myMLModule.models.decision_tree import TreeRegressor
from myMLModule.model_selection import train_test_split
from myMLModule.preprocessing.feature import polynomial_feature
from myMLModule.preprocessing.scaler import min_max_scaler
from myMLModule.metrics.regression import MeanSqaureError, r2_score
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
    data = pd.read_csv("./波士顿房价预测/BostonHousing.csv")
    data = data.dropna(axis=0)
    # print(data.head(10))
    # print(data.columns)
    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    # show_relevance(X, y)
    print(X.columns)
    
    X = X.drop(columns=['indus' ,'chas', 'rad', 'tax', 'b'])
    print(X.columns)
    
    X, y = np.array(X), np.array(y)
    print(X.shape)
 
    X = min_max_scaler(X)
    X = polynomial_feature(X, degrees=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.shape)
    
    # model = LinearRegression(n_epoches=6000, learning_rate=0.05, batch_size=len(X)//2)
    # model.fit(X_train, y_train)
    # y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)
    
    # print("线性回归模型")
    # print(f"训练集")
    # print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    # print(f"R2: {r2_score(y_train, y_train_pred)}")
    
    # print(f"测试集")
    # print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    # print(f"R2: {r2_score(y_test, y_test_pred)}")
    
    # model.loss_plot()
    
    # 实验后min_batch = 12, gain_threshold = 1时最佳
    model = TreeRegressor(max_depth=10, min_batch=12, gain_threshold=1)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
        
    print("回归决策树")
    print(f"训练集")
    print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    print(f"R2: {r2_score(y_train, y_train_pred)}")
        
    print(f"测试集")
    print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    print(f"R2: {r2_score(y_test, y_test_pred)}")
    
    # train_scores = []
    # test_scores = []
    
    # for i in range(0, 5):
    #     model = TreeRegressor(max_depth=100, min_batch=2, gain_threshold=i)
    #     model.fit(X_train, y_train)
    #     y_train_pred = model.predict(X_train)
    #     y_test_pred = model.predict(X_test)
        
    #     print("回归决策树")
    #     print(f"训练集")
    #     print(f"MSE: {MeanSqaureError(y_train, y_train_pred)}")
    #     print(f"R2: {r2_score(y_train, y_train_pred)}")
    #     train_scores.append((i, float(MeanSqaureError(y_train, y_train_pred)), float(r2_score(y_train, y_train_pred))))
        
    #     print(f"测试集")
    #     print(f"MSE: {MeanSqaureError(y_test, y_test_pred)}")
    #     print(f"R2: {r2_score(y_test, y_test_pred)}")
    #     test_scores.append((i, float(MeanSqaureError(y_test, y_test_pred)), float(r2_score(y_test, y_test_pred))))
    
    # for i, j in zip(train_scores, test_scores):
    #     print(i, j)