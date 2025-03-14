import numpy as np

def Accuracy(y, y_pred):
    return np.mean(y==y_pred)