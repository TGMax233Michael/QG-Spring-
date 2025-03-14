import numpy as np

def r2_score(y, y_pred):
    return 1 - (np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2))

def MeanSqaureError(y, y_pred):
    return np.mean((y-y_pred)**2)