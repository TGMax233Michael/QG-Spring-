import numpy as np

# R² = 1 - (Σ(y - y_pred)² / Σ(y - ȳ)²)
# R² ∈ [0, 1]
# R² -> 1 拟合效果好
def r2_score(y, y_pred):
    return 1 - (np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2))

# MSE = 1 / N * Σ(y - y_pred)²
def MeanSqaureError(y, y_pred):
    return np.mean((y-y_pred)**2)