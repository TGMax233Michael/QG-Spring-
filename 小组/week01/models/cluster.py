import numpy as np

class MyKMeans:
    """
        KMeans Model
        
        Attributes:
            n_clusters: number of clusters
            n_epoches: number of iterations
            tolerance: centroid movement tolerance
            
    """
    def __init__(self, n_clusters, n_epoches=200, tolerance=1e-5):
        self.n_clusters = n_clusters
        self.n_epoches = n_epoches
        self.tolerance = tolerance
        self.labels = None
        self.centroids = None
    
    def _init_centroids(self, X: np.ndarray):
        self.centroids =  X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
    
    def _calc_distance(self, X: np.ndarray):
        return np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))
    
    def fit(self, X: np.ndarray):
        self._init_centroids(X)
        self.labels = np.zeros(shape=(X.shape[0]))
        
        for epoch in range(self.n_epoches):
            distances = self._calc_distance(X)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([np.mean(X[i == self.labels], axis=0) for i in range(self.n_clusters)])
            
            if np.allclose(new_centroids, self.centroids, rtol=0, atol=self.tolerance):
                break
            
            self.centroids = new_centroids        
            