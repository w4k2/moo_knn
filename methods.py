import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class ClassWeightedKNN(BaseEstimator, ClassifierMixin):
    """
    Class Weighted k-NN.  
    """
    def __init__(self, k_neighbors=5, weight=None, normalization=False):
        self.k_neighbors = k_neighbors
        self.weight = weight # <0, 1>
        self.normalization = normalization

    def fit(self, X, y, sample_weight=None):
        self.X_, self.y_ = (
            np.copy(X),
            np.copy(y),
        )

        self.classes_ = np.unique(y)
    
    def predict(self, X_test):
        if self.normalization:
            scaler = StandardScaler()
            self.X_ = scaler.fit_transform(self.X_)
            X_test = scaler.fit_transform(X_test)

        distances = cdist(self.X_, X_test)  

        if self.weight == 0:
            weight = 0
        else:
            weight = 1/self.weight
            
        distances[self.y_==1] = distances[self.y_==1] * weight

        sorted_distances = np.argsort(distances, axis=0)
        n_distances = sorted_distances[:int(self.k_neighbors)]
        y_s = self.y_[n_distances]
        preds, counts = mode(y_s, axis=0, keepdims=True)
        preds = preds.reshape(X_test.shape[0])

        return preds

class CowSplit(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, weights=None, centroids=None):
        self.k = k
        self.weights = weights
        self.centroids = centroids

    def split(self, X):
        # Assign all samples to regions
        distances = cdist(self.centroids, X)
        regions = np.argmin(distances, axis=0)

        return regions

    def fit(self, X, y):
        regions = self.split(X)
        self.knns = []
        # Initialize and fit k-NNs for every regions
        for i in range(len(self.centroids)):
            knn = ClassWeightedKNN(n_neighbors=self.k)
            knn.fit(X[regions==i], y[regions==i])
            self.knns.append(knn)
    
    def predict(self, X):
        regions = self.split(X)
        preds = []
        # Predict probas in their regions
        for i in range(len(self.centroids)):
            pred = self.knns[i].predict(X[regions==i])
            preds.append(pred)

        return preds, regions
        
