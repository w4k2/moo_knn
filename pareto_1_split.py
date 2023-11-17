import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.datasets import make_classification
from methods import *

classes_weights = [[1, 1], [0.9, 0.1], [0.95, 0.05], [0.99, 0.01], [0.995, 0.005]]
n_datasets = 10
n_splits = 2
n_repeats = 5

for data_idx in range(n_datasets):
    plt.figure()
    for weight_idx, weight in enumerate(classes_weights):
        X, y = make_classification(n_samples=1000, n_classes=2 , weights=weight, random_state=42*data_idx)
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42*data_idx)

        gen = rskf.split(X, y)
        train, test = next(gen)
        parameters = np.load(f"results/split/opt_variables_0_{data_idx}_{weight[0]}.npy", allow_pickle=True)
        scores = np.zeros((len(parameters), 2))

        for param_idx, param in enumerate(parameters):
            knn = ClassWeightedKNN(k_neighbors=param["x01"], weight=param["x02"])
            knn.fit(X[train], y[train])
            y_pred = knn.predict(X[test])
            scores[param_idx, 0] = precision_score(y[test], y_pred, zero_division=0.0)
            scores[param_idx, 1] = recall_score(y[test], y_pred)

            
        plt.scatter(scores[:,0], scores[:,1])
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.legend(["minority weight-1", "minority weight-0.1", "minority weight-0.05", "minority weight-0.01", "minority weight-0.005"])
        plt.savefig(f"figures/split/exp1_{data_idx}.png")
