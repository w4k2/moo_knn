import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from methods import *
import os

n_datasets = 10
n_splits = 2
n_repeats = 5

datasets_dir = "datasets"

datasets = os.listdir(datasets_dir)

for data_idx, dataset in enumerate(datasets):
    dataset_name = dataset.split(".")[0]
    plt.figure()
    path = os.path.join(datasets_dir, dataset)
    
    data = np.genfromtxt(path, delimiter=',')
    print(data.shape)
    X = data[:,:-1]
    y = data[:,-1].astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 2 and counts[0]/counts[1] > 2.5:

        parameters = np.load(f"results/genetic/opt_variables_0_{dataset}.npy", allow_pickle=True)
        scores = np.zeros((3))

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42*data_idx)
        gen = rskf.split(X, y)
        train, test = next(gen)

        # for param_idx, param in enumerate(parameters):
        parameters = parameters.item()
        knn = ClassWeightedKNN(k_neighbors=parameters["x01"], weight=parameters["x02"])
        knn.fit(X[train], y[train])
        y_pred = knn.predict(X[test])
        scores[0] = precision_score(y[test], y_pred, zero_division=0.0)
        scores[1] = recall_score(y[test], y_pred)
        scores[2] = balanced_accuracy_score(y[test], y_pred)

            
        fig, ax = plt.subplots(1, 1, figsize=(13, 10))
        ax.scatter(scores[0], scores[1])
        ax.set_xlabel("precision")
        ax.set_ylabel("recall")
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.suptitle(f"{dataset_name} -- {scores[2]}")
        plt.tight_layout()
        plt.savefig(f"figures/genetic/exp3_{dataset_name}.png")
        plt.close()
        
