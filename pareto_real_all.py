import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.datasets import make_classification
from methods import *
import os
from sklearn.neighbors import KNeighborsClassifier

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

        parameters_ref = np.load(f"results/real_reference/opt_variables_0_{dataset}.npy", allow_pickle=True)
        scores_ref = np.zeros((len(parameters_ref), 2))

        parameters_weighted = np.load(f"results/real/opt_variables_0_{dataset}.npy", allow_pickle=True)
        scores_weighted = np.zeros((len(parameters_weighted), 2))

        parameters_bac = np.load(f"results/genetic/opt_variables_0_{dataset}.npy", allow_pickle=True)
        scores_bac = np.zeros((3))

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42*data_idx)
        gen = rskf.split(X, y)
        train, test = next(gen)

        for param_idx, param in enumerate(parameters_ref):
            knn = KNeighborsClassifier(n_neighbors=param["x01"], weights="distance")
            knn.fit(X[train], y[train])
            y_pred = knn.predict(X[test])
            scores_ref[param_idx, 0] = precision_score(y[test], y_pred, zero_division=0.0)
            scores_ref[param_idx, 1] = recall_score(y[test], y_pred)

        for param_idx, param in enumerate(parameters_weighted):
            knn = ClassWeightedKNN(k_neighbors=param["x01"], weight=param["x02"])
            knn.fit(X[train], y[train])
            y_pred = knn.predict(X[test])
            scores_weighted[param_idx, 0] = precision_score(y[test], y_pred, zero_division=0.0)
            scores_weighted[param_idx, 1] = recall_score(y[test], y_pred)

        parameters_bac = parameters_bac.item()
        knn = ClassWeightedKNN(k_neighbors=parameters_bac["x01"], weight=parameters_bac["x02"])
        knn.fit(X[train], y[train])
        y_pred = knn.predict(X[test])
        scores_bac[0] = precision_score(y[test], y_pred, zero_division=0.0)
        scores_bac[1] = recall_score(y[test], y_pred)
            
        marker_size = 150
        fig, ax = plt.subplots(1, 1, figsize=(8, 8*0.618))
        ax.scatter(scores_weighted[:,0], scores_weighted[:,1], marker="s", c="cornflowerblue", s=marker_size)
        ax.scatter(scores_bac[0], scores_bac[1], marker="*", c="gold", s=marker_size)
        ax.scatter(scores_ref[:,0], scores_ref[:,1], marker="x", c="tomato", s=marker_size)
        ax.set_xlabel("precision")
        ax.set_ylabel("recall")
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.suptitle(f"{dataset_name}")
        fig.legend(["weighted k-NN", "genetic BAC", "sklearn k-NN"], frameon=False)
        plt.tight_layout()
        plt.savefig(f"figures/full/exp_all_{dataset_name}.png")
        plt.savefig(f"figures/full/exp_all_{dataset_name}.eps")
        plt.close()
        #exit()
        
