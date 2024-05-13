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

# ratio x [k <1, 5>]
collected_k = np.zeros((len(classes_weights), 5))

# ratio x weight x parameters number
collected_weight = np.zeros((len(classes_weights), 2)) 

for data_idx in range(n_datasets):
    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    for weight_idx, weight in enumerate(classes_weights):
        X, y = make_classification(n_samples=1000, n_classes=2 , weights=weight, random_state=42*data_idx)
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42*data_idx)

        gen = rskf.split(X, y)
        train, test = next(gen)
        parameters = np.load(f"results/split/opt_variables_0_{data_idx}_{weight[1]}.npy", allow_pickle=True)
        scores = np.zeros((len(parameters), 2))

        for param_idx, param in enumerate(parameters):
            collected_k[weight_idx, param["x01"]-1] += 1
            collected_weight[weight_idx, 0] += param["x02"]
            collected_weight[weight_idx, 1] += 1

print(collected_k)

a = collected_weight[:, 0] / collected_weight[:, 1]
print(1/a)
print(collected_weight[:, 1])