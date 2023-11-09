import numpy as np
import matplotlib.pyplot as plt

classes_weights = [[1, 1], [0.1, 0.9], [0.05, 0.95], [0.01, 0.99], [0.005, 0.995]]
n_datasets = 10


for data_idx in range(n_datasets):
    plt.figure()
    for weight_idx, weight in enumerate(classes_weights):
        results = np.load(f"results/without_cv/opt_metrics_{data_idx}_{weight[0]}.npy")*-1
        
        plt.scatter(results[:,0], results[:,1])
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.legend(["minority weight-1", "minority weight-0.1", "minority weight-0.05", "minority weight-0.01", "minority weight-0.005"])
    plt.savefig(f"figures/without_cv/exp1_{data_idx}.png")