from methods import *
import numpy as np
from problem import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold
import os

n_splits = 2
n_repeats = 5
datasets_dir = "datasets"

datasets = os.listdir(datasets_dir)

for data_idx, dataset in enumerate(datasets):
    path = os.path.join(datasets_dir, dataset)
    
    data = np.genfromtxt(path, delimiter=',')
    print(data.shape)
    X = data[:,:-1]
    y = data[:,-1].astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 2 and counts[0]/counts[1] > 2.5:
    
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42*data_idx)

        for fold, (train, test) in enumerate(rskf.split(X, y)):

            problem = KnnOptProblem(X[train], y[train])

            algorithm = NSGA2(
                pop_size=40,
                n_offsprings=10,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                )

            termination = get_termination("n_gen", 50)

            res = minimize(problem,
                        algorithm,
                        termination,
                        seed=42,
                        save_history=True,
                        verbose=True)
            
            np.save(f"results/real/opt_variables_{fold}_{dataset}", res.X)
            np.save(f"results/real/opt_metrics_{fold}_{dataset}", res.F)

    else:
        print(f"{dataset} is balanced or multiclass")
