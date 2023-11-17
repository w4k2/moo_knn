from methods import *
from sklearn.datasets import make_classification
import numpy as np
from problem import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold

classes_weights = [[1, 1], [0.9, 0.1], [0.95, 0.05], [0.99, 0.01], [0.995, 0.005]]
n_datasets = 10
n_splits = 2
n_repeats = 5

for weight_idx, weight in enumerate(classes_weights):
    for data_idx in range(n_datasets):
        X, y = make_classification(n_samples=1000, n_classes=2 , weights=weight, random_state=42*data_idx)

        problem = KnnOptProblem(X, y)

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
            
        np.save(f"results/without_cv/opt_variables_{data_idx}_{weight[1]}", res.X)
        np.save(f"results/without_cv/opt_metrics_{data_idx}_{weight[1]}", res.F)
