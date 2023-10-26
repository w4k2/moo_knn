import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.model_selection import RepeatedStratifiedKFold
from methods import *
from sklearn.metrics import precision_score, recall_score
from pymoo.core.variable import Integer, Real

class KnnOptProblem(ElementwiseProblem):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        variables = dict()
        variables[f"x01"] = Integer(bounds=(1,5))
        variables[f"x02"] = Real(bounds=(0, 1))
        
        
        super().__init__(vars=variables, 
                         n_obj=2, # Liczba celi (objectives)
                         xl=np.array([1, 0]), # Dolna granica
                         xu=np.array([5, 1])) # Górna granica
        
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Set variables
        x = np.array([x[f"x01"], x[f"x02"]])
        k, weight = x
        n_splits = 2
        n_repeats = 5
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        X = np.copy(self.X)
        y = np.copy(self.y)

        # FOLDS x [precision x recall]
        scores = np.zeros((n_splits * n_repeats, 2))
        for fold, (train, test) in enumerate(rskf.split(X, y)):
            knn = ClassWeightedKNN(k_neighbors=k, weight=weight)
            knn.fit(X[train], y[train])
            y_pred = knn.predict(X[test])
            scores[fold, 0] = precision_score(y[test], y_pred, zero_division=0.0)
            scores[fold, 1] = recall_score(y[test], y_pred)

        out["F"] = np.mean(scores, axis=0)*-1