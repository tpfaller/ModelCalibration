from typing import Literal, Mapping
from betacal import BetaCalibration
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
import ml_insights as mli

class BetaCalibrationWrapper(BetaCalibration):
    def __init__(self, parameters="abm"):
        super().__init__(parameters)

    def predict_proba(self, x):
        return self.predict(x)
    
class PlattScaler(LogisticRegression):
    def __init__(self, penalty: Literal['l1', 'l2', 'elasticnet'] | None = "l2", *, dual: bool = False, tol: float = 0.0001, C: float = 1, fit_intercept: bool = True, intercept_scaling: float = 1, class_weight: Mapping | str | None = None, random_state: int | RandomState | None = None, solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = "lbfgs", max_iter: int = 100, multi_class: Literal['auto', 'ovr', 'multinomial'] = "auto", verbose: int = 0, warm_start: bool = False, n_jobs: int | None = None, l1_ratio: float | None = None) -> None:
        super().__init__(penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    def predict_proba(self, x):
        return super().predict_proba(x)[:,1]