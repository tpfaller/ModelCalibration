from typing import Literal, Mapping

import numpy as np
import matplotlib.pyplot as plt
from betacal import BetaCalibration
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import ml_insights as mli

class SplineCalibrationWrapper(mli.SplineCalib):
    def __init__(self):
        super().__init__()

    def fit(self, x,y):
        x = x.squeeze()
        super().fit(x, y)

    def predict_proba(self, x):
        x = x.squeeze()
        return self.calibrate(x)


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
    

#custom functions to generate reliability diagram
def calc_bins(y_test, preds):
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)  
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (y_test[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]
        else:
            bin_accs[bin] = 0.
            bin_confs[bin] = 0.

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_calibration_metrics(y_test, preds):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_test, preds)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    return ECE,MCE


def plot_calibration_curves(filename, y_true, uncalib, platt, beta, spline, bins=10):
    prob_pos, probs_uncalib = calibration_curve(y_true, uncalib, n_bins=bins)
    pos_platt, probs_platt = calibration_curve(y_true, platt, n_bins=bins)
    pos_beta, probs_beta = calibration_curve(y_true, beta, n_bins=bins)
    pos_spline, probs_spline = calibration_curve(y_true, spline, n_bins=bins)

    plt.plot(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10), color="black", linestyle="--")
    plt.plot(probs_uncalib, prob_pos, color="red", marker='o')
    plt.plot(probs_platt ,pos_platt, color="green", marker='x')
    plt.plot(probs_beta ,pos_beta, color="blue", marker='*')
    plt.plot(probs_spline ,pos_spline, color="purple", marker='^')
    plt.legend(["perfect", "uncalibrated", "platt", "beta", "spline"])
    plt.xlabel('Mittlere geschätzte Wahrscheinlichkeit')
    plt.ylabel('Mittlere accuracy')
    plt.xlim([.0, 1.0])
    plt.ylim([.0, 1.0])
    plt.grid()
    plt.savefig(filename)
    