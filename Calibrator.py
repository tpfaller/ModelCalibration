from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import ml_insights as mli
from betacal import BetaCalibration

class Calibrator:
    def __init__(self) -> None:
        self.platt_scaler = LogisticRegression(C=99999999999, solver='lbfgs')
        self.beta_calibrator = BetaCalibration()
        self.spline_calibrator = mli.SplineCalib(
            penalty='l2',                   
            knot_sample_size=40,
            cv_spline=5,
            unity_prior=False,
            unity_prior_weight=128
            )
        
        self.calibrator = dict()
        self.calibrator = {
            "platt_scaler": self.platt_scaler,
            "beta_calibrator": self.beta_calibrator,
            "spline_calibrator": self.spline_calibrator
        }

        self.confidence_scores = list()
        self.predictions = list()
        self.ground_truth = list()

    def fit(self, pred_proba_train: np.ndarray, real_proba_train: np.ndarray) -> None:
        for calibrator in self.calibrator.values():
            calibrator.fit(pred_proba_train.reshape(-1,1), real_proba_train)
    
    def transform(self, pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        calibrated_predictions = dict()
        for name, calibrator in self.calibrator.items():
            calibrated_predictions[name] = calibrator.predict_proba(pred_proba)
        return calibrated_predictions

    def fit_transform(self, pred_proba_train: np.ndarray, real_proba_train: np.ndarray) -> Dict[str, np.ndarray]:
        self.fit(pred_proba_train, real_proba_train)
        self.transform(pred_proba=pred_proba_train)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self.confidence_scores.append(torch.nn.functional.softmax(output, dim=1))
        self.predictions.append(torch.argmax(output, dim=1))
        self.ground_truth.append(target)

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np_ground_truth = np.concatenate([x.numpy() for x in self.ground_truth])
        np_predictions = np.concatenate([x.numpy() for x in self.predictions])
        np_confidence = np.concatenate([x[:,0].numpy() for x in self.confidence_scores])
        return np_ground_truth, np_predictions, np_confidence
        