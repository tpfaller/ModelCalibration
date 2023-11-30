from sklearn.linear_model import LogisticRegression
import ml_insights as mli
from betacal import BetaCalibration


def platt_scaling(pred_proba_train, real_proba_train, pred_proba_val, real_proba_val):
    # Fit logistic regression
    lr = LogisticRegression(C=99999999999, solver='lbfgs')
    lr.fit(pred_proba_train.reshape(-1,1), real_proba_train)

    # Transform predicted probabilities into calibrated probabilities
    calib_proba_train = lr.predict_proba(pred_proba_train)
    calib_proba_val = lr.predict_proba(pred_proba_val)
    return calib_proba_train, calib_proba_val


def beta_scaling(pred_proba_train, real_proba_train, pred_proba_val, real_proba_val):
    # Fit beta calibrator
    bc = BetaCalibration()
    bc.fit(pred_proba_train.reshape(-1,1), real_proba_train)

    # Transform predicted probabilities into calibrated probabilities
    calib_proba_train = bc.predict_proba(pred_proba_train)
    calib_proba_val = bc.predict_proba(pred_proba_val)

    return calib_proba_train, calib_proba_val


def spline_calibration(pred_proba_train, real_proba_train, pred_proba_val, real_proba_val):
    # Fit spline calibrator
    splinecalib = mli.SplineCalib(penalty='l2',
                                knot_sample_size=40,
                                cv_spline=5,
                                unity_prior=False,
                                unity_prior_weight=128)
    splinecalib.fit(pred_proba_train, real_proba_train)

    # Transform predicted probabilities into calibrated probabilities
    calib_proba_train = splinecalib.predict_proba(pred_proba_train)
    calib_proba_val = splinecalib.predict_proba(pred_proba_val)

    return calib_proba_train, calib_proba_val


def main():
    pass


if __name__ == '__main__:':
    main()