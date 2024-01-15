import os
import argparse

import numpy as np
import torch
import mlflow
from tqdm import tqdm

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import models
from calibrators import PlattScaler, BetaCalibrationWrapper, SplineCalibrationWrapper, get_calibration_metrics, plot_calibration_curves


def get_dataset(ratio: float=1.0, reduce: bool=False, dataset:str="synthetic"):
    X, y = make_classification(
    n_samples=20000, n_features=20, n_informative=2, n_redundant=2, random_state=42
    )

    train_samples = 2000  # Samples used for training the models
    ratio = 1 / ratio

    if not reduce:
        train_samples = int(ratio * train_samples)

    X_train, X_first, y_train, y_first = train_test_split(
        X,
        y,
        shuffle=True,
        stratify=y,
        train_size=train_samples,
        test_size=2000,
    )

    all_negatives, all_positives = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
    reduced_positives = all_positives[:int(all_positives.shape[0] / ratio)]
    X_train = np.concatenate([X_train[all_negatives], X_train[reduced_positives]])
    y_train = np.concatenate([y_train[all_negatives], y_train[reduced_positives]])


    X_test, X_val, y_test, y_val = train_test_split(
        X_first,
        y_first,
        stratify=y_first,
        shuffle=True,
        test_size=1000,
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_metrics(y_true: np.ndarray, proba: np.ndarray):
    y_pred = np.where(proba > .5 , 1, 0)
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=np.nan)
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, zero_division=np.nan)
    f1_score = (2 * precision * recall) / (precision + recall)
    aucroc = metrics.roc_auc_score(y_true=y_true, y_score=proba)
    return acc, precision, recall, f1_score, aucroc


def train(model, optimizer, X_train, y_train, criterion, device, epoch):
    optimizer.zero_grad()
    predictions = model(torch.Tensor(X_train).to(device))
    target=torch.stack([1-torch.Tensor(y_train), torch.Tensor(y_train)],axis=1).squeeze().to(device)
    loss = criterion(predictions, target)
    loss.backward()
    optimizer.step()

    # y_pred = torch.argmax(predictions, 1).numpy()
    proba = predictions.detach().numpy()[:, 1]
    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_train, proba=proba)
    ece, mce = get_calibration_metrics(y_test=y_train, preds=proba)
    metrics = {
            "Train Loss": loss.item(),
            "Train Accuracy": acc,
            "Train Precision": precision,
            "Train Recall": recall,
            "Train F1 Score": f1_score,
            "Train AUC": aucroc,
            "Train ECE": ece,
            "Train MCE": mce
        }
    mlflow.log_metrics(metrics, step=epoch)
    return metrics


def train_calibrator(calibrator, proba_test, y_test):
    # Fit Calibrator on Test Data
    calibrator.fit(proba_test.reshape(-1, 1), y_test)
    return calibrator


def eval_test(model, X_test, y_test, criterion, device, epoch):
    with torch.no_grad():
        predictions_test = model(torch.Tensor(X_test).to(device))
        target=torch.stack([1-torch.Tensor(y_test), torch.Tensor(y_test)],axis=1).squeeze().to(device)
        loss = criterion(predictions_test, target)
    
    proba_test = predictions_test.numpy()[:, 1]
    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_test, proba=proba_test)
    ece, mce = get_calibration_metrics(y_test=y_test, preds=proba_test)
    metrics = {
            "Test Loss": loss.item(),
            "Test Accuracy": acc,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1 Score": f1_score,
            "Test AUC": aucroc,
            "Test ECE": ece,
            "Test MCE": mce
        }
    mlflow.log_metrics(metrics, step=epoch)

    return metrics, proba_test


def eval_calibrated_test(calibrator, calibratorname, proba_test, y_test, epoch):
    calibrated_proba_test = calibrator.predict_proba(proba_test.reshape(-1, 1))

    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_test, proba=calibrated_proba_test)
    ece, mce = get_calibration_metrics(y_test=y_test, preds=calibrated_proba_test)
    metrics = {
            f"Test {calibratorname} Accuracy": acc,
            f"Test {calibratorname} Precision": precision,
            f"Test {calibratorname} Recall": recall,
            f"Test {calibratorname} F1 Score": f1_score,
            f"Test {calibratorname} AUC": aucroc,
            f"Test {calibratorname} ECE": ece,
            f"Test {calibratorname} MCE": mce
        }
    mlflow.log_metrics(metrics ,step=epoch)
    return metrics


def eval_validation(model, X_val, y_val, criterion, device, epoch):
    with torch.no_grad():
        prediction = model(torch.Tensor(X_val).to(device))
        target=torch.stack([1-torch.Tensor(y_val), torch.Tensor(y_val)],axis=1).squeeze().to(device)
        loss = criterion(prediction, target)

    proba_val = prediction.numpy()[:, 1]
    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_val, proba=proba_val)
    ece, mce = get_calibration_metrics(y_test=y_val, preds=proba_val)
    metrics = {
            "Val Loss": loss.item(),
            "Val Accuracy": acc,
            "Val Precision": precision,
            "Val Recall": recall,
            "Val F1 Score": f1_score,
            "Val AUC": aucroc,
            "Val ECE": ece,
            "Val MCE": mce,
        }
    mlflow.log_metrics(metrics ,step=epoch)
    return metrics, proba_val


def eval_calibrated_validation(calibrator, calibratorname, proba_val, y_val, epoch):
    calibrated_proba_val = calibrator.predict_proba(proba_val.reshape(-1, 1))
    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_val, proba=calibrated_proba_val)
    ece, mce = get_calibration_metrics(y_test=y_val, preds=calibrated_proba_val)
    metrics = {
            f"Val {calibratorname} Accuracy": acc,
            f"Val {calibratorname} Precision": precision,
            f"Val {calibratorname} Recall": recall,
            f"Val {calibratorname} F1 Score": f1_score,
            f"Val {calibratorname} AUC": aucroc,
            f"Val {calibratorname} ECE": ece,
            f"Val {calibratorname} MCE": mce,
        }
    mlflow.log_metrics(metrics ,step=epoch)
    return metrics, calibrated_proba_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=0.6)
    args = parser.parse_args()


    # Parameter List
    learning_rate = 0.01
    device = "cpu"
    epochs = args.epochs

    dataset="synthetic"
    ratio = args.ratio
    reduce = True
    out_dir = f"output/results/{dataset}_{ratio}{'_red' if reduce else ''}"

    X_train, X_test, X_val, y_train, y_test, y_val = get_dataset(ratio=ratio, reduce=reduce,dataset=dataset)
    print(f"Train Ratio:  {np.sum(y_train) / (y_train.shape[0] - np.sum(y_train))}")
    print(f"Test Ratio:  {np.sum(y_test) / (y_test.shape[0] - np.sum(y_test))}")
    print(f"Val Ratio:  {np.sum(y_val) / (y_val.shape[0] - np.sum(y_val))}")

    print(out_dir)

    mlflow.log_params(
        {
            "train_pos": np.sum(y_train), 
            "train_neg": y_train.shape[0] - np.sum(y_train),
            "test_pos": np.sum(y_test), 
            "test_neg": y_test.shape[0] - np.sum(y_test),
            "val_pos": np.sum(y_val), 
            "val_neg": y_val.shape[0] - np.sum(y_val),
            "ratio": ratio, 
            "reduce": reduce
        }
    )


    model = models.LogRegression(X_train.shape[1])
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    
    calibrators = {
        "platt_scaler": PlattScaler(),
        "beta_calibrator": BetaCalibrationWrapper(),
        "spline_calibrator": SplineCalibrationWrapper()
    }

    highest_auc = .0
    for epoch in tqdm(range(epochs)):
        m_train = train(model, optim, X_train, y_train, criterion, device, epoch)
        m_test, proba_test = eval_test(model, X_test, y_test, criterion, device, epoch)
        m_val, proba_val = eval_validation(model, X_val, y_val, criterion, device, epoch)
        
        
        if m_val["Val AUC"] > highest_auc:
            highest_auc = m_val["Val AUC"]
            metrics = {}
            metrics["epoch"] = epoch
            metrics["proba_val"] = proba_val
            metrics["y_true"] = y_val
            metrics.update(m_train)
            metrics.update(m_test)
            metrics.update(m_val)
            for name, calibrator in calibrators.items():
                calibrator = train_calibrator(calibrator, proba_test, y_test)
                m_cal_test = eval_calibrated_test(calibrator, name, proba_test, y_test, epoch)
                m_cal_val, cal_proba = eval_calibrated_validation(calibrator, name, proba_val, y_val, epoch)
                metrics.update(m_cal_test)
                metrics.update(m_cal_val)
                metrics[f"{name}_proba"] = cal_proba

    if highest_auc > .0:
        print("Datset Ratio: ", args.ratio)
        print("Highest AUC: ", highest_auc)
        print("Val ECE: ", metrics["Val ECE"])
        print("Val platt_scaler ECE: ", metrics["Val platt_scaler ECE"])
        print("Val beta_calibrator ECE: ", metrics["Val beta_calibrator ECE"])
        print("Val spline_calibrator ECE: ", metrics["Val spline_calibrator ECE"])
        os.makedirs(out_dir, exist_ok=True)
        calib_file = os.path.join(out_dir, f"calibration_curve_{metrics['epoch']}.png")

        plot_calibration_curves(
            filename=calib_file, 
            y_true=metrics["y_true"], 
            uncalib=metrics["proba_val"],
            platt=metrics["platt_scaler_proba"],
            beta=metrics["beta_calibrator_proba"],
            spline=metrics["spline_calibrator_proba"]
            )


if __name__ == '__main__':
    with mlflow.start_run():
        main()
