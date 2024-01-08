import os
import argparse
from collections import OrderedDict

import numpy as np
import torch
import mlflow
from tqdm import tqdm

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torchvision
from torch import nn

import models
from calibrators import PlattScaler, BetaCalibrationWrapper, SplineCalibrationWrapper, get_calibration_metrics, plot_calibration_curves
from image_functions.presets import ClassificationPresetTrain, ClassificationPresetEval

def get_metrics(y_true: np.ndarray, proba: np.ndarray):
    y_pred = np.where(proba > .5 , 1, 0)
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=np.nan)
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, zero_division=np.nan)
    f1_score = (2 * precision * recall) / (precision + recall)
    aucroc = metrics.roc_auc_score(y_true=y_true, y_score=proba)
    return acc, precision, recall, f1_score, aucroc


def train(model, optimizer, loader, y_train, criterion, device, epoch):
    losses = []
    predicted_probas = []
    y_true = []
    for image, target in loader:
        optimizer.zero_grad()
        image, target = image.to(device), target.to(device)
        predictions = model(image)
        loss = criterion(predictions[:, 1], target.to(torch.float32))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        predicted_probas.append(predictions.detach().numpy()[:, 1])
        y_true.append(target.detach().numpy())

    proba = np.concatenate(predicted_probas, axis=0)
    y_train = np.concatenate(y_true, axis=0)
    acc, precision, recall, f1_score, aucroc = get_metrics(y_true=y_train, proba=proba)
    ece, mce = get_calibration_metrics(y_test=y_train, preds=proba)
    metrics = {
            "Train Loss": sum(losses) / len(losses),
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


def eval_test(model, loader, y_test, criterion, device, epoch):
    image, target = next(iter(loader))
    with torch.no_grad():
        predictions_test = model(image.to(device))
        loss = criterion(predictions_test[:, 1], target.to(torch.float32))
    
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


def eval_validation(model, loader, y_val, criterion, device, epoch):
    image, target = next(iter(loader))
    with torch.no_grad():
        prediction = model(image.to(device))
        loss = criterion(prediction[:, 1], target.to(torch.float32))

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
    # Parameter List
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--ratio", type=str, default="1.0", choices=["1.0", "0.6", "0.2"])
    args = parser.parse_args()

    learning_rate = args.lr
    device = "cpu"
    epochs = args.epochs

    reduce = True
    out_dir = f"output/results/images/ratio_{args.ratio}"

    model = torchvision.models.get_model("mobilenet_v3_small", weights="DEFAULT")
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier= nn.Sequential(OrderedDict([
        ("0", nn.Linear(in_features=576, out_features=1024, bias=True)),
        ("1", nn.Hardswish()),
        ("2", nn.Dropout(p=0.2, inplace=True)),
        ("3", nn.Linear(in_features=1024, out_features=2, bias=True)),
        ("4", nn.Sigmoid())
    ]))
    model.to(device)


    train_preproc = ClassificationPresetTrain(resize_size=args.image_size, crop_size=args.image_size)
    test_preproc = ClassificationPresetEval(resize_size=args.image_size, crop_size=args.image_size)

    train_set = torchvision.datasets.ImageFolder("data/train_1.0", transform=train_preproc)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)


    test_set = torchvision.datasets.ImageFolder("data/test", transform=test_preproc)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    _, y_test = next(iter(test_loader))
    y_test = y_test.numpy()

    val_set = torchvision.datasets.ImageFolder("data/val", transform=test_preproc)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    _, y_val = next(iter(val_loader))
    y_val = y_val.numpy()

    print(f"Train Ratio:  {args.ratio}")
    print(f"Test Ratio:  {np.sum(y_test) / (y_test.shape[0] - np.sum(y_test))}")
    print(f"Val Ratio:  {np.sum(y_val) / (y_val.shape[0] - np.sum(y_val))}")

    print(out_dir)

    mlflow.log_params(
        {

            "test_pos": np.sum(y_test), 
            "test_neg": y_test.shape[0] - np.sum(y_test),
            "val_pos": np.sum(y_val), 
            "val_neg": y_val.shape[0] - np.sum(y_val),
            "ratio": args.ratio, 
            "reduce": reduce, 
            "lr": learning_rate,
            "wd": args.weight_decay
        }
    )

    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCELoss()
    
    calibrators = {
        "platt_scaler": PlattScaler(),
        "beta_calibrator": BetaCalibrationWrapper(),
        "spline_calibrator": SplineCalibrationWrapper()
    }

    highest_auc = .0
    for epoch in tqdm(range(epochs)):
        m_train = train(model, optim, train_loader, None, criterion, device, epoch)
        m_test, proba_test = eval_test(model, test_loader, y_test, criterion, device, epoch)
        m_val, proba_val = eval_validation(model, val_loader, y_val, criterion, device, epoch)
        
        
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
        print("Highest AUC: ", highest_auc)
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
