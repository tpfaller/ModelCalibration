import datetime
import os
import time

from sklearn import metrics

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn

from torchvision.transforms.functional import InterpolationMode


import mlflow


def get_metrics(target: torch.Tensor, output: torch.Tensor):
    if target.ndim == 2:
        target = target.max(dim=1)[1]
    labels = torch.argmax(output, 1)
    y_true, y_pred = target.numpy(), labels.numpy()
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    return acc, precision


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, precision = get_metrics(target=target, output=output)
        print(acc, precision)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc"].update(acc, n=batch_size)
        metric_logger.meters["precision"].update(precision, n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    # print(f"{header} Acc@1 {metric_logger.acc.global_avg:.3f}") # Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            
            acc, precision = get_metrics(target=target, output=output)

            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc"].update(acc, n=batch_size)
            metric_logger.meters["precision"].update(precision, n=batch_size)

    print(f"{header} Acc {metric_logger.acc.global_avg:.3f}") # Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger


def change_head(model, model_name, num_classes):
    if "resnet" in model_name:
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    elif "efficientnet" in model_name:
        model.classifier = nn.Sequential(
            nn.Dropout(p=.2, inplace=True),
            nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        )
    elif "mobilenet" in model_name:
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
    else:
        assert True, "Something went wrong."
    return model


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
 
            backend=args.backend,
            use_v2=args.use_v2,
        ),
    )

    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    print("Creating data loaders")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    mlflow.log_params({
        "learning_rate": args.lr,
        "model": args.model,
        "batch_size": args.batch_size,
        "optimizer": args.opt,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay
    })

    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights)
    model = change_head(model=model, model_name=args.model, num_classes=num_classes)
    model.to(device)

    class_weights = torch.Tensor([1.23796424, 5.20231214])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    parameters = model.parameters()
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    val_logger = evaluate(model, criterion, data_loader_test, device=device)

    mlflow.log_metrics({
        "Val-Accuracy": val_logger.acc.global_avg,
        "Val-Loss":  val_logger.loss.global_avg
    }, step=0)

    print("Start training")
    start_time = time.time()
    for epoch in range(0, args.epochs):
        train_logger = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args)
        val_logger = evaluate(model, criterion, data_loader_test, device=device)

        mlflow.log_metrics({
            "Train-Accuracy": train_logger.acc.global_avg,
            "Train-Precision": train_logger.precision.global_avg,
            "Train-Loss":  train_logger.loss.global_avg,
            "Val-Accuracy": val_logger.acc.global_avg,
            "Val-Precision": val_logger.precision.global_avg,
            "Val-Loss":  val_logger.loss.global_avg
        }, step=epoch + 1)

        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }

            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    with mlflow.start_run():
        main(args)
