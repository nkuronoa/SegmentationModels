# coding: utf-8
import argparse
import os
import pathlib
import time
from dataclasses import dataclass
from logging import (CRITICAL, DEBUG, ERROR, INFO, WARNING, FileHandler,
                     Formatter, StreamHandler, getLogger)
from logging.handlers import RotatingFileHandler

import mlflow
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from augment import (get_preprocessing, get_training_augmentation,
                     get_validation_augmentation)
from dataset import Dataset
from model import load_segment_model
from util import DatasetPath, ModelParam, Output


@dataclass
class TrainParam:
    epoch: int
    batch: int
    lr: float


def create_logger(loglevel: str, logpath: str):
    logger = getLogger(__name__)
    # set loglevel
    logger.setLevel(DEBUG)

    # set handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(loglevel.upper())

    # set log output format
    handler_format = Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(handler_format)

    # set file handler
    file_handler = FileHandler(logpath)
    #file_handler = RotatingFileHandler(logpath, maxBytes=100000, backupCount=10)
    file_handler.setLevel(loglevel.upper())
    file_handler_format = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def train(model, train_loader, valid_loader, train_param, output_param, logger):
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5), ]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=train_param.lr), ])
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0

    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []

    start = time.perf_counter()

    for i in range(train_param.epoch):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, output_param.modelpath)
            print('Model saved!')

        logger.debug("epoch:{}, lr:{}".format(i + 1, scheduler.get_last_lr()))
        scheduler.step()

    end = time.perf_counter()

    elapsed_time = end - start
    print("elapsed train time:{} sec".format(elapsed_time))

    logger.debug(output_param)
    mlflow.set_tracking_uri(output_param.mldir)
    mlflow.set_experiment(output_param.experiment)
    with mlflow.start_run() as run:
        # save param with mlflow
        mlflow.log_metric("traintime", elapsed_time)
        mlflow.log_metric("max_score", max_score)
        mlflow.log_metrics({"Loss/train": train_logs["dice_loss"], "Score/train": train_logs["iou_score"]})
        mlflow.log_metrics({"Loss/val": valid_logs["dice_loss"], "Score/val": valid_logs["iou_score"]})

        mlflow.log_params({"epochs": train_param.epoch, "lr": train_param.lr, "batch": train_param.batch})

        mlflow.log_artifact(os.path.join(output_param.logdir, output_param.logname))


def arg_parse():
    parser = argparse.ArgumentParser("Pytorch segmentation sample")
    parser.add_argument("--input", "-i", default="../../../Datasets/PublicDatasets/CamVid", type=str, help="public dataset name")
    parser.add_argument("--epoch", "-e", default=50, type=int, help="training epochs")
    parser.add_argument("--batch", "-b", default=8, type=int, help="train batch")
    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--modeltype", "-mt", default="UNet++", type=str, choices=["UNet++", "FPN", "DeepLabv3+"], help="model type")
    parser.add_argument(
        "-encoder",
        default="resnet34",
        type=str,
        choices=[
            "mobilenet_v2",
            "resnet34",
            "resnet50",
            "densenet121",
            "efficientnet-b0",
            "efficientnet-b4"],
        help="encoder model type")
    parser.add_argument("-model", default="model", type=str, help="output model path")
    parser.add_argument("-mldir", default="mllogs/train/mlruns", type=str, help="mlflow dir")
    parser.add_argument("--experiment", "-ex", default="test", type=str, help="mlflow experiment")
    parser.add_argument("--loglevel", default="debug", type=str, choices=["debug", "info", "warning", "error", "critical"], help="log level")
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    # initialize
    inputpath = pathlib.Path(args.input)
    train_set = DatasetPath(inputpath / "train", inputpath / "trainannot")
    val_set = DatasetPath(inputpath / "val", inputpath / "valannot")
    classes = [item.strip() for item in open(inputpath / "class.txt", "r")]

    model_param = ModelParam(args.modeltype, args.encoder)
    train_param = TrainParam(args.epoch, args.batch, args.learning_rate)

    modelpath = os.path.join(args.model, model_param.modeltype + "_" + model_param.encoder + ".pth")
    output_param = Output(modelpath, args.experiment, args.mldir)

    # logger
    logger = create_logger(args.loglevel, os.path.join(output_param.logdir, output_param.logname))
    logger.info(classes)

    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation

    model = load_segment_model(model_param.modeltype, model_param.encoder, model_param.weights, n_classes, model_param.activation)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_param.encoder, model_param.weights)

    train_dataset = Dataset(
        str(train_set.x_path),
        str(train_set.y_path),
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    valid_dataset = Dataset(
        str(val_set.x_path),
        str(val_set.y_path),
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_param.batch, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    train(model, train_loader, valid_loader, train_param, output_param, logger)


if __name__ == "__main__":
    main()
