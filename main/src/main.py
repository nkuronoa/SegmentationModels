# coding: utf-8
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from dataclasses import dataclass
from dataset import Dataset
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from augment import get_preprocessing, get_training_augmentation, get_validation_augmentation
from model import load_segment_model
from torch.utils.tensorboard import SummaryWriter

from util import load_list
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL


@dataclass
class DatasetPath:
    x_path: pathlib.Path
    y_path: pathlib.Path


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def train(model, train_loader, valid_loader, epochs, modelpath, log_dir="logs"):

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

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

    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    max_score = 0

    #train accuracy, train loss, val_accuracy, val_loss をグラフ化できるように設定．
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []

    for i in range(0, epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        x_epoch_data.append(i)
        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        writer.add_scalar("Loss/train", train_logs['dice_loss'], i)
        writer.add_scalar("Score/train", train_logs['iou_score'], i)
        writer.add_scalar("Loss/val", valid_logs['dice_loss'], i)
        writer.add_scalar("Score/val", valid_logs['iou_score'], i)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, modelpath)
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if i == 50:
            optimizer.param_groups[0]['lr'] = 5e-6
            print('Decrease decoder learning rate to 5e-6!')

        if i == 75:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')

    writer.close()


def create_logger(loglevel: str):
    logger = getLogger(__name__)
    # set loglevel
    logger.setLevel(DEBUG)

    # set handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(loglevel.upper())

    # set log output format
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    return logger


def arg_parse():
    parser = argparse.ArgumentParser("Pytorch segmentation sample")
    parser.add_argument("--input", "-i", default="../../../Datasets/PublicDatasets/CamVid", type=str, help="public dataset name")
    parser.add_argument("--epoch", "-e", default=50, type=int, help="training epochs")
    parser.add_argument("--modeltype", "-mt", default="UNet++", type=str, choices=["UNet++", "FPN", "DeepLabv3+"], help="model type")
    parser.add_argument("-encoder", default="resnet34", type=str, choices=["efficientnet-b4", "resnet34"], help="encoder model type")
    parser.add_argument("-model", default="model", type=str, help="output model path")
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
    #test_set = DatasetPath(inputpath / "test", inputpath / "testannot")

    # logger
    logger = create_logger(args.loglevel)

    #print('the number of image/label in the train: {}'.format(len(train_set.x_path.glob("*.png"))))
    #print('the number of image/label in the validation: {}'.format(len(val_set.x_path.glob("*.png"))))
    #print('the number of image/label in the test: '.format(len(test_set.x_path.glob("*.png"))))
    #train_dataset = Dataset(str(train_set.x_path), str(train_set.y_path), classes=['car', 'pedestrian'])

    #dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])

    """
    image, mask = train_dataset[4] # get some sample
    visualize(
        image=image, 
        cars_mask=mask[..., 0].squeeze(),
        pedestrian_mask=mask[..., 1].squeeze(),
        background_mask=mask[..., 2].squeeze(),
    )

    cars_mask=mask[..., 0].squeeze()
    print(cars_mask.shape)
    """
    encoder = args.encoder
    encoder_weights = 'imagenet'
    #classes = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
    logger.info(classes)

    activation = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation

    #model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes, activation=activation)
    #model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes, activation=activation)
    model = load_segment_model(args.modeltype, encoder, encoder_weights, n_classes, activation)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    modelpath = os.path.join(args.model, args.modeltype + "_" + encoder + ".pth")

    #train(model, train_loader, valid_loader, args.epoch, modelpath)


if __name__ == "__main__":
    main()
