# coding: utf-8
import argparse
import os
import pathlib

import mlflow
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from augment import (get_preprocessing, get_training_augmentation,
                     get_validation_augmentation)
from dataset import Dataset
from util import DatasetPath, ModelParam, Output, visualize


def arg_parse():
    parser = argparse.ArgumentParser("Pytorch segmentation sample")
    parser.add_argument("--input", "-i", default="../../../Datasets/PublicDatasets/CamVid", type=str, help="public dataset name")
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
    parser.add_argument("-mldir", default="mllogs/eval/mlruns", type=str, help="mlflow dir")
    parser.add_argument("--experiment", "-ex", default="test", type=str, help="mlflow experiment")
    parser.add_argument("--loglevel", default="debug", type=str, choices=["debug", "info", "warning", "error", "critical"], help="log level")
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    inputpath = pathlib.Path(args.input)
    test_set = DatasetPath(inputpath / "test", inputpath / "testannot")

    classes = [item.strip() for item in open(inputpath / "class.txt", "r")]

    model_param = ModelParam(args.modeltype, args.encoder)

    modelpath = os.path.join(args.model, model_param.modeltype + "_" + model_param.encoder + ".pth")
    output_param = Output(modelpath, args.experiment, args.mldir)

    # load best saved checkpoint
    best_model = torch.load(modelpath)

    # print(best_model)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_param.encoder, model_param.weights)

    # create test dataset
    test_dataset = Dataset(
        str(test_set.x_path),
        str(test_set.y_path),
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    test_dataloader = DataLoader(test_dataset)

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5), ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )

    logs = test_epoch.run(test_dataloader)

    print("dice_loss:{}, iou_score:{}".format(logs['dice_loss'], logs['iou_score']))

    mlflow.set_tracking_uri(output_param.mldir)
    mlflow.set_experiment(output_param.experiment)
    with mlflow.start_run() as run:
        # save result with mlflow
        mlflow.log_metrics({"dice_loss": logs['dice_loss'], "iou_score": logs['iou_score']})

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(str(test_set.x_path), str(test_set.y_path), classes=classes)

    for i in range(3):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        pr_mask = np.transpose(pr_mask, (1, 2, 0))

        gt_mask_gray = np.zeros((gt_mask.shape[0], gt_mask.shape[1]))

        for ii in range(gt_mask.shape[2]):
            gt_mask_gray = gt_mask_gray + 1 / gt_mask.shape[2] * ii * gt_mask[:, :, ii]

        pr_mask_gray = np.zeros((pr_mask.shape[0], pr_mask.shape[1]))
        for ii in range(pr_mask.shape[2]):
            pr_mask_gray = pr_mask_gray + 1 / pr_mask.shape[2] * ii * pr_mask[:, :, ii]

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask_gray,
            predicted_mask=pr_mask_gray
        )


if __name__ == "__main__":
    main()
