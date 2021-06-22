# coding: utf-8
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pathlib
from dataclasses import dataclass
from dataset import Dataset
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from augment import get_preprocessing, get_training_augmentation, get_validation_augmentation


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


def main():
    parser = argparse.ArgumentParser("Pytorch segmentation sample")
    parser.add_argument("--input", "-i", default="../../../Datasets/PublicDatasets/CamVid", type=str, help="public dataset name")
    parser.add_argument("-encoder", default="efficientnet-b4", type=str, help="encoder model type")
    parser.add_argument("-model", default="model", type=str, help="output model path")
    args = parser.parse_args()
    print(args.input)

    inputpath = pathlib.Path(args.input)
    test_set = DatasetPath(inputpath / "test", inputpath / "testannot")

    #print('the number of image/label in the train: {}'.format(len(train_set.x_path.glob("*.png"))))
    #print('the number of image/label in the validation: {}'.format(len(val_set.x_path.glob("*.png"))))
    #print('the number of image/label in the test: '.format(len(test_set.x_path.glob("*.png"))))
    #train_dataset = Dataset(str(train_set.x_path), str(train_set.y_path), classes=['car', 'pedestrian'])

    #dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])

    # load best saved checkpoint
    modelpath = os.path.join(args.model, args.encoder + ".pth")
    best_model = torch.load(modelpath)

    #print(best_model)

    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, encoder_weights)
    classes = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
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

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(str(test_set.x_path), str(test_set.y_path), classes=classes)

    for i in range(9):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        pr_mask = np.transpose(pr_mask, (1, 2, 0))

        gt_mask_gray = np.zeros((gt_mask.shape[0],gt_mask.shape[1]))

        for ii in range(gt_mask.shape[2]):
            gt_mask_gray = gt_mask_gray + 1/gt_mask.shape[2]*ii*gt_mask[:,:,ii]

        pr_mask_gray = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))
        for ii in range(pr_mask.shape[2]):
            pr_mask_gray = pr_mask_gray + 1/pr_mask.shape[2]*ii*pr_mask[:,:,ii]

        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask_gray, 
            predicted_mask=pr_mask_gray
        )



    """
    encoder = args.encoder
    encoder_weights = 'imagenet'
    classes = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
    activation = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation

    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=n_classes, #len(CLASSES),
        activation=activation,
    )
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

    modelpath = os.path.join(args.model, encoder + ".pth")

    train(model, train_loader, valid_loader, args.epoch, modelpath)
    """


if __name__ == "__main__":
    main()
