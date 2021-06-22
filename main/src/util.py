import matplotlib.pyplot as plt
from dataclasses import dataclass
import pathlib


@dataclass
class DatasetPath:
    x_path: pathlib.Path
    y_path: pathlib.Path


@dataclass
class ModelParam:
    modeltype: str
    encoder: str
    weights: str = "imagenet"
    activation: str = "softmax2d"  # could be None for logits or 'softmax2d' for multicalss segmentation


@dataclass
class Output:
    modelpath: str
    experiment: str
    mldir: str
    logdir: str = "logs"
    logname: str = "log.txt"


def load_multi_list(filepath: str, sep="\t"):
    item_list = []
    with open(filepath, "r") as f:
        for item in f:
            tmp_list = item.rstrip().split(sep)
            item_list.append(tmp_list)

    return item_list


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
