import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import unique_path


def show_samples(path: Path) -> None:
    """
    Show a grid of images from a directory.

    Args:
        path (Path): The path to the directory containing the images.
    """

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(5, 5, figsize=(18, 18))

    for class_id, folder in enumerate(path.rglob("[!.]*")):
        if folder.is_dir():
            class_images = list(folder.glob("*"))
            samples = random.sample(class_images, 5)

            for col, image_path in enumerate(samples):
                image = cv2.imread(str(image_path))
                ax[class_id, col].imshow(image)
                ax[class_id, col].set_title("class_" + str(class_id))
                ax[class_id, col].set_axis_off()

    fig.show()


def show_distributions(
    base_dir: str, label_map: Optional[dict[int, int]] = None
) -> None:
    """
    Print the distribution of images in a folder.

    Args:
        base_dir: The path to the folder containing the images
    """

    data_types = ["train", "val", "test"]
    data_path = Path(base_dir)
    class_images, total_images = {}, {}
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle("The distribution across all the data", fontsize=16)

    for col, data_type in enumerate(data_types):
        type_path = data_path / data_type

        total_images[data_type] = sum(
            1 for x in type_path.rglob("[!.]*") if x.is_file()
        )
        
        print(f"Total number of images in {data_type} set: {total_images[data_type]}")

        class_images[data_type] = [0] * (
            max(label_map.values()) + 1 if label_map else 5
        )

        if label_map:
            for original_class, new_class in label_map.items():
                class_path = type_path / str(original_class)
                class_images[data_type][new_class] += sum(
                    1 for x in class_path.rglob("[!.]*") if x.is_file()
                )
        else:
            for i in range(5):
                class_path = type_path / str(i)
                class_images[data_type][i] = sum(
                    1 for x in class_path.rglob("[!.]*") if x.is_file()
                )
                
        print(f"Class distribution in {data_type} set: {class_images[data_type]}")

        distribution = [
            (class_id, round(num_images / total_images[data_type] * 100, 2))
            for class_id, num_images in enumerate(class_images[data_type])
        ]

        x_values = [t[0] for t in distribution]
        y_values = [t[1] for t in distribution]

        sns.barplot(
            x=x_values, y=y_values, errorbar=None, ax=ax[col], palette="husl"
        ).set(title=data_type)
        ax[col].yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    columns: list[str],
    *,
    matrix_plot_path: Path | None = None
) -> None:
    """
    Plot a confusion matrix.

    Args:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.
        columns (list[str]): The names of the classes.
    """

    class_count = len(columns)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt="g", cmap="Blues", cbar=False, ax=ax)

    ax.set_xticks(np.arange(class_count) + 0.5)
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticks(np.arange(class_count) + 0.5)
    ax.set_yticklabels(columns, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    if matrix_plot_path is not None:
        fig.savefig(matrix_plot_path)
    else:
        fig.show()
