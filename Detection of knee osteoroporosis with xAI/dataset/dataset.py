import os
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from configs.config import BaseConfig
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ImageDataset(Dataset):
    """
    Custom Dataset class for loading and transforming images for training, validation, and testing.

    Args:
        config (BaseConfig): Configuration object containing dataset parameters.
        mode (Literal["train", "val", "test"]): Mode of the dataset (train, val, or test).
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        undersample (Optional[Dict[int, int]], optional): Dictionary specifying the number of samples to undersample for each class. Defaults to None.
    """

    def __init__(
        self,
        config: BaseConfig,
        mode: Literal["train", "val", "test"],
        *,
        augment: bool = False,
        undersample: Optional[dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.__config = config
        self.mode = mode
        self.undersample = undersample
        self.num_of_samples: dict[int, int] = {}
        self.transform = self._get_transform(augment)
        self.__dataset: list[tuple[Image.Image, int]] = []

        self._prepare_data()

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, label = self.__dataset[index]
        img = self.transform(img)
        label = torch.tensor(label)
        label = nn.functional.one_hot(label, len(self.config.label_map_legend))

        return img, label

    def _get_transform(self, augment: bool) -> v2.Compose:
        """
        Get the transformation pipeline for the dataset.

        Args:
            augment (bool): Whether to apply data augmentation.

        Returns:
            v2.Compose: The transformation pipeline.
        """
        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment:
            augment_transforms = [
                v2.RandomRotation(degrees=20),
                v2.RandomResizedCrop(size=self.config.img_size, scale=(0.9, 1.0)),
            ]
            return v2.Compose(augment_transforms + base_transforms)
        else:
            return v2.Compose(base_transforms)

    def _prepare_data(self) -> None:
        """
        Prepare the dataset by loading images and their corresponding labels.
        """
        root_dir = Path(self.config.data_path) / self.mode
        for root, dirs, files in track(
            os.walk(root_dir), description=f"Loading {self.mode} data..."
        ):
            dirs.sort()

            if dirs:
                continue

            label = self._get_label_from_dir(root)

            if label not in self.num_of_samples:
                self.num_of_samples[label] = 0

            for file in files:
                if self._should_undersample(label):
                    break

                img_path = Path(root) / file
                self._add_image_to_dataset(img_path, label)

    def _get_label_from_dir(self, dir_path: str) -> int:
        """
        Get the label from the directory path.

        Args:
            dir_path (str): The directory path.

        Returns:
            int: The label.
        """
        label = int(Path(dir_path).name)
        
        if self.config.label_map:
            label = self.config.label_map.get(label, 0)
            
        return label

    def _should_undersample(self, label: int) -> bool:
        """
        Check if the current label should be undersampled.

        Args:
            label (int): The label to check.

        Returns:
            bool: Whether the label should be undersampled.
        """
        return (
            self.undersample
            and label in self.undersample
            and self.num_of_samples[label] >= self.undersample[label]
        )

    def _add_image_to_dataset(self, img_path: Path, label: int) -> None:
        """
        Add an image to the dataset.

        Args:
            img_path (Path): The path to the image.
            label (int): The label of the image.
        """
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            self.__dataset.append((img, label))
            
        self.num_of_samples[label] += 1

    @property
    def config(self) -> BaseConfig:
        return self.__config

    @property
    def dataset(self) -> list[tuple[Image.Image, int]]:
        return self.__dataset
