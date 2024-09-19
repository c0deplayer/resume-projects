import os
import random
import re
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from configs.constants import TARGET_LAYERS


class EarlyStopper:
    """
    Early stopping to stop the training when the validation metric is not improving.
    """

    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        *,
        verbose: bool = False,
    ):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (Literal["min", "max"]): Whether to look for a minimum or maximum in the monitored metric.
            verbose (bool): If True, prints a message for each validation metric check.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = (
            torch.tensor([min_delta]) if mode == "max" else torch.tensor([-min_delta])
        )
        self.counter = 0
        self.best_score = (
            torch.tensor(float("inf")) if mode == "min" else torch.tensor(float("-inf"))
        )
        self.verbose = verbose
        self.__model_state_dict = None

    def early_stop(self, validation_metric: float) -> bool:
        """
        Check if training should be stopped early.

        Args:
            validation_metric (float): The current value of the monitored metric.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        current = torch.tensor([validation_metric])

        if self.monitor_metric(current - self.min_delta, self.best_score):
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1

            if self.verbose:
                msg = f"The metric has not improved for {self.counter} epochs"

                if self.counter >= self.patience:
                    msg += ". Stopping training"

                print(msg)

        return self.counter >= self.patience

    def save_model(self, model: nn.Module, path: Path):
        """
        Save the model state dictionary to a file.

        Args:
            model (nn.Module): The model to save.
            path (Path): The file path where the model state dictionary will be saved.
        """
        model = model.to("cpu")
        torch.save(model.state_dict(), path)
        self.__model_state_dict = model.state_dict()

    @property
    def monitor_metric(self) -> Callable:
        """
        Get the comparison function based on the mode.

        Returns:
            Callable: The comparison function (torch.lt or torch.gt).
        """
        return self.mode_dict[self.mode]

    @property
    def model_state_dict(self) -> dict:
        """
        Get the saved model state dictionary.

        Returns:
            dict: The saved model state dictionary.
        """
        return self.__model_state_dict


def unique_path(path: Path) -> Path:
    """
    Generate a unique file path by appending a counter to the filename if a file with the same name exists.

    Args:
        path (Path): The original path to be made unique.

    Returns:
        Path: A unique path that does not clash with existing files.

    Example:
        original_path = Path('path/to/file.txt')
        unique_path = unique_path(original_path)
    """
    filename = path.stem
    extension = path.suffix
    directory = path.parent
    counter = 1

    while path.exists():
        path = directory / f"{filename}({counter}){extension}"
        counter += 1

    return path


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def seed_everything(
    seed: Optional[int] = None, workers: bool = False, verbose: bool = True
) -> int:
    """
    Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed (Optional[int]): The integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
        workers (bool): If set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence.
        verbose (bool): Whether to print a message on each rank with the seed being set.

    Returns:
        int: The seed used.
    """
    if seed is None:
        env_seed = os.getenv("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
            if verbose:
                print(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
                if verbose:
                    print(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if workers:
        os.environ["PL_SEED_WORKERS"] = "1"

    os.environ["PL_GLOBAL_SEED"] = str(seed)

    if verbose:
        print(f"Seed set to {seed}")

    return seed


def make_weights_for_balanced_classes(
    images: tuple[Any, int], n_classes: int
) -> list[float]:
    """
    Calculate weights for each image to balance classes in a dataset.

    Args:
        images (Tuple[Any, int]): A tuple where the first element is the image and the second element is the class index.
        n_classes (int): The number of classes.

    Returns:
        List[float]: A list of weights for each image.
    """
    count_per_class = [0] * n_classes

    for _, image_class in images:
        count_per_class[image_class] += 1

    weights = [1.0 / float(count_per_class[image_class]) for _, image_class in images]

    return weights


def get_nested_attr(obj: Any, attr_list: list[str]) -> Any:
    """
    Retrieve a nested attribute from an object using a list of attribute names.
    Supports indexed attributes (e.g., 'features[8]').

    Args:
        obj (Any): The object from which to retrieve the attribute.
        attr_list (List[str]): A list of attribute names.

    Returns:
        Any: The nested attribute.
    """
    pattern = re.compile(r"(\w+)\[(\d+)\]")

    def _get_attr(obj: Any, attr: str) -> Any:
        match = pattern.match(attr)

        if match:
            attr_name, index = match.groups()
            try:
                return getattr(obj, attr_name)[int(index)]
            except (AttributeError, IndexError, TypeError) as e:
                raise AttributeError(f"Error accessing {attr}: {e}")
        try:
            return getattr(obj, attr)
        except AttributeError as e:
            raise AttributeError(f"Error accessing {attr}: {e}")

    return reduce(_get_attr, attr_list, obj)


def get_target_layers(model: nn.Module, model_name: str) -> list[Any]:
    """
    Retrieve the target layers for a given model based on its name.

    Args:
        model (nn.Module): The model instance.
        model_name (str): The name of the model.

    Returns:
        List[Any]: A list containing the target layer.

    Raises:
        ValueError: If the model name is not supported.
    """
    layers = TARGET_LAYERS.get(model_name)

    if not layers:
        raise ValueError(f"Model {model_name} is not supported.")

    features_layer = get_nested_attr(model, layers)

    return [features_layer]


def get_image_from_test_dataset(
    dataset_path: Optional[Path] = Path("./dataset/knee-osteoarthritis/test").resolve(),
) -> Path:
    """
    Retrieve a random image path from the test dataset.

    Args:
        dataset_path (Optional[Path]): The path to the test dataset directory. Defaults to TEST_DATASET_PATH.

    Returns:
        Path: A random image path from the test dataset.

    Raises:
        FileNotFoundError: If the test dataset directory is empty.
    """
    image_paths = list(dataset_path.rglob("*"))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in the test dataset directory: {dataset_path}"
        )

    return random.choice(image_paths)
