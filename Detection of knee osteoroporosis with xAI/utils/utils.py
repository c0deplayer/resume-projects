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
    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        *,
        verbose: bool = False,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = (
            torch.Tensor([min_delta * 1])
            if self.monitor_metric == torch.gt
            else torch.tensor([min_delta * -1])
        )
        self.counter = 0
        torch_inf = torch.tensor(torch.inf)
        self.best_score = -torch_inf if mode == "max" else torch_inf
        self.verbose = verbose
        self.__model_state_dict = None

    def early_stop(self, validation_metric: float) -> bool:
        current = torch.Tensor([validation_metric])
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

        return False

    def save_model(self, model: torch.nn.Module, path: Path):
        model = model.to("cpu")
        torch.save(model.state_dict(), path)
        self.__model_state_dict = model.state_dict()

    @property
    def monitor_metric(self) -> Callable:
        return self.mode_dict[self.mode]

    @property
    def model_state_dict(self) -> dict[str, int]:
        return self.__model_state_dict


def unique_path(path: Path) -> Path:
    """
    This function takes a 'path' and checks if a file with the same name exists. If it does, it appends a counter
    in parentheses to the file name to make it unique. It continues to increment the counter until a unique path is found.

    Args:
        path (Path): The original path to be made unique.

    Returns:
        Path: A unique path that does not clash with existing files.

    Example:
        ```
        original_path = Path('path/to/file.txt')
        unique_path = unique_path(original_path)
        ```
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename}({str(counter)}){extension}"
        counter += 1

    return Path(path)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def seed_everything(
    seed: Optional[int] = None, workers: bool = False, verbose: bool = True
) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.
        verbose: Whether to print a message on each rank with the seed being set.

    """
    if seed is None:
        env_seed = os.environ.get("GLOBAL_SEED")
        if env_seed is None:
            seed = 0
            print(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
                print(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def make_weights_for_balanced_classes(
    images: tuple[Any, int], n_classes: int
) -> list[int]:
    # n_images = len(images)
    count_per_class = [0] * n_classes

    for _, image_class in images:
        count_per_class[image_class] += 1

    # weight_per_class = [0.0] * n_classes

    # for i in range(n_classes):
    #     weight_per_class[i] = float(n_images) / float(count_per_class[i])

    weights = [1.0 / float(count_per_class[image_class]) for _, image_class in images]

    # weights = [0] * n_images

    # for idx, (_, image_class) in enumerate(images):
    #     weights[idx] = weight_per_class[image_class]

    return weights


def get_nested_attr(obj: Any, attr_list: list[str]) -> Any:
    """
    Retrieve a nested attribute from an object using a list of attribute names.
    Supports indexed attributes (e.g., 'features[8]').

    Args:
        obj: The object from which to retrieve the attribute.
        attr_list: A list of attribute names.

    Returns:
        The nested attribute.
    """

    def _get_attr(obj: Any, attr: str) -> Any:
        match = re.match(r"(\w+)\[(\d+)\]", attr)

        if match:
            attr_name, index = match.groups()
            return getattr(obj, attr_name)[int(index)]

        return getattr(obj, attr)

    return reduce(_get_attr, attr_list, obj)


def get_target_layers(model: nn.Module, model_name: str) -> list[Any]:
    """
    Retrieve the target layers for a given model based on its name.

    Args:
        model: The model instance.
        model_name: The name of the model.

    Returns:
        A list containing the target layer.
    """
    layers = TARGET_LAYERS.get(model_name)

    if not layers:
        raise ValueError(f"Model {model_name} is not supported.")

    features_layer = get_nested_attr(model, layers)

    return [features_layer]


def get_image_from_test_dataset() -> Path:
    test_dataset_path = Path("./dataset/knee-osteoarthritis/test").resolve()

    return random.choice(list(test_dataset_path.rglob("*")))
