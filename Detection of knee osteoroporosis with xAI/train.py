import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import neptune
import numpy as np
import torch
import torch.nn as nn
from configs.constants import ACTIVATIONS, CONFIGS, MODELS, OPTIMIZERS
from dataset.dataset import ImageDataset
from dotenv import load_dotenv
from model import build_model, train_loop
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError
from rich.progress import track
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary
from utils import utils
from utils.visualize import plot_confusion_matrix


def cli_main() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for knee osteoporosis detection"
    )
    parser.add_argument("--config", "-c", type=str, help="Path to the config file")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="The model to train",
        choices=["ConvNeXt", "EfficientNet", "EfficientNetV2", "DenseNet"],
        required=True,
    )
    parser.add_argument(
        "-n",
        "--neptune",
        help="Flag for using Neptune for logging",
        action="store_true",
    )
    parser.add_argument(
        "--exp-name",
        help="Name of the experiment",
    )

    return parser.parse_args()


def initialize_neptune(
    args: argparse.Namespace, model_name: str, config_file: str, neptune_token: str
) -> tuple[neptune.Run, neptune.ModelVersion]:
    """
    Initialize Neptune for logging and model versioning.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model_name (str): The name of the model.
        config_file (str): Path to the configuration file.
        neptune_token (str): Neptune API token.

    Returns:
        Tuple[neptune.Run, neptune.ModelVersion]: Neptune run and model version objects.
    """
    if not args.exp_name:
        raise ValueError("Please provide the name of the experiment")

    run = neptune.init_run(
        name=args.exp_name,
        project="codeplayer/Detection-of-knee-osteoroporosis-with-xAI",
        api_token=neptune_token,
        source_files=["**/*.py", config_file],
        dependencies="/Users/codeplayer/Różne rzeczy/Resume Projects/pyproject.toml",
    )

    try:
        neptune.init_model(
            name=args.exp_name,
            key=f"{model_name[0].upper()}{model_name[-4:].upper()}",
            project="codeplayer/Detection-of-knee-osteoroporosis-with-xAI",
            api_token=neptune_token,
        )
    except NeptuneModelKeyAlreadyExistsError:
        pass

    model_version = neptune.init_model_version(
        model=f"DOFKO-{model_name[0].upper()}{model_name[-4:].upper()}",
        project="codeplayer/Detection-of-knee-osteoroporosis-with-xAI",
        api_token=neptune_token,
    )

    model_version["run/id"] = run["sys/id"].fetch()
    model_version["run/url"] = run.get_url()

    return run, model_version


def prepare_dataloaders(
    config: Any, class_count: int, g: torch.Generator
) -> tuple[DataLoader, DataLoader]:
    """
    Prepare the training and validation dataloaders.

    Args:
        config (Any): Configuration object.
        class_count (int): Number of classes.
        g (torch.Generator): Random generator for reproducibility.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    train_dataset = ImageDataset(
        config,
        mode="train",
        augment=config.augment_v2,
        undersample=config.num_of_samples,
    )
    val_dataset = ImageDataset(
        config,
        mode="val",
    )

    if config.augment:
        weights = utils.make_weights_for_balanced_classes(
            train_dataset.dataset, class_count
        )
        weights = torch.tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
    else:
        sampler = None

    train_dataloader = DataLoader(
        batch_size=config.batch_size,
        dataset=train_dataset,
        shuffle=False if sampler else config.shuffle,
        pin_memory=True,
        generator=g,
        sampler=sampler,
        worker_init_fn=lambda _: np.random.seed(config.seed),
    )
    val_dataloader = DataLoader(
        batch_size=config.batch_size,
        dataset=val_dataset,
        shuffle=config.shuffle,
        pin_memory=True,
        generator=g,
        worker_init_fn=lambda _: np.random.seed(config.seed),
    )

    return train_dataloader, val_dataloader


def main():
    load_dotenv()
    args = cli_main()

    config_file = f"{Path().resolve()}/configs/{args.model}/{args.config}"
    config = CONFIGS[args.model].from_yaml_file(file=config_file)
    class_count = len(config.label_map_legend)
    neptune_token = os.environ["NEPTUNE_API_TOKEN"]

    utils.seed_everything(config.seed)

    model = build_model(
        class_count=class_count,
        model=MODELS[args.model],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        activation=ACTIVATIONS[config.activation],
        trainable_model=config.trainable_model,
    )

    optimizer = OPTIMIZERS[config.optimizer](
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        # momentum=config.momentum,
    )
    criterion = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_dataloader, val_dataloader = prepare_dataloaders(config, class_count, g)

    checkpoint_filepath = (
        Path(config.checkpoint_path).resolve()
        / f"{args.model}/best-model-{model.name}-{datetime.now().strftime('%Y%m%d')}.pth"
    )

    checkpoint_filepath = utils.unique_path(checkpoint_filepath)

    os.makedirs(checkpoint_filepath.parent, exist_ok=True)

    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.3, mode="min")
    early_stopper = utils.EarlyStopper(patience=12, verbose=True, mode="min")
    device = utils.get_device()

    model_train_params = dict(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        epochs=config.epochs,
        class_count=class_count,
        scheduler=scheduler,
        device=device,
        early_stopper=early_stopper,
        checkpoint_filepath=checkpoint_filepath,
    )

    if args.neptune:
        run, model_version = initialize_neptune(
            args, model.name, config_file, neptune_token
        )
        model_train_params["neptune_experiment"] = run

    summary(model, input_size=next(iter(train_dataloader))[0].size())

    model = train_loop(**model_train_params)

    val_true_list, val_pred_list = [], []
    for img, label in track(val_dataloader):
        with torch.inference_mode():
            img, label = img.to(device), label.type(torch.FloatTensor).to(device)
            output = model(img)

            val_true_list.append(label.detach().cpu().numpy())
            val_pred_list.append(output.detach().cpu().numpy())

    val_true = np.argmax(np.concatenate(val_true_list, axis=0), axis=1)
    val_pred = np.argmax(np.concatenate(val_pred_list, axis=0), axis=1)

    matrix_plot_path = utils.unique_path(
        Path(f"./plots/{model.name}/confusion_matrix.png")
    )

    os.makedirs(matrix_plot_path.parent, exist_ok=True)

    plot_confusion_matrix(
        val_true,
        val_pred,
        columns=config.label_map_legend.values(),
        matrix_plot_path=matrix_plot_path,
    )

    if args.neptune:
        model_version["model"].upload(str(checkpoint_filepath))
        model_version.change_stage("staging")

        run["val/confusion_matrix"].upload(str(matrix_plot_path))

        run.stop()
        model_version.stop()


if __name__ == "__main__":
    main()
