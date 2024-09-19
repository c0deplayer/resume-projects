import re
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from configs.constants import ACTIVATIONS, WEIGHTS
from rich.progress import track
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)
from utils.utils import EarlyStopper


def build_model(
    model: torchvision.models,
    class_count: int,
    activation: nn.Module = ACTIVATIONS["relu"],
    trainable_model: bool = False,
    hidden_size: int = 256,
    dropout: float = 0.4,
) -> nn.Module:
    """
    Build a model with a custom classifier.

    Args:
        model (torchvision.models): The base model to use.
        class_count (int): The number of output classes.
        activation (nn.Module): The activation function to use.
        trainable_model (bool): Whether the base model's parameters should be trainable.
        hidden_size (int): The size of the hidden layer in the classifier.
        dropout (float): The dropout rate to use in the classifier.

    Returns:
        nn.Module: The model with the custom classifier.
    """
    model_name = model.__name__.lower()
    model_name = re.sub(r"[^a-zA-Z0-9]", "", model_name)
    pretrained_model = model(weights=WEIGHTS[model_name])

    if "densenet" in model_name:
        for param in pretrained_model.parameters():
            param.requires_grad = trainable_model

        num_features = pretrained_model.classifier.in_features
        # pretrained_model.classifier = nn.Sequential(
        #     nn.Linear(num_features, hidden_size),
        #     activation(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, class_count),
        # )
        pretrained_model.classifier = nn.Linear(num_features, class_count)
    elif "efficientnet" in model_name or "efficientnetv2" in model_name:
        for param in pretrained_model.parameters():
            param.requires_grad = trainable_model

        num_features = pretrained_model.classifier[-1].in_features
        pretrained_model.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, class_count),
        )
        # pretrained_model.classifier = nn.Linear(num_features, class_count)
    elif "convnext" in model_name:
        for name, param in pretrained_model.named_parameters():
            if "7" not in name:
                param.requires_grad = False

        num_features = pretrained_model.classifier[-1].in_features
        # pretrained_model.classifier[-1] = nn.Linear(num_features, hidden_size)
        # pretrained_model.classifier.append(activation())
        # pretrained_model.classifier.append(nn.Dropout(dropout))
        # pretrained_model.classifier.append(nn.Linear(hidden_size, class_count))
        # pretrained_model.classifier = nn.Sequential(
        #     nn.Flatten(start_dim=1, end_dim=-1),
        #     nn.Linear(num_features, hidden_size),
        #     activation(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, class_count),
        # )
        pretrained_model.classifier[-1] = nn.Linear(num_features, class_count)
    else:
        raise NotImplementedError()

    pretrained_model.name = model_name

    return pretrained_model


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_count: int,
    epochs: int,
    *,
    device: torch.device,
    neptune_experiment: Optional[Any] = None,
    scheduler: Optional[LRScheduler] = None,
    early_stopper: Optional[EarlyStopper] = None,
    checkpoint_filepath: Optional[str] = None,
) -> nn.Module:
    for epoch in range(epochs):
        if not next(model.parameters()).is_mps or not next(model.parameters()).is_cuda:
            model = model.to(device)

        train_loss, train_accuracy = __train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            class_count=class_count,
        )

        val_loss, val_metrics = __validation(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            class_count=class_count,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch: [{epoch + 1}/{epochs}] Loss {train_loss} | Accuracy: {train_accuracy} "
            f"| Validation Loss: {val_loss} | Validation F2: {val_metrics[0]} | "
            f"Validation Accuracy: {val_metrics[1]} | Validation Recall: {val_metrics[2]} | Validation Precision: {val_metrics[3]}"
        )

        if neptune_experiment is not None:
            __log_neptune_metrics(
                neptune_experiment,
                train_loss,
                train_accuracy,
                val_loss,
                val_metrics,
                epoch,
            )

        if early_stopper is not None and checkpoint_filepath is not None:
            __handle_early_stopping(early_stopper, model, val_loss, checkpoint_filepath)

    return model


def __log_neptune_metrics(
    neptune_experiment: Any,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_metrics: list,
    epoch: int,
) -> None:
    """
    Log the metrics to Neptune.

    Args:
        neptune_experiment (Any): Neptune experiment for logging metrics.
        train_loss (float): The training loss.
        train_accuracy (float): The training accuracy.
        val_loss (float): The validation loss.
        val_metrics (list): The validation metrics.
        epoch (int): The current epoch.
    """
    neptune_experiment["train/loss"].append(train_loss)
    neptune_experiment["train/accuracy"].append(train_accuracy)
    neptune_experiment["val/loss"].append(val_loss)
    neptune_experiment["val/f2-score"].append(val_metrics[0])
    neptune_experiment["val/precision"].append(val_metrics[3])
    neptune_experiment["val/recall"].append(val_metrics[2])
    neptune_experiment["val/accuracy"].append(val_metrics[1])

    msg = (
        f" End of epoch {epoch} val_loss: {val_loss} - val_f2: {val_metrics[0]} — val_precision: "
        f"{val_metrics[3]}, — val_recall: {val_metrics[2]} — val_accuracy: {val_metrics[1]}"
    )
    neptune_experiment[f"Epoch End Metrics (each step)"].log(msg)


def __handle_early_stopping(
    early_stopper: EarlyStopper,
    model: nn.Module,
    val_loss: float,
    checkpoint_filepath: str,
) -> None:
    """
    Handle early stopping and model checkpointing.

    Args:
        early_stopper (EarlyStopper): Early stopping utility.
        model (nn.Module): The model to save.
        val_loss (float): The validation loss.
        checkpoint_filepath (str): Filepath to save the model checkpoint.
    """
    if (
        val_loss < early_stopper.best_score and early_stopper.monitor_metric == torch.lt
    ) or (
        val_loss > early_stopper.best_score and early_stopper.monitor_metric == torch.gt
    ):
        early_stopper.save_model(model, checkpoint_filepath)

    if early_stopper.early_stop(val_loss):
        model.load_state_dict(early_stopper.model_state_dict)

    return model


def __train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: torch.utils.data.DataLoader,
    class_count: int,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        class_count (int): The number of classes.
        device (torch.device): The device to train on.

    Returns:
        Tuple[float, float]: The average training loss and training accuracy.
    """
    model.train()
    accuracy = MulticlassAccuracy(average="micro", num_classes=class_count).to(device)

    total_train_loss = 0.0
    for img, label in track(train_loader, description="Training..."):
        img, label = img.to(device), label.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        target = torch.argmax(label, dim=1)
        accuracy.update(preds=output, target=target)
        total_train_loss += loss.item()

    total_train_loss = round(total_train_loss / len(train_loader), 4)
    train_accuracy = round(accuracy.compute().item(), 4)

    return total_train_loss, train_accuracy


def __validation(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    class_count: int,
    *,
    device: torch.device,
) -> tuple[float, tuple[float, float, float, float]]:
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        criterion (nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        class_count (int): The number of classes.
        device (torch.device): The device to validate on.

    Returns:
        Tuple[float, Tuple[float, float, float, float]]: The average validation loss and validation metrics (F2 score, accuracy, recall, precision).
    """
    model.eval()
    f2_score = MulticlassFBetaScore(
        average="macro", num_classes=class_count, beta=2.0
    ).to(device)
    recall = MulticlassRecall(average="macro", num_classes=class_count).to(device)
    precision = MulticlassPrecision(average="macro", num_classes=class_count).to(device)
    accuracy = MulticlassAccuracy(average="micro", num_classes=class_count).to(device)

    total_val_loss = 0.0
    for img, label in track(val_loader, description="Validation..."):
        with torch.inference_mode():
            img, label = img.to(device), label.type(torch.FloatTensor).to(device)
            output = model(img)
            val_loss = criterion(output, label)

            target = torch.argmax(label, dim=1)
            f2_score.update(preds=output, target=target)
            precision.update(preds=output, target=target)
            recall.update(preds=output, target=target)
            accuracy.update(preds=output, target=target)

            total_val_loss += val_loss.item()

    total_val_loss = round(total_val_loss / len(val_loader), 4)
    val_metrics = (
        round(f2_score.compute().item(), 4),
        round(accuracy.compute().item(), 4),
        round(recall.compute().item(), 4),
        round(precision.compute().item(), 4),
    )

    return total_val_loss, val_metrics
