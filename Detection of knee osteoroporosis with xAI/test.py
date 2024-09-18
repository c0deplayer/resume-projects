import argparse
from pathlib import Path

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)

from configs.constants import ACTIVATIONS, CONFIGS, MODELS
from dataset.dataset import ImageDataset
from model import build_model
from utils import utils
from utils.visualize import plot_confusion_matrix


def cli_main():
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
    parser.add_argument("--weights", "-w", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = cli_main()

    config_file = f"{Path().resolve()}/configs/{args.model}/{args.config}"
    config = CONFIGS[args.model].from_yaml_file(file=config_file)
    class_count = len(config.label_map_legend)
    device = utils.get_device()

    utils.seed_everything(config.seed)

    model = build_model(
        class_count=class_count,
        model=MODELS[args.model],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        activation=ACTIVATIONS[config.activation],
        trainable_model=config.trainable_model,
    )

    model.load_state_dict(
        torch.load(f"./checkpoints/{args.model}/{args.weights}", weights_only=True)
    )
    model.eval()

    test_dataset = ImageDataset(
        config,
        mode="test",
    )

    test_dataloader = DataLoader(
        batch_size=config.batch_size,
        dataset=test_dataset,
        shuffle=config.shuffle,
        pin_memory=True,
    )

    summary(model, input_size=next(iter(test_dataloader))[0].size())

    f2_score = MulticlassFBetaScore(
        average="macro", num_classes=class_count, beta=2.0
    ).to(device)
    recall = MulticlassRecall(average="macro", num_classes=class_count).to(device)
    precision = MulticlassPrecision(average="macro", num_classes=class_count).to(device)
    accuracy = MulticlassAccuracy(average="micro", num_classes=class_count).to(device)

    test_true_list, test_pred_list = [], []
    for img, label in track(test_dataloader):
        with torch.inference_mode():
            if (
                not next(model.parameters()).is_mps
                or not next(model.parameters()).is_cuda
            ):
                model = model.to(device)

            img, label = img.to(device), label.type(torch.FloatTensor).to(device)
            output = model(img)

            target = torch.argmax(label, dim=1)

            f2_score.update(preds=output, target=target)
            precision.update(preds=output, target=target)
            recall.update(preds=output, target=target)
            accuracy.update(preds=output, target=target)

            test_true_list.append(label.detach().cpu().numpy())
            test_pred_list.append(output.detach().cpu().numpy())

    test_metrics = (
        round(f2_score.compute().item(), 4),
        round(accuracy.compute().item(), 4),
        round(recall.compute().item(), 4),
        round(precision.compute().item(), 4),
    )

    test_true = np.argmax(np.concatenate(test_true_list, axis=0), axis=1)
    test_pred = np.argmax(np.concatenate(test_pred_list, axis=0), axis=1)

    print(
        f"Test F2 Score: {test_metrics[0]} | Test Accuracy: {test_metrics[1]} | Test Recall: {test_metrics[2]} | Test Precision: {test_metrics[3]}"
    )

    plot_confusion_matrix(
        test_true,
        test_pred,
        columns=config.label_map_legend.values(),
    )
