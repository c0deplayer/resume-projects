import torch.optim
import torchvision
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMElementWise,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.base_cam import BaseCAM
from torchvision.models.convnext import ConvNeXt_Small_Weights, convnext_small
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.models.efficientnet import (
    EfficientNet_B3_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_b3,
    efficientnet_v2_s,
)

from .config import (
    BaseConfig,
    ConvNeXtConfig,
    DenseNetConfig,
    EfficientNetConfig,
    EfficientNetV2Config,
)

CONFIGS: dict[str, BaseConfig] = {
    "DenseNet": DenseNetConfig,
    "ConvNeXt": ConvNeXtConfig,
    "EfficientNet": EfficientNetConfig,
    "EfficientNetV2": EfficientNetV2Config,
}

OPTIMIZERS: dict[str, torch.optim.Optimizer] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SDG": torch.optim.SGD,
    # "Lion": lion,
}

MODELS: dict[str, torch.nn.Module] = {
    "DenseNet": densenet121,
    "ConvNeXt": convnext_small,
    "EfficientNet": efficientnet_b3,
    "EfficientNetV2": efficientnet_v2_s,
}

WEIGHTS: dict[str, torchvision.models.Weights] = {
    "densenet121": DenseNet121_Weights.IMAGENET1K_V1,
    "convnextsmall": ConvNeXt_Small_Weights.IMAGENET1K_V1,
    "efficientnetb3": EfficientNet_B3_Weights.IMAGENET1K_V1,
    "efficientnetv2s": EfficientNet_V2_S_Weights.IMAGENET1K_V1,
}

ACTIVATIONS: dict[str, torch.nn.Module] = {
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}

CAM_METHODS: dict[str, BaseCAM] = {
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "ScoreCAM": ScoreCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "EigenGradCAM": EigenGradCAM,
    "GradCAMElementWise": GradCAMElementWise,
    "HiResCAM": HiResCAM,
}

TARGET_LAYERS: dict[str, list[str]] = {
    "efficientnetb3": ["features[8]"],
    "densenet121": ["features", "denseblock4", "denselayer16"],
    "efficientnetv2s": ["features[7]"],
}
