import torch.optim
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
from torchvision.models.convnext import ConvNeXt_Small_Weights, convnext_small
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.models.efficientnet import (
    EfficientNet_B3_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_b3,
    efficientnet_v2_s,
)

from .config import (
    ConvNeXtConfig,
    DenseNetConfig,
    EfficientNetConfig,
    EfficientNetV2Config,
)

CONFIGS = {
    "DenseNet": DenseNetConfig,
    "ConvNeXt": ConvNeXtConfig,
    "EfficientNet": EfficientNetConfig,
    "EfficientNetV2": EfficientNetV2Config,
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SDG": torch.optim.SGD,
    # "Lion": lion,
}

MODELS = {
    "DenseNet": densenet121,
    "ConvNeXt": convnext_small,
    "EfficientNet": efficientnet_b3,
    "EfficientNetV2": efficientnet_v2_s,
}

WEIGHTS = {
    "densenet121": DenseNet121_Weights.IMAGENET1K_V1,
    "convnextsmall": ConvNeXt_Small_Weights.IMAGENET1K_V1,
    "efficientnetb3": EfficientNet_B3_Weights.IMAGENET1K_V1,
    "efficientnetv2s": EfficientNet_V2_S_Weights.IMAGENET1K_V1,
}

ACTIVATIONS = {"relu": torch.nn.ReLU, "silu": torch.nn.SiLU, "gelu": torch.nn.GELU}

CAM_METHODS = {
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

TARGET_LAYERS = {
    "efficientnetb3": ["features[8]"],
    "densenet121": ["features", "denseblock4", "denselayer16"],
    "efficientnetv2s": ["features[7]"]
}
