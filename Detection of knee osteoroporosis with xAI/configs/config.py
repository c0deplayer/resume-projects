from dataclasses import dataclass, field

from dataclass_wizard import YAMLWizard


@dataclass
class EDA(YAMLWizard, key_transform="SNAKE"):
    data_path: str
    class_count: int
    label_map: dict[int, int] | None


@dataclass(kw_only=True)
class BaseConfig(YAMLWizard, key_transform="SNAKE"):
    data_path: str
    checkpoint_path: str
    label_map: dict[int, int] | None
    label_map_legend: dict[int, str]

    num_of_samples: dict[int, int] | None = field(default=None)
    img_size: tuple[int, int]
    shuffle: bool
    augment: bool
    augment_v2: bool

    epochs: int
    batch_size: int
    hidden_size: int
    dropout: float
    trainable_model: bool
    learning_rate: float
    activation: str
    optimizer: str
    weight_decay: float
    momentum: float | None = field(default=None)

    seed: int

    def __post_init__(self):
        self.img_shape = (*self.img_size, 3)


@dataclass
class DenseNetConfig(BaseConfig):
    pass


@dataclass
class ConvNeXtConfig(BaseConfig):
    pass


@dataclass
class EfficientNetConfig(BaseConfig):
    pass


@dataclass
class EfficientNetV2Config(BaseConfig):
    pass
