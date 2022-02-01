from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConfigDatasets:
    name: str
    data_path: str
    module: str = "torchvision.datasets"
    class_: str = "ImageFolder"
    transforms: Dict = field(default_factory=lambda: {"ToTensor": None})


@dataclass
class CIFAR10Config(ConfigDatasets):
    name = "cifar10"


@dataclass
class ImageNetteConfig(ConfigDatasets):
    name = "imagenette"
    module = "datasets.imagenet.imagenette"
    class_ = "ImageNette"
    transforms = {
        "Resize": 266,
        "CenterCrop": 256,
        "ToTensor": None,
    }


@dataclass
class ConfigModels:
    name: str
    class_: str
    module: str = "torchvision.models"
    preprocessing: Dict = field(default_factory=dict)
    pretrained: Optional[bool] = True
    state_dict_path: Optional[str] = None
    device: str = "cpu"


@dataclass
class ImageNetteGoogLeNetConfig(ConfigModels):
    name = "imagenette_googlenet"
    class_ = "googlenet"
    preprocessing = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "axis": -3,
    }


@dataclass
class CIFAR10GoogLeNetConfig(ConfigModels):
    name = "cifar10_googlenet"
    module = "models.googlenet"
    class_ = "GoogLeNet"
    preprocessing = {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "axis": -3,
    }


@dataclass
class Mode:
    name: str


@dataclass
class AccuracyMode:
    name = "accuracy"


@dataclass
class NormalRunMode:
    name = "normal_run"
    results_path = "../results/"


@dataclass
class Config:
    dataset: ConfigDatasets
    model: ConfigModels
