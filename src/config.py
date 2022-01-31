from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigDatasets:
    path: str
    module: str
    transforms: dict[str, Optional[any]]


@dataclass
class ConfigModels:
    module: str
    model: str
    preprocessing: Optional[dict] = None
    pretrained: Optional[bool] = True
    state_dict_path: Optional[str] = None
    download: Optional[bool] = False
    device: Optional[str] = "cpu"


@dataclass
class Config:
    dataset: ConfigDatasets
    model: ConfigModels
