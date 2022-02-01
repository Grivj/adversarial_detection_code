import os
import pickle
import sys
from datetime import datetime
from importlib import import_module
from time import time

from torchvision.datasets import ImageFolder

from config import ConfigDatasets, ConfigModels


def append_basedir():
    currentdir = os.path.dirname(os.path.realpath(__file__))
    basedir = os.path.dirname(currentdir)
    sys.path.append(basedir)


def get_transforms(transforms_config: dict) -> list:
    module = import_module("torchvision.transforms")
    transforms = []
    for cls, value in transforms_config.items():
        cls = getattr(module, cls)
        transforms.append(cls(value) if value else cls())

    if len(transforms) > 1:
        from torchvision.transforms import Compose

        return Compose(transforms)
    return transforms[0]


def get_dataset(config: ConfigDatasets):
    append_basedir()
    module = import_module(config.module)
    class_ = getattr(module, config.class_)
    return class_(
        root=config.data_path,
        transform=get_transforms(config.transforms),
    )


def get_loader(config: ConfigDatasets):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset=get_dataset(config),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def get_model(config: ConfigModels):
    append_basedir()
    module = import_module(config.module)
    model = getattr(module, config.class_)
    if config.pretrained and config.state_dict_path is None:
        return model(pretrained=True).eval()
    if config.state_dict_path is not None and config.pretrained:
        import torch

        model = model()
        model.load_state_dict(torch.load(config.state_dict_path))
        return model.eval()
    return model.eval()


def get_fmodel(config: ConfigModels):
    model = get_model(config)
    from foolbox.models.pytorch import PyTorchModel

    return PyTorchModel(
        model=model,
        preprocessing=config.preprocessing if "preprocessing" in config else None,
        device=config.device,
        bounds=(0, 1),
    )


def timer(func):
    def wrap_timer(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_timer


def create_pickle(path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)


def append_pickle(path: str, data: dict[any]) -> None:
    with open(path, "ab") as f:
        pickle.dump(data, f)


def read_pickle(path: str) -> list[any]:
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data
