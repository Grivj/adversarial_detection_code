import random
from dataclasses import dataclass
from typing import List, Optional

import foolbox as fb
import torch

from config import AttackConfig
from utils import append_basedir


@dataclass
class Attack:
    def __init__(self, config: AttackConfig, targets: Optional[List[int]]):
        append_basedir()
        self.class_ = getattr(fb.attacks, config.class_)
        self.epsilon = config.epsilon
        self.targets = targets

    def __call__(
        self, model: any, image: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        target = (
            fb.criteria.TargetedMisclassification(
                target_classes=torch.tensor(
                    [int(random.choice([l for l in self.targets if int(l) != label]))]
                ).to(image.device)
            )
            if self.targets
            else fb.criteria.Misclassification(label)
        )

        return self.class_()(model, image, target, epsilons=self.epsilon)[0]
