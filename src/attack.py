import abc
import random
from dataclasses import dataclass, field
from typing import List, Optional

import foolbox as fb
import torch


@dataclass
class Attack:
    module: abc.ABCMeta
    epsilon: float
    targets: Optional[List[int]] = field(default_factory=list)

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

        return self.module()(model, image, target, epsilons=self.epsilon)[0]
