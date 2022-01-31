from dataclasses import dataclass

import torch


@dataclass
class Masker:
    x: torch.Tensor
    standard_deviations: torch.Tensor

    def _mask(self):
        """Returns a clamped noised (gaussian noise) version of self.x"""
        return torch.clamp(
            input=(
                self.x
                + torch.randn(self.x.size(), device=self.x.device)
                * self.standard_deviations[self._index]
            ),
            min=0,
            max=1,
        )

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        if self._index >= len(self.standard_deviations) - 1:
            raise StopIteration()
        self._index += 1
        if self._index == 0:
            return self.x
        return self._mask()

    def __len__(self):
        return len(self.standard_deviations)
