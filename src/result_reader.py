from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.metric import Metric
from src.utils import read_pickle


@dataclass
class ResultReader:
    pickle_path: str

    def __post_init__(self):
        # loading pickle file
        self.data = read_pickle(self.pickle_path)
        self.standard_deviations = self.data[1]["standard_deviations"]

        # generating a Metric object for each batch
        self.metrics = [
            Metric(d["logits"], d["labels"]) for d in self.data if "batch_id" in d
        ]
        # retrieving distance metrics
        self.l2 = self._to_df(
            torch.stack([d["distances_l2"] for d in self.data if "batch_id" in d])
        )
        self.PSNR = self._to_df(
            torch.stack([d["distances_PSNR"] for d in self.data if "batch_id" in d])
        )
        self.attack_l2 = torch.Tensor(
            [d["attack_distance_l2"] for d in self.data if "attack_distance_l2" in d]
        )
        self.attack_PSNR = torch.Tensor(
            [
                d["attack_distance_PSNR"]
                for d in self.data
                if "attack_distance_PSNR" in d
            ]
        )

        self.cached_accuracies = None
        self.cached_consistencies = None
        self.cached_pairwise_distances = None
        self.cached_std_differences = None

    @property
    def accuracies(self) -> pd.DataFrame:
        if self.cached_accuracies is None:
            self.cached_accuracies = self._to_df(
                torch.stack([metric.accuracy for metric in self.metrics])
            )
        return self.cached_accuracies

    @property
    def consistencies(self) -> pd.DataFrame:
        if self.cached_consistencies is None:
            self.cached_consistencies = self._to_df(
                torch.stack([metric.consistency for metric in self.metrics])
            )
        return self.cached_consistencies

    @property
    def pairwise_distances(self) -> pd.DataFrame:
        if self.cached_pairwise_distances is None:
            self.cached_pairwise_distances = self._to_df(
                torch.stack([metric.pairwise_distance for metric in self.metrics])
            )
        return self.cached_pairwise_distances

    @property
    def std_differences(self) -> pd.DataFrame:
        if self.cached_std_differences is None:
            self.cached_std_differences = self._to_df(
                torch.stack([metric.std_difference for metric in self.metrics])
            )
        return self.cached_std_differences

    def _to_df(self, data: torch.Tensor) -> pd.DataFrame:
        return pd.DataFrame(
            data.numpy(),
            columns=[
                str(round(v, 3)).lstrip("0")
                for v in np.asarray(self.standard_deviations)
            ],
        )
