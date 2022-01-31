from dataclasses import dataclass

import torch
from torch.nn import PairwiseDistance
from torch.nn.functional import softmax


@dataclass
class Metric:
    logits: torch.Tensor
    labels: torch.Tensor

    @property
    def prediction(self) -> torch.Tensor:
        return torch.max(self.logits, 1)[1]

    @property
    def accuracy(self) -> torch.Tensor:
        return self.prediction == self.labels

    @property
    def consistency(self) -> torch.Tensor:
        """
        Compares the first prediction (i.e. with 0-std mask)
        to the rest of the predictions (i.e. with n-std masks)
        Consistency is when the [1:] predictions equals the [0] prediction.
        """
        assert len(self.logits.shape) == 2 and self.logits.shape[0] > 1
        return self.prediction[0] == self.prediction

    @property
    def softmax(self) -> torch.Tensor:
        return softmax(self.logits, dim=-1)

    @property
    def pairwise_distance(self) -> torch.Tensor:
        """
        Computes the L1-norm between the 0-th softmax and (n+1)-softmax
        Used to show the difference in predictions between n-batches predictions, ex:
            self.softmax[0] is the reference
            self.softmax[1:] are the softmax we compute the L1-norm wrt the reference
        """
        pdist = PairwiseDistance(p=1)
        distances = [torch.tensor(0)]
        for softmax in self.softmax[1:]:
            distances.append(pdist(self.softmax[0], softmax))
        return torch.tensor(distances)

    @property
    def std_difference(self) -> torch.Tensor:
        """
        Computes the standard deviation of the difference between the 0-th logits and (n+1)-logits
        Used to show the difference in logits between n-batches logits, ex:
            self.logits[0] is the reference
            self.logits[1:] are the logits we compute the std difference wrt the reference
        """
        differences = [torch.tensor(0)]
        for logits in self.logits[1:]:
            differences.append((self.logits[0] - logits).std(dim=-1))
        return torch.tensor(differences)


@dataclass
class Distance:
    x: torch.Tensor
    y: torch.Tensor

    @property
    def l2(self) -> torch.Tensor:
        """L2 norm. Euclidean distance"""
        return self._norm(p=2)

    @property
    def PSNR(self) -> torch.Tensor:
        "Computes the Peak Signal-To-Noise Ratio (PSNR) between x and y."
        mse = torch.mean((self.x - self.y) ** 2)
        if not mse:
            return torch.tensor(100.00)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def _norm(self, p: int = 2) -> torch.Tensor:
        assert self.x.shape == self.y.shape
        dist = torch.cdist(self.x.flatten(1), self.y.flatten(1), p=p).diagonal().cpu()
        return dist.mean()
