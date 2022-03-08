from dataclasses import dataclass
from typing import Optional

import torch
from tqdm.notebook import tqdm

# from attack import Attack
# from config import Config
from masker import Masker
from metric import Distance, Metric
from pickle_utils import append_pickle, create_pickle
from utils import timer


@dataclass
class Runner:
    device: Optional[str] = "cpu"

    def __init__(self, model, loader, device: str = "cpu"):
        self.model = model
        self.loader = loader
        self.device = torch.device(device)

    @timer
    @torch.no_grad()
    def run_accuracy(self, n_batches: Optional[int] = 0) -> None:
        accuracies = []
        for batch_idx, batch in tqdm(
            enumerate(self.loader),
            total=n_batches or len(self.loader),
        ):
            if n_batches and n_batches <= batch_idx:
                break
            accuracies.append(self.predict_batch(batch).accuracy)
        accuracies = torch.cat(accuracies)
        accuracy = accuracies.float().mean().item()
        print(
            f"Accuracy of {(accuracy * 100):.2f}% ({accuracies.sum()}/{accuracies.numel()})"
        )

    @torch.no_grad()
    def predict_batch(self, batch: list[list[torch.Tensor]]) -> Metric:
        """
        Run the batch threw the model
        """
        x, y = self.unpack_batch_to_device(batch)
        return Metric(self.model(x), y)

    def unpack_batch_to_device(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Sends (x, y) to self.device if needed"""
        x, y = batch
        if self.device != x.device:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def get_one_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.unpack_batch_to_device(next(iter(self.loader)))

    @timer
    def run(self, pickle_path: str, standard_deviations: list[float]):
        """
        Run #n_batches number of batches threw the model.
        Each input is predicted #standard_deviations amount of time(s).
        Finally, the results are appended into a pickle file -> pickle_path.
        """
        standard_deviations = torch.as_tensor(standard_deviations)

        # initialize the pickle file at {pickle_path}.
        create_pickle(pickle_path)
        # append pickle file with standard_deviations.
        append_pickle(pickle_path, {"standard_deviations": standard_deviations})

        for batch_id, batch in tqdm(enumerate(self.loader), total=len(self.loader)):
            x, y = self.unpack_batch_to_device(batch)

            logits = []
            distances_l2 = []
            distances_PSNR = []

            masker = Masker(x, standard_deviations)
            for mask in masker:
                logits.append(self.model(mask))
                d = Distance(x, mask)
                distances_l2.append(d.l2)
                distances_PSNR.append(d.PSNR)

            # append pickle file with batch data.
            append_pickle(
                pickle_path,
                {
                    "batch_id": batch_id,
                    "logits": torch.cat(logits).cpu(),
                    "labels": y.cpu(),
                    "distances_l2": torch.tensor(distances_l2).cpu(),
                    "distances_PSNR": torch.tensor(distances_PSNR).cpu(),
                },
            )

    # def run_with_attack(
    #     self,
    #     pickle_path: str,
    #     standard_deviations: torch.Tensor,
    #     attack: Attack,
    # ):
    #     """
    #     Run #n_batches number of batches threw the model.
    #     Generate adversarial examples with {attack: Attack}
    #     Each input is predicted #standard_deviations amount of time(s).
    #     Finally, the results are appended into a pickle file -> pickle_path.
    #     """

    #     # initialize the pickle file at {pickle_path}.
    #     create_pickle(pickle_path)
    #     # append pickle file with standard_deviations.
    #     append_pickle(pickle_path, {"standard_deviations": standard_deviations})

    #     append_pickle(
    #         pickle_path,
    #         {
    #             "attack_module": attack.class_.__class__.__name__,
    #             "attack_epsilon": attack.epsilon,
    #             "attack_targeted": bool(attack.targets),
    #         },
    #     )

    #     for batch_id, batch in tqdm(enumerate(self.loader), total=len(self.loader)):
    #         x, y = self.unpack_batch_to_device(batch)

    #         # generating adversarial examples
    #         a = attack(self.model, x, y)
    #         attack_distance = Distance(x, a)
    #         del x

    #         logits = []
    #         distances_l2 = []
    #         distances_PSNR = []

    #         masker = Masker(a, standard_deviations)
    #         for mask in masker:
    #             logits.append(self.model(mask))
    #             d = Distance(a, mask)
    #             distances_l2.append(d.l2)
    #             distances_PSNR.append(d.PSNR)

    #         # append pickle file with batch data.
    #         append_pickle(
    #             pickle_path,
    #             {
    #                 "batch_id": batch_id,
    #                 "logits": torch.cat(logits).cpu(),
    #                 "labels": y.cpu(),
    #                 "distances_l2": torch.tensor(distances_l2).cpu(),
    #                 "distances_PSNR": torch.tensor(distances_PSNR).cpu(),
    #                 "attack_distance_l2": attack_distance.l2.item(),
    #                 "attack_distance_PSNR": attack_distance.PSNR.item(),
    #             },
    #         )
