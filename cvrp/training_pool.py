"""Fixed CVRP training instances with separate cross-epoch route archives."""

from dataclasses import dataclass
from typing import Optional

import torch

from solution_graph import SolutionArchive
from utils import gen_instance


@dataclass
class PersistentTrainingInstance:
    demands: torch.Tensor
    distances: torch.Tensor
    archive: Optional[SolutionArchive] = None

    def __post_init__(self):
        self.demands = self.demands.detach().clone().cpu()
        self.distances = self.distances.detach().clone().cpu()

    def materialize(self, device):
        return self.demands.to(device), self.distances.to(device)

    def get_archive(self, capacity=50, max_solutions=None, max_rounds=None):
        if self.archive is None:
            self.archive = SolutionArchive(
                self.demands, capacity=capacity,
                max_solutions=max_solutions, max_rounds=max_rounds,
            )
        elif (self.archive.max_solutions != max_solutions or
              self.archive.max_rounds != max_rounds):
            raise ValueError("archive limits cannot change between visits")
        return self.archive


def create_training_pool(count, n_customers):
    if count < 1:
        raise ValueError("pool size must be positive")
    return [PersistentTrainingInstance(*gen_instance(n_customers, "cpu"))
            for _ in range(count)]


def iter_pool_batches(pool, steps, batch_size):
    if len(pool) != steps * batch_size:
        raise ValueError("pool size must equal steps * batch_size")
    order = torch.randperm(len(pool)).tolist()
    for offset in range(0, len(order), batch_size):
        yield [pool[index] for index in order[offset:offset + batch_size]]
