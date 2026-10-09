"""Persistent per-instance state used by KL-only TSP training."""

from dataclasses import dataclass
from typing import Optional

import torch

from solution_graph import InstanceSearchState
from utils import gen_pyg_data


@dataclass
class PersistentTrainingInstance:
    """One fixed TSP instance with visit-local search state."""

    coordinates: torch.Tensor
    state: Optional[InstanceSearchState] = None
    visits: int = 0

    def __post_init__(self):
        if self.coordinates.dim() != 2 or self.coordinates.size(-1) != 2:
            raise ValueError("TSP coordinates must have shape [n_nodes, 2]")
        # The pool itself is intentionally CPU-resident.  A problem graph is
        # reconstructed on the requested device only while the instance is in
        # the current optimizer batch.
        self.coordinates = self.coordinates.detach().clone().cpu()

    @property
    def n_nodes(self):
        return self.coordinates.size(0)

    def materialize(self, k_sparse, device):
        """Construct this fixed instance's problem graph on ``device``."""
        return gen_pyg_data(
            self.coordinates.to(device),
            k_sparse=k_sparse,
            start_node=0,
        )

    def start_visit(
        self,
        initial_heatmap,
        archive_max_solutions=None,
        archive_max_rounds=None,
    ):
        """Start from H0 and an empty archive, matching fresh test instances."""
        if initial_heatmap.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("initial heatmap shape does not match the instance")
        self.state = InstanceSearchState(
            initial_heatmap.detach().cpu(),
            archive_device="cpu",
            archive_path_dtype=torch.int32,
            archive_max_solutions=archive_max_solutions,
            archive_max_rounds=archive_max_rounds,
        )
        return self.state

    def finish_visit(self):
        self.visits += 1


def create_training_pool(count, n_nodes):
    """Create a fixed random instance pool that lives for the whole run."""
    if count < 1:
        raise ValueError("training pool size must be positive")
    coordinates = torch.rand((count, n_nodes, 2), device="cpu")
    return [PersistentTrainingInstance(item) for item in coordinates]


def refresh_training_pool(pool, fraction):
    """Replace a controlled fraction of coordinates before a new epoch."""
    if not pool:
        raise ValueError("training pool cannot be empty")
    if not 0 <= fraction <= 1:
        raise ValueError("pool refresh fraction must be between 0 and 1")
    replace_count = int(round(len(pool) * fraction))
    if replace_count == 0:
        return 0

    indices = torch.randperm(len(pool))[:replace_count].tolist()
    n_nodes = pool[0].n_nodes
    for index in indices:
        pool[index] = PersistentTrainingInstance(torch.rand(n_nodes, 2))
    return replace_count


def iter_pool_batches(pool, steps, batch_size):
    """Shuffle and yield every pool member exactly once in one epoch."""
    if not pool:
        raise ValueError("training pool cannot be empty")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    required = steps * batch_size
    if len(pool) != required:
        raise ValueError(
            "training pool size must equal steps * batch_size "
            f"({len(pool)} != {steps} * {batch_size})"
        )

    indices = torch.randperm(len(pool)).tolist()
    for offset in range(0, required, batch_size):
        yield [pool[index] for index in indices[offset : offset + batch_size]]
