"""Persistent per-instance state used by KL-only TSP training."""

from dataclasses import dataclass
from typing import Optional

import torch

from solution_graph import InstanceSearchState
from utils import gen_pyg_data


@dataclass
class PersistentTrainingInstance:
    """One fixed TSP instance and its graph/heatmap history across epochs."""

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

    def get_or_create_state(self, initial_heatmap, memory_strength=0.0):
        """Return persistent state re-anchored by the latest model H0."""
        if self.state is None:
            self.state = InstanceSearchState(
                initial_heatmap.detach().cpu(),
                archive_device="cpu",
                archive_path_dtype=torch.int32,
            )
        elif self.state.current_heatmap.shape != initial_heatmap.shape:
            raise ValueError("persistent heatmap shape does not match the instance")
        else:
            self.state.reanchor(
                initial_heatmap.detach().cpu(),
                memory_strength=memory_strength,
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
