"""Replaceable constrained solution samplers for iterative heatmap refinement.

The solution-graph learner only depends on this small interface.  ACO remains
the default sampler, but another constrained decoder can be plugged in by
returning the same ``SolutionBatch`` fields.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from aco import ACO


@dataclass
class SolutionBatch:
    """One population sampled from a heatmap.

    ``sampled_paths`` and ``sampled_costs`` always correspond to one another.
    ``feasible_paths`` and ``feasible_costs`` are the paths admitted to the
    cumulative solution graph.  For the ACO+NLS implementation they are the
    locally improved, valid TSP tours and their actual costs.
    """

    sampled_costs: torch.Tensor
    feasible_costs: torch.Tensor
    log_probs: Optional[torch.Tensor]
    sampled_paths: torch.Tensor
    feasible_paths: torch.Tensor


class ACOSolutionSampler:
    """Constraint-aware ACO sampler implementing the solution sampler API."""

    def __init__(
        self,
        n_solutions,
        heatmap,
        distances,
        device,
        local_search="nls",
    ):
        self._aco = ACO(
            n_ants=n_solutions,
            heuristic=heatmap,
            distances=distances,
            device=device,
            local_search=local_search,
        )

    @property
    def n_solutions(self):
        return self._aco.n_ants

    @property
    def best_cost(self):
        return self._aco.lowest_cost

    def set_heatmap(self, heatmap):
        self._aco.set_heuristic(heatmap)

    def sample(
        self,
        inference=False,
        require_log_probs=False,
        local_search_inference=None,
    ):
        (
            sampled_costs,
            feasible_costs,
            log_probs,
            sampled_paths,
            feasible_paths,
        ) = self._aco.sample_iteration(
            inference=inference,
            require_prob=require_log_probs,
            local_search_inference=local_search_inference,
        )
        return SolutionBatch(
            sampled_costs=sampled_costs,
            feasible_costs=feasible_costs,
            log_probs=log_probs,
            sampled_paths=sampled_paths,
            feasible_paths=feasible_paths,
        )
