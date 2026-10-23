"""Persistent tour archive and solution-edge graphs for H0-to-H1 training.

The default local graph retains strong candidate and sampled city edges as
nodes. Compatible edge pairs sharing a city are relations, with observed tour
fragments supplying quality-weighted evidence. Older whole-tour hypergraph
helpers remain for checkpoint and ablation compatibility. ``SolutionArchive``
retains feasible sampled tours across visits to the same training instance.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


EPS = 1e-10


class SolutionArchive:
    """Keep all feasible sampled tours with their costs and sampling rounds."""

    def __init__(
        self,
        n_nodes: int,
        storage_device=None,
        path_dtype=None,
        max_solutions=None,
        max_rounds=None,
    ):
        self.n_nodes = n_nodes
        self.storage_device = (
            None if storage_device is None else torch.device(storage_device)
        )
        self.path_dtype = path_dtype
        if max_solutions is not None and max_solutions < 1:
            raise ValueError("archive solution limit must be positive")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("archive round limit must be positive")
        self.max_solutions = max_solutions
        self.max_rounds = max_rounds
        if path_dtype is not None:
            if path_dtype not in (torch.int16, torch.int32, torch.int64):
                raise ValueError("archive path dtype must be an integer dtype")
            if n_nodes - 1 > torch.iinfo(path_dtype).max:
                raise ValueError(
                    f"{path_dtype} cannot represent node index {n_nodes - 1}"
                )
        self._paths = []
        self._costs = []
        self._rounds = []
        self._next_round = 0
        # Each sampling round contributes one compact edge-key population to
        # the online hypergraph accumulator.  Historical paths remain available
        # for inspection, but H1 no longer rebuilds their full incidence graph.
        self._round_edge_keys = []
        self._round_costs = []
        self._online_statistics_cache = {}

    def _replace_storage(self, paths, costs, rounds):
        """Replace compact storage after explicit archive pruning."""
        self._paths = [paths]
        self._costs = [costs]
        self._rounds = [rounds]

    def _enforce_limits(self):
        """Bound raw round history and full-tour storage when requested."""
        history_trimmed = False
        if (
            self.max_rounds is not None
            and len(self._round_edge_keys) > self.max_rounds
        ):
            overflow = len(self._round_edge_keys) - self.max_rounds
            del self._round_edge_keys[:overflow]
            del self._round_costs[:overflow]
            history_trimmed = True

        if not self._paths:
            if history_trimmed:
                self._online_statistics_cache.clear()
            return

        # The common case is still below both limits. Avoid concatenating all
        # archived tours until pruning or a round-history trim is necessary.
        if not history_trimmed and (
            self.max_solutions is None or len(self) <= self.max_solutions
        ):
            return

        paths, costs, rounds = self.tensors()
        selection = torch.arange(paths.size(0), device=paths.device)
        if self.max_rounds is not None:
            first_retained_round = max(self._next_round - self.max_rounds, 0)
            selection = selection.index_select(
                0,
                torch.nonzero(
                    rounds.index_select(0, selection) >= first_retained_round,
                    as_tuple=False,
                ).flatten(),
            )

        if (
            self.max_solutions is not None
            and selection.numel() > self.max_solutions
        ):
            selected_costs = costs.index_select(0, selection.to(costs.device))
            selected_rounds = rounds.index_select(0, selection.to(rounds.device))
            weights = compute_quality_weights(
                selected_costs,
                selected_rounds,
                num_rounds=self._next_round,
            )
            keep = torch.topk(
                weights,
                k=self.max_solutions,
                largest=True,
                sorted=True,
            ).indices.to(selection.device)
            selection = selection.index_select(0, keep)

        if selection.numel() != paths.size(0) or history_trimmed:
            self._replace_storage(
                paths.index_select(0, selection),
                costs.index_select(0, selection.to(costs.device)),
                rounds.index_select(0, selection.to(rounds.device)),
            )
            self._online_statistics_cache.clear()

    def __len__(self):
        return sum(paths.size(0) for paths in self._paths)

    @property
    def num_rounds(self):
        return self._next_round

    def add(self, paths: torch.Tensor, costs: torch.Tensor):
        """Append one feasible solution population to the archive.

        Store every sampled feasible tour as observed, including identical,
        rotated, and reversed tours. Each occurrence retains its own cost and
        sampling round.

        Args:
            paths: ``[n_nodes, n_ants]`` sampled tours.
            costs: ``[n_ants]`` objective values of the sampled tours.
        """
        if paths.dim() != 2 or paths.size(0) != self.n_nodes:
            raise ValueError(
                f"paths must have shape [{self.n_nodes}, n_ants], got {tuple(paths.shape)}"
            )
        if costs.dim() != 1 or costs.size(0) != paths.size(1):
            raise ValueError("costs must contain one value for every sampled tour")
        if not torch.isfinite(costs).all():
            raise ValueError("solution costs must all be finite")

        tours = paths.transpose(0, 1).detach().clone()
        expected_nodes = torch.arange(
            self.n_nodes, device=tours.device, dtype=tours.dtype
        ).expand_as(tours)
        if not torch.equal(torch.sort(tours, dim=1).values, expected_nodes):
            raise ValueError(
                "every archived TSP solution must be a permutation of all nodes"
            )
        detached_costs = costs.detach().clone()
        observation_u = tours
        observation_v = torch.roll(tours, shifts=-1, dims=1)
        observation_lo = torch.minimum(observation_u, observation_v)
        observation_hi = torch.maximum(observation_u, observation_v)
        observation_edge_keys = observation_lo * self.n_nodes + observation_hi
        self._round_edge_keys.append(
            observation_edge_keys.to(device="cpu", dtype=torch.int32)
        )
        self._round_costs.append(
            detached_costs.to(device="cpu", dtype=torch.float32)
        )

        if self.storage_device is not None:
            tours = tours.to(self.storage_device)
        if self.path_dtype is not None:
            tours = tours.to(dtype=self.path_dtype)

        if self.storage_device is not None:
            detached_costs = detached_costs.to(self.storage_device)
        rounds = torch.full(
            (tours.size(0),),
            self._next_round,
            dtype=torch.long,
            device=tours.device,
        )
        self._paths.append(tours)
        self._costs.append(detached_costs)
        self._rounds.append(rounds)
        self._next_round += 1
        self._enforce_limits()
        return {
            "added": tours.size(0),
            "duplicates": 0,
        }

    def online_edge_statistics(
        self,
        temperature=0.75,
        age_decay=0.1,
        uniform_mix=0.01,
        elite_ratio=0.25,
    ):
        """Incrementally update direct and edge-solution-edge statistics.

        Quality weights are normalized within each sampling round.  Historical
        round mass is exponentially decayed before the next round is added.
        Consequently each new call processes only newly appended populations,
        instead of rebuilding a growing incidence graph from every old tour.
        The sufficient statistics intentionally live on CPU so test-time H1
        updates do not bounce small TSP matrices between CPU and GPU.
        """
        if not self._round_edge_keys:
            raise RuntimeError("cannot aggregate an empty solution archive")
        if not 0 < elite_ratio <= 1:
            raise ValueError("elite ratio must be in (0, 1]")

        cache_key = (
            float(temperature),
            float(age_decay),
            float(uniform_mix),
            float(elite_ratio),
        )
        cache = self._online_statistics_cache.get(cache_key)
        if cache is None:
            flat_size = self.n_nodes * self.n_nodes
            cache = {
                "processed_rounds": 0,
                "direct": torch.zeros(flat_size, dtype=torch.float32),
                "propagated_numerator": torch.zeros(
                    flat_size, dtype=torch.float32
                ),
                "propagated_mass": torch.zeros(flat_size, dtype=torch.float32),
            }
            self._online_statistics_cache[cache_key] = cache

        decay = float(torch.exp(torch.tensor(-float(age_decay))))
        for round_index in range(
            cache["processed_rounds"],
            len(self._round_edge_keys),
        ):
            cache["direct"].mul_(decay)
            cache["propagated_numerator"].mul_(decay)
            cache["propagated_mass"].mul_(decay)

            edge_keys = self._round_edge_keys[round_index].to(dtype=torch.long)
            costs = self._round_costs[round_index]
            elite_count = max(
                1,
                int(torch.ceil(torch.tensor(costs.numel() * elite_ratio))),
            )
            elite_indices = torch.topk(
                costs,
                k=elite_count,
                largest=False,
                sorted=False,
            ).indices
            edge_keys = edge_keys.index_select(0, elite_indices)
            costs = costs.index_select(0, elite_indices)
            rounds = torch.zeros(costs.numel(), dtype=torch.long)
            weights = compute_quality_weights(
                costs,
                rounds,
                num_rounds=1,
                temperature=temperature,
                age_decay=0.0,
                uniform_mix=uniform_mix,
            ).to(dtype=torch.float32)
            flat_keys = edge_keys.reshape(-1)
            incidence_weights = weights.repeat_interleave(self.n_nodes)

            cache["direct"].index_add_(0, flat_keys, incidence_weights)
            solution_context = cache["direct"][edge_keys].mean(dim=1)
            context_weights = (
                weights * solution_context
            ).repeat_interleave(self.n_nodes)
            cache["propagated_numerator"].index_add_(
                0,
                flat_keys,
                context_weights,
            )
            cache["propagated_mass"].index_add_(
                0,
                flat_keys,
                incidence_weights,
            )
            cache["processed_rounds"] = round_index + 1

        direct = cache["direct"].reshape(self.n_nodes, self.n_nodes)
        propagated = (
            cache["propagated_numerator"]
            / cache["propagated_mass"].clamp_min(EPS)
        ).reshape(self.n_nodes, self.n_nodes)
        return direct, propagated

    def tensors(self, device=None):
        if not self._paths:
            raise RuntimeError("cannot build a solution graph from an empty archive")

        paths = torch.cat(self._paths, dim=0)
        costs = torch.cat(self._costs, dim=0)
        rounds = torch.cat(self._rounds, dim=0)
        if device is not None:
            paths = paths.to(device)
            costs = costs.to(device)
            rounds = rounds.to(device)
        return paths, costs, rounds

    def build_incidence(self, device=None):
        """Return the compact edge-node/solution-hyperedge incidence graph."""
        paths, costs, rounds = self.tensors(device=device)
        # Persistent training archives use a compact integer dtype on CPU.
        # Incidence keys and tensor indices must be computed in int64.
        paths = paths.to(dtype=torch.long)
        n_solutions, n_nodes = paths.shape

        # TSP edges are undirected.  Canonicalising (u, v) lets reversed tours
        # share the same edge node in the cumulative solution graph.
        u = paths
        v = torch.roll(paths, shifts=-1, dims=1)
        lo = torch.minimum(u, v)
        hi = torch.maximum(u, v)
        edge_keys = (lo * n_nodes + hi).reshape(-1)

        unique_keys, edge_incidence = torch.unique(
            edge_keys, sorted=True, return_inverse=True
        )
        edge_u = torch.div(unique_keys, n_nodes, rounding_mode="floor")
        edge_v = unique_keys.remainder(n_nodes)
        solution_incidence = torch.arange(
            n_solutions, device=paths.device
        ).repeat_interleave(n_nodes)

        return {
            "edge_u": edge_u,
            "edge_v": edge_v,
            "edge_incidence": edge_incidence,
            "solution_incidence": solution_incidence,
            "costs": costs,
            "rounds": rounds,
            "n_solutions": n_solutions,
        }


class InstanceSearchState:
    """Heatmap and cumulative hypergraph state owned by one problem instance.

    Training carries only the archive into a new visit of the same instance.
    Search and the sole refinement both start from that visit's fresh H0.
    """

    def __init__(
        self,
        initial_heatmap,
        archive_device=None,
        archive_path_dtype=None,
        archive_max_solutions=None,
        archive_max_rounds=None,
        archive=None,
        refinement_prior_heatmap=None,
    ):
        if initial_heatmap.dim() != 2:
            raise ValueError("initial heatmap must be a matrix")
        if initial_heatmap.size(0) != initial_heatmap.size(1):
            raise ValueError("initial heatmap must be square")

        self.initial_heatmap = initial_heatmap
        self.current_heatmap = initial_heatmap
        if refinement_prior_heatmap is not None and (
            refinement_prior_heatmap.shape != initial_heatmap.shape
        ):
            raise ValueError("refinement prior shape must match the heatmap")
        self.refinement_prior_heatmap = (
            initial_heatmap
            if refinement_prior_heatmap is None
            else refinement_prior_heatmap
        )
        if archive is not None and archive.n_nodes != initial_heatmap.size(0):
            raise ValueError("inherited archive node count must match the heatmap")
        self.archive = archive if archive is not None else SolutionArchive(
            n_nodes=initial_heatmap.size(0),
            storage_device=archive_device,
            path_dtype=archive_path_dtype,
            max_solutions=archive_max_solutions,
            max_rounds=archive_max_rounds,
        )
        self.round_index = 0

    def add_feasible_solutions(self, paths, costs):
        self.archive.add(paths, costs)

    def advance(self, next_heatmap):
        if next_heatmap.shape != self.current_heatmap.shape:
            raise ValueError("next heatmap shape must match the current heatmap")
        self.current_heatmap = next_heatmap
        self.round_index += 1


def normalize_heatmap_rows(heatmap):
    """Normalize a square heatmap over non-self transitions."""
    if heatmap.dim() != 2 or heatmap.size(0) != heatmap.size(1):
        raise ValueError("heatmap must be a square matrix")
    mask = 1.0 - torch.eye(
        heatmap.size(0),
        device=heatmap.device,
        dtype=heatmap.dtype,
    )
    normalized = heatmap.clamp_min(0.0) * mask
    normalized = normalized / normalized.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(EPS)
    return normalized + EPS * mask


def compute_quality_weights(
    costs,
    rounds,
    num_rounds,
    temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
):
    """Convert minimization costs into normalized quality/age weights.

    Lower-ranked-cost and more recent solutions receive more weight.  A small
    uniform component prevents poor or old solutions from losing all influence
    and helps preserve exploration.
    """
    if temperature <= 0:
        raise ValueError("quality temperature must be positive")
    if age_decay < 0:
        raise ValueError("age decay must be non-negative")
    if not 0 <= uniform_mix <= 1:
        raise ValueError("uniform mix must be between 0 and 1")

    costs = costs.to(dtype=torch.float32) if not costs.is_floating_point() else costs
    rounds = rounds.to(device=costs.device, dtype=costs.dtype)
    # Preserve objective-value differences while preventing nearly identical
    # sampled-tour costs from producing a numerically one-hot target.
    best_cost = costs.min()
    mean_cost = costs.mean()
    scale_floor = mean_cost.abs().clamp_min(1.0) * 1e-4
    quality_scale = (mean_cost - best_cost).clamp_min(scale_floor)
    normalized_gap = ((costs - best_cost) / quality_scale).clamp(max=20.0)
    latest_round = max(float(num_rounds - 1), 0.0)
    age = latest_round - rounds
    logits = -normalized_gap / temperature - age_decay * age
    weights = torch.softmax(logits, dim=0)
    uniform = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
    return (1.0 - uniform_mix) * weights + uniform_mix * uniform


def build_learnable_solution_graph(
    previous_heatmap,
    distances,
    archive: SolutionArchive,
    max_solutions=None,
    temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
):
    """Build the sparse edge-solution graph consumed by the learned updater.

    TSP edges are graph nodes and archived tours are solution nodes.  Incidence
    indices represent the two message-passing directions without materialising
    an edge clique. All archived tours are used by default; an optional cap
    selects the highest quality/recency entries when memory must be limited.
    """
    if max_solutions is not None and max_solutions < 1:
        raise ValueError("maximum solution-graph size must be positive")
    if distances.shape != previous_heatmap.shape:
        raise ValueError("distance matrix shape must match the heatmap")

    # Rank on the archive's storage device (normally CPU), then transfer the
    # selected tours to the neural updater's device.
    paths, costs, rounds = archive.tensors()
    paths = paths.to(dtype=torch.long)
    costs = costs.to(dtype=torch.float32)
    rounds = rounds.to(dtype=torch.long)
    all_weights = compute_quality_weights(
        costs,
        rounds,
        num_rounds=archive.num_rounds,
        temperature=temperature,
        age_decay=age_decay,
        uniform_mix=uniform_mix,
    ).to(dtype=previous_heatmap.dtype)

    if max_solutions is None:
        selected = torch.arange(paths.size(0), device=paths.device)
    else:
        selected = torch.topk(
            all_weights,
            k=min(int(max_solutions), paths.size(0)),
            largest=True,
            sorted=True,
        ).indices
    paths = paths.index_select(0, selected).to(previous_heatmap.device)
    costs = costs.index_select(0, selected).to(
        device=previous_heatmap.device,
        dtype=previous_heatmap.dtype,
    )
    rounds = rounds.index_select(0, selected).to(previous_heatmap.device)
    solution_weights = all_weights.index_select(0, selected).to(
        device=previous_heatmap.device,
        dtype=previous_heatmap.dtype,
    )
    solution_weights = solution_weights / solution_weights.sum().clamp_min(EPS)

    n_solutions, n_nodes = paths.shape
    u = paths
    v = torch.roll(paths, shifts=-1, dims=1)
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    flat_edge_keys = (lo * n_nodes + hi).reshape(-1)
    unique_keys, edge_incidence = torch.unique(
        flat_edge_keys,
        sorted=True,
        return_inverse=True,
    )
    edge_u = torch.div(unique_keys, n_nodes, rounding_mode="floor")
    edge_v = unique_keys.remainder(n_nodes)
    solution_incidence = torch.arange(
        n_solutions,
        device=paths.device,
    ).repeat_interleave(n_nodes)

    normalized_heatmap = normalize_heatmap_rows(previous_heatmap.detach())
    heatmap_feature = 0.5 * (
        normalized_heatmap[edge_u, edge_v]
        + normalized_heatmap[edge_v, edge_u]
    )

    distances = distances.detach().to(
        device=previous_heatmap.device,
        dtype=previous_heatmap.dtype,
    )
    inverse_distance = distances[edge_u, edge_v].clamp_min(EPS).reciprocal()
    inverse_distance = inverse_distance / inverse_distance.max().clamp_min(EPS)

    incidence_quality = solution_weights.index_select(0, solution_incidence)
    quality_support = previous_heatmap.new_zeros(unique_keys.numel())
    quality_support.index_add_(0, edge_incidence, incidence_quality)
    quality_support = quality_support / quality_support.max().clamp_min(EPS)

    occurrence = previous_heatmap.new_zeros(unique_keys.numel())
    occurrence.index_add_(
        0,
        edge_incidence,
        previous_heatmap.new_ones(edge_incidence.numel()),
    )
    occurrence = occurrence / float(max(n_solutions, 1))
    edge_features = torch.stack(
        (heatmap_feature, inverse_distance, quality_support, occurrence),
        dim=-1,
    )

    best_cost = costs.min()
    mean_cost = costs.mean()
    scale_floor = mean_cost.abs().clamp_min(1.0) * 1e-4
    quality_scale = (mean_cost - best_cost).clamp_min(scale_floor)
    normalized_gap = ((costs - best_cost) / quality_scale).clamp(min=0.0)
    quality_score = normalized_gap.add(1.0).reciprocal()
    latest_round = max(archive.num_rounds - 1, 0)
    ages = latest_round - rounds.to(dtype=previous_heatmap.dtype)
    recency = torch.exp(-float(age_decay) * ages.clamp_min(0.0))
    solution_features = torch.stack(
        (solution_weights, quality_score, recency),
        dim=-1,
    )

    return {
        "edge_features": edge_features,
        "solution_features": solution_features,
        "edge_incidence": edge_incidence,
        "solution_incidence": solution_incidence,
        "incidence_weights": incidence_quality,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "n_nodes": n_nodes,
        "n_solutions": n_solutions,
    }


def quality_target_heatmap(
    previous_heatmap,
    archive: SolutionArchive,
    distances=None,
    temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
):
    """Build H_(t+1) by fixed quality-weighted hypergraph aggregation.

    Direct edge support is first aggregated from all solution hyperedges.  A
    second edge->solution->edge pass propagates context between TSP edges that
    occur in the same archived solutions.  Direct support, propagated support,
    inverse-distance support are normalized before convex mixing. The
    preceding heatmap supplies only shape, device, and dtype; its values are
    not mixed into the result. This function has no trainable parameters, so
    its output can safely serve as a detached KL pseudo-label.
    """
    if not 0 <= propagation_strength <= 1:
        raise ValueError("propagation strength must be between 0 and 1")
    if not 0 <= distance_prior_strength <= 1:
        raise ValueError("distance prior strength must be between 0 and 1")

    direct_upper, propagated_upper = archive.online_edge_statistics(
        temperature=temperature,
        age_decay=age_decay,
        uniform_mix=uniform_mix,
        elite_ratio=elite_ratio,
    )
    direct_upper = direct_upper.to(
        device=previous_heatmap.device,
        dtype=previous_heatmap.dtype,
    )
    direct_occupancy = direct_upper + direct_upper.transpose(0, 1)

    propagated_occupancy = None
    if propagation_strength > 0:
        propagated_upper = propagated_upper.to(
            device=previous_heatmap.device,
            dtype=previous_heatmap.dtype,
        )
        propagated_occupancy = (
            propagated_upper + propagated_upper.transpose(0, 1)
        )

    n_nodes = previous_heatmap.size(0)
    mask = 1.0 - torch.eye(
        n_nodes, device=previous_heatmap.device, dtype=previous_heatmap.dtype
    )
    def normalize_rows(matrix):
        matrix = matrix.clamp_min(0.0) * mask
        return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(EPS)

    graph_target = normalize_rows(direct_occupancy)
    if propagated_occupancy is not None:
        graph_target = (
            (1.0 - propagation_strength) * graph_target
            + propagation_strength * normalize_rows(propagated_occupancy)
        )

    if distances is not None and distance_prior_strength > 0:
        distances = distances.to(
            device=previous_heatmap.device,
            dtype=previous_heatmap.dtype,
        )
        if distances.shape != previous_heatmap.shape:
            raise ValueError("distance matrix shape must match the heatmap")
        distance_target = normalize_rows(distances.clamp_min(EPS).reciprocal())
        graph_target = (
            (1.0 - distance_prior_strength) * graph_target
            + distance_prior_strength * distance_target
        )

    return normalize_rows(graph_target) + EPS * mask


def graph_refined_heatmap(
    fresh_h0_heatmap,
    archive: SolutionArchive,
    layers=2,
    structure="whole_tour",
    quality_mode="inverse_cost",
    quality_temperature=0.1,
    candidate_only=False,
):
    """Parameter-free weighted GCN on the archived edge graph.

    A graph node represents an undirected TSP edge. In ``inverse_cost`` mode,
    its initial scalar feature is the mean of 1 / tour_cost over all archived
    tour occurrences containing it.
    ``whole_tour`` relates all edges in a tour, as in the original teacher.
    ``local`` relates only consecutive tour edges and gives each edge a
    self-loop. Both use symmetric-normalized messages without dense cliques.
    ``relative`` quality mode uses cost-relative route weights and a
    quality-weighted edge occurrence frequency. It does not decay old visits.
    ``candidate_only`` builds graph nodes only for edges present in H0's
    candidate support and masks the directed output to that same support.
    Historical tours have exactly the same statistical role as new tours: no
    age or round decay is applied. The returned H1 contains only the graph
    result; fresh H0 values are never blended into it.
    """
    if fresh_h0_heatmap.dim() != 2 or (
        fresh_h0_heatmap.size(0) != fresh_h0_heatmap.size(1)
    ):
        raise ValueError("fresh H0 heatmap must be square")
    if layers < 1:
        raise ValueError("GCN layers must be positive")
    if structure not in {"whole_tour", "local"}:
        raise ValueError("unknown solution-edge graph structure")
    if quality_mode not in {"inverse_cost", "relative"}:
        raise ValueError("unknown solution quality mode")
    if quality_temperature <= 0:
        raise ValueError("quality temperature must be positive")
    if len(archive) == 0:
        raise ValueError("the edge graph needs at least one sampled solution")

    paths, costs, _ = archive.tensors()
    n_nodes = fresh_h0_heatmap.size(0)
    if paths.size(1) != n_nodes:
        raise ValueError("archive and heatmap node counts must match")
    paths = paths.to(device=fresh_h0_heatmap.device, dtype=torch.long)
    costs = costs.to(
        device=fresh_h0_heatmap.device,
        dtype=fresh_h0_heatmap.dtype,
    )
    if quality_mode == "inverse_cost":
        solution_weights = costs.clamp_min(EPS).reciprocal()
    else:
        scale = (costs.mean() * quality_temperature).clamp_min(EPS)
        solution_weights = torch.exp(
            -(costs - costs.min()) / scale
        ).clamp_min(EPS)
    n_solutions = paths.size(0)
    next_nodes = torch.roll(paths, shifts=-1, dims=1)
    edge_u = torch.minimum(paths, next_nodes)
    edge_v = torch.maximum(paths, next_nodes)
    edge_keys = (edge_u * n_nodes + edge_v).reshape(-1)
    candidate = fresh_h0_heatmap.detach() > (10.0 * EPS)
    if candidate_only:
        keep = (candidate | candidate.T)[edge_u, edge_v].reshape(-1)
        if not bool(keep.any()):
            raise ValueError(
                "candidate-only graph has no observed candidate-edge support"
            )
        edge_keys = edge_keys[keep]
    unique_keys, edge_incidence = torch.unique(
        edge_keys, sorted=True, return_inverse=True
    )
    solution_incidence = torch.arange(
        n_solutions, device=paths.device
    ).repeat_interleave(n_nodes)
    if candidate_only:
        solution_incidence = solution_incidence[keep]
    incidence_weights = solution_weights.index_select(
        0, solution_incidence
    )
    n_edges = unique_keys.numel()
    support_sum = solution_weights.new_zeros(n_edges)
    support_sum.index_add_(0, edge_incidence, incidence_weights)
    if quality_mode == "inverse_cost":
        counts = solution_weights.new_zeros(n_edges)
        counts.index_add_(0, edge_incidence, torch.ones_like(incidence_weights))
        initial = support_sum / counts.clamp_min(1.0)
    else:
        initial = support_sum / solution_weights.sum().clamp_min(EPS)

    # In the whole-tour graph, every tour contributes an n_nodes-edge clique.
    # In the local graph, each occurrence contributes two consecutive-neighbor
    # relations and one self-loop, so both degrees have a closed form.
    if candidate_only:
        if structure == "whole_tour":
            retained_per_solution = torch.bincount(
                solution_incidence, minlength=n_solutions,
            ).to(dtype=solution_weights.dtype)
            degree = support_sum.new_zeros(n_edges)
            degree.index_add_(
                0, edge_incidence,
                incidence_weights * retained_per_solution.index_select(
                    0, solution_incidence,
                ),
            )
        else:
            occurrence_edges = torch.full(
                (n_solutions * n_nodes,), -1,
                device=paths.device, dtype=torch.long,
            )
            occurrence_edges[keep] = edge_incidence
            neighbors = torch.roll(
                occurrence_edges.reshape(n_solutions, n_nodes),
                shifts=-1, dims=1,
            ).reshape(-1)
            valid_pair = (occurrence_edges >= 0) & (neighbors >= 0)
            local_source = occurrence_edges[valid_pair]
            local_neighbor = neighbors[valid_pair]
            pair_weights = solution_weights.repeat_interleave(n_nodes)[valid_pair]
            degree = support_sum.clone()
            degree.index_add_(0, local_source, pair_weights)
            degree.index_add_(0, local_neighbor, pair_weights)
    else:
        degree_multiplier = n_nodes if structure == "whole_tour" else 3
        degree = degree_multiplier * support_sum
    degree_sqrt = degree.clamp_min(EPS).sqrt()
    if structure == "local" and not candidate_only:
        local_source = edge_incidence
        local_neighbor = torch.roll(
            edge_incidence.reshape(n_solutions, n_nodes), shifts=-1, dims=1,
        ).reshape(-1)
    hidden = initial
    for _ in range(layers):
        normalized_edges = hidden / degree_sqrt
        if structure == "whole_tour":
            solution_messages = hidden.new_zeros(n_solutions)
            solution_messages.index_add_(
                0,
                solution_incidence,
                normalized_edges.index_select(0, edge_incidence),
            )
            edge_messages = hidden.new_zeros(n_edges)
            edge_messages.index_add_(
                0,
                edge_incidence,
                incidence_weights
                * solution_messages.index_select(0, solution_incidence),
            )
        else:
            edge_messages = support_sum * normalized_edges
            edge_messages.index_add_(
                0, local_source,
                (pair_weights if candidate_only else incidence_weights)
                * normalized_edges.index_select(0, local_neighbor),
            )
            edge_messages.index_add_(
                0, local_neighbor,
                (pair_weights if candidate_only else incidence_weights)
                * normalized_edges.index_select(0, local_source),
            )
        hidden = 0.5 * hidden + 0.5 * (edge_messages / degree_sqrt)

    output_u = torch.div(unique_keys, n_nodes, rounding_mode="floor")
    output_v = unique_keys.remainder(n_nodes)
    graph_heatmap = hidden.new_zeros((n_nodes, n_nodes))
    graph_heatmap[output_u, output_v] = hidden
    graph_heatmap[output_v, output_u] = hidden
    if candidate_only:
        graph_heatmap = graph_heatmap.masked_fill(~candidate, 0.0)
    graph_heatmap = normalize_heatmap_rows(graph_heatmap)
    return graph_heatmap


def build_local_edge_gcn_graph(
    heatmap,
    distances,
    archive: SolutionArchive,
    max_solutions=None,
    quality_mode="relative_diversity",
    quality_temperature=0.1,
    diversity_strength=0.5,
    candidate_only=False,
    pheromone=None,
    incident_topk=6,
):
    """Build a sparse edge-node graph with compatible TSP fragments.

    Each city keeps its strongest ``incident_topk`` edges after merging
    pheromone/heuristic candidates with quality-weighted sampled edges.  Every
    pair of retained edges sharing a city is a relation; observed consecutive
    tour fragments add evidence to that relation, while unseen compatible
    fragments keep zero evidence and a small nonzero message-passing weight.
    """
    if heatmap.ndim != 2 or heatmap.size(0) != heatmap.size(1):
        raise ValueError("H0 heatmap must be square")
    if distances.shape != heatmap.shape:
        raise ValueError("distance matrix shape must match the heatmap")
    if max_solutions is not None and max_solutions < 1:
        raise ValueError("maximum solution-graph size must be positive")
    if quality_mode not in {"inverse_cost", "relative", "relative_diversity"}:
        raise ValueError("unknown solution quality mode")
    if quality_temperature <= 0:
        raise ValueError("quality temperature must be positive")
    if not 0 <= diversity_strength <= 1:
        raise ValueError("diversity strength must be between 0 and 1")
    if incident_topk < 1:
        raise ValueError("incident top-k must be positive")
    if pheromone is not None and pheromone.shape != heatmap.shape:
        raise ValueError("pheromone shape must match the heatmap")

    paths, costs, _ = archive.tensors(device=heatmap.device)
    paths = paths.to(dtype=torch.long)
    costs = costs.to(dtype=heatmap.dtype)
    n_solutions, n_nodes = paths.shape
    if n_nodes != heatmap.size(0):
        raise ValueError("archive and heatmap node counts must match")
    if max_solutions is not None and n_solutions > max_solutions:
        selected = torch.topk(costs, k=max_solutions, largest=False).indices
        paths = paths.index_select(0, selected)
        costs = costs.index_select(0, selected)
        n_solutions = paths.size(0)

    next_city = torch.roll(paths, shifts=-1, dims=1)
    lo = torch.minimum(paths, next_city)
    hi = torch.maximum(paths, next_city)
    keys = (lo * n_nodes + hi).reshape(-1)
    if quality_mode == "inverse_cost":
        weights = costs.clamp_min(EPS).reciprocal()
    else:
        scale = (costs.mean() * quality_temperature).clamp_min(EPS)
        weights = torch.exp(-(costs - costs.min()) / scale).clamp_min(EPS)
    novelty = torch.ones_like(weights)
    if quality_mode == "relative_diversity" and diversity_strength > 0:
        # Count undirected tour-edge occurrences across the archive. A tour
        # dominated by repeated edges contributes less than one with rarer
        # edges, without discarding any feasible solution from the archive.
        _, all_edge_ids, all_edge_counts = torch.unique(
            keys, sorted=True, return_inverse=True, return_counts=True,
        )
        edge_rarity = all_edge_counts[all_edge_ids].to(weights.dtype).rsqrt()
        route_rarity = edge_rarity.reshape(n_solutions, n_nodes).mean(dim=1)
        relative_rarity = route_rarity / route_rarity.mean().clamp_min(EPS)
        novelty = (
            1.0 + diversity_strength * (relative_rarity - 1.0)
        ).clamp(0.5, 1.5)
        weights = weights * novelty
    occurrence_weights = weights[:, None].expand(-1, n_nodes).reshape(-1)
    observed_keys, observed_ids = torch.unique(
        keys, sorted=True, return_inverse=True,
    )
    observed_support = weights.new_zeros(observed_keys.numel())
    observed_support.index_add_(0, observed_ids, occurrence_weights)
    observed_counts = weights.new_zeros(observed_keys.numel())
    observed_counts.index_add_(0, observed_ids, torch.ones_like(occurrence_weights))

    candidate = (heatmap.detach() > 10.0 * EPS)
    candidate = candidate | candidate.T
    candidate.fill_diagonal_(False)
    q = heatmap.detach().clamp_min(0)
    if pheromone is not None:
        q = q * pheromone.detach().to(device=q.device, dtype=q.dtype).clamp_min(0)
    q = torch.maximum(q, q.T).masked_fill(~candidate, 0.0)
    topk = min(incident_topk, n_nodes - 1)
    candidate_values, candidate_neighbors = torch.topk(
        q.masked_fill(~candidate, -torch.inf), k=topk, dim=-1,
    )
    candidate_cities = torch.arange(n_nodes, device=paths.device)[:, None].expand_as(
        candidate_neighbors
    )
    candidate_valid = torch.isfinite(candidate_values)
    candidate_lo = torch.minimum(candidate_cities, candidate_neighbors)
    candidate_hi = torch.maximum(candidate_cities, candidate_neighbors)
    candidate_keys = (
        candidate_lo * n_nodes + candidate_hi
    )[candidate_valid]
    observed_allowed = (
        candidate[torch.div(observed_keys, n_nodes, rounding_mode="floor"),
                  observed_keys.remainder(n_nodes)]
        if candidate_only else torch.ones_like(observed_keys, dtype=torch.bool)
    )
    merged_keys = torch.unique(
        torch.cat((candidate_keys, observed_keys[observed_allowed])), sorted=True,
    )
    if merged_keys.numel() == 0:
        raise ValueError("local edge graph has no candidate or sampled edges")
    observed_positions = torch.searchsorted(observed_keys, merged_keys)
    observed_positions = observed_positions.clamp_max(observed_keys.numel() - 1)
    merged_observed = observed_keys[observed_positions] == merged_keys
    merged_support = torch.where(
        merged_observed, observed_support[observed_positions],
        torch.zeros_like(merged_keys, dtype=weights.dtype),
    )
    merged_u = torch.div(merged_keys, n_nodes, rounding_mode="floor")
    merged_v = merged_keys.remainder(n_nodes)
    # Normalizing by the population weight makes sampled support comparable to
    # the heuristic/pheromone candidate score during top-D selection.
    merged_scores = q[merged_u, merged_v] + merged_support / weights.sum().clamp_min(EPS)
    incident_scores = q.new_full((n_nodes, n_nodes), -torch.inf)
    incident_scores[merged_u, merged_v] = merged_scores
    incident_scores[merged_v, merged_u] = merged_scores
    retained_values, retained_neighbors = torch.topk(
        incident_scores, k=topk, dim=-1,
    )
    retained_cities = torch.arange(n_nodes, device=paths.device)[:, None].expand_as(
        retained_neighbors
    )
    retained = torch.isfinite(retained_values)
    retained_lo = torch.minimum(retained_cities, retained_neighbors)
    retained_hi = torch.maximum(retained_cities, retained_neighbors)
    unique_keys = torch.unique(
        (retained_lo * n_nodes + retained_hi)[retained], sorted=True,
    )
    n_edges = unique_keys.numel()
    selected_positions = torch.searchsorted(observed_keys, unique_keys)
    selected_positions = selected_positions.clamp_max(observed_keys.numel() - 1)
    selected_observed = observed_keys[selected_positions] == unique_keys
    support = torch.where(
        selected_observed, observed_support[selected_positions],
        torch.zeros(n_edges, device=weights.device, dtype=weights.dtype),
    )
    counts = torch.where(
        selected_observed, observed_counts[selected_positions],
        torch.zeros(n_edges, device=weights.device, dtype=weights.dtype),
    )
    edge_u = torch.div(unique_keys, n_nodes, rounding_mode="floor")
    edge_v = unique_keys.remainder(n_nodes)
    inverse_distance = distances.detach()[edge_u, edge_v].clamp_min(EPS).reciprocal()
    inverse_distance = inverse_distance / inverse_distance.max().clamp_min(EPS)
    # Candidate-only nodes still need a positive starting score in H1.
    edge_prior = support + 0.01 * weights.mean() * inverse_distance
    edge_features = torch.stack((
        support / support.max().clamp_min(EPS),
        counts / n_solutions,
        inverse_distance,
    ), dim=-1)

    # Enumerate only local, compatible edge pairs: O(sum_v degree(v)^2), not
    # the dense O(n_edges^2) graph. All pairs receive a relation, regardless
    # of whether that fragment has yet appeared in an archived tour.
    incident = [[] for _ in range(n_nodes)]
    for edge_id, (u, v) in enumerate(zip(edge_u.tolist(), edge_v.tolist())):
        incident[u].append(edge_id)
        incident[v].append(edge_id)
    pair_keys_list = []
    shared_list = []
    for city, edge_ids_at_city in enumerate(incident):
        for i, source in enumerate(edge_ids_at_city):
            for target in edge_ids_at_city[i + 1:]:
                pair_keys_list.append(source * n_edges + target)
                shared_list.append(city)
    pair_keys = torch.tensor(pair_keys_list, device=paths.device, dtype=torch.long)
    shared_city = torch.tensor(shared_list, device=paths.device, dtype=torch.long)
    order = torch.argsort(pair_keys)
    pair_keys = pair_keys[order]
    shared_city = shared_city[order]
    pair_source = torch.div(pair_keys, n_edges, rounding_mode="floor")
    pair_target = pair_keys.remainder(n_edges)

    selected_at_occurrence = torch.searchsorted(unique_keys, keys)
    selected_at_occurrence = selected_at_occurrence.clamp_max(n_edges - 1)
    valid_occurrence = (unique_keys[selected_at_occurrence] == keys).reshape(
        n_solutions, n_nodes
    )
    occurrence = selected_at_occurrence.reshape(n_solutions, n_nodes)
    next_occurrence = torch.roll(occurrence, shifts=-1, dims=1)
    valid_fragments = valid_occurrence & torch.roll(
        valid_occurrence, shifts=-1, dims=1
    )
    fragment_source = torch.minimum(occurrence, next_occurrence)[valid_fragments]
    fragment_target = torch.maximum(occurrence, next_occurrence)[valid_fragments]
    fragment_keys = fragment_source * n_edges + fragment_target
    fragment_position = torch.searchsorted(pair_keys, fragment_keys)
    fragment_position = fragment_position.clamp_max(pair_keys.numel() - 1)
    if not bool((pair_keys[fragment_position] == fragment_keys).all()):
        raise RuntimeError("sampled tour contains an incompatible edge fragment")
    fragment_weights = weights[:, None].expand(-1, n_nodes)[valid_fragments]
    fragment_best = (costs == costs.min())[:, None].expand(-1, n_nodes)[valid_fragments]
    fragment_count = weights.new_zeros(pair_keys.numel())
    fragment_count.index_add_(0, fragment_position, torch.ones_like(fragment_weights))
    fragment_support = weights.new_zeros(pair_keys.numel())
    fragment_support.index_add_(0, fragment_position, fragment_weights)
    fragment_max = weights.new_zeros(pair_keys.numel())
    fragment_max.scatter_reduce_(
        0, fragment_position, fragment_weights, reduce="amax", include_self=True,
    )
    fragment_best_count = weights.new_zeros(pair_keys.numel())
    fragment_best_count.index_add_(
        0, fragment_position, fragment_best.to(weights.dtype),
    )
    shared_length_a = distances.detach()[edge_u[pair_source], edge_v[pair_source]]
    shared_length_b = distances.detach()[edge_u[pair_target], edge_v[pair_target]]
    outer_a = torch.where(
        edge_u[pair_source] == shared_city, edge_v[pair_source], edge_u[pair_source],
    )
    outer_b = torch.where(
        edge_u[pair_target] == shared_city, edge_v[pair_target], edge_u[pair_target],
    )
    outer_length = distances.detach()[outer_a, outer_b]
    length_ratio = (
        torch.minimum(shared_length_a, shared_length_b)
        / torch.maximum(shared_length_a, shared_length_b).clamp_min(EPS)
    )
    angle_cosine = (
        (shared_length_a.square() + shared_length_b.square() - outer_length.square())
        / (2 * shared_length_a * shared_length_b).clamp_min(EPS)
    ).clamp(-1.0, 1.0)
    relation_features = torch.stack((
        fragment_count / n_solutions,
        fragment_support / weights.sum().clamp_min(EPS),
        fragment_support / weights.sum().clamp_min(EPS) / fragment_count.clamp_min(1),
        fragment_max / weights.sum().clamp_min(EPS),
        (fragment_best_count > 0).to(weights.dtype),
        length_ratio,
        angle_cosine,
    ), dim=-1)
    # Nonobserved compatible relations remain traversable by the GCN.
    relation_weight = 0.05 + (
        fragment_support / weights.mean().clamp_min(EPS)
        * (n_nodes / (2.0 * n_solutions))
    )
    edge_index = torch.stack((
        torch.cat((pair_source, pair_target)),
        torch.cat((pair_target, pair_source)),
    ))
    edge_weight = torch.cat((relation_weight, relation_weight))
    relation_features = torch.cat((relation_features, relation_features), dim=0)

    # Best quartile supplies a quality target distinct from all-tour support.
    elite_count = max(1, (n_solutions + 3) // 4)
    elite = torch.zeros(n_solutions, device=paths.device, dtype=torch.bool)
    elite[torch.topk(costs, k=elite_count, largest=False).indices] = True
    elite_occurrences = elite.repeat_interleave(n_nodes)
    selected_elite_occurrences = valid_occurrence.reshape(-1) & elite_occurrences
    target_support = weights.new_zeros(n_edges)
    target_support.index_add_(
        0, selected_at_occurrence[selected_elite_occurrences],
        occurrence_weights[selected_elite_occurrences],
    )
    target_support = target_support + 0.01 * edge_prior
    quality_target = weights.new_zeros((n_nodes, n_nodes))
    quality_target[edge_u, edge_v] = target_support
    quality_target[edge_v, edge_u] = target_support
    if candidate_only:
        quality_target = quality_target.masked_fill(~candidate, 0.0)
        if not bool((quality_target.sum(dim=-1) > 0).all()):
            raise ValueError("candidate-only local graph leaves a city without support")

    return {
        "edge_features": edge_features,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "relation_features": relation_features,
        "edge_prior": edge_prior,
        "solution_weights": weights,
        "solution_novelty": novelty,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "candidate": candidate if candidate_only else None,
        "quality_target": normalize_heatmap_rows(quality_target),
        "n_nodes": n_nodes,
        "n_solutions": n_solutions,
    }


def rowwise_heatmap_kl(student_heatmap, target_heatmap, support_mask=None):
    """KL(target || student), averaged over normalized TSP decision rows.

    The target should be detached by the caller when it is a bootstrapped
    pseudo-label.  Naming the arguments by their optimization roles avoids the
    otherwise easy-to-miss reversal in ``torch.nn.functional.kl_div``.
    """
    n_nodes = student_heatmap.size(0)
    mask = 1.0 - torch.eye(
        n_nodes, device=student_heatmap.device, dtype=student_heatmap.dtype
    )
    if support_mask is not None:
        mask = mask * support_mask.to(
            device=student_heatmap.device,
            dtype=student_heatmap.dtype,
        )
    target = target_heatmap.clamp_min(EPS) * mask
    student = student_heatmap.clamp_min(EPS) * mask
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return F.kl_div(
        torch.log(student.clamp_min(EPS)),
        target,
        reduction="batchmean",
    )


def refinement_distillation_kl(previous_heatmap, refined_heatmap):
    """Teach H_t to imitate the detached, graph-refined H_(t+1).

    The stop-gradient is essential: this term updates the earlier heatmap
    instead of pulling the refined heatmap back to its input or allowing both
    sides to collapse to the same uninformative distribution.
    """
    # With an explicitly sparse candidate graph, entries outside H0's support
    # are constant epsilon padding and cannot be learned. The default full
    # graph has trainable predictions for every non-self edge.
    support_mask = previous_heatmap.detach() > (10.0 * EPS)
    return rowwise_heatmap_kl(
        previous_heatmap,
        refined_heatmap.detach(),
        support_mask=support_mask,
    )
