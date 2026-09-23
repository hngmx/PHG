"""Cumulative solution-hypergraph utilities for dynamic DeepACO heatmaps.

The graph requested by the solution-graph variant has one node for every TSP
edge observed in the sampled tours.  The graph relation between two edge nodes
is the set of sampled solutions containing both edges.  Materialising that
clique graph costs O(num_solutions * n_nodes**2), so this module stores the
equivalent hypergraph incidence representation instead:

    edge node <-> sampled solution (hyperedge)

Two edge nodes are related exactly when they share at least one solution
hyperedge.  All distinct sampled solutions from all rounds remain in
``SolutionArchive``; rotation/reversal duplicates refresh recency in place.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


EPS = 1e-10


class SolutionArchive:
    """Keep unique feasible tours, matching costs, and latest sampling rounds."""

    def __init__(
        self,
        n_nodes: int,
        storage_device=None,
        path_dtype=None,
        deduplicate=True,
    ):
        self.n_nodes = n_nodes
        self.storage_device = (
            None if storage_device is None else torch.device(storage_device)
        )
        self.path_dtype = path_dtype
        self.deduplicate = deduplicate
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
        self._tour_locations = {}
        # Each sampling round contributes one compact edge-key population to
        # the online hypergraph accumulator.  Historical paths remain available
        # for inspection, but H1 no longer rebuilds their full incidence graph.
        self._round_edge_keys = []
        self._round_costs = []
        self._online_statistics_cache = {}

    def __len__(self):
        return sum(paths.size(0) for paths in self._paths)

    @property
    def num_rounds(self):
        return self._next_round

    def add(self, paths: torch.Tensor, costs: torch.Tensor):
        """Append one feasible solution population to the archive.

        Tours that differ only by cyclic rotation or reversal represent the
        same undirected TSP solution.  Only one copy is stored; seeing it again
        refreshes its round and keeps its best matching cost.

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
        # Rotate every tour to start at node 0, then choose the lexicographically
        # smaller orientation.  With distinct nodes, comparing the second and
        # final node is sufficient to choose between the two orientations.
        offsets = torch.arange(self.n_nodes, device=tours.device)
        positions = (offsets.unsqueeze(0) + tours.argmin(dim=1, keepdim=True))
        positions = positions.remainder(self.n_nodes)
        tours = tours.gather(1, positions)
        if self.n_nodes > 2:
            reversed_tours = torch.cat(
                (tours[:, :1], torch.flip(tours[:, 1:], dims=(1,))),
                dim=1,
            )
            use_reverse = tours[:, 1] > tours[:, -1]
            tours = torch.where(use_reverse.unsqueeze(1), reversed_tours, tours)

        detached_costs = costs.detach().clone()
        if self.deduplicate:
            key_tours = tours.to(device="cpu", dtype=torch.int32).contiguous()
            batch_keys = {}
            for index, row in enumerate(key_tours):
                batch_keys[row.numpy().tobytes()] = index

            observation_indices = list(batch_keys.values())
            observation_selection = torch.tensor(
                observation_indices,
                device=tours.device,
                dtype=torch.long,
            )
            observation_tours = tours.index_select(0, observation_selection)
            observation_costs = detached_costs.index_select(
                0, observation_selection.to(detached_costs.device)
            )

            new_keys = []
            new_indices = []
            for key, index in batch_keys.items():
                location = self._tour_locations.get(key)
                if location is None:
                    new_keys.append(key)
                    new_indices.append(index)
                    continue

                chunk_index, row_index = location
                stored_cost = self._costs[chunk_index][row_index]
                candidate_cost = detached_costs[index].to(stored_cost.device)
                self._costs[chunk_index][row_index] = torch.minimum(
                    stored_cost,
                    candidate_cost.to(dtype=stored_cost.dtype),
                )
                self._rounds[chunk_index][row_index] = self._next_round

            if not new_indices:
                observation_u = observation_tours
                observation_v = torch.roll(
                    observation_tours, shifts=-1, dims=1
                )
                observation_lo = torch.minimum(observation_u, observation_v)
                observation_hi = torch.maximum(observation_u, observation_v)
                observation_edge_keys = (
                    observation_lo * self.n_nodes + observation_hi
                )
                self._round_edge_keys.append(
                    observation_edge_keys.to(device="cpu", dtype=torch.int32)
                )
                self._round_costs.append(
                    observation_costs.to(device="cpu", dtype=torch.float32)
                )
                self._next_round += 1
                return {"added": 0, "duplicates": paths.size(1)}

            selection = torch.tensor(new_indices, device=tours.device)
            tours = tours.index_select(0, selection)
            detached_costs = detached_costs.index_select(
                0, selection.to(detached_costs.device)
            )
        else:
            new_keys = []
            observation_tours = tours
            observation_costs = detached_costs

        observation_u = observation_tours
        observation_v = torch.roll(observation_tours, shifts=-1, dims=1)
        observation_lo = torch.minimum(observation_u, observation_v)
        observation_hi = torch.maximum(observation_u, observation_v)
        observation_edge_keys = observation_lo * self.n_nodes + observation_hi
        self._round_edge_keys.append(
            observation_edge_keys.to(device="cpu", dtype=torch.int32)
        )
        self._round_costs.append(
            observation_costs.to(device="cpu", dtype=torch.float32)
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
        if self.deduplicate:
            chunk_index = len(self._paths) - 1
            for row_index, key in enumerate(new_keys):
                self._tour_locations[key] = (chunk_index, row_index)
        self._next_round += 1
        return {
            "added": tours.size(0),
            "duplicates": paths.size(1) - tours.size(0),
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

    The state lives for all inner refinement rounds of a training or inference
    episode.  It is intentionally not shared across unrelated TSP instances.
    """

    def __init__(
        self,
        initial_heatmap,
        archive_device=None,
        archive_path_dtype=None,
    ):
        if initial_heatmap.dim() != 2:
            raise ValueError("initial heatmap must be a matrix")
        if initial_heatmap.size(0) != initial_heatmap.size(1):
            raise ValueError("initial heatmap must be square")

        self.initial_heatmap = initial_heatmap
        self.current_heatmap = initial_heatmap
        self.archive = SolutionArchive(
            n_nodes=initial_heatmap.size(0),
            storage_device=archive_device,
            path_dtype=archive_path_dtype,
        )
        self.round_index = 0

    def add_feasible_solutions(self, paths, costs):
        self.archive.add(paths, costs)

    def reanchor(self, model_heatmap, memory_strength=0.5):
        """Fuse the latest model H0 into this instance's persistent heatmap.

        The archive remains untouched.  Re-anchoring closes the training loop:
        parameter improvements can influence the next population instead of
        merely imitating a search trajectory created by an early checkpoint.
        """
        if not 0 <= memory_strength <= 1:
            raise ValueError("memory strength must be between 0 and 1")
        if model_heatmap.shape != self.current_heatmap.shape:
            raise ValueError("model heatmap shape must match the current heatmap")

        model_heatmap = model_heatmap.detach().to(
            device=self.current_heatmap.device,
            dtype=self.current_heatmap.dtype,
        )
        model_heatmap = normalize_heatmap_rows(model_heatmap)
        memory_heatmap = normalize_heatmap_rows(self.current_heatmap.detach())
        self.current_heatmap = normalize_heatmap_rows(
            (1.0 - memory_strength) * model_heatmap
            + memory_strength * memory_heatmap
        )
        return self.current_heatmap

    def advance(self, next_heatmap):
        if next_heatmap.shape != self.current_heatmap.shape:
            raise ValueError("next heatmap shape must match the current heatmap")
        self.current_heatmap = next_heatmap
        self.round_index += 1


def _scatter_mean(values, index, output_size):
    output = values.new_zeros((output_size, values.size(-1)))
    output.index_add_(0, index, values)
    counts = values.new_zeros((output_size, 1))
    counts.index_add_(0, index, values.new_ones((values.size(0), 1)))
    return output / counts.clamp_min(1.0)


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
    # NLS costs from producing a numerically one-hot target.
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
    max_solutions=128,
    temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
):
    """Build the sparse edge-solution graph consumed by the learned updater.

    TSP edges are graph nodes and archived tours are solution nodes.  Incidence
    indices represent the two message-passing directions without materialising
    an edge clique.  Only the highest quality/recency archive entries are kept
    so the neural updater has a bounded memory and runtime footprint.
    """
    if max_solutions < 1:
        raise ValueError("maximum solution-graph size must be positive")
    if distances.shape != previous_heatmap.shape:
        raise ValueError("distance matrix shape must match the heatmap")

    # Rank on the archive's storage device (normally CPU), then transfer only
    # the bounded selected subset to the neural updater's device.
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

    selected_count = min(int(max_solutions), paths.size(0))
    selected = torch.topk(
        all_weights,
        k=selected_count,
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
    prior_strength=0.05,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
):
    """Build H_(t+1) by fixed quality-weighted hypergraph aggregation.

    Direct edge support is first aggregated from all solution hyperedges.  A
    second edge->solution->edge pass propagates context between TSP edges that
    occur in the same archived solutions.  Direct support, propagated support,
    inverse-distance support, and the preceding heatmap are normalized before
    convex mixing.  This function has no trainable parameters, so its output
    can safely serve as a detached KL pseudo-label.
    """
    if not 0 <= prior_strength <= 1:
        raise ValueError("quality prior strength must be between 0 and 1")
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

    previous = normalize_rows(previous_heatmap)
    refined = (1.0 - prior_strength) * graph_target + prior_strength * previous
    return normalize_rows(refined) + EPS * mask


def graph_refined_heatmap(
    previous_heatmap,
    archive: SolutionArchive,
    distances=None,
    temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    prior_strength=0.05,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
):
    """Public name for the deterministic solution-graph heatmap update."""
    return quality_target_heatmap(
        previous_heatmap,
        archive,
        distances=distances,
        temperature=temperature,
        age_decay=age_decay,
        uniform_mix=uniform_mix,
        elite_ratio=elite_ratio,
        prior_strength=prior_strength,
        propagation_strength=propagation_strength,
        distance_prior_strength=distance_prior_strength,
    )


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
    # H0 is produced only on the compressed candidate graph.  Entries outside
    # that support are constant epsilon padding and cannot be learned, so KL is
    # evaluated only where H0 has trainable candidate-edge predictions.
    support_mask = previous_heatmap.detach() > (10.0 * EPS)
    return rowwise_heatmap_kl(
        previous_heatmap,
        refined_heatmap.detach(),
        support_mask=support_mask,
    )


def future_solution_quality_kl(predicted_heatmap, future_target_heatmap):
    """Train a graph updater to anticipate the detached final archive target."""
    return rowwise_heatmap_kl(
        predicted_heatmap,
        future_target_heatmap.detach(),
    )
