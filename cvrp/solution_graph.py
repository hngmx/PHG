"""Persistent CVRP routes and a local graph of compatible directed edges.

Depot revisits are real route transitions.  Only the trailing depot padding
introduced by batched ACO sampling is removed from archived solutions.
"""

import torch
from torch.nn import functional as F


EPS = 1e-10


def normalize_heatmap_rows(heatmap):
    mask = ~torch.eye(heatmap.size(0), dtype=torch.bool, device=heatmap.device)
    scores = heatmap.masked_fill(~mask, 0).clamp_min(0)
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(EPS)


def rowwise_heatmap_kl(student, target):
    """KL(target || student) over outgoing, non-self transitions."""
    student = normalize_heatmap_rows(student)
    target = normalize_heatmap_rows(target).detach()
    return F.kl_div(torch.log(student.clamp_min(EPS)), target, reduction="batchmean")


class SolutionArchive:
    """Keep valid routes for one instance, with optional history limits."""

    def __init__(self, demands, capacity=50, max_solutions=None, max_rounds=None):
        self.demands = demands.detach().cpu().clone()
        self.n_nodes = self.demands.numel()
        self.capacity = capacity
        if max_solutions is not None and max_solutions < 1:
            raise ValueError("max_solutions must be positive")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.max_solutions = max_solutions
        self.max_rounds = max_rounds
        self.max_length = 2 * self.n_nodes + 1
        self.paths = torch.empty((0, self.max_length), dtype=torch.int32)
        self.lengths = torch.empty(0, dtype=torch.long)
        self.costs = torch.empty(0, dtype=torch.float32)
        self.rounds = torch.empty(0, dtype=torch.long)
        self.num_rounds = 0

    def __len__(self):
        return self.costs.numel()

    def add(self, paths, costs):
        if paths.ndim != 2 or costs.shape != (paths.size(1),):
            raise ValueError("expected paths [steps, ants] and costs [ants]")
        if not torch.isfinite(costs).all():
            raise ValueError("all route costs must be finite")
        paths = paths.detach().cpu().long()
        rows = torch.zeros((paths.size(1), self.max_length), dtype=torch.int32)
        lengths = torch.empty(paths.size(1), dtype=torch.long)
        expected = torch.arange(1, self.n_nodes)
        for ant in range(paths.size(1)):
            raw = paths[:, ant]
            customers = torch.nonzero(raw != 0, as_tuple=False).flatten()
            if customers.numel() != self.n_nodes - 1:
                raise ValueError("each customer must appear exactly once")
            length = int(customers[-1]) + 2
            route = raw[:length]
            if length > self.max_length or route[0] != 0 or route[-1] != 0:
                raise ValueError("CVRP route must start and end at the depot")
            if not torch.equal(torch.sort(route[route != 0]).values, expected):
                raise ValueError("invalid CVRP customer permutation")
            load = 0
            for node in route[1:]:
                if node == 0:
                    if load == 0:
                        raise ValueError("empty vehicle route or depot padding")
                    load = 0
                else:
                    load += int(self.demands[node])
                    if load > self.capacity:
                        raise ValueError("vehicle capacity exceeded")
            rows[ant, :length] = route.to(torch.int32)
            lengths[ant] = length
        self.paths = torch.cat((self.paths, rows))
        self.lengths = torch.cat((self.lengths, lengths))
        self.costs = torch.cat((self.costs, costs.detach().cpu().float()))
        self.rounds = torch.cat((
            self.rounds,
            torch.full((rows.size(0),), self.num_rounds, dtype=torch.long),
        ))
        self.num_rounds += 1
        keep = torch.arange(len(self))
        if self.max_rounds is not None:
            keep = keep[self.rounds[keep] >= self.num_rounds - self.max_rounds]
        if self.max_solutions is not None and keep.numel() > self.max_solutions:
            best = torch.topk(self.costs[keep], self.max_solutions, largest=False).indices
            keep = keep[best]
        if keep.numel() != len(self):
            self.paths = self.paths[keep]
            self.lengths = self.lengths[keep]
            self.costs = self.costs[keep]
            self.rounds = self.rounds[keep]

    def tensors(self, device):
        if not len(self):
            raise ValueError("cannot build a graph from an empty archive")
        return (self.paths.to(device=device, dtype=torch.long),
                self.lengths.to(device), self.costs.to(device))


def _solution_weights(costs, keys, route_ids, mode, temperature, diversity):
    if mode == "inverse_cost":
        weights = costs.clamp_min(EPS).reciprocal()
    else:
        scale = (costs.mean() * temperature).clamp_min(EPS)
        weights = torch.exp(-(costs - costs.min()) / scale).clamp_min(EPS)
    if mode == "relative_diversity" and diversity:
        _, inverse, counts = torch.unique(
            keys, sorted=True, return_inverse=True, return_counts=True,
        )
        rarity = counts[inverse].to(costs.dtype).rsqrt()
        route_rarity = torch.zeros_like(costs)
        route_rarity.index_add_(0, route_ids, rarity)
        route_counts = torch.bincount(route_ids, minlength=costs.numel())
        route_rarity /= route_counts.clamp_min(1)
        relative = route_rarity / route_rarity.mean().clamp_min(EPS)
        weights *= (1 + diversity * (relative - 1)).clamp(0.5, 1.5)
    return weights


def build_local_edge_graph(
    heatmap, distances, demands, archive, pheromone=None, incident_topk=6,
    max_solutions=None, quality_mode="relative_diversity",
    quality_temperature=0.1, diversity_strength=0.5,
):
    """Directed CVRP analogue of the TSP local solution-edge graph."""
    n = heatmap.size(0)
    if heatmap.shape != (n, n) or distances.shape != (n, n):
        raise ValueError("heatmap and distances must be square and aligned")
    if incident_topk < 1 or quality_temperature <= 0:
        raise ValueError("invalid graph selection or weight temperature")
    if quality_mode not in {"inverse_cost", "relative", "relative_diversity"}:
        raise ValueError("unknown quality mode")
    if not 0 <= diversity_strength <= 1:
        raise ValueError("diversity strength must be in [0, 1]")
    paths, lengths, costs = archive.tensors(heatmap.device)
    if max_solutions is not None and costs.numel() > max_solutions:
        if max_solutions < 1:
            raise ValueError("max_solutions must be positive")
        selected = torch.topk(costs, max_solutions, largest=False).indices
        paths, lengths, costs = paths[selected], lengths[selected], costs[selected]
    m, width = paths.shape
    positions = torch.arange(width - 1, device=paths.device)[None, :]
    valid_edges = positions < (lengths - 1)[:, None]
    route_ids = torch.arange(m, device=paths.device)[:, None].expand_as(valid_edges)[valid_edges]
    keys = (paths[:, :-1] * n + paths[:, 1:])[valid_edges]
    weights = _solution_weights(
        costs, keys, route_ids, quality_mode, quality_temperature,
        diversity_strength,
    )
    observed, observed_ids = torch.unique(keys, sorted=True, return_inverse=True)
    support = weights.new_zeros(observed.numel())
    support.index_add_(0, observed_ids, weights[route_ids])
    frequency = weights.new_zeros(observed.numel())
    frequency.index_add_(0, observed_ids, torch.ones_like(route_ids, dtype=weights.dtype))

    q = heatmap.detach().clamp_min(0)
    if pheromone is not None:
        q = q * pheromone.detach().to(q.device).clamp_min(0)
    q = q.clone()
    q.fill_diagonal_(-torch.inf)
    topk = min(incident_topk, n - 1)
    initial = torch.topk(q, topk, dim=1).indices
    candidate_keys = (
        torch.arange(n, device=q.device)[:, None] * n + initial
    ).flatten()
    merged = torch.unique(torch.cat((candidate_keys, observed)), sorted=True)
    observed_pos = torch.searchsorted(observed, merged).clamp_max(observed.numel() - 1)
    merged_support = torch.where(
        observed[observed_pos] == merged,
        support[observed_pos], torch.zeros_like(merged, dtype=weights.dtype),
    )
    scores = q.new_full((n, n), -torch.inf)
    scores[merged // n, merged % n] = (
        q[merged // n, merged % n] + merged_support / weights.sum().clamp_min(EPS)
    )
    retained = torch.topk(scores, topk, dim=1).indices
    edge_keys = torch.unique(
        (torch.arange(n, device=q.device)[:, None] * n + retained).flatten(),
        sorted=True,
    )
    edge_u, edge_v = edge_keys // n, edge_keys % n
    observed_pos = torch.searchsorted(observed, edge_keys).clamp_max(observed.numel() - 1)
    is_observed = observed[observed_pos] == edge_keys
    edge_support = torch.where(
        is_observed, support[observed_pos], torch.zeros_like(edge_keys, dtype=weights.dtype),
    )
    edge_counts = torch.where(
        is_observed, frequency[observed_pos],
        torch.zeros_like(edge_keys, dtype=weights.dtype),
    )
    inverse_distance = distances.detach()[edge_u, edge_v].clamp_min(EPS).reciprocal()
    inverse_distance /= inverse_distance.max().clamp_min(EPS)
    prior = edge_support + 0.01 * weights.mean() * inverse_distance
    node_demands = demands.to(device=q.device, dtype=q.dtype)
    edge_features = torch.stack((
        edge_support / edge_support.max().clamp_min(EPS),
        edge_counts / m,
        inverse_distance,
        node_demands[edge_v] / archive.capacity,
        ((edge_u == 0) | (edge_v == 0)).to(q.dtype),
    ), dim=1)

    # Each incoming edge (a,b) can precede each outgoing edge (b,c).
    sources, targets = [], []
    for city in range(n):
        incoming = torch.nonzero(edge_v == city, as_tuple=False).flatten()
        outgoing = torch.nonzero(edge_u == city, as_tuple=False).flatten()
        if incoming.numel() and outgoing.numel():
            sources.append(incoming.repeat_interleave(outgoing.numel()))
            targets.append(outgoing.repeat(incoming.numel()))
    if sources:
        source, target = torch.cat(sources), torch.cat(targets)
    else:
        source = target = torch.empty(0, dtype=torch.long, device=q.device)
    pair_keys = source * edge_keys.numel() + target
    order = torch.argsort(pair_keys)
    pair_keys, source, target = pair_keys[order], source[order], target[order]
    pair_count = pair_keys.numel()

    valid_fragments = positions[:, :-1] < (lengths - 2)[:, None]
    first_keys = paths[:, :-2] * n + paths[:, 1:-1]
    second_keys = paths[:, 1:-1] * n + paths[:, 2:]
    first_pos = torch.searchsorted(edge_keys, first_keys).clamp_max(edge_keys.numel() - 1)
    second_pos = torch.searchsorted(edge_keys, second_keys).clamp_max(edge_keys.numel() - 1)
    valid_fragments &= (edge_keys[first_pos] == first_keys) & (edge_keys[second_pos] == second_keys)
    fragment_pairs = (first_pos * edge_keys.numel() + second_pos)[valid_fragments]
    fragment_routes = torch.arange(m, device=q.device)[:, None].expand_as(valid_fragments)[valid_fragments]
    fragment_count = weights.new_zeros(pair_count)
    fragment_support = weights.new_zeros(pair_count)
    fragment_max = weights.new_zeros(pair_count)
    fragment_best = weights.new_zeros(pair_count)
    if fragment_pairs.numel():
        pair_pos = torch.searchsorted(pair_keys, fragment_pairs).clamp_max(pair_count - 1)
        if not torch.equal(pair_keys[pair_pos], fragment_pairs):
            raise RuntimeError("archived fragment is missing its compatible graph relation")
        fragment_count.index_add_(0, pair_pos, torch.ones_like(pair_pos, dtype=q.dtype))
        fragment_support.index_add_(0, pair_pos, weights[fragment_routes])
        fragment_max.scatter_reduce_(
            0, pair_pos, weights[fragment_routes], reduce="amax", include_self=True,
        )
        fragment_best.index_add_(
            0, pair_pos, (costs[fragment_routes] == costs.min()).to(q.dtype),
        )
    length_a = distances.detach()[edge_u[source], edge_v[source]]
    length_b = distances.detach()[edge_u[target], edge_v[target]]
    outer_length = distances.detach()[edge_u[source], edge_v[target]]
    length_ratio = torch.minimum(length_a, length_b) / torch.maximum(length_a, length_b).clamp_min(EPS)
    angle_cosine = ((length_a.square() + length_b.square() - outer_length.square()) /
                    (2 * length_a * length_b).clamp_min(EPS)).clamp(-1, 1)
    normalized_support = fragment_support / weights.sum().clamp_min(EPS)
    relation_features = torch.stack((
        fragment_count / m,
        normalized_support,
        normalized_support / fragment_count.clamp_min(1),
        fragment_max / weights.sum().clamp_min(EPS),
        (fragment_best > 0).to(q.dtype),
        length_ratio,
        angle_cosine,
    ), dim=1)
    relation_weight = 0.05 + (
        fragment_support / weights.mean().clamp_min(EPS) *
        (n / (2.0 * m))
    )

    elite_count = max(1, (m + 3) // 4)
    elite = torch.zeros(m, dtype=torch.bool, device=q.device)
    elite[torch.topk(costs, elite_count, largest=False).indices] = True
    chosen_pos = torch.searchsorted(edge_keys, keys).clamp_max(edge_keys.numel() - 1)
    chosen_mask = elite[route_ids] & (edge_keys[chosen_pos] == keys)
    target_support = weights.new_zeros(edge_keys.numel())
    target_support.index_add_(0, chosen_pos[chosen_mask], weights[route_ids[chosen_mask]])
    quality_target = q.new_zeros((n, n))
    quality_target[edge_u, edge_v] = target_support + 0.01 * prior
    # A compatible route fragment has a direction, while the encoder passes
    # evidence both ways between the two transition nodes.
    message_index = torch.stack((
        torch.cat((source, target)), torch.cat((target, source)),
    ))
    return {
        "edge_features": edge_features,
        "edge_index": message_index,
        "edge_weight": torch.cat((relation_weight, relation_weight)),
        "relation_features": torch.cat((relation_features, relation_features)),
        "edge_prior": prior,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "quality_target": normalize_heatmap_rows(quality_target),
        "n_nodes": n,
        "n_solutions": m,
    }
