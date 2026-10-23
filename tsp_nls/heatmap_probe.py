"""Paired, read-only H0 / graph-prior / H1 sampling during training.

One H0 ACO population is admitted to a separate diagnostic archive per
instance and epoch. All subsequent probes start from its *same* pheromone,
without changing that pheromone or admitting probe tours to the archive.
"""

import json
import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aco import ACO
from solution_sampler import ACOSolutionSampler
from training_pool import create_training_pool


ARMS = ("h0", "prior", "h1")
COMPARISONS = (("h1", "h0"), ("prior", "h0"), ("h1", "prior"))


@contextmanager
def isolated_seed(seed):
    """Keep diagnostic draws from changing training, validation, or CUDA RNG."""
    import random

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _ratio_summary(instance_records, numerator, denominator, metric, seed):
    ratios = np.asarray([
        record["arms"][numerator][metric] / record["arms"][denominator][metric]
        for record in instance_records
    ], dtype=np.float64)
    if not np.isfinite(ratios).all():
        raise ValueError("non-finite paired route-length ratio")
    rng = np.random.default_rng(seed)
    if len(ratios) == 1:
        lower = upper = float(ratios[0])
    else:
        bootstrap = ratios[rng.integers(len(ratios), size=(2000, len(ratios)))].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "mean_ratio": float(ratios.mean()),
        "ci95": [float(lower), float(upper)],
        "wins": int((ratios < 1.0).sum()),
        "ties": int(np.isclose(ratios, 1.0, atol=1e-8).sum()),
        "instances": int(len(ratios)),
    }


class HeatmapProbe:
    def __init__(
        self, *, nodes, count, repeats, seed, output, ants, k_sparse,
        local_search, refinement_mode, max_solution_graph_solutions,
        max_archive_solutions, max_archive_rounds, graph_structure,
        graph_layers, graph_quality_mode, graph_quality_temperature,
        graph_diversity_strength, graph_candidate_only,
    ):
        if count < 1 or repeats < 1:
            raise ValueError("probe instances and repeats must be positive")
        if refinement_mode != "local_gcn":
            raise ValueError("paired graph-prior probe requires local_gcn")
        self.nodes = nodes
        self.repeats = repeats
        self.seed = seed
        self.output = output
        self.ants = ants
        self.k_sparse = k_sparse
        self.local_search = local_search
        self.graph_kwargs = dict(
            max_solutions=max_solution_graph_solutions,
            refinement_mode=refinement_mode,
            return_components=True,
            graph_structure=graph_structure,
            graph_layers=graph_layers,
            graph_quality_mode=graph_quality_mode,
            graph_quality_temperature=graph_quality_temperature,
            graph_diversity_strength=graph_diversity_strength,
            graph_candidate_only=graph_candidate_only,
        )
        self.archive_kwargs = dict(
            archive_max_solutions=max_archive_solutions,
            archive_max_rounds=max_archive_rounds,
        )
        os.makedirs(output, exist_ok=False)
        with isolated_seed(seed):
            self.pool = create_training_pool(count, nodes)
        self.history = []
        self.config = {
            "nodes": nodes, "instances": count, "repeats": repeats,
            "seed": seed, "ants": ants, "k_sparse": k_sparse,
            "local_search": local_search, "sampling_backend": "torch",
            "local_search_inference": False,
            "pheromone_during_probes": "frozen_after_shared_s0",
            "graph": self.graph_kwargs,
            "archive": self.archive_kwargs,
        }
        self._write_json("config.json", self.config)

    def _write_json(self, name, value):
        path = os.path.join(self.output, name)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)

    def _sample_arm(self, heatmap, distances, pheromone, seed):
        # A fresh ACO object makes all three arms independent while retaining
        # the exact same post-S0 pheromone. No update_best/update_pheromone.
        with isolated_seed(seed):
            aco = ACO(
                distances=distances, n_ants=self.ants,
                pheromone=pheromone.clone(), heuristic=heatmap,
                device=distances.device, local_search=self.local_search,
            )
            raw_costs, _, raw_paths = aco.sample(inference=False, require_prob=False)
            if self.local_search is None:
                improved_costs = raw_costs
            else:
                improved_paths = aco.local_search(raw_paths, inference=False)
                improved_costs = aco.gen_path_costs(improved_paths)
        return float(raw_costs.mean()), float(improved_costs.mean())

    @torch.no_grad()
    def evaluate(self, epoch, model):
        was_training = model.training
        model.eval()
        records = []
        try:
            for instance_index, instance in enumerate(self.pool):
                data, distances = instance.materialize(self.k_sparse, model.device)
                heatmap = model.reshape(data, model(data)) + 1e-10
                state = instance.start_visit(heatmap, **self.archive_kwargs)
                with isolated_seed(self.seed + 10000 + instance_index):
                    sampler = ACOSolutionSampler(
                        self.ants, heatmap, distances, model.device,
                        local_search=self.local_search,
                    )
                    s0 = sampler.sample(inference=False, require_log_probs=False)
                state.add_feasible_solutions(s0.feasible_paths, s0.feasible_costs)
                h1, prior, _, _ = model.refine_heatmap(
                    state.initial_heatmap.to(model.device), distances,
                    state.archive, graph_pheromone=sampler.pheromone,
                    **self.graph_kwargs,
                )
                pheromone = sampler.pheromone.detach().clone()
                maps = {"h0": heatmap.detach(), "prior": prior.detach(), "h1": h1.detach()}
                measurements = {arm: {"raw": [], "nls": []} for arm in ARMS}
                for repeat in range(self.repeats):
                    paired_seed = self.seed + 100000 + instance_index * 100 + repeat
                    for arm in ARMS:
                        raw, improved = self._sample_arm(
                            maps[arm], distances, pheromone, paired_seed,
                        )
                        measurements[arm]["raw"].append(raw)
                        measurements[arm]["nls"].append(improved)
                records.append({
                    "instance": instance_index,
                    "archive_rounds": state.archive.num_rounds,
                    "shared_s0_raw_mean": float(s0.sampled_costs.mean()),
                    "shared_s0_improved_mean": float(s0.feasible_costs.mean()),
                    "arms": {
                        arm: {
                            "raw": float(np.mean(measurements[arm]["raw"])),
                            "nls": float(np.mean(measurements[arm]["nls"])),
                        }
                        for arm in ARMS
                    },
                })
                instance.finish_visit()
        finally:
            model.train(was_training)

        summary = {
            "epoch": epoch,
            "arms": {
                arm: {
                    metric: float(np.mean([
                        record["arms"][arm][metric] for record in records
                    ]))
                    for metric in ("raw", "nls")
                }
                for arm in ARMS
            },
            "ratios": {
                f"{numerator}_over_{denominator}": {
                    metric: _ratio_summary(
                        records, numerator, denominator, metric,
                        self.seed + epoch * 10 + metric_index,
                    )
                    for metric_index, metric in enumerate(("raw", "nls"))
                }
                for numerator, denominator in COMPARISONS
            },
        }
        self.history.append(summary)
        self._write_json(f"epoch_{epoch:03d}_instances.json", records)
        self._write_json("summary.json", self.history)
        self.plot()
        ratios = summary["ratios"]
        print(
            f"epoch {epoch} paired probe: "
            + ", ".join(
                f"{name} raw={value['raw']['mean_ratio']:.5f} "
                f"NLS={value['nls']['mean_ratio']:.5f}"
                for name, value in ratios.items()
            ),
            flush=True,
        )
        return summary

    def plot(self):
        epochs = [item["epoch"] for item in self.history]
        for numerator, denominator in COMPARISONS:
            name = f"{numerator}_over_{denominator}"
            figure, axis = plt.subplots(figsize=(8, 4.5))
            for metric, color, label in (
                ("raw", "#1f77b4", "Raw tours"),
                ("nls", "#d62728", "After NLS"),
            ):
                values = [item["ratios"][name][metric] for item in self.history]
                means = [value["mean_ratio"] for value in values]
                lows = [value["ci95"][0] for value in values]
                highs = [value["ci95"][1] for value in values]
                axis.plot(epochs, means, marker="o", color=color, label=label)
                axis.fill_between(epochs, lows, highs, color=color, alpha=0.15)
            axis.axhline(1, color="black", linestyle="--", linewidth=1)
            axis.set(
                title=f"TSP{self.nodes}: {numerator.upper()} / {denominator.upper()}",
                xlabel="Training epoch", ylabel="Paired mean route-length ratio",
            )
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            destination = os.path.join(self.output, f"{name}.png")
            temporary = destination + ".tmp.png"
            figure.savefig(temporary, dpi=160)
            plt.close(figure)
            os.replace(temporary, destination)
