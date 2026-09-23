import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from net import Net
from solution_graph import InstanceSearchState
from solution_sampler import ACOSolutionSampler
from utils import load_test_dataset


EPS = 1e-10
device = "cuda:0" if torch.cuda.is_available() else "cpu"
PRETRAINED_DIR = "../pretrained/tsp_nls"


def default_checkpoint_candidates(nodes):
    """Return the original root best-checkpoint path."""
    return [os.path.join(PRETRAINED_DIR, f"tsp{nodes}-best.pt")]


def resolve_checkpoint_path(nodes, requested=None):
    """Honor --model, otherwise use the original root best checkpoint."""
    if requested is not None:
        return requested
    return default_checkpoint_candidates(nodes)[0]


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    distances,
    n_ants,
    evaluation_rounds,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.05,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=128,
    sampling_backend="torch",
    sampler_factory=ACOSolutionSampler,
):
    """Keep one instance's graph and heatmap through all search rounds."""
    model.eval()
    # ACO remains CPU-resident. H0 and the learned solution-graph residual run
    # on the model device; persistent search state returns to CPU between
    # rounds. Local search is deliberately disabled.
    initial_heatmap = (model.reshape(pyg_data, model(pyg_data)) + EPS).cpu()
    distances_cpu = distances.cpu()
    state = InstanceSearchState(initial_heatmap)
    sampler = sampler_factory(
        n_solutions=n_ants,
        heatmap=state.current_heatmap,
        distances=distances_cpu,
        device="cpu",
        local_search=None,
    )

    requested = set(evaluation_rounds)
    results = {}
    max_round = max(evaluation_rounds)
    for round_idx in range(1, max_round + 1):
        sampler.set_heatmap(state.current_heatmap)
        solutions = sampler.sample(
            inference=sampling_backend == "numba",
            require_log_probs=False,
        )
        state.add_feasible_solutions(
            solutions.feasible_paths,
            solutions.feasible_costs,
        )

        if round_idx in requested:
            results[round_idx] = sampler.best_cost

        if round_idx < max_round:
            next_heatmap = model.refine_heatmap(
                state.current_heatmap.to(model.device),
                distances_cpu.to(model.device),
                state.archive,
                quality_temperature=quality_temperature,
                age_decay=age_decay,
                uniform_mix=uniform_mix,
                elite_ratio=elite_ratio,
                prior_strength=quality_prior_strength,
                propagation_strength=propagation_strength,
                distance_prior_strength=distance_prior_strength,
                max_solutions=max_solution_graph_solutions,
            )
            state.advance(next_heatmap.detach().cpu())

    return torch.tensor(
        [results[round_idx] for round_idx in evaluation_rounds],
        dtype=torch.float32,
    )


@torch.no_grad()
def test(
    dataset,
    model,
    n_ants,
    evaluation_rounds,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.05,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=128,
    sampling_backend="torch",
):
    sum_results = torch.zeros(size=(len(evaluation_rounds),))
    start = time.time()
    for pyg_data, distances in tqdm(dataset):
        sum_results += infer_instance(
            model,
            pyg_data,
            distances,
            n_ants,
            evaluation_rounds,
            quality_temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            quality_prior_strength=quality_prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
            max_solution_graph_solutions=max_solution_graph_solutions,
            sampling_backend=sampling_backend,
        )
    duration = time.time() - start
    return sum_results / len(dataset), duration


def main(
    n_node,
    model_file,
    k_sparse=None,
    n_ants=48,
    evaluation_rounds=None,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.05,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=128,
    sampling_backend="torch",
    test_size=None,
):
    k_sparse = k_sparse or n_node // 10
    evaluation_rounds = evaluation_rounds or list(range(1, 11))
    test_list = load_test_dataset(
        n_node, k_sparse, device, start_node=0
    )
    if test_size is not None:
        test_list = test_list[:test_size]
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of instances:", len(test_list))
    print("device:", "cpu" if device == "cpu" else device + "+cpu")
    print("sampling backend:", sampling_backend)

    net_tsp = Net().to(device)
    incompatible = net_tsp.load_state_dict(
        torch.load(model_file, map_location=device), strict=False
    )
    if incompatible.unexpected_keys:
        print(
            "ignored obsolete trainable graph-updater parameters; "
            "the current updater uses the PHG-ACO residual architecture"
        )
    if incompatible.missing_keys:
        print(
            "initialized missing PHG-ACO graph-updater parameters; "
            "use a checkpoint trained on this branch for learned residuals"
        )

    avg_aco_best, duration = test(
        test_list,
        net_tsp,
        n_ants,
        evaluation_rounds,
        quality_temperature=quality_temperature,
        age_decay=age_decay,
        uniform_mix=uniform_mix,
        elite_ratio=elite_ratio,
        quality_prior_strength=quality_prior_strength,
        propagation_strength=propagation_strength,
        distance_prior_strength=distance_prior_strength,
        max_solution_graph_solutions=max_solution_graph_solutions,
        sampling_backend=sampling_backend,
    )
    print("total duration:", duration)
    for round_idx, average_cost in zip(evaluation_rounds, avg_aco_best):
        print(f"T={round_idx}, average cost is {average_cost}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file; defaults to the original root "
            "../pretrained/tsp_nls/tsp{nodes}-best.pt"
        ),
    )
    parser.add_argument("-a", "--ants", type=int, default=48,
                        help="Number of ants")
    parser.add_argument(
        "--k_sparse",
        type=int,
        default=None,
        help="Candidate neighbors per node; must match training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for reproducible checkpoint comparison",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="ACO/solution-graph rounds at which results are reported",
    )
    parser.add_argument(
        "--quality_temperature",
        type=float,
        default=0.75,
        help="Softmax temperature used to weight archived solution quality",
    )
    parser.add_argument(
        "--age_decay",
        type=float,
        default=0.1,
        help="Exponential log-weight decay per archived round",
    )
    parser.add_argument(
        "--uniform_mix",
        type=float,
        default=0.01,
        help="Uniform mixture preserving influence from diverse solutions",
    )
    parser.add_argument(
        "--elite_ratio",
        type=float,
        default=0.25,
        help="Best-cost fraction of each population used to update H1",
    )
    parser.add_argument(
        "--quality_prior_strength",
        type=float,
        default=0.05,
        help="Previous-heatmap prior used in the graph-refined heatmap",
    )
    parser.add_argument(
        "--propagation_strength",
        type=float,
        default=0.1,
        help="Edge-solution-edge aggregation strength used to construct H1",
    )
    parser.add_argument(
        "--distance_prior_strength",
        type=float,
        default=0.1,
        help="Normalized inverse-distance mixture used to construct H1",
    )
    parser.add_argument(
        "--sampling_backend",
        choices=("numba", "torch"),
        default="torch",
        help=(
            "Tour-construction backend. 'torch' is reproducible and faster "
            "for TSP50 on the tested machine; 'numba' can help at larger scales"
        ),
    )
    parser.add_argument(
        "--max_solution_graph_solutions",
        type=int,
        default=128,
        help="Maximum number of quality-ranked archive tours used by the learned graph updater",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Evaluate only the first N instances (default: full test set)",
    )
    opt = parser.parse_args()

    if not opt.iterations or min(opt.iterations) < 1:
        parser.error("all --iterations values must be positive")
    if sorted(set(opt.iterations)) != opt.iterations:
        parser.error("--iterations must be unique and sorted")
    if opt.quality_temperature <= 0:
        parser.error("--quality_temperature must be positive")
    if opt.age_decay < 0:
        parser.error("--age_decay must be non-negative")
    if opt.k_sparse is not None and not 1 <= opt.k_sparse < opt.nodes:
        parser.error("--k_sparse must be in [1, nodes - 1]")
    if not 0 <= opt.uniform_mix <= 1:
        parser.error("--uniform_mix must be between 0 and 1")
    if not 0 < opt.elite_ratio <= 1:
        parser.error("--elite_ratio must be in (0, 1]")
    if not 0 <= opt.quality_prior_strength <= 1:
        parser.error("--quality_prior_strength must be between 0 and 1")
    if not 0 <= opt.propagation_strength <= 1:
        parser.error("--propagation_strength must be between 0 and 1")
    if not 0 <= opt.distance_prior_strength <= 1:
        parser.error("--distance_prior_strength must be between 0 and 1")
    if opt.test_size is not None and opt.test_size < 1:
        parser.error("--test_size must be positive")
    if opt.max_solution_graph_solutions < 1:
        parser.error("--max_solution_graph_solutions must be positive")

    filepath = resolve_checkpoint_path(opt.nodes, opt.model)
    if not os.path.isfile(filepath):
        searched = ", ".join(default_checkpoint_candidates(opt.nodes))
        print(f"Checkpoint file '{filepath}' not found! Searched: {searched}")
        raise SystemExit(1)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    main(
        opt.nodes,
        filepath,
        k_sparse=opt.k_sparse,
        n_ants=opt.ants,
        evaluation_rounds=opt.iterations,
        quality_temperature=opt.quality_temperature,
        age_decay=opt.age_decay,
        uniform_mix=opt.uniform_mix,
        elite_ratio=opt.elite_ratio,
        quality_prior_strength=opt.quality_prior_strength,
        propagation_strength=opt.propagation_strength,
        distance_prior_strength=opt.distance_prior_strength,
        max_solution_graph_solutions=opt.max_solution_graph_solutions,
        sampling_backend=opt.sampling_backend,
        test_size=opt.test_size,
    )
