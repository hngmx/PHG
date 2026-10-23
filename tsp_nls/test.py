"""Evaluate repeated ACO sampling on a fixed initial heatmap per instance."""

import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from aco import ACO
from net import Net
from utils import default_k_sparse, load_test_dataset


EPS = 1e-10
device = "cuda:0" if torch.cuda.is_available() else "cpu"
PRETRAINED_DIR = "../pretrained/tsp_nls"


def default_checkpoint_candidates(nodes):
    """Return the root best-checkpoint path."""
    return [os.path.join(PRETRAINED_DIR, f"tsp{nodes}-best.pt")]


def resolve_checkpoint_path(nodes, requested=None):
    """Honor --model, otherwise use the root best checkpoint."""
    if requested is not None:
        return requested
    return default_checkpoint_candidates(nodes)[0]


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    distances,
    n_ants,
    n_iterations=10,
    local_search=None,
    sampling_backend="numba",
    sampler_factory=None,
    return_training_metrics=False,
):
    """Run the original DeepACO ACO search on one H0."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    if return_training_metrics and n_iterations < 5:
        raise ValueError("DeepACO training metrics require at least 5 ACO iterations")
    model.eval()
    heatmap = (model.reshape(pyg_data, model(pyg_data)) + EPS).cpu()
    if sampler_factory is None:
        aco = ACO(
            n_ants=n_ants,
            heuristic=heatmap,
            distances=distances.cpu(),
            device="cpu",
            local_search=local_search,
        )
    else:
        # Allow controlled sampler substitutes for experiments and unit tests.
        sampler = sampler_factory(
            n_solutions=n_ants,
            heatmap=heatmap,
            distances=distances.cpu(),
            device="cpu",
            local_search=local_search,
        )
    best_costs = []
    training_metrics = []
    if return_training_metrics:
        # The original DeepACO validation first probes the sampler without
        # changing pheromone or the cumulative best, then runs ACO for 1+4
        # iterations. Keep the checkpoint curve at 10 actual ACO iterations.
        if sampler_factory is None:
            probe_costs = aco.sample(
                inference=sampling_backend == "numba", require_prob=False,
            )[0]
        else:
            probe_costs = sampler.probe(inference=sampling_backend == "numba")
        training_metrics.extend((float(probe_costs.mean()), float(probe_costs.min())))
    for iteration in range(n_iterations):
        if sampler_factory is None:
            best_cost = aco.run(1, inference=sampling_backend == "numba")
        else:
            sampler.sample(
                inference=sampling_backend == "numba",
                require_log_probs=False,
                local_search_inference=False if local_search is not None else None,
            )
            best_cost = sampler.best_cost
        best_costs.append(float(best_cost))
        if return_training_metrics and iteration in (0, 4):
            training_metrics.append(float(best_cost))
    if return_training_metrics:
        return best_costs, training_metrics
    return best_costs


@torch.no_grad()
def test(
    dataset,
    model,
    n_ants,
    n_iterations=10,
    local_search=None,
    sampling_backend="numba",
):
    """Return mean cumulative best costs across instances, one per ACO round."""
    if not dataset:
        raise ValueError("test dataset cannot be empty")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    total_costs = np.zeros(n_iterations, dtype=np.float64)
    start = time.time()
    for pyg_data, distances in tqdm(dataset):
        best_costs = infer_instance(
            model,
            pyg_data,
            distances,
            n_ants,
            n_iterations=n_iterations,
            local_search=local_search,
            sampling_backend=sampling_backend,
        )
        total_costs += best_costs
    return (total_costs / len(dataset)).tolist(), time.time() - start


def main(
    n_node,
    model_file,
    k_sparse=None,
    n_ants=48,
    n_iterations=10,
    local_search=None,
    sampling_backend="numba",
    test_size=None,
):
    k_sparse = default_k_sparse(n_node) if k_sparse is None else k_sparse
    test_list = load_test_dataset(n_node, k_sparse, device, start_node=0)
    if test_size is not None:
        test_list = test_list[:test_size]
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of instances:", len(test_list))
    print("device:", "cpu" if device == "cpu" else device + "+cpu")

    net_tsp = Net().to(device)
    incompatible = net_tsp.load_state_dict(
        torch.load(model_file, map_location=device), strict=False
    )
    essential_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("solution_graph_net.")
        and not key.startswith("local_edge_gcn.")
    ]
    if essential_missing:
        raise RuntimeError(
            "checkpoint is missing initial-heatmap parameters: "
            + ", ".join(essential_missing)
        )
    if incompatible.unexpected_keys:
        print("ignored unused checkpoint parameters:", incompatible.unexpected_keys)

    average_costs, duration = test(
        test_list,
        net_tsp,
        n_ants,
        n_iterations=n_iterations,
        local_search=local_search,
        sampling_backend=sampling_backend,
    )
    print("total duration:", duration)
    for iteration, average_cost in enumerate(average_costs, start=1):
        print(f"T={iteration}, average cost is {average_cost}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="Checkpoint path; defaults to ../pretrained/tsp_nls/tsp{nodes}-best.pt",
    )
    parser.add_argument("-a", "--ants", type=int, default=48,
                        help="Number of ants sampled at each ACO iteration")
    parser.add_argument(
        "-i", "--iterations", type=int, default=10,
        help="ACO iterations per instance on fixed H0 (default: 10)",
    )
    parser.add_argument(
        "--k_sparse", type=int, default=None,
        help="Candidate neighbors per node (default: max(1, nodes // 10)); must match training",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed for reproducible ACO sampling",
    )
    parser.add_argument(
        "--sampling_backend", choices=("numba", "torch"), default="numba",
        help="Tour-construction backend (default: original DeepACO Numba inference)",
    )
    parser.add_argument(
        "--local_search", choices=("none", "2opt", "nls"), default="nls",
        help="Local search used after ACO sampling (default: nls)",
    )
    parser.add_argument(
        "--test_size", type=int, default=None,
        help="Evaluate only the first N instances (default: full test set)",
    )
    opt = parser.parse_args()

    if opt.nodes < 2:
        parser.error("nodes must be at least 2")
    if opt.ants < 1:
        parser.error("--ants must be positive")
    if opt.iterations < 1:
        parser.error("--iterations must be positive")
    if opt.k_sparse is not None and not 1 <= opt.k_sparse < opt.nodes:
        parser.error("--k_sparse must be in [1, nodes - 1]")
    if opt.test_size is not None and opt.test_size < 1:
        parser.error("--test_size must be positive")
    opt.local_search = None if opt.local_search == "none" else opt.local_search

    filepath = resolve_checkpoint_path(opt.nodes, opt.model)
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found!")
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
        n_iterations=opt.iterations,
        local_search=opt.local_search,
        sampling_backend=opt.sampling_backend,
        test_size=opt.test_size,
    )
