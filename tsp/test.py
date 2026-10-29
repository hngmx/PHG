"""Test PHG-trained TSP H0 with the original notebook's ACO checkpoints."""

import argparse
import os
import time

import torch

from aco import ACO
from graph_net import GraphNet
from train import DEFAULT_OUTPUT, EPS, original_profile
from utils import load_test_dataset


ORIGINAL_TEST_CHECKPOINTS = (1, 10, 20, 30, 40, 50, 100)


@torch.no_grad()
def infer_instance(model, data, distances, n_ants, checkpoints):
    model.eval()
    h0 = model.reshape(data, model(data)) + EPS
    aco = ACO(
        n_ants=n_ants,
        heuristic=h0,
        distances=distances,
        device=str(distances.device),
    )
    results = []
    previous = 0
    for checkpoint in checkpoints:
        results.append(float(aco.run(checkpoint - previous)))
        previous = checkpoint
    return results


@torch.no_grad()
def test(dataset, model, n_ants, checkpoints):
    totals = torch.zeros(len(checkpoints), dtype=torch.float64)
    start = time.time()
    for data, distances in dataset:
        totals += torch.tensor(
            infer_instance(model, data, distances, n_ants, checkpoints),
            dtype=torch.float64,
        )
    return (totals / len(dataset)).tolist(), time.time() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ants", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(ORIGINAL_TEST_CHECKPOINTS))
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--graph_layers", type=int, default=2)
    options = parser.parse_args()
    if not options.checkpoints or any(
        checkpoint < 1 for checkpoint in options.checkpoints
    ) or sorted(set(options.checkpoints)) != options.checkpoints:
        parser.error("--checkpoints must be strictly increasing positive integers")
    default_ants, default_k = original_profile(options.nodes)
    n_ants = default_ants if options.ants is None else options.ants
    k_sparse = default_k if options.k_sparse is None else options.k_sparse
    model_path = options.model or os.path.join(
        DEFAULT_OUTPUT, f"tsp{options.nodes}-best.pt"
    )
    if not os.path.isfile(model_path):
        parser.error(f"model checkpoint not found: {model_path}")
    torch.manual_seed(options.seed)
    device = torch.device(
        options.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    dataset = load_test_dataset(options.nodes, k_sparse, device)
    if options.test_size is not None:
        dataset = dataset[:options.test_size]
    if not dataset:
        parser.error("test dataset cannot be empty")
    model = GraphNet(graph_layers=options.graph_layers).to(device)
    incompatible = model.load_state_dict(
        torch.load(model_path, map_location=device), strict=False,
    )
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("local_edge_gcn.")
    ]
    if missing or incompatible.unexpected_keys:
        parser.error(
            "incompatible checkpoint: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    averages, duration = test(dataset, model, n_ants, options.checkpoints)
    print(f"problem scale: {options.nodes}")
    print(f"checkpoint: {model_path}")
    print(f"number of instances: {len(dataset)}")
    print(f"total duration: {duration}")
    for checkpoint, average in zip(options.checkpoints, averages):
        print(f"T={checkpoint}, average cost is {average}.")
