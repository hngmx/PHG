"""Evaluate a trained CVRP H0 with ordinary DeepACO ACO search.

The PHG graph and archives are training-only.  Optionally compare a released
DeepACO checkpoint on the same instances, seeds, ants, and iteration budget.
"""

import argparse
import csv
from pathlib import Path

import torch

from aco import ACO
from graph_net import GraphNet
from net import Net as OriginalNet
from utils import gen_instance, gen_pyg_data


EPS = 1e-10
ROOT = Path(__file__).resolve().parents[1]


def load_instances(n_customers, count, seed, dataset_path=None):
    source = Path(dataset_path) if dataset_path else (
        ROOT / "data" / "cvrp" / f"testDataset-{n_customers}.pt"
    )
    if source.exists():
        dataset = torch.load(source, map_location="cpu", weights_only=True)
        if count > len(dataset):
            raise ValueError(f"requested {count} instances but {source} has {len(dataset)}")
        return [(item[0].clone(), item[1:].clone()) for item in dataset[:count]]
    if dataset_path:
        raise FileNotFoundError(source)
    with torch.random.fork_rng(devices=[]):
        # Keep generated test instances separate from train.py's validation pool.
        torch.manual_seed(seed + 300000)
        return [gen_instance(n_customers, "cpu") for _ in range(count)]


@torch.no_grad()
def infer_instance(model, demands, distances, n_ants, iterations, device):
    model.eval()
    demands = demands.to(device)
    distances = distances.to(device)
    pyg = gen_pyg_data(demands, distances, device)
    h0 = model(pyg).reshape_as(distances) + EPS
    aco = ACO(
        distances=distances, demand=demands, n_ants=n_ants,
        heuristic=h0, device=str(device),
    )
    return [float(aco.run(1)) for _ in range(iterations)]


def evaluate(model, instances, n_ants, iterations, device, seed):
    totals = torch.zeros(iterations, dtype=torch.float64)
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    for index, (demands, distances) in enumerate(instances):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + index)
            totals += torch.tensor(
                infer_instance(model, demands, distances, n_ants, iterations, device),
                dtype=torch.float64,
            )
    return (totals / len(instances)).tolist()


def load_model(path, device, graph):
    model = (GraphNet() if graph else OriginalNet()).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def save_results(path, phg, baseline=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "cvrp_iterations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "PHG_H0"] + (["DeepACO"] if baseline else []))
        for index, cost in enumerate(phg):
            writer.writerow([index + 1, cost] + ([baseline[index]] if baseline else []))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = list(range(1, len(phg) + 1))
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, phg, marker="o", label="PHG-trained H0")
    if baseline:
        plt.plot(x, baseline, marker="s", label="DeepACO")
    plt.xlabel("ACO iteration")
    plt.ylabel("Mean best CVRP route length")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "cvrp_iterations.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes", type=int, help="number of customers, excluding depot")
    parser.add_argument("--model", default=None, help="PHG-trained best checkpoint")
    parser.add_argument("--baseline_model", default=None,
                        help="optional original DeepACO checkpoint")
    parser.add_argument("--ants", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None,
                        help="optional directory for CSV and PNG trajectory")
    args = parser.parse_args()
    if min(args.nodes, args.iterations, args.test_size) < 1:
        parser.error("nodes, iterations, and test_size must be positive")
    device = torch.device(args.device)
    ants = args.ants or (20 if args.nodes == 20 else 50)
    model_path = args.model or (
        ROOT / "pretrained" / "cvrp" / "phg_current" / f"cvrp{args.nodes}-best.pt"
    )
    instances = load_instances(args.nodes, args.test_size, args.seed, args.dataset)
    phg = evaluate(
        load_model(model_path, device, graph=True), instances,
        ants, args.iterations, device, args.seed,
    )
    baseline = None
    if args.baseline_model:
        baseline = evaluate(
            load_model(args.baseline_model, device, graph=False), instances,
            ants, args.iterations, device, args.seed,
        )
    for index, cost in enumerate(phg):
        line = f"iteration {index + 1}: PHG_H0={cost:.6f}"
        if baseline is not None:
            line += f" DeepACO={baseline[index]:.6f}"
        print(line)
    if args.output:
        save_results(args.output, phg, baseline)


if __name__ == "__main__":
    main()
