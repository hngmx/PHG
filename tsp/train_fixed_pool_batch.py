"""Train PHG TSP with one fixed pool, revisited in full batches.

The 50-instance pool is sampled once at the start of each independent run.
Every optimizer step revisits all 50 instances, and each instance retains its
own tour archive across passes and epochs.  The original trainer is untouched.
"""

import argparse
import hashlib
import os
import time

import torch

from aco import ACO
from graph_net import GraphNet
from solution_graph import SolutionArchive, refinement_distillation_kl, rowwise_heatmap_kl
from train import EPS, reinforce_cost_loss, validation
from utils import gen_pyg_data, load_val_dataset


def train_pool_pass(model, optimizer, pool, archives, n_ants, k_sparse):
    """Visit every fixed instance once and make one averaged gradient update."""
    model.train()
    optimizer.zero_grad()
    batch_size = len(pool)
    totals = {"reinforce": 0.0, "kl": 0.0, "graph": 0.0}
    for index in torch.randperm(batch_size).tolist():
        coordinates = pool[index]
        data, distances = gen_pyg_data(coordinates, k_sparse=k_sparse)
        h0 = model.reshape(data, model(data)) + EPS
        aco = ACO(
            n_ants=n_ants,
            heuristic=h0,
            distances=distances,
            device=str(coordinates.device),
        )
        costs, feasible_costs, log_probs, paths, feasible_paths = (
            aco.sample_iteration(require_prob=True)
        )
        archive = archives[index]
        archive.add(feasible_paths, feasible_costs)
        h1, graph = model.refine_heatmap(
            h0,
            distances,
            archive,
            quality_mode="relative_diversity",
            quality_temperature=0.1,
            diversity_strength=0.5,
            pheromone=aco.pheromone,
        )
        reinforce = reinforce_cost_loss(costs, log_probs)
        kl = refinement_distillation_kl(h0, h1)
        graph_loss = rowwise_heatmap_kl(h1, graph["quality_target"])
        loss = reinforce + 0.03 * kl + graph_loss
        (loss / batch_size).backward()
        totals["reinforce"] += reinforce.detach().item()
        totals["kl"] += kl.detach().item()
        totals["graph"] += graph_loss.detach().item()
    optimizer.step()
    return {name: value / batch_size for name, value in totals.items()}


def train(args):
    if args.pool_size != args.batch_size:
        raise ValueError("the fixed pool must contain exactly one full batch")
    if min(args.pool_size, args.steps, args.epochs, args.ants) < 1:
        raise ValueError("pool size, steps, epochs, and ants must be positive")
    if not 1 <= args.k_sparse < args.nodes:
        raise ValueError("k_sparse must be between 1 and nodes - 1")
    if os.path.exists(args.output):
        if os.listdir(args.output):
            raise FileExistsError(f"output directory is not empty: {args.output}")
    else:
        os.makedirs(args.output)

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = GraphNet(graph_layers=args.graph_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    val_data = load_val_dataset(args.nodes, args.k_sparse, device)
    if args.validation_size is not None:
        val_data = val_data[:args.validation_size]
    if not val_data:
        raise ValueError("validation dataset cannot be empty")

    pool = torch.rand((args.pool_size, args.nodes, 2), device=device)
    archives = [
        SolutionArchive(
            args.nodes, storage_device="cpu", path_dtype=torch.int32
        )
        for _ in range(args.pool_size)
    ]
    pool_hash = hashlib.sha256(pool.detach().cpu().numpy().tobytes()).hexdigest()
    print(
        f"CONFIG nodes={args.nodes} seed={args.seed} ants={args.ants} "
        f"k_sparse={args.k_sparse} pool_size={args.pool_size} "
        f"batch_size={args.batch_size} steps={args.steps} "
        f"unique_training_instances={args.pool_size} "
        f"optimizer_updates_per_epoch={args.steps} graph_layers={args.graph_layers} "
        f"lr={args.lr} fixed_pool_across_epochs=True persistent_archives=True "
        f"pool_sha256={pool_hash}",
        flush=True,
    )
    best_score = float("inf")
    best_state = None
    start = time.time()
    for epoch in range(args.epochs + 1):
        if epoch:
            totals = {"reinforce": 0.0, "kl": 0.0, "graph": 0.0}
            for step in range(args.steps):
                values = train_pool_pass(
                    model, optimizer, pool, archives, args.ants, args.k_sparse
                )
                for name, value in values.items():
                    totals[name] += value
                print(
                    f"epoch {epoch} fixed_pool_pass {step + 1}/{args.steps}: "
                    f"{args.pool_size} reused instances",
                    flush=True,
                )
            losses = {name: value / args.steps for name, value in totals.items()}
        metrics = validation(model, val_data, args.ants)
        print(f"epoch {epoch}: {metrics}", flush=True)
        if epoch:
            print(f"epoch {epoch} losses: {losses}", flush=True)
            torch.save(model.state_dict(), os.path.join(args.output, f"tsp{args.nodes}-last.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(args.output, f"tsp{args.nodes}-init.pt"))
        if metrics[-1] <= best_score:
            best_score = metrics[-1]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    best_path = os.path.join(args.output, f"tsp{args.nodes}-best.pt")
    temporary_path = f"{best_path}.{os.getpid()}.tmp"
    torch.save(best_state, temporary_path)
    os.replace(temporary_path, best_path)
    print(f"total duration: {time.time() - start:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int)
    parser.add_argument("--ants", type=int, required=True)
    parser.add_argument("--k_sparse", type=int, required=True)
    parser.add_argument("--pool_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--validation_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
