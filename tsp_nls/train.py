import argparse
from contextlib import contextmanager
import os
import random
import time

import numpy as np
import torch

from net import Net, REFINEMENT_MODES
from solution_graph import (
    InstanceSearchState,
    population_quality_kl,
    refinement_distillation_kl,
)
from solution_sampler import ACOSolutionSampler
from training_pool import (
    create_training_pool,
    iter_pool_batches,
    refresh_training_pool,
)
from utils import load_val_dataset


EPS = 1e-10
DEFAULT_GRAPH_ROUNDS = 1
VALIDATION_SAMPLING_ROUNDS = 2
PRETRAINED_DIR = "../pretrained/tsp_nls"
TSP100_FINETUNE_PROFILE = "tsp100_finetune"


def sampling_rounds_for_refinements(graph_rounds):
    """The H0 -> H1 visit samples once before and once after refinement."""
    if graph_rounds != DEFAULT_GRAPH_ROUNDS:
        raise ValueError("training supports exactly one H0 -> H1 refinement")
    return VALIDATION_SAMPLING_ROUNDS


def pool_refresh_fraction_for_mode(mode, mixed_fraction=0.5):
    """Resolve fixed, fully refreshed, and mixed training-pool policies."""
    if not 0 <= mixed_fraction <= 1:
        raise ValueError("mixed pool refresh fraction must be between 0 and 1")
    if mode == "fixed":
        return 0.0
    if mode == "refresh":
        return 1.0
    if mode == "mixed":
        return mixed_fraction
    raise ValueError(f"unknown training pool mode: {mode}")


def resolve_training_profile(
    nodes,
    profile="standard",
    *,
    lr=None,
    epochs=None,
    k_sparse=None,
    train_pool_size=None,
    pretrained=None,
    output=None,
):
    """Resolve reproducible CLI defaults without hiding user overrides."""
    if profile not in {"standard", TSP100_FINETUNE_PROFILE}:
        raise ValueError(f"unknown training profile: {profile}")

    if profile == TSP100_FINETUNE_PROFILE:
        if nodes != 100:
            raise ValueError(
                f"{TSP100_FINETUNE_PROFILE} is restricted to TSP100"
            )
        defaults = {
            "lr": 1e-4,
            "epochs": 3,
            "k_sparse": 10,
            "train_pool_size": 800,
            "pretrained": os.path.join(PRETRAINED_DIR, "tsp100-best.pt"),
            "output": os.path.join(
                PRETRAINED_DIR, "optimized_v3_k10_finetune"
            ),
        }
    else:
        defaults = {
            "lr": 3e-4,
            "epochs": 20,
            "k_sparse": None,
            "train_pool_size": 800,
            "pretrained": None,
            "output": PRETRAINED_DIR,
        }

    supplied = {
        "lr": lr,
        "epochs": epochs,
        "k_sparse": k_sparse,
        "train_pool_size": train_pool_size,
        "pretrained": pretrained,
        "output": output,
    }
    return {
        name: defaults[name] if value is None else value
        for name, value in supplied.items()
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_random_seed(seed):
    """Run deterministic validation without changing the training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def train_instance(
    model,
    optimizer,
    data,
    n_ants,
    k_sparse,
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    kl_weight=1.0,
    path_cost_kl_weight=1.0,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.5,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    history_heatmap_weight=0.5,
    refinement_mode="full",
    local_search=None,
    sampler_factory=ACOSolutionSampler,
):
    """Train H0 distillation and the learnable solution-graph correction."""
    model.train()
    sum_loss = 0.0
    count = 0

    for training_instance in data:
        pyg_data, distances = training_instance.materialize(k_sparse, device)
        heu_vec = model(pyg_data)
        initial_heatmap = model.reshape(pyg_data, heu_vec) + EPS
        state = training_instance.start_visit(
            initial_heatmap,
            archive_max_solutions=max_archive_solutions,
            archive_max_rounds=max_archive_rounds,
            history_heatmap_weight=history_heatmap_weight,
        )

        sampler = sampler_factory(
            n_solutions=n_ants,
            heatmap=state.current_heatmap.to(device),
            distances=distances,
            device=device,
            local_search=local_search,
        )

        # ``graph_rounds`` counts heatmap refinements.  Sampling once more
        # ensures the terminal H_graph_rounds is evaluated by a real ACO
        # population instead of existing only as an unsampled KL prediction.
        for sampling_round in range(
            sampling_rounds_for_refinements(graph_rounds)
        ):
            sampler.set_heatmap(state.current_heatmap.to(device))
            solutions = sampler.sample(
                inference=False,
                require_log_probs=False,
            )

            # Only admit a sampled path together with its own objective value
            # so edge-level quality credit remains consistent.
            state.add_feasible_solutions(
                solutions.feasible_paths,
                solutions.feasible_costs,
            )
            if sampling_round == graph_rounds:
                path_cost_loss = population_quality_kl(
                    refined_heatmap,
                    solutions.feasible_paths,
                    solutions.feasible_costs,
                    temperature=quality_temperature,
                    elite_ratio=elite_ratio,
                    uniform_mix=uniform_mix,
                )
                break

            refined_heatmap, _, _, _ = model.refine_heatmap(
                state.refinement_prior_heatmap.to(model.device),
                distances,
                state.archive,
                quality_temperature=quality_temperature,
                age_decay=age_decay,
                uniform_mix=uniform_mix,
                elite_ratio=elite_ratio,
                prior_strength=quality_prior_strength,
                propagation_strength=propagation_strength,
                distance_prior_strength=distance_prior_strength,
                max_solutions=max_solution_graph_solutions,
                refinement_mode=refinement_mode,
                return_components=True,
            )
            # H1 supervises the trainable H0.
            # The complete teacher, including the learned residual, is detached
            # by the helper so this loss updates only the initial GNN.
            kl_loss = refinement_distillation_kl(
                initial_heatmap, refined_heatmap
            )
            # Inputs to the graph updater are detached search state. Retaining
            # H1 lets S1's actual costs supervise the updater without sending
            # gradients into H0 or the discrete search.
            # The search state outlives this optimizer step.  Keeping it on
            # CPU and detached prevents stale autograd graphs and persistent
            # GPU growth while preserving the instance's heatmap trajectory.
            state.advance(refined_heatmap.detach().cpu())

        sum_loss += kl_weight * kl_loss + path_cost_kl_weight * path_cost_loss
        count += 1
        training_instance.finish_visit()

    sum_loss = sum_loss / count
    optimizer.zero_grad()
    sum_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(), max_norm=3.0, norm_type=2
    )
    optimizer.step()


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    distances,
    n_ants,
    graph_rounds=VALIDATION_SAMPLING_ROUNDS,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.5,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    refinement_mode="full",
    local_search=None,
    sampler_factory=ACOSolutionSampler,
):
    """Solve one instance while retaining its heatmap and graph state."""
    if graph_rounds != VALIDATION_SAMPLING_ROUNDS:
        raise ValueError("validation must sample H0 and H1 exactly once each")
    model.eval()
    heu_vec = model(pyg_data)
    initial_heatmap = model.reshape(pyg_data, heu_vec) + EPS
    state = InstanceSearchState(
        initial_heatmap,
        archive_max_solutions=max_archive_solutions,
        archive_max_rounds=max_archive_rounds,
    )

    sampler = sampler_factory(
        n_solutions=n_ants,
        heatmap=state.current_heatmap.cpu(),
        distances=distances.cpu(),
        device="cpu",
        local_search=local_search,
    )

    baseline = None
    best_sample_cost = None
    best_aco_1 = None
    for round_idx in range(graph_rounds):
        sampler.set_heatmap(state.current_heatmap.cpu())
        solutions = sampler.sample(
            inference=False,
            require_log_probs=False,
        )

        if round_idx == 0:
            baseline = solutions.sampled_costs.mean()
            best_sample_cost = solutions.sampled_costs.min()
            best_aco_1 = sampler.best_cost

        state.add_feasible_solutions(
            solutions.feasible_paths,
            solutions.feasible_costs,
        )
        if round_idx + 1 < graph_rounds:
            next_heatmap = model.refine_heatmap(
                state.current_heatmap.to(model.device),
                distances.to(model.device),
                state.archive,
                quality_temperature=quality_temperature,
                age_decay=age_decay,
                uniform_mix=uniform_mix,
                elite_ratio=elite_ratio,
                prior_strength=quality_prior_strength,
                propagation_strength=propagation_strength,
                distance_prior_strength=distance_prior_strength,
                max_solutions=max_solution_graph_solutions,
                refinement_mode=refinement_mode,
            )
            state.advance(next_heatmap.detach().cpu())

    best_aco_t = sampler.best_cost
    return np.array(
        [baseline.item(), best_sample_cost.item(), best_aco_1, best_aco_t]
    )


def train_epoch(
    n_ants,
    k_sparse,
    steps_per_epoch,
    net,
    optimizer,
    training_pool,
    batch_size=1,
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    kl_weight=1.0,
    path_cost_kl_weight=1.0,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.5,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    history_heatmap_weight=0.5,
    refinement_mode="full",
    local_search=None,
):
    for training_batch in iter_pool_batches(
        training_pool,
        steps=steps_per_epoch,
        batch_size=batch_size,
    ):
        train_instance(
            net,
            optimizer,
            training_batch,
            n_ants,
            k_sparse,
            graph_rounds=graph_rounds,
            kl_weight=kl_weight,
            path_cost_kl_weight=path_cost_kl_weight,
            quality_temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            quality_prior_strength=quality_prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            history_heatmap_weight=history_heatmap_weight,
            refinement_mode=refinement_mode,
            local_search=local_search,
        )


@torch.no_grad()
def validation(
    n_ants,
    net,
    val_dataset,
    graph_rounds=VALIDATION_SAMPLING_ROUNDS,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.5,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    refinement_mode="full",
    local_search=None,
    validation_seed=12345,
):
    stats = []
    with isolated_random_seed(validation_seed):
        for data, distances in val_dataset:
            stats.append(
                infer_instance(
                    net,
                    data,
                    distances,
                    n_ants,
                    graph_rounds=graph_rounds,
                    quality_temperature=quality_temperature,
                    age_decay=age_decay,
                    uniform_mix=uniform_mix,
                    elite_ratio=elite_ratio,
                    quality_prior_strength=quality_prior_strength,
                    propagation_strength=propagation_strength,
                    distance_prior_strength=distance_prior_strength,
                    max_solution_graph_solutions=max_solution_graph_solutions,
                    max_archive_solutions=max_archive_solutions,
                    max_archive_rounds=max_archive_rounds,
                    refinement_mode=refinement_mode,
                    local_search=local_search,
                )
            )
    return [value.item() for value in np.stack(stats).mean(0)]


def train(
    n_node,
    n_ants,
    steps_per_epoch,
    epochs,
    k_sparse=None,
    batch_size=20,
    test_size=None,
    pretrained=None,
    savepath="../pretrained/tsp_nls",
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    validation_rounds=VALIDATION_SAMPLING_ROUNDS,
    kl_weight=1.0,
    path_cost_kl_weight=1.0,
    quality_temperature=0.75,
    age_decay=0.1,
    uniform_mix=0.01,
    elite_ratio=0.25,
    quality_prior_strength=0.5,
    propagation_strength=0.1,
    distance_prior_strength=0.1,
    train_pool_size=800,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    history_heatmap_weight=0.5,
    refinement_mode="full",
    local_search=None,
    train_pool_mode="fixed",
    pool_refresh_fraction=0.5,
    seed=1234,
):
    seed_everything(seed)
    sampling_rounds_for_refinements(graph_rounds)
    if validation_rounds != VALIDATION_SAMPLING_ROUNDS:
        raise ValueError("validation must sample H0 and H1 exactly once each")
    if refinement_mode == "h0":
        raise ValueError("h0 refinement mode is evaluation-only")
    if refinement_mode not in REFINEMENT_MODES:
        raise ValueError(f"unknown refinement mode: {refinement_mode}")
    if not 0 <= history_heatmap_weight <= 1:
        raise ValueError("history heatmap weight must be between 0 and 1")
    if (
        max_solution_graph_solutions is not None
        and max_archive_solutions is not None
        and max_solution_graph_solutions > max_archive_solutions
    ):
        raise ValueError(
            "learned graph solution limit cannot exceed archive capacity"
        )
    k_sparse = n_node // 10 if k_sparse is None else k_sparse
    required_pool_size = steps_per_epoch * batch_size
    if train_pool_size is None:
        train_pool_size = required_pool_size
    if train_pool_size != required_pool_size:
        raise ValueError(
            "training pool size must equal steps_per_epoch * batch_size "
            f"({train_pool_size} != {steps_per_epoch} * {batch_size})"
        )

    os.makedirs(savepath, exist_ok=True)
    net = Net().to(device)
    if pretrained:
        incompatible = net.load_state_dict(
            torch.load(pretrained, map_location=device), strict=False
        )
        if incompatible.missing_keys:
            print("initialized new parameters:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print(
                "ignored obsolete trainable graph-updater parameters:",
                incompatible.unexpected_keys,
            )

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    training_pool = create_training_pool(train_pool_size, n_node)
    resolved_refresh_fraction = pool_refresh_fraction_for_mode(
        train_pool_mode,
        mixed_fraction=pool_refresh_fraction,
    )
    print(
        "training coordinate pool:",
        f"mode={train_pool_mode}, size={len(training_pool)}, "
        f"refresh_fraction={resolved_refresh_fraction}; every current pool "
        "member is visited once per epoch; surviving members retain their "
        "solution archive and H1",
    )
    val_list = load_val_dataset(n_node, k_sparse, device, start_node=0)
    if test_size is not None:
        val_list = val_list[:test_size]

    stats = validation(
        n_ants,
        net,
        val_list,
        graph_rounds=validation_rounds,
        quality_temperature=quality_temperature,
        age_decay=age_decay,
        uniform_mix=uniform_mix,
        elite_ratio=elite_ratio,
        quality_prior_strength=quality_prior_strength,
        propagation_strength=propagation_strength,
        distance_prior_strength=distance_prior_strength,
        max_solution_graph_solutions=max_solution_graph_solutions,
        max_archive_solutions=max_archive_solutions,
        max_archive_rounds=max_archive_rounds,
        refinement_mode=refinement_mode,
        local_search=local_search,
        validation_seed=seed + 100000,
    )
    val_results = [stats]
    best_result = (stats[-1], stats[-2], stats[-3])
    best_path = os.path.join(savepath, f"tsp{n_node}-best.pt")
    last_path = os.path.join(savepath, f"tsp{n_node}-last.pt")
    init_path = os.path.join(savepath, f"tsp{n_node}-init.pt")
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in net.state_dict().items()
    }
    best_epoch = 0
    # Save initialization under an explicit name.  The public best checkpoint
    # is replaced atomically only after the whole run finishes, so an aborted
    # run cannot overwrite a previously trained best model with epoch 0.
    torch.save(best_state, init_path)
    print("epoch 0:", stats)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        if epoch > 1:
            refreshed = refresh_training_pool(
                training_pool,
                resolved_refresh_fraction,
            )
            if refreshed:
                print(f"refreshed training-pool coordinates: {refreshed}")
        start = time.time()
        train_epoch(
            n_ants,
            k_sparse,
            steps_per_epoch,
            net,
            optimizer,
            training_pool,
            batch_size=batch_size,
            graph_rounds=graph_rounds,
            kl_weight=kl_weight,
            path_cost_kl_weight=path_cost_kl_weight,
            quality_temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            quality_prior_strength=quality_prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            history_heatmap_weight=history_heatmap_weight,
            refinement_mode=refinement_mode,
            local_search=local_search,
        )
        sum_time += time.time() - start
        stats = validation(
            n_ants,
            net,
            val_list,
            graph_rounds=validation_rounds,
            quality_temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            quality_prior_strength=quality_prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            refinement_mode=refinement_mode,
            local_search=local_search,
            validation_seed=seed + 100000,
        )
        print(f"epoch {epoch}:", stats)
        val_results.append(stats)
        scheduler.step()
        curr_result = (stats[-1], stats[-2], stats[-3])
        if curr_result <= best_result:
            best_result = curr_result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
            best_epoch = epoch
        torch.save(net.state_dict(), last_path)

    temporary_best_path = f"{best_path}.{os.getpid()}.tmp"
    torch.save(best_state, temporary_best_path)
    os.replace(temporary_best_path, best_path)
    print("\ntotal training duration:", sum_time)
    print("best validation epoch:", best_epoch)
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar="N", type=int, help="Problem scale")
    parser.add_argument(
        "--profile",
        choices=("standard", TSP100_FINETUNE_PROFILE),
        default="standard",
        help=(
            "Training defaults. tsp100_finetune uses k=10 and a low learning "
            "rate with the 800-instance pool; explicit flags override it"
        ),
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=None,
        help="Learning rate (profile default: 3e-4 or 1e-4 for TSP100 fine-tuning)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="The device used to train neural networks",
    )
    parser.add_argument("-p", "--pretrained", type=str, default=None,
                        help="Path to a pretrained DeepACO model")
    parser.add_argument("-a", "--ants", type=int, default=48,
                        help="Number of ants (matches test.py default)")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                        help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=40,
                        help="Steps per epoch (default: 40 with batch size 20)")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=None,
        help="Epochs to run (profile default: 20 or 3 for TSP100 fine-tuning)",
    )
    parser.add_argument(
        "--k_sparse",
        type=int,
        default=None,
        help="Candidate neighbors per node (TSP100 fine-tuning default: 10)",
    )
    parser.add_argument(
        "--train_pool_size",
        type=int,
        default=None,
        help=(
            "Fixed pool size; must equal steps * batch_size (default: 800)"
        ),
    )
    parser.add_argument(
        "--train_pool_mode",
        choices=("fixed", "refresh", "mixed"),
        default="fixed",
        help="Reuse coordinates, refresh all each epoch, or refresh a fraction",
    )
    parser.add_argument(
        "--pool_refresh_fraction",
        type=float,
        default=0.5,
        help="Fraction replaced per epoch when --train_pool_mode mixed",
    )
    parser.add_argument(
        "--history_heatmap_weight",
        type=float,
        default=0.5,
        help=(
            "Previous H1 share in the normalized refinement prior on a "
            "revisit; S0 still samples from the fresh H0 (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Training seed; validation uses a fixed isolated derived seed",
    )
    parser.add_argument(
        "-r",
        "--graph_rounds",
        type=int,
        default=DEFAULT_GRAPH_ROUNDS,
        help="Fixed at 1: each training visit refines H0 to H1 once",
    )
    parser.add_argument(
        "--validation_rounds",
        type=int,
        default=VALIDATION_SAMPLING_ROUNDS,
        help="Fixed at 2 ACO populations: one from H0 and one from H1",
    )
    parser.add_argument(
        "--kl_weight",
        "--distill_kl_weight",
        dest="kl_weight",
        type=float,
        default=1.0,
        help="Weight of KL(detached refined heatmap || H0)",
    )
    parser.add_argument(
        "--path_cost_kl_weight",
        type=float,
        default=1.0,
        help="Weight of KL(actual cost-weighted ACO edges || sampling heatmap)",
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
        default=0.5,
        help=(
            "Previous-heatmap share in the deterministic update; default "
            "0.5 gives a 1:1 mix with the new graph target"
        ),
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
        "--max_solution_graph_solutions",
        type=int,
        default=None,
        help="Optional learned-graph tour cap; default keeps all archived tours",
    )
    parser.add_argument(
        "--max_archive_solutions",
        type=int,
        default=None,
        help="Optional cap on unique tours retained by each instance archive",
    )
    parser.add_argument(
        "--max_archive_rounds",
        type=int,
        default=None,
        help="Optional cap on ACO populations retained by each instance archive",
    )
    parser.add_argument(
        "--refinement_mode",
        choices=REFINEMENT_MODES,
        default="full",
        help=(
            "Ablation mode: deterministic uses fixed aggregation, learned "
            "uses only the residual over the current heatmap, and full uses both"
        ),
    )
    parser.add_argument(
        "--local_search",
        choices=("none", "2opt", "nls"),
        default="none",
        help="Local-search baseline applied equally in training and validation",
    )
    parser.add_argument("-t", "--test_size", type=int, default=None,
                        help="Number of instances used for validation")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Directory used to store checkpoints (selected by the profile)",
    )
    opt = parser.parse_args()

    try:
        resolved = resolve_training_profile(
            opt.nodes,
            opt.profile,
            lr=opt.lr,
            epochs=opt.epochs,
            k_sparse=opt.k_sparse,
            train_pool_size=opt.train_pool_size,
            pretrained=opt.pretrained,
            output=opt.output,
        )
    except ValueError as exc:
        parser.error(str(exc))

    opt.lr = resolved["lr"]
    opt.epochs = resolved["epochs"]
    opt.k_sparse = resolved["k_sparse"]
    opt.train_pool_size = resolved["train_pool_size"]
    opt.pretrained = resolved["pretrained"]
    opt.output = resolved["output"]
    opt.local_search = None if opt.local_search == "none" else opt.local_search

    if opt.graph_rounds != DEFAULT_GRAPH_ROUNDS:
        parser.error("--graph_rounds must be 1 (H0 -> H1 only)")
    if opt.validation_rounds != VALIDATION_SAMPLING_ROUNDS:
        parser.error("--validation_rounds must be 2 (sample H0 and H1)")
    required_pool_size = opt.steps * opt.batch_size
    if opt.train_pool_size != required_pool_size:
        parser.error(
            "--train_pool_size must equal --steps * --batch_size "
            f"({required_pool_size}) so every instance is visited once per epoch"
        )
    if not 0 <= opt.pool_refresh_fraction <= 1:
        parser.error("--pool_refresh_fraction must be between 0 and 1")
    if not 0 <= opt.history_heatmap_weight <= 1:
        parser.error("--history_heatmap_weight must be between 0 and 1")
    if opt.k_sparse is not None and not 1 <= opt.k_sparse < opt.nodes:
        parser.error("--k_sparse must be in [1, nodes - 1]")
    if opt.kl_weight < 0:
        parser.error("--kl_weight must be non-negative")
    if opt.path_cost_kl_weight < 0:
        parser.error("--path_cost_kl_weight must be non-negative")
    if opt.quality_temperature <= 0:
        parser.error("--quality_temperature must be positive")
    if opt.age_decay < 0:
        parser.error("--age_decay must be non-negative")
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
    if opt.max_solution_graph_solutions is not None and opt.max_solution_graph_solutions < 1:
        parser.error("--max_solution_graph_solutions must be positive")
    if opt.max_archive_solutions is not None and opt.max_archive_solutions < 1:
        parser.error("--max_archive_solutions must be positive")
    if opt.max_archive_rounds is not None and opt.max_archive_rounds < 1:
        parser.error("--max_archive_rounds must be positive")
    if (
        opt.max_solution_graph_solutions is not None
        and opt.max_archive_solutions is not None
        and opt.max_solution_graph_solutions > opt.max_archive_solutions
    ):
        parser.error(
            "--max_solution_graph_solutions cannot exceed "
            "--max_archive_solutions"
        )
    if opt.refinement_mode == "h0":
        parser.error(
            "--refinement_mode h0 cannot be trained because it has no "
            "refined H1 target"
        )

    if opt.pretrained is not None and not os.path.isfile(opt.pretrained):
        parser.error(f"pretrained checkpoint not found: {opt.pretrained}")

    os.makedirs(opt.output, exist_ok=True)
    lr = opt.lr
    device = opt.device

    print(
        "training configuration:",
        {
            "profile": opt.profile,
            "nodes": opt.nodes,
            "k_sparse": opt.k_sparse or opt.nodes // 10,
            "lr": opt.lr,
            "epochs": opt.epochs,
            "ants": opt.ants,
            "graph_rounds": opt.graph_rounds,
            "validation_rounds": opt.validation_rounds,
            "train_pool_size": opt.train_pool_size,
            "train_pool_mode": opt.train_pool_mode,
            "pool_refresh_fraction": (
                opt.pool_refresh_fraction
                if opt.train_pool_mode == "mixed"
                else pool_refresh_fraction_for_mode(opt.train_pool_mode)
            ),
            "history_heatmap_weight": opt.history_heatmap_weight,
            "max_solution_graph_solutions": opt.max_solution_graph_solutions,
            "max_archive_solutions": opt.max_archive_solutions,
            "max_archive_rounds": opt.max_archive_rounds,
            "refinement_mode": opt.refinement_mode,
            "local_search": opt.local_search or "none",
            "pretrained": opt.pretrained,
            "output": opt.output,
        },
    )

    train(
        opt.nodes,
        opt.ants,
        opt.steps,
        opt.epochs,
        k_sparse=opt.k_sparse,
        batch_size=opt.batch_size,
        test_size=opt.test_size,
        pretrained=opt.pretrained,
        savepath=opt.output,
        graph_rounds=opt.graph_rounds,
        validation_rounds=opt.validation_rounds,
        kl_weight=opt.kl_weight,
        path_cost_kl_weight=opt.path_cost_kl_weight,
        quality_temperature=opt.quality_temperature,
        age_decay=opt.age_decay,
        uniform_mix=opt.uniform_mix,
        elite_ratio=opt.elite_ratio,
        quality_prior_strength=opt.quality_prior_strength,
        propagation_strength=opt.propagation_strength,
        distance_prior_strength=opt.distance_prior_strength,
        train_pool_size=opt.train_pool_size,
        max_solution_graph_solutions=opt.max_solution_graph_solutions,
        max_archive_solutions=opt.max_archive_solutions,
        max_archive_rounds=opt.max_archive_rounds,
        refinement_mode=opt.refinement_mode,
        local_search=opt.local_search,
        train_pool_mode=opt.train_pool_mode,
        pool_refresh_fraction=opt.pool_refresh_fraction,
        history_heatmap_weight=opt.history_heatmap_weight,
        seed=opt.seed,
    )
