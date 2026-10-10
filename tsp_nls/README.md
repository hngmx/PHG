## PHG-ACO: persistent hypergraph-guided ACO for TSP

This variant treats heatmap construction as an iterative, per-instance search
process. A training instance retains its feasible-solution archive and last H1
across visits, while each visit still runs only one refinement:

```text
problem graph -> H0 -> ACO sampling S0 -> solution hypergraph -> H1
              -> ACO sampling S1 -> stop
```

`graph_rounds` is fixed at one heatmap refinement, while training and
validation each sample two ACO populations: `ACO(H0) -> H1 -> ACO(H1)`.
The H1 population is included in the terminal archive used for training.
Testing is S0-only: it generates H0 once per instance, then runs 10 ACO
iterations on that fixed heatmap by default. Pheromone and the cumulative best
tour carry across these iterations, but testing never constructs H1. It reports
the average cumulative best path length after each ACO iteration. No mode
generates H2.

For a repeated training-pool instance, the current model always recomputes H0
and ACO(H0) still samples S0. The archive from prior visits is reused, including
the previous terminal S1 population. Only H1 construction uses a history prior:

```text
prior = (1 - history_heatmap_weight) * normalize(H0_new)
      + history_heatmap_weight * normalize(stopgrad(H1_previous))
H1_new = refine(prior, inherited_archive + S0_new)
S1_new = ACO(H1_new)
save(inherited_archive + S0_new + S1_new, detach(H1_new))
```

The default history weight is `0.5`. The first visit has no historical H1, so
its refinement prior is simply H0. Both the historical H1 and the new H0 are
detached before forming the refinement prior; the H0 distillation loss still
trains the current initial GNN. Historical heatmaps never replace the fresh H0
used by S0, and no extra H2 is generated.

The initial GNN heatmap is sparse and therefore compresses the candidate
solution space. A sampler is accessed through the interface in
`solution_sampler.py`; ACO sampling is used directly and local search is
disabled by default but available as a matched baseline. A different constrained decoder can return the same
`SolutionBatch` fields.

Distinct feasible ACO tours are retained across visits to the same training
instance. Cyclic rotations and reversed
orientations of the same undirected tour are stored once; observing a duplicate
refreshes its recency instead of multiplying its weight:

- a graph node is one undirected TSP edge observed in a sampled tour;
- a sampled tour is a hyperedge connecting all of its TSP-edge nodes;
- two edge nodes are therefore related by the set of archived tours containing
  both of them, without materialising the equivalent quadratic clique graph.

The path stored in the graph is always paired with its own cost. The archive
validates that every stored TSP tour is a permutation of all nodes.

PHG-ACO first produces a deterministic base heatmap from cumulative
feasible-tour evidence, then applies a bounded residual predicted by a
learnable solution-graph GNN. The combined `H_(t+1)` becomes the next sampler
heatmap.
Only the best-cost `elite_ratio` fraction (default `0.25`) of each newly
sampled population updates H1. Feasible tours are admitted to the archive;
optional quality/recency pruning applies only when a cap is configured.
Elite tours are explicitly quality-weighted:

```text
w_i = softmax(-(q_i - q_best) / (temperature * quality_scale))
```

A scale floor prevents nearly identical ACO tour costs from producing a
numerical one-hot target. A small uniform mixture (default `0.01`) preserves limited
diversity among the elite set. Compact unique
paths remain available for inspection, while the hot path processes only the
new population and incrementally updates sufficient edge and propagation
statistics. Previous statistics receive exponential `age_decay`, avoiding a
full rebuild from every old path at every round. The deterministic update first
aggregates direct edge support and then applies an edge-to-solution-to-edge
pass, so edges are related through sampled solutions containing them. Direct
support, propagated support, inverse-distance
support, and the preceding heatmap are row-normalized before convex mixing.
`--propagation_strength` controls the second pass and
`--distance_prior_strength` controls the geometric prior; both default to
`0.1`. `--quality_prior_strength` defaults to `0.5`, so the final
deterministic update mixes the new graph target and the preceding heatmap at
`1:1`.

By default, each training instance's archive retains all unique tours and all
sampling rounds across its visits,
and the learned updater uses every archived tour. Optional positive limits can
be set for archive tours, archive rounds, and learned-graph tours; pruning
rebuilds the deduplication index and online statistics when needed. Unbounded
history can require substantial CPU and GPU memory on large or long searches.
The learned updater runs
two sparse edge-to-solution-to-edge message-passing layers. Edge nodes use the
current heatmap probability, inverse distance, quality support, and occurrence
frequency. Solution nodes use quality weight, normalized path quality, and
recency. The output is a symmetric residual bounded to `[-0.25, 0.25]`; its
final projection is zero-initialized, so a new updater starts exactly from the
deterministic base heatmap.

Training uses two KL objectives for the single H0 -> H1 refinement:

```text
L_H0 = KL(stopgrad(P_H1) || P_H0)
L_cost = KL(stopgrad(P_elite_edges_from_ACO(H1)) || P_H1)
L = kl_weight * L_H0 + path_cost_kl_weight * L_cost
```

`P_H0` and `P_H1` are row-normalized heatmap probabilities with self-loops
and non-trainable padding outside H0's compressed candidate graph masked.
The first target is detached so `L_H0` is the only objective that updates the
GNN+MLP producing H0.
`L_cost` trains the learned H1 correction against quality-weighted elite edges
from S1, without constructing another heatmap. The same
statistic is reported for
H0 only as a detached diagnostic metric, so it cannot send an extra gradient
to the initial network. With `graph_rounds=1`, the initial-network
path is exactly `H0 -> sample S0 -> build hypergraph -> H1 -> KL(H1 || H0)`.
There is no REINFORCE or entropy loss.

The standard training policy uses a fixed coordinate pool. Every epoch
shuffles the pool and visits every instance exactly once, with no omission,
duplication, or replacement. Therefore, an instance participates exactly as
many times as the number of training epochs. Every visit starts by generating
H0 from the latest model, then inherits only that instance's archive and
previous H1 as described above. No search state is shared between different
instances.
The standard profile uses 800 fixed coordinates, batch size 20, 40 steps per
epoch, 20 epochs, and one graph refinement plus a terminal ACO population.
Each of the 800
instances therefore participates exactly 20 times.
Pool coordinates and cross-visit compressed archive paths are kept on CPU;
only the current optimizer batch is materialized on the training device.
The learned graph updater still uses every archived tour by default, so memory
and per-visit computation can increase over epochs. Fresh validation and test
instances do not have historical archives or H1 heatmaps; this difference
should be considered when interpreting held-out results.

Coordinate-pool overfitting can be measured with three matched policies:

```raw
$ python3 train.py 100 --train_pool_mode fixed
$ python3 train.py 100 --train_pool_mode refresh
$ python3 train.py 100 --train_pool_mode mixed --pool_refresh_fraction 0.5
```

All policies still visit exactly 800 instances per epoch. `fixed` repeats the
same coordinates, `refresh` replaces all coordinates after each epoch, and
`mixed` replaces the requested fraction while retaining the rest. Replaced
instances lose their history; surviving instances keep it.

Testing runs only the first part of the training visit under
`torch.no_grad()`: problem graph, trained initial GNN, H0, and repeated ACO
sampling within S0. It does not create a solution archive, build a hypergraph,
generate H1, calculate KL, or update model parameters. ACO stays on CPU and
H0 is generated on the model device. Within each instance, the same ACO sampler
retains its pheromone and best tour for all iterations; each new instance starts
with a fresh sampler. No NLS or 2-opt improvement is applied
by default; either can be enabled explicitly.
Tour construction uses the seed-controlled PyTorch sampler by default. The
optional `--sampling_backend numba` selects the original inference constructor; it can
be faster at larger scales, but its thread-local random stream is not strictly
reproducible and its thread-pool overhead made TSP50 slower in measurement.
Action log-probabilities are skipped because the S0-only test does not need
them. Training still uses its two KL objectives internally, but no longer
prints or records their per-epoch values.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

TSP100 fine-tuning preset:
```raw
$ python3 train.py 100 --profile tsp100_finetune
```

This explicit profile starts from `../pretrained/tsp_nls/tsp100-best.pt` and
uses `k_sparse=10`, `lr=1e-4`, 3 epochs, 48 ants,
1 graph refinement (2 ACO populations), 2 validation sampling rounds, and a
fixed pool of 800 instances. It writes checkpoints to
`../pretrained/tsp_nls/optimized_v3_k10_finetune`. Individual command-line
flags still override profile values. The profile is deliberately restricted
to TSP100, but the revised pipeline still requires multi-seed validation.

TSP200:
```raw
$ python3 train.py 200
```

TSP500:
```raw
$ python3 train.py 500 --graph_rounds 1 --kl_weight 1.0
```

TSP1000:
```raw
$ python3 train.py 1000 --graph_rounds 1 --kl_weight 1.0
```

`--graph_rounds` must be 1; training samples H0 and H1 once each.
`--train_pool_size` controls the fixed pool size,
defaults to 800, and must equal `--steps * --batch_size` so every pool member
is visited exactly once per epoch.
`--profile standard` retains the general training defaults. `--elite_ratio`
controls graph-update admission.
`--path_cost_kl_weight` controls supervision from actual ACO costs, and
`--max_solution_graph_solutions`, `--max_archive_solutions`, and
`--max_archive_rounds` are optional positive caps. By default none of these
three limits is applied. The first visit starts with an empty archive, while
later visits of the same training instance reuse it.
`--history_heatmap_weight` controls the old H1 share in the normalized prior
used only for H1 construction. Its default is `0.5`; `0` keeps the archive but
uses only the new H0 as the refinement prior. This is distinct from
`--quality_prior_strength`, which mixes the refinement prior with the
hypergraph target in the deterministic update.
`--validation_rounds` must be 2 ACO populations. `--quality_temperature`,
`--age_decay`, and `--uniform_mix` control quality weighting.
`--quality_prior_strength` smooths the quality target with the preceding
heatmap; its default `0.5` gives equal weight to both. `--seed` fixes training
and uses an isolated fixed validation seed, so
every checkpoint is evaluated with the same random stream. Epoch-0 parameters
are saved as `tsp{N}-init.pt`; `tsp{N}-best.pt` is atomically replaced only when
the full run completes. Old DeepACO checkpoints can be supplied with `--pretrained`.
Missing solution-graph parameters are initialized with a zero output layer, so
old DeepACO checkpoints start from deterministic heatmap refinement.

The test entry point deliberately does not expose H1 refinement ablations:
every test instance stops after S0, which contains multiple ACO iterations.
Local search remains disabled by default
and can be enabled explicitly:

```raw
$ python3 test.py 100 --local_search none
$ python3 test.py 100 --local_search nls
```

The same `--local_search none|2opt|nls` option is available in `train.py`, so
checkpoints can be evaluated with matched search settings.

### Testing

When `--model` is omitted, testing loads the root checkpoint
`../pretrained/tsp_nls/tsp{nodes}-best.pt`. Pass `--model` to select a
different checkpoint explicitly; only its initial-heatmap GNN is used by the
S0-only test. Testing defaults to `--seed 1234`, allowing two checkpoints to
be compared with the same sampling randomness. By default, each instance runs
10 ACO iterations on its fixed H0. Output retains the original aggregate
format: total duration followed by `T=1` through `T=10` average cumulative
best path lengths, with no separate result line for each instance. Here `T`
counts ACO iterations within S0, not H0/H1 heatmap refinements. Use
`--iterations N` to change this count.

When training uses a non-default candidate size, pass the same value at test
time, for example:

```raw
$ python3 test.py 100 --k_sparse 10
```

This is an H0/S0 evaluation, not an evaluation of H1 or the learned
solution-graph updater. Training still uses H1 and validates its terminal
S1 cost when selecting the best checkpoint; keep that distinction in mind
when interpreting S0-only test results.

TSP200:
```
$ python3 test.py 200
```

TSP500:
```
$ python3 test.py 500
```

TSP1000:
```
$ python3 test.py 1000
```
