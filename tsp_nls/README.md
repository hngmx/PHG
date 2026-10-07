## PHG-ACO: persistent hypergraph-guided ACO for TSP

This variant treats heatmap construction as an iterative, per-instance search
process. An instance owns its current heatmap, feasible-solution archive, and
cumulative solution hypergraph during one search visit:

```text
problem graph -> H0 -> constrained sampling -> solution hypergraph -> H1
              -> constrained sampling -> larger hypergraph -> H2 -> ...
```

`graph_rounds` counts heatmap refinements, not ACO populations. With the
default three refinements, training samples four populations so every heatmap
is evaluated: `ACO(H0) -> H1 -> ACO(H1) -> H2 -> ACO(H2) -> H3 -> ACO(H3)`.
The final H3 population is included in the terminal archive used for training.

The initial GNN heatmap is sparse and therefore compresses the candidate
solution space. A sampler is accessed through the interface in
`solution_sampler.py`; ACO sampling is used directly and local search is
disabled by default but available as a matched baseline. A different constrained decoder can return the same
`SolutionBatch` fields.

Distinct feasible ACO tours are retained in the bounded visit-local solution
hypergraph. Cyclic rotations and reversed
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
sampled population updates H1. Feasible tours are admitted before bounded
quality/recency pruning. Elite tours are explicitly quality-weighted:

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
`0.1`.

The full visit archive is hard-capped at 256 unique tours and 10 raw sampling
rounds. Older raw rounds are discarded and the deduplication index plus online
statistics are rebuilt after pruning, so the underlying archive cannot grow
without bound. The learned updater selects at most 128 quality-ranked tours
from that bounded archive and runs
two sparse edge-to-solution-to-edge message-passing layers. Edge nodes use the
current heatmap probability, inverse distance, quality support, and occurrence
frequency. Solution nodes use quality weight, normalized path quality, and
recency. The output is a symmetric residual bounded to `[-0.25, 0.25]`; its
final projection is zero-initialized, so a new updater starts exactly from the
deterministic base heatmap.

Training uses three KL objectives:

```text
L_H0 = weighted_mean_t KL(stopgrad(P_(t+1)) || P_H0)
L_future = weighted_mean_t KL(stopgrad(P_final_archive) || P_(t+1))
L_cost = mean_t KL(stopgrad(P_elite_edges_from_actual_ACO_t) || P_t)
L = kl_weight * L_H0 + future_kl_weight * L_future
    + path_cost_kl_weight * L_cost
```

`P_H0` and `P_(t+1)` are row-normalized heatmap probabilities with self-loops
and non-trainable padding outside H0's compressed candidate graph masked.
Every increasingly refined graph heatmap supervises the initial GNN heatmap.
The standard profile can weight later rounds as `1, 2, ..., graph_rounds`.
The legacy TSP100 fine-tuning preset instead uses equal round weights
(`kl_round_power=0`). It must be revalidated with the revised archive and
cost-supervision pipeline. The
first target is detached so `L_H0` updates only the GNN+MLP that produces H0.
After all rounds, the final archive creates a detached future-quality target;
`L_future` trains each intermediate learned graph correction to anticipate
that structure. `L_cost` evaluates every sampling heatmap, including H0 and
the terminal H3, against the quality-weighted elite edges from the population
it actually produced. There is no REINFORCE or entropy loss.

The standard training policy uses a fixed coordinate pool. Every epoch
shuffles the pool and visits every instance exactly once, with no omission,
duplication, or replacement. Therefore, an instance participates exactly as
many times as the number of training epochs. Every visit starts from the latest model H0 and an
empty solution archive, exactly like a fresh test instance. Heatmaps and tours
accumulate only through the visit-local refinement sequence.
The standard profile uses 400 fixed coordinates, batch size 20, 20 steps per
epoch, 20 epochs, and three graph refinements plus a terminal ACO population.
Each of the 400
instances therefore participates exactly 20 times.
Pool coordinates and visit-local compressed archive paths are kept on CPU;
only the current optimizer batch is materialized on the training device.
States are never shared between visits or different TSP instances.

Coordinate-pool overfitting can be measured with three matched policies:

```raw
$ python3 train.py 100 --train_pool_mode fixed
$ python3 train.py 100 --train_pool_mode refresh
$ python3 train.py 100 --train_pool_mode mixed --pool_refresh_fraction 0.5
```

All policies still visit exactly 400 instances per epoch. `fixed` repeats the
same coordinates, `refresh` replaces all coordinates after each epoch, and
`mixed` replaces the requested fraction while retaining the rest.

Testing executes the same sampling, archive, hypergraph aggregation, and
heatmap-update loop under `torch.no_grad()`.  It does not compute KL, call
backward, or update model parameters. ACO, the archive, and visit-local
heatmaps remain on CPU. H0 and the learned solution-graph residual run on the
model device. No NLS or 2-opt improvement is applied by default; either can
be enabled explicitly for matched comparisons.
Tour construction uses the seed-controlled PyTorch sampler by default. The
optional `--sampling_backend numba` selects the original inference constructor; it can
be faster at larger scales, but its thread-local random stream is not strictly
reproducible and its thread-pool overhead made TSP50 slower in measurement.
Action log-probabilities are skipped because none of the KL objectives consumes
them.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

TSP100 fine-tuning preset:
```raw
$ python3 train.py 100 --profile tsp100_finetune
```

This explicit profile starts from `../pretrained/tsp_nls/tsp100-best.pt` and
uses `k_sparse=10`, `lr=1e-4`, 3 epochs, 48 ants,
3 graph refinements (4 ACO populations), 5 validation rounds, a fixed pool of 400 instances, and
equal KL weights across graph rounds. It writes checkpoints to
`../pretrained/tsp_nls/optimized_v3_k10_finetune`. Individual command-line
flags still override profile values. The profile is deliberately restricted
to TSP100, but the revised pipeline still requires multi-seed validation.

TSP200:
```raw
$ python3 train.py 200
```

TSP500:
```raw
$ python3 train.py 500 --graph_rounds 3 --kl_weight 1.0
```

TSP1000:
```raw
$ python3 train.py 1000 --graph_rounds 3 --kl_weight 1.0
```

`--graph_rounds` controls the number of heatmap refinements; training samples
one additional ACO population so the terminal heatmap is evaluated.
`--train_pool_size` controls the fixed pool size,
defaults to 400, and must equal `--steps * --batch_size` so every pool member
is visited exactly once per epoch.
`--profile standard` retains the general training defaults. `--elite_ratio`
controls graph-update admission, and `--kl_round_power` controls the
later-round KL weighting.
`--future_kl_weight` controls final-archive supervision,
`--path_cost_kl_weight` controls supervision from actual ACO costs, and
`--max_solution_graph_solutions` bounds the learned graph to 128 tours by
default. `--max_archive_solutions` and `--max_archive_rounds` bound the full
underlying archive.
`--validation_rounds` controls validation search depth. `--quality_temperature`,
`--age_decay`, and `--uniform_mix` control quality weighting.
`--quality_prior_strength` smooths the quality target with the preceding
heatmap. `--seed` fixes training and uses an isolated fixed validation seed, so
every checkpoint is evaluated with the same random stream. Epoch-0 parameters
are saved as `tsp{N}-init.pt`; `tsp{N}-best.pt` is atomically replaced only when
the full run completes. Old DeepACO checkpoints can be supplied with `--pretrained`.
Missing solution-graph parameters are initialized with a zero output layer, so
old DeepACO checkpoints start from deterministic heatmap refinement.

Refinement ablations use the same ACO budget and checkpoint interface:

```raw
$ python3 test.py 100 --refinement_mode h0
$ python3 test.py 100 --refinement_mode deterministic
$ python3 test.py 100 --refinement_mode learned
$ python3 test.py 100 --refinement_mode full
```

`h0` keeps the original heatmap fixed, `deterministic` uses only fixed archive
aggregation, `learned` applies only the learned residual over the current
heatmap, and `full` combines deterministic aggregation with that residual.
The `h0` mode is evaluation-only; the other modes can be trained separately.

Local search is an explicit matched-baseline option and remains disabled by
default:

```raw
$ python3 test.py 100 --refinement_mode h0 --local_search none
$ python3 test.py 100 --refinement_mode full --local_search none
$ python3 test.py 100 --refinement_mode h0 --local_search nls
$ python3 test.py 100 --refinement_mode full --local_search nls
```

The same `--local_search none|2opt|nls` option is available in `train.py`, so
baseline and PHG-ACO checkpoints can be trained with matched search settings.

### Testing

When `--model` is omitted, testing loads the root checkpoint
`../pretrained/tsp_nls/tsp{nodes}-best.pt`. Legacy checkpoints do not contain
the learned solution-graph updater, so the loader initializes its output at
zero and reproduces deterministic refinement. Use a checkpoint trained on
this branch to evaluate the full PHG-ACO model. Pass `--model` to select it
explicitly. Testing defaults to `--seed 1234`, allowing two checkpoints to be
compared with the same sampling randomness.

When training uses a non-default candidate size, pass the same value at test
time, for example:

```raw
$ python3 test.py 100 --k_sparse 10 --iterations 1 2 3 5 10
```

Results produced by the earlier parameter-free updater are not reported as
PHG-ACO results. Retrain the learned updater and report paired seeds before
comparing it with DeepACO or the fixed-aggregation branch.

TSP200:
```
$ python3 test.py 200 --iterations 1 2 3 5 10
```

TSP500:
```
$ python3 test.py 500 --iterations 1 2 3 5 10
```

TSP1000:
```
$ python3 test.py 1000 --iterations 1 2 3 5 10
```
