## PHG-ACO: persistent hypergraph-guided ACO for TSP

This variant treats heatmap construction as an iterative, per-instance search
process. A training instance retains its feasible-solution archive across
visits, while each visit still runs only one refinement:

```text
problem graph -> H0 -> ACO sampling S0 -> solution hypergraph -> H1
              -> KL(stopgrad(H1) || H0) -> stop
```

`graph_rounds` is fixed at one heatmap refinement. Each training visit samples
only one ACO population from H0 and constructs H1 without sampling from it.
H0-only validation runs 10 ACO iterations on a fixed H0 and selects the best
checkpoint, matching testing.
Testing is S0-only: it generates H0 once per instance, then runs 10 ACO
iterations on that fixed heatmap by default. Pheromone and the cumulative best
tour carry across these iterations, but testing never constructs H1. It reports
the average cumulative best path length after each ACO iteration. No mode
generates H2.

For a repeated training-pool instance, the current model always recomputes H0
and ACO(H0) samples S0. The archive from prior visits is reused, including
the previous S0 populations. H1 uses the current fresh H0, not the previous H1:

```text
graph_result = GCN(inherited_archive + S0_new)
H1_new = normalize(0.9 * graph_result + 0.1 * normalize(H0_new))
save(inherited_archive + S0_new)
```

The graph result and H0 are both row-normalized before mixing. H1 is detached
as the distillation target, so the H0 loss trains the current initial GNN.
Historical H1 heatmaps are neither stored nor mixed into later visits, and no
extra H2 is generated.

By default, the initial GNN receives a full directed candidate graph:
`k_sparse = N - 1` neighbors per node (49 for TSP50, 99 for TSP100), excluding
self-loops. This increases GNN computation and memory relative to the former
nearest-neighbor graph. A sampler is accessed through the interface in
`solution_sampler.py`; ACO sampling is used directly and local search is
disabled by default but available as a matched baseline. A different constrained decoder can return the same
`SolutionBatch` fields.

Distinct feasible ACO tours are retained across visits to the same training
instance. Cyclic rotations and reversed
orientations of the same undirected tour are stored once; observing a duplicate
does not add another statistical sample:

- a graph node is one undirected TSP edge observed in a sampled tour;
- a sampled tour is a hyperedge connecting all of its TSP-edge nodes;
- two edge nodes are therefore related by the set of archived tours containing
  both of them, without materialising the equivalent quadratic clique graph.

The path stored in the graph is always paired with its own cost. The archive
validates that every stored TSP tour is a permutation of all nodes.

H1 is constructed from the statistical edge graph using a two-layer,
parameter-free weighted GCN. For an edge node `e`, its initial value uses
**all distinct archived tours** containing that edge:

```text
x_e = mean_{s contains e} (1 / cost(s))
w_ef = sum_{s contains both e and f} (1 / cost(s))
```

The GCN uses symmetric degree normalization and self-loops. Edge-to-solution
to-edge incidence propagation computes the same weighted co-occurrence
messages without materializing a quadratic edge clique. Two propagation
layers produce edge scores, which are written back to a symmetric TSP heatmap
and row-normalized. There is **no age or visit-round decay**: old and new
sampled solutions contribute according to their actual tour lengths alone.
`--quality_prior_strength` defaults to `0.1`, mixing 90% graph result with 10%
of the normalized current H0 on every visit. H1 is a detached teacher for H0,
not a heatmap used for another ACO sample.

By default, each training instance's archive retains every sampled feasible
tour across visits, including repeats, rotations, and reversals, with its
original sampled order and matching cost. Repeated observations count again
in the statistical graph. Optional positive limits can be set for archive
tours and rounds; pruning trims stored observations when needed. Unbounded history can
require substantial CPU/GPU memory and increasing graph-construction time.
The optional learned updater runs
two sparse edge-to-solution-to-edge message-passing layers. Edge nodes use the
current heatmap probability, inverse distance, quality support, and occurrence
frequency. Solution nodes use quality weight, normalized path quality, and
recency (with age decay fixed to zero in the current training path). The
output is a symmetric residual bounded to `[-0.25, 0.25]`; its
final projection is zero-initialized. With KL as the only training objective,
this residual receives no gradient. Training therefore defaults to the
deterministic hypergraph update; `learned`/`full` can only use an already
trained, frozen residual from a supplied checkpoint.

Training uses only H1-to-H0 distillation for the single refinement:

```text
L_H0 = KL(stopgrad(P_H1) || P_H0)
L = kl_weight * L_H0
```

`P_H0` and `P_H1` are row-normalized heatmap probabilities with self-loops
masked. If a smaller candidate graph is explicitly selected, non-trainable
padding outside it is also masked.
The H1 target is detached, so this loss updates only the GNN+MLP that produces
H0. It does not train the hypergraph residual network, and there is no
REINFORCE, S1 cost loss, or entropy loss.

The standard training policy uses a fixed coordinate pool. Every epoch
shuffles the pool and visits every instance exactly once, with no omission,
duplication, or replacement. Therefore, an instance participates exactly as
many times as the number of training epochs. Every visit starts by generating
H0 from the latest model, then inherits only that instance's solution archive.
No search state is shared between different instances.
The standard profile uses 400 fixed coordinates, batch size 20, 20 steps per
epoch, 20 epochs, and one ACO population followed by one graph refinement.
Each of the 400
instances therefore participates exactly 20 times.
Pool coordinates and cross-visit compressed archive paths are kept on CPU;
only the current optimizer batch is materialized on the training device.
The graph uses every archived tour by default, so memory and per-visit
computation can increase over epochs. Fresh validation and test
instances do not have historical archives; this difference
should be considered when interpreting held-out results.

Coordinate-pool overfitting can be measured with three matched policies:

```raw
$ python3 train.py 100 --train_pool_mode fixed
$ python3 train.py 100 --train_pool_mode refresh
$ python3 train.py 100 --train_pool_mode mixed --pool_refresh_fraction 0.5
```

All policies still visit exactly 400 instances per epoch. `fixed` repeats the
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
Action log-probabilities are not needed in training or testing because the
only training objective is heatmap KL. Training does not print or record
per-epoch KL values.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

TSP100 fine-tuning preset:
```raw
$ python3 train.py 100 --profile tsp100_finetune
```

This explicit profile starts from `../pretrained/tsp_nls/tsp100-best.pt` and
uses `k_sparse=99` (full TSP100 graph), `lr=1e-4`, 3 epochs, 48 ants,
1 S0 ACO population, 1 graph refinement, H0-only validation, and a
fixed pool of 400 instances. It writes checkpoints to
`../pretrained/tsp_nls/optimized_v3_full_finetune`. Individual command-line
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

`--graph_rounds` must be 1; training samples S0 once and constructs H1 once.
The default candidate size is `N - 1`, matching a full graph in both training
and testing. An explicit `--k_sparse` overrides this default.
`--kl_weight` is the only loss weight and must be positive.
`--train_pool_size` controls the fixed pool size,
defaults to 400, and must equal `--steps * --batch_size` so every pool member
is visited exactly once per epoch.
`--profile standard` retains the general training defaults. All sampled
archived tours, including repeated routes, contribute to the statistical graph; there is no elite-only
filter or round-age decay.
`--max_solution_graph_solutions`, `--max_archive_solutions`, and
`--max_archive_rounds` are optional positive caps. By default none of these
three limits is applied. The first visit starts with an empty archive, while
later visits of the same training instance reuse it.
The checkpoint is selected by H0-only T=10 validation; validation does not
sample S1 or build H1.
The four values printed on each training epoch line match the original
DeepACO validation definitions: mean and minimum cost of an initial probe
population that does not advance ACO, then cumulative best after one and
five ACO iterations. The H0-only T=10 value remains the checkpoint criterion
but is not one of the four printed numbers. The test output format is unchanged.
`--quality_prior_strength` smooths the GCN graph output with the current fresh
H0; its default `0.1` gives 90% weight to the graph and 10% to H0. `--seed` fixes training
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

This is an H0/S0 evaluation, not an evaluation of H1 or the optional learned
solution-graph updater. Both validation and testing use this H0-only path.

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
