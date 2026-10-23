## PHG-ACO TSP: H0-to-H1 solution-edge graph training

This variant treats heatmap construction as an iterative, per-instance search
process. A training instance retains its feasible-solution archive across
visits, while each visit still runs only one refinement:

```text
problem graph -> H0 -> ACO sampling S0 -> local edge graph -> trainable GCN -> H1
              -> elite-route supervision for GCN + KL(stopgrad(H1) || H0)
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
the previous S0 populations. H1 is recomputed from the accumulated solution
graph. H0 and current ACO pheromone guide which graph nodes are retained, but
their values are not numerically blended into H1:

```text
graph_result = local_GCN(inherited_archive + S0_new)
H1_new = normalize(graph_result)
save(inherited_archive + S0_new)
```

The graph result is row-normalized directly. H1 is detached only for the H0
distillation loss; a separate archive-quality loss trains the local GCN.
Historical H1 heatmaps are neither stored nor mixed into later visits, and no
extra H2 is generated.

By default, the initial GNN receives a nearest-neighbor candidate
graph: `k_sparse = max(1, N // 10)` (5 for TSP50, 10 for TSP100), excluding
self-loops. Every sampled TSP tour still visits all N nodes; noncandidate
edges have a tiny fallback heuristic so construction can always finish.
Pass `--k_sparse N-1` explicitly to use a full graph. A sampler is accessed through the interface in
`solution_sampler.py`; ACO sampling is used directly and NLS is enabled by
default. A different constrained decoder can return the same
`SolutionBatch` fields.

Every feasible ACO tour is retained across visits to the same training
instance. Repeated, rotated, and reversed tours are stored as separate
observations with their own costs, so each occurrence contributes to the
statistical graph:

- a graph node is a retained undirected TSP city edge. Per city, the strongest
  six edges are selected from the union of `pheromone × H0` top-six candidates
  and sampled edges scored by quality-weighted support; a candidate need not
  have appeared in a sampled tour;
- two retained edge nodes are connected whenever they share a city, including
  compatible fragments not yet seen in an archived tour;
- each relation has seven features: fragment frequency, weighted support,
  mean and maximum tour weight, best-tour indicator, edge-length ratio, and
  cosine of the turn angle. Unseen fragments have zero evidence features but
  retain their geometric features and a small positive message weight;
- archived tours provide edge occurrence, route-quality weights, fragment
  evidence and the elite-route target. Repeated tours still contribute, but
  the default structural novelty factor reduces their combined influence.

The path stored in the graph is always paired with its own cost. The archive
validates that every stored TSP tour is a permutation of all nodes.

The default H1 uses a two-layer trainable GCN (`GCNConv`) over this local edge
graph. Each edge node starts with its normalized archive-quality support,
occurrence frequency, and inverse distance. A small learned gate uses the
seven relation features to modulate each positive message weight; `GCNConv`
then performs degree-normalized neighbor aggregation with self-loops. The GCN
predicts a bounded log-scale adjustment to edge support. Unseen candidate
edges receive a small distance-based prior, so they can acquire nonzero H1
mass. The undirected edge scores are copied into an N-by-N heatmap and
row-normalized. H0 values are not node features or blended into H1; with
`--graph_candidate_only`, H0 restricts the candidate support.

The older parameter-free teacher remains available with
`--refinement_mode deterministic`. Its whole-tour co-occurrence graph uses:

```text
x_e = mean_{s contains e} (1 / cost(s))
w_ef = sum_{s contains both e and f} (1 / cost(s))
```

That legacy mode computes weighted edge co-occurrence through edge-to-solution
to-edge incidence propagation without materializing the clique. The default
local GCN never makes whole-tour cliques. Neither mode decays older visits.
H1 is used as a teacher for H0 and is not sampled for another ACO population.

### Optional paired H0 / graph-prior / H1 training probe

The default training and official H0-only validation/test paths are unchanged.
Pass `--probe_instances 10 --probe_repeats 3` to add a diagnostic after every
training epoch. It creates 10 separate fixed instances outside the training
pool and official validation/test datasets, and retains each diagnostic
instance's archive across epochs. On each epoch and instance, one common H0
ACO+NLS population enters that archive and produces the post-S0 pheromone and
the two graph-derived heatmaps. Starting from that identical, frozen
pheromone, H0, the parameter-free graph prior, and learned H1 each produce
three independent 48-ant populations. These probe populations never update
pheromone or enter the archive, and all diagnostic random draws are isolated
from training. Both raw sampled-tour length and post-NLS tour length are
recorded. The diagnostic uses Torch construction and the *training* NLS
budget for every arm; it is not a substitute for the official test.

For each epoch, `heatmap_probe/summary.json` contains paired instance-mean
ratios and instance-bootstrap 95% intervals. The three charts
`h1_over_h0.png`, `prior_over_h0.png`, and `h1_over_prior.png` show raw and
post-NLS ratios; below one favors the numerator. Per-instance data are saved
as `epoch_NNN_instances.json`. Use `--probe_output` to choose a separate
result folder. An H1/H0 gain alone does not isolate the contribution of
trainable graph relations: H1/prior is the more relevant immediate control,
and a relation-disabled retraining ablation would be needed for a stronger
causal claim.

For optional legacy graph modes, `--graph_structure local` connects only
consecutive tour edges (plus self-loops); `whole_tour` keeps co-occurrence.
The `--graph_structure` setting applies only to legacy modes. `--graph_layers`
sets the default GCN depth or the fixed legacy aggregation depth.
`--graph_quality_mode relative` assigns each sampled tour a weight
`exp(-(cost(s) - min_cost) / (mean_cost * temperature))`; an edge's initial
feature includes its accumulated weighted support, normalized by the
largest edge support.
The default `local_gcn` mode instead uses `relative_diversity`: it multiplies
this relative-quality weight by a bounded novelty factor. Novelty is the
mean inverse square root of the archive frequency of a tour's undirected
edges, normalized by the archive mean; `--graph_diversity_strength 0.5`
controls its contribution and the factor is clipped to `[0.5, 1.5]`.
Thus low-cost, structurally distinct tours contribute more, while repeated
tours remain in the graph and are relatively downweighted when more diverse
tours exist. Both edge support and
local GCN adjacency weights use the combined tour weight. Set
`--graph_quality_mode relative` to remove novelty, or
`--graph_quality_mode inverse_cost` to reproduce the former local-GCN
inverse-length weighting. Legacy refinement modes still default to
`inverse_cost`.
`--graph_candidate_only` removes noncandidate edge nodes before propagation
and restricts the directed H1 target to H0's trainable candidate support.
These options never deduplicate tours or decay older visits. Validation and
testing continue to use H0 alone.

By default, each training instance's archive retains every sampled feasible
tour across visits, including repeats, rotations, and reversals, with its
original sampled order and matching cost. Repeated observations count again
in the statistical graph. Optional positive limits can be set for archive
tours and rounds; pruning trims stored observations when needed. Unbounded history can
require substantial CPU/GPU memory and increasing graph-construction time.
The legacy optional learned updater runs
two sparse edge-to-solution-to-edge message-passing layers. Edge nodes use the
current heatmap probability, inverse distance, quality support, and occurrence
frequency. Solution nodes use quality weight, normalized path quality, and
recency (with age decay fixed to zero in the current training path). The
output is a symmetric residual bounded to `[-0.25, 0.25]`; its
final projection is zero-initialized. This residual is still frozen in the
legacy `learned`/`full` modes. The default `local_gcn` is a separate module
that receives gradients from elite-route supervision.

Training can use H1-to-H0 distillation, an S0 REINFORCE objective, or both:

```text
L_H0 = KL(stopgrad(P_H1) || P_H0)
L_GCN = KL(P_elite || P_H1)
L_RL = mean_a[(L(s_a) - b_{-a}) * sum_t log p_H0(a_t | a_<t)]
b_{-a} = mean_{j != a} L(s_j)
L = kl_weight * L_H0 + graph_supervision_weight * L_GCN
    + reinforce_weight * L_RL
```

`P_H0` and `P_H1` are row-normalized heatmap probabilities with self-loops
masked. If a smaller candidate graph is explicitly selected, non-trainable
padding outside it is also masked.
`P_elite` is the row-normalized edge mass of the best-cost quartile of archived
tours, with 1% edge-prior smoothing so unseen retained edges have support.
This target is detached. `L_GCN` updates the local GCN; `L_H0` and `L_RL`
update the initial GNN+MLP. The REINFORCE term uses sampled S0 tours before
optional local search. With `--kl_weight 0`, no archive or H1 is constructed
and the graph loss is inactive. No mode samples S1 in training.
There is no S1 cost loss or entropy loss.

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

The default `fixed` pool retains coordinates and their accumulated archives
throughout training. `refresh` and `mixed` remain optional training policies;
replaced instances lose their history, while surviving instances keep it.

Testing runs only the first part of the training visit under
`torch.no_grad()`: problem graph, trained initial GNN, H0, and repeated ACO
sampling within S0. It does not create a solution archive, build a hypergraph,
generate H1, calculate KL, or update model parameters. ACO stays on CPU and
H0 is generated on the model device. Within each instance, the same ACO sampler
retains its pheromone and best tour for all iterations; each new instance starts
with a fresh sampler. NLS improves sampled tours by default in both training
and testing; use `--local_search none` to disable it explicitly.
Tour construction now defaults to the original DeepACO Numba inference sampler
for both validation and testing. Use `--sampling_backend torch` for the slower
PyTorch sampler when needed. Numba's threaded random stream is not strictly
reproducible from the Python, NumPy, and PyTorch seeds.
Validation and testing advance pheromones and track the best tour through the
original `ACO.run()` loop, one iteration at a time. The training-only sampler
interface remains available for collecting tours and action log-probabilities.
Training computes action log-probabilities for the S0 REINFORCE term when its
weight is positive. Testing never computes action log-probabilities. The
default four-value training output is unchanged. Add `--show_loss_values` to
print a separate line after each trained epoch with the instance-mean raw KL,
REINFORCE, and GCN supervision losses and their weighted contributions. Inactive objectives
display zero. Epoch 0 has no training losses to report. The switch does not
change optimization, validation, testing, or checkpoint selection.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

TSP100 fine-tuning preset:
```raw
$ python3 train.py 100 --profile tsp100_finetune
```

This explicit profile starts from `../pretrained/tsp_nls/tsp100-best.pt` and
uses `k_sparse=10`, `lr=1e-4`, 3 epochs, 48 ants,
1 S0 ACO population, 1 graph refinement, H0-only validation, and a
fixed pool of 400 instances. It writes checkpoints to
`../pretrained/tsp_nls/optimized_v3_full_finetune` (an unchanged legacy
directory name). Individual command-line
flags still override profile values. The profile is deliberately restricted
to TSP100, but the revised pipeline still requires multi-seed validation.

TSP200:
```raw
$ python3 train.py 200
```

TSP500:
```raw
$ python3 train.py 500 --graph_rounds 1
```

TSP1000:
```raw
$ python3 train.py 1000 --graph_rounds 1
```

`--graph_rounds` must be 1; training samples S0 once and, when KL is enabled,
constructs H1 once. REINFORCE-only training skips the H1 construction.
The default candidate size is `max(1, N // 10)` in both training and testing.
An explicit `--k_sparse` overrides this default.
`--kl_weight` and `--reinforce_weight` are non-negative, and at least one must
be positive. Every size defaults to `(0.03, 0.1)` for `(KL, REINFORCE)`,
matching the TSP50 setting from the single-seed T=10 comparison. Its benefit
outside TSP50 has not yet been validated. Explicit flags override either weight, including
`--reinforce_weight 0` to disable REINFORCE.
The default is `--refinement_mode local_gcn`, trained with
`--graph_supervision_weight 1.0`. The local graph stays sparse because each
city selects six incident edges before all compatible relations are formed.
`--graph_candidate_only` and the archive caps further control its size.
Existing checkpoints can still be tested through H0, but the changed H1 graph
and relation gate require retraining before drawing performance conclusions.
`--train_pool_size` controls the fixed pool size,
defaults to 400, and must equal `--steps * --batch_size` so every pool member
is visited exactly once per epoch.
`--profile standard` retains the general training defaults. All sampled
archived tours, including repeated routes, contribute to the statistical graph;
the best-cost quartile defines only the GCN supervision target. There is no
round-age decay.
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
H1 contains only the normalized GCN graph output; there is no H0/H1 blending
parameter. `--seed` fixes training and uses an isolated validation seed, but
the default Numba sampler does not guarantee the same random stream for every
checkpoint. Epoch-0 parameters
are saved as `tsp{N}-init.pt`; `tsp{N}-best.pt` is atomically replaced only when
the full run completes. Old DeepACO checkpoints can be supplied with `--pretrained`.
Missing local GCN parameters are initialized with a zero output layer, so an
old DeepACO checkpoint initially makes H1 from archive edge support before the
GCN learns a correction.

The test entry point deliberately does not expose H1 refinement ablations:
every test instance stops after S0, which contains multiple ACO iterations.
NLS is enabled by default and can be disabled explicitly:

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
be compared under the same seed settings, although the default Numba sampler
does not guarantee identical random streams. By default, each instance runs
10 ACO iterations on its fixed H0. Output retains the original aggregate
format: total duration followed by `T=1` through `T=10` average cumulative
best path lengths, with no separate result line for each instance. Here `T`
counts ACO iterations within S0, not H0/H1 heatmap refinements. Use
`--iterations N` to change this count.

When training uses a non-default candidate size, pass the same value at test
time, for example:

```raw
$ python3 test.py 100 --k_sparse 99
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
