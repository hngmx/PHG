## Per-instance iterative solution-space compression for TSP

This variant treats heatmap construction as an iterative, per-instance search
process.  An instance owns its current heatmap, feasible-solution archive, and
cumulative solution hypergraph across both inner rounds and later epochs:

```text
problem graph -> H0 -> constrained sampling -> solution hypergraph -> H1
              -> constrained sampling -> larger hypergraph -> H2 -> ...
```

The initial GNN heatmap is sparse and therefore compresses the candidate
solution space.  A sampler is accessed through the interface in
`solution_sampler.py`; ACO+NLS is the default, but a different constrained
decoder can return the same `SolutionBatch` fields.

Every distinct feasible NLS tour from every round is retained in the
instance's cumulative solution hypergraph. Cyclic rotations and reversed
orientations of the same undirected tour are stored once; observing a duplicate
refreshes its recency instead of multiplying its weight:

- a graph node is one undirected TSP edge observed in a sampled tour;
- a sampled tour is a hyperedge connecting all of its TSP-edge nodes;
- two edge nodes are therefore related by the set of archived tours containing
  both of them, without materialising the equivalent quadratic clique graph.

The path stored in the graph is always paired with its own cost.  In
particular, an NLS tour is paired with its NLS cost; a post-NLS cost is never
assigned to the different pre-NLS path.  The archive validates that every
stored TSP tour is a permutation of all nodes.

Fixed, parameter-free hypergraph aggregation produces `H_(t+1)` from the
cumulative feasible-tour evidence and uses it as the next sampler heatmap.
Only the best-cost `elite_ratio` fraction (default `0.25`) of each newly
sampled population updates H1. All feasible tours are still retained in the
archive. Elite tours are explicitly quality-weighted:

```text
w_i = softmax(-(q_i - q_best) / (temperature * quality_scale))
```

A scale floor prevents nearly identical NLS costs from producing a numerical
one-hot target. A small uniform mixture (default `0.01`) preserves limited
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

The only training loss is KL distillation:

```text
L = kl_weight * weighted_mean_t KL(stopgrad(P_(t+1)) || P_H0).
```

`P_H0` and `P_(t+1)` are row-normalized heatmap probabilities with self-loops
and non-trainable padding outside H0's compressed candidate graph masked.
Every increasingly refined graph heatmap supervises the initial GNN heatmap.
The standard profile can weight later rounds as `1, 2, ..., graph_rounds`.
The validated TSP100 fine-tuning profile instead uses equal round weights
(`kl_round_power=0`), which was more reliable in the k=10 experiments. The
target is detached, and H1 has no trainable parameters, so KL
updates only the GNN+MLP that produces H0.  There is no REINFORCE loss, second
quality KL, or entropy loss.

Training uses a rolling persistent instance pool. Instances are revisited and
retain their detached heatmap and solution archive, but are replaced after
`max_instance_visits` visits (default `5`) to prevent a small fixed pool from
dominating generalization. Before every repeat visit, the latest model H0 is
row-normalized and fused with the stored heatmap:

```text
H_start = (1 - state_memory_strength) * H0_new
          + state_memory_strength * H_memory
```

The default memory strength is `0.5`, closing the loop between parameter
updates and later pseudo-label generation without discarding instance history.
Pool coordinates, persistent heatmaps, and compressed archive paths are kept
on CPU; only the current optimizer batch is materialized on the training
device. States are never shared between different TSP instances.

Testing executes the same sampling, archive, hypergraph aggregation, and
heatmap-update loop under `torch.no_grad()`.  It does not compute KL, call
backward, or update model parameters. Only H0's neural forward pass uses the
GPU; parameter-free H1 refinement remains on CPU to avoid per-round transfers.
Tour construction uses the seed-controlled PyTorch sampler by default, while
NLS retains the bounded iterative-round budget. The optional
`--sampling_backend numba` selects the original inference constructor; it can
be faster at larger scales, but its thread-local random stream is not strictly
reproducible and its thread-pool overhead made TSP50 slower in measurement.
Action log-probabilities are skipped because the KL-only objective does not
consume them.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

Recommended TSP100 fine-tuning experiment:
```raw
$ python3 train.py 100 --profile tsp100_finetune
```

This explicit profile starts from `../pretrained/tsp_nls/tsp100-best.pt` and
uses the measured configuration: `k_sparse=10`, `lr=1e-4`, 3 epochs, 48 ants,
3 graph rounds, 5 validation rounds, a rolling pool of 800 instances, and
equal KL weights across graph rounds. It writes checkpoints to
`../pretrained/tsp_nls/optimized_v3_k10_finetune`. Individual command-line
flags still override profile values. The profile is deliberately restricted
to TSP100 because the same defaults have not been validated at other scales.

TSP200:
```raw
$ python3 train.py 200 --graph_rounds 3 --kl_weight 1.0 --train_pool_size 400
```

TSP500:
```raw
$ python3 train.py 500 --graph_rounds 3 --kl_weight 1.0
```

TSP1000:
```raw
$ python3 train.py 1000 --graph_rounds 3 --kl_weight 1.0
```

`--graph_rounds` controls how many new ant populations are added on each visit
to a training instance. `--train_pool_size` controls the rolling pool size; by
default it is `steps * batch_size * max_instance_visits`.
`--profile standard` retains the general training defaults. `--state_memory_strength` controls H0 re-anchoring, `--max_instance_visits`
controls pool replacement, `--elite_ratio` controls graph-update admission,
and `--kl_round_power` controls the later-round KL weighting.
`--validation_rounds` controls validation search depth. `--quality_temperature`,
`--age_decay`, and `--uniform_mix` control quality weighting.
`--quality_prior_strength` smooths the quality target with the preceding
heatmap. `--seed` fixes training and uses an isolated fixed validation seed, so
every checkpoint is evaluated with the same random stream. Epoch-0 parameters
are saved as `tsp{N}-init.pt`; `tsp{N}-best.pt` is atomically replaced only when
the full run completes. Old DeepACO checkpoints can be supplied with `--pretrained`.
Checkpoint parameters from the former trainable graph updater are ignored
because H1 is now generated by fixed aggregation.

### Testing

When `--model` is omitted, testing loads the original root checkpoint
`../pretrained/tsp_nls/tsp{nodes}-best.pt`. Pass `--model` to select any other
checkpoint explicitly, including
`../pretrained/tsp_nls/optimized_v3_k10_finetune/tsp100-best.pt`. Testing
defaults to `--seed 1234`, allowing two checkpoints to be compared with the
same sampling randomness.

When training uses a non-default candidate size, pass the same value at test
time, for example:

```raw
$ python3 test.py 100 --k_sparse 10 --iterations 1 2 3 5 10
```

On the 1280-instance TSP100 test set, the fine-tuned checkpoint reduced T=10
from `7.769109` to `7.761188` at seed 1234. At seed 2234 the models were
effectively tied (`7.760764` versus `7.760729`). These two paired runs support
using the fine-tuned checkpoint but do not establish statistical significance;
report multiple paired seeds when presenting final results. Wall-clock time is
not treated as a model-quality guarantee because it varied with system load.

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
