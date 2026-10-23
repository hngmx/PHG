# Original TSP parameters with PHG-ACO graph training

The original `train.ipynb` and `test.ipynb` are retained. `train.py` is the
new runnable version using the current `tsp_nls` idea:

```text
original TSP GNN -> H0 -> one complete ACO iteration S0
-> update pheromone and record S0 tours -> local solution-edge graph
-> trainable GCN -> H1 -> elite-edge supervision + KL(H1 detached || H0)
```

The original TSP REINFORCE objective is also kept. The same S0 tours supply its
loss and update ACO pheromone once, before H1 is built. H1 uses the persistent
archive when `--fixed_pool` is selected, the post-S0 pheromone for candidate
selection, relative route quality, structural diversity, and the current
relation-gated local GCN. It is not mixed with H0. PHG training, validation,
and testing use the original TSP random-start rule: each ant independently
chooses its starting city. Validation and testing use H0 alone; neither samples
another population from H1. The graph
builder and KL definitions are shared with `tsp_nls` to keep the two
implementations consistent.

The original notebook's defaults take priority: learning rate `3e-4`, 128
fresh random instances per epoch, 5 epochs, one optimizer step per instance,
and 5 ACO validation iterations. TSP20 uses 20 ants and 10 neighbors; TSP100
uses 20 ants and 20 neighbors; TSP500 uses 50 ants and 50 neighbors. The
original TSP REINFORCE coefficient is 1.0. The new KL and graph supervision
coefficients default to 0.03 and 1.0 respectively. `--fixed_pool` is optional
for repeated visits and a persistent per-instance archive; the default remains
the original fresh-instance policy. Unlike `tsp_nls`, this TSP variant has no
NLS and keeps the original Torch sampling backend and test checkpoints.

Run from this directory:

```text
python train.py 100
python test.py 100 --model ../pretrained/tsp/phg_current_sampling/tsp100-best.pt
```

New checkpoints go to `../pretrained/tsp/phg_current_sampling` so the notebook's
original `../pretrained/tsp/tsp{N}.pt` and earlier PHG checkpoints are not
overwritten. `test.py` uses the
original test notebook's ACO checkpoints: 1, 10, 20, 30, 40, 50, and 100.
