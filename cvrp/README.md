# CVRP with PHG training

`train.py` applies the current PHG H0 -> solution-edge graph -> H1 training
scheme to the **non-NLS** DeepACO CVRP solver. `net.py` and the original
capacity-constrained route sampler remain the H0 backbone. The learned H0
is trained from directly sampled route lengths (REINFORCE). Valid sampled
routes are also stored in a separate archive for each fixed training instance;
their edge evidence trains a local edge GCN to produce H1, and detached H1
distills into H0. Each visit starts a fresh ACO state. Test-time search uses
only H0 and the original ACO pheromone update; it does not invoke H1.

CVRP paths may revisit depot `0` between vehicles. The archive retains these
transitions and removes only trailing `0` padding caused by batched sampling.
It checks that each customer occurs once and that each vehicle load respects
the original capacity of 50. The graph uses **directed** transition nodes,
because the original CVRP ACO uses directed heuristic and pheromone entries.
Compatible `(a,b)` -> `(b,c)` transitions form relations; observed fragments
add quality-weighted evidence. Each node has support, frequency, inverse
distance, destination demand, and a depot-edge flag. The H1 quality target
comes from the best quartile of archived routes.

The default fixed pool has 400 instances (`20 steps x 20 instances`), 20
epochs, and the original 20-ant CVRP20 / 50-ant larger-CVRP profiles. Route
archives grow across epochs unless `--max_archive_solutions` or
`--max_archive_rounds` is set. `--max_solution_graph_solutions` caps the
number of archive routes used to build one graph. This CVRP adaptation does
not use the paper's 12 node / five global features, residual lookahead
supervision, or test-time residual pheromone loop.

From the repository root, with the `deepaco` environment active:

```powershell
python cvrp/train.py 20 --pretrained pretrained/cvrp/cvrp20.pt
python cvrp/test.py 20 --baseline_model pretrained/cvrp/cvrp20.pt --output outputs/cvrp20_phg_compare
```

Training writes `cvrp20-init.pt`, `cvrp20-last.pt`, and `cvrp20-best.pt` under
`pretrained/cvrp/phg_current/`. The test command uses the best checkpoint by
default and writes mean best-route-length trajectories as CSV and PNG. It
uses `data/cvrp/testDataset-20.pt` if present; otherwise it generates a fixed
held-out test pool separate from training and validation. `--test_size`,
`--ants`, `--iterations`, `--seed`, and `--device` control the paired test.
`--model` selects another PHG checkpoint. There is no need to edit the
original `train.ipynb` to run this training path.
