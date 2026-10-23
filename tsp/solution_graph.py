"""Reuse the PHG-ACO solution graph with the original TSP network."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp_nls.solution_graph import (  # noqa: E402
    SolutionArchive,
    build_local_edge_gcn_graph,
    normalize_heatmap_rows,
    refinement_distillation_kl,
    rowwise_heatmap_kl,
)


__all__ = [
    "SolutionArchive",
    "build_local_edge_gcn_graph",
    "normalize_heatmap_rows",
    "refinement_distillation_kl",
    "rowwise_heatmap_kl",
]
