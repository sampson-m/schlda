"""scHLDA — hierarchical topic models for annotated single-cell count data.

One identity topic per annotated cell type, plus a set of shared activity
topics every cell may use.

    import scanpy as sc
    import schlda

    adata = sc.read_h5ad("pbmc.h5ad")
    res = schlda.fit_hlda(adata, "cell_type", n_activity_topics=2,
                          layer="counts", output_dir="results/pbmc", seed=0)

    schlda.structure_plot(res, save="results/pbmc/structure.pdf")

The plots take a theta from any method, so fits from other models can be drawn
the same way and compared on a shared cell order (see ``cell_order``).
"""

from .model import HLDAResult, fit_hlda
from .parallel import fit_hlda_parallel
from .plotting import cell_order, structure_plot, theta_heatmap

__all__ = ["fit_hlda", "fit_hlda_parallel", "HLDAResult", "structure_plot", "theta_heatmap", "cell_order"]
__version__ = "0.3.0"
