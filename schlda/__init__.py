"""scHLDA — hierarchical topic models for annotated single-cell count data.

Every annotated cell type carries its own identity topic; activity topics are shared by the cell types that may
use them.

    import scanpy as sc
    import schlda

    adata = sc.read_h5ad("pbmc.h5ad")
    res = schlda.fit_hlda(adata, "cell_type", n_activity_topics=2,
                          layer="counts", output_dir="results/pbmc", seed=0)

The plots take a theta from any method, so fits from other models can be drawn the same way and compared on a
shared cell order (see ``cell_order``).
"""

from ._geometry import Geometry
from ._likelihood import likelihood_curve
from .model import HLDAResult, fit_hlda, geometry
from .plotting import cell_order, structure_plot, theta_heatmap

__all__ = ["fit_hlda", "geometry", "likelihood_curve", "Geometry", "HLDAResult", "structure_plot", "theta_heatmap", "cell_order"]
__version__ = "1.0.0"
