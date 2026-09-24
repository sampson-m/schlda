"""End-to-end HLDA vignette on simulated data.

Runs in well under a minute and needs no input files: it simulates four cell
types plus one shared activity program, fits HLDA, and checks that the fit
recovered the program that was planted.

    python examples/vignette.py [output_dir]

The point of the simulation is that the truth is known, so every number printed
below can be checked against it. On your own data, skip to `fit_hlda` — the
call is the same.
"""

import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")  # write figures to file rather than opening a window
import matplotlib.pyplot as plt  # noqa: E402

import schlda

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "vignette_out")
rng = np.random.default_rng(0)

# ── 1. Simulate ────────────────────────────────────────────────────────────
# Four cell types, each with its own expression program over 300 genes, plus
# one activity program ("stress") that a third of the cells in every type
# switch on to some degree. That last part is what HLDA is for: a program that
# cuts across cell types rather than defining one.

CELL_TYPES = ["T_cell", "B_cell", "NK_cell", "Monocyte"]
N_PER_TYPE = [200, 120, 90, 60]
N_GENES = 300

programs = {}
for i, name in enumerate(CELL_TYPES + ["stress"]):
    p = rng.gamma(0.3, 1.0, N_GENES) + 1e-3
    p[i * 50:(i + 1) * 50] *= 60  # each program has ~50 marker genes
    programs[name] = p / p.sum()

counts, labels, true_stress = [], [], []
for name, n in zip(CELL_TYPES, N_PER_TYPE):
    for _ in range(n):
        # a third of cells carry the stress program, at varying strength
        w = rng.beta(2, 4) if rng.random() < 0.33 else 0.0
        profile = (1 - w) * programs[name] + w * programs["stress"]
        counts.append(rng.multinomial(rng.integers(700, 1400), profile))
        labels.append(name)
        true_stress.append(w)

counts = np.array(counts)
gene_names = [f"gene_{i}" for i in range(N_GENES)]

# ── 2. Build the AnnData the way a real one usually looks ──────────────────
# Raw counts kept in a layer, a normalised matrix in .X. This is the standard
# scanpy layout, and it is why `layer="counts"` matters below.

adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
adata.var_names = gene_names
adata.layers["counts"] = adata.X.copy()
adata.X = sp.csr_matrix(adata.X.multiply(1e4 / counts.sum(1, keepdims=True)))
adata.obs["cell_type"] = pd.Categorical(labels, categories=CELL_TYPES)

print(f"simulated {adata.n_obs} cells x {adata.n_vars} genes")
print(adata.obs["cell_type"].value_counts().to_string(), "\n")

# HLDA needs raw counts and refuses anything else rather than rounding it.
# The check runs before any output is written, so the directory named here is
# never created.
try:
    schlda.fit_hlda(adata, "cell_type", 1, output_dir=OUT / "not_created",
                  n_loops=2, burn_in=0, thin=1)
except ValueError as err:
    print(f"as expected, the normalised .X is rejected:\n  {err}\n")

# ── 3. Fit ─────────────────────────────────────────────────────────────────
# One identity topic per cell type is created automatically from the
# annotation. n_activity_topics is the only structural choice you make.
#
# n_loops/burn_in are kept short here so the vignette finishes quickly. For
# real data the defaults (3000/1000/30) are a more sensible starting point.

res = schlda.fit_hlda(
    adata,
    "cell_type",
    n_activity_topics=1,
    layer="counts",
    output_dir=OUT,
    init="data_driven",   # or "random"
    n_loops=600,
    burn_in=300,
    thin=10,
    seed=0,
)

print(f"\n{res}")
print(f"identity topics: {res.identity_topics}")
print(f"activity topics: {res.activity_topics}\n")

# ── 3b. The same fit, in parallel and with the depth-scaled prior ──────────
# fit_hlda_parallel returns the same object. n_threads splits the cells into
# blocks swept concurrently; prior_scale=0.1 gives every cell a prior of
# 0.1 x its token count, 70 % of it on the identity topic.

res_par = schlda.fit_hlda_parallel(
    adata,
    "cell_type",
    n_activity_topics=1,
    layer="counts",
    output_dir=OUT / "parallel",
    n_loops=600,
    burn_in=300,
    thin=10,
    seed=0,
    n_threads=2,
    prior_scale=0.1,
    identity_share=0.7,
)
print(f"\nparallel fit: {res_par}  ({res_par.params['seconds_per_sweep']:.3f} s/sweep)")

# ── 4. Inspect ─────────────────────────────────────────────────────────────
# beta is genes x topics: the top genes of a topic say what it represents.

print("top 5 genes per topic")
for topic in res.beta.columns:
    top = res.beta[topic].nlargest(5)
    print(f"  {topic:>9}: {', '.join(top.index)}")

# theta is cells x topics: mean activity usage per cell type.
usage = res.theta.groupby(res.cell_types.values, observed=True).mean()
print("\nmean topic usage by cell type")
print(usage.round(3).to_string())

# ── 5. Did it recover the planted program? ─────────────────────────────────
# The activity topic should match the stress program in gene space, and its
# per-cell weight should track the strength each cell was simulated with.

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

v1 = res.beta["V1"].to_numpy()
print(f"\ncos(V1 genes, true stress program) = {cosine(v1, programs['stress']):.3f}")
print(f"corr(V1 per-cell weight, true weight) = "
      f"{np.corrcoef(res.theta['V1'], true_stress)[0, 1]:.3f}")
print("(both near 1.0 means the activity topic found what was planted)")

# ── 6. Plot ────────────────────────────────────────────────────────────────
# structure_plot: one stacked bar per cell, grouped by cell type, sorted within
# each group by activity weight. theta_heatmap: the same matrix as topics x
# cells, which stays readable when there are many topics.

schlda.structure_plot(
    res, title="HLDA: 4 cell types + 1 activity topic",
    save=OUT / "structure.png",
)
schlda.theta_heatmap(
    res, cells_per_group=5, title="Topic usage (meta-cells of 5)",
    save=OUT / "heatmap.png",
)

# Both return (fig, ax), so anything not exposed as an argument can be set
# afterwards:
fig, ax = schlda.structure_plot(res, sort_by="V1", legend=False)
ax.set_ylabel("stress program usage")
fig.savefig(OUT / "structure_custom.png", dpi=150, bbox_inches="tight")

# ── 6b. Compare against another theta on the same cells ────────────────────
# structure_plot takes a theta from any method. Here the second theta is the
# simulation truth, but it could just as well be an LDA or NMF fit. cell_order
# is computed once and passed to both plots, so column i is the same cell in
# each panel and differences between the panels are real differences.

truth = pd.DataFrame(0.0, index=adata.obs_names, columns=CELL_TYPES + ["stress"])
for i, (name, w) in enumerate(zip(labels, true_stress)):
    truth.iloc[i, truth.columns.get_loc(name)] = 1 - w
    truth.iloc[i, -1] = w

order = schlda.cell_order(res)
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
schlda.structure_plot(res, order=order, ax=axes[0], title="HLDA fit", legend=False)
schlda.structure_plot(res_par, order=order, ax=axes[1], title="parallel fit, depth-scaled prior", legend=False)
schlda.structure_plot(
    truth, cell_types=adata.obs["cell_type"], activity_topics=["stress"],
    order=order, ax=axes[2], title="simulation truth",
)
fig.savefig(OUT / "structure_compare.png", dpi=150, bbox_inches="tight")

# ── 7. What ended up on disk, and on the object ────────────────────────────
print(f"\nwritten to {OUT.resolve()}")
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(OUT)}  ({path.stat().st_size:,} bytes)")

print("\nthe AnnData was annotated in place:")
print(f"  adata.obsm['schlda_theta']  {adata.obsm['schlda_theta'].shape}")
print(f"  adata.varm['schlda_beta']   {adata.varm['schlda_beta'].shape}")
print(f"  adata.uns['schlda']         {sorted(adata.uns['schlda'])}")
print("\nso adata.write_h5ad(...) carries the fit with it.")
