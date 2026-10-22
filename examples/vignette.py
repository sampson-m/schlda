"""End-to-end HLDA vignette on simulated data.

Runs in a couple of minutes and needs no input files. It simulates five cell
types in two lineage branches, each branch with a shared program, plus one
stress program active in every type; fits the default geometry (one identity
topic per type plus shared activity topics) and a geometry given as a flat
file, where each lineage program is an activity topic only its branch may use;
records the likelihood trace and saves the posterior samples; and plots.

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


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── 1. Simulate a lineage tree ─────────────────────────────────────────────
# Five cell types in two branches of the haematopoietic tree:
#
#     lymphoid: T_cell, B_cell, NK_cell        myeloid: Monocyte, Dendritic
#
# Each type has its own program; each branch has a lineage program expressed
# by every member (think of the shared lymphocyte or myeloid machinery); and a
# stress program is switched on by a third of the cells of every type.

LINEAGE = {"lymphoid": ["T_cell", "B_cell", "NK_cell"], "myeloid": ["Monocyte", "Dendritic"]}
CELL_TYPES = [t for ts in LINEAGE.values() for t in ts]
N_PER_TYPE = {"T_cell": 180, "B_cell": 110, "NK_cell": 80, "Monocyte": 110, "Dendritic": 60}
BLOCKS = CELL_TYPES + list(LINEAGE) + ["stress"]      # 8 gene blocks of 50
N_GENES = 50 * len(BLOCKS)

programs = {}
for i, name in enumerate(BLOCKS):
    p = rng.gamma(0.3, 1.0, N_GENES) + 1e-3
    p[i * 50:(i + 1) * 50] *= 60                        # ~50 marker genes per program
    programs[name] = p / p.sum()

branch_of = {t: b for b, ts in LINEAGE.items() for t in ts}
counts, labels, true_w = [], [], []                   # true_w: (type, lineage, stress) weights per cell
for name in CELL_TYPES:
    for _ in range(N_PER_TYPE[name]):
        w_lin = rng.beta(6, 14)                          # ~30 % of each cell is its lineage program
        w_str = (rng.beta(2, 4) if rng.random() < 0.33 else 0.0) * (1 - w_lin)
        w = (1 - w_lin - w_str, w_lin, w_str)
        profile = w[0] * programs[name] + w[1] * programs[branch_of[name]] + w[2] * programs["stress"]
        counts.append(rng.multinomial(rng.integers(700, 1400), profile))
        labels.append(name)
        true_w.append(w)
true_stress = np.array([w[2] for w in true_w])

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
    schlda.fit_hlda(adata, "cell_type", 1, output_dir=OUT / "not_created", n_loops=2, burn_in=0, thin=1)
except ValueError as err:
    print(f"as expected, the normalised .X is rejected:\n  {err}\n")

# ── 3. Fit the default geometry ────────────────────────────────────────────
# One identity topic per cell type is created from the annotation, plus
# n_activity_topics shared by every cell. n_loops/burn_in are kept short so the
# vignette finishes quickly; for real data the defaults (3000/1000/30) are a
# more sensible starting point. Every chain starts from a uniform random
# assignment, so on a new dataset compare two or three seeds.

FIT = dict(layer="counts", n_loops=600, burn_in=300, thin=10, seed=0)

res = schlda.fit_hlda(adata, "cell_type", n_activity_topics=1, output_dir=OUT / "default", **FIT)

print(f"\n{res}")
print(f"identity topics: {res.identity_topics}")
print(f"activity topics: {res.activity_topics}")
print(f"cos(V1 genes, true stress program) = {cosine(res.beta['V1'].to_numpy(), programs['stress']):.3f}")
print("with one shared topic, each type's topic absorbs its lineage program:")
for t in ["T_cell", "Monocyte"]:
    print(f"  cos({t} topic, {branch_of[t]} program) = {cosine(res.beta[t].to_numpy(), programs[branch_of[t]]):.2f}")

# ── 3b. The same fit on two threads ───────────────────────────────────────
# n_threads splits the cells into blocks swept concurrently (1 = the exact
# serial sampler). A run repeats exactly for a fixed seed and n_threads.

res_par = schlda.fit_hlda(adata, "cell_type", n_activity_topics=1, output_dir=OUT / "parallel",
                          n_threads=2, verbose=False, **FIT)
print(f"\ntwo-thread fit: {res_par}  ({res_par.params['seconds_per_sweep']:.3f} s/sweep)")

# ── 3c. Activity topics from a flat file ──────────────────────────────────
# `activity=` names the activity topics each cell type may use. The lineage
# programs are what this is for: a program shared by some types but not
# others. One row per cell type, topics separated by ';':
#
#     cell_type,activity_topics
#     T_cell,stress; lymphoid
#     ...
#     Monocyte,stress; myeloid
#
# The same table transposed (columns topic, cell_types, with "all") is also
# accepted, as is a dict in either orientation.

activity = pd.DataFrame({"cell_type": CELL_TYPES,
                         "activity_topics": [f"stress; {branch_of[t]}" for t in CELL_TYPES]})
activity.to_csv(OUT / "activity.csv", index=False)
print("\n" + open(OUT / "activity.csv").read())

g = schlda.geometry(adata, "cell_type", activity=OUT / "activity.csv")   # inspect before fitting
print(g.describe())

res_tree = schlda.fit_hlda(adata, "cell_type", activity=OUT / "activity.csv", output_dir=OUT / "lineage",
                           verbose=False, **FIT)
print(f"\nlineage geometry: {res_tree}")
for k in ["stress", "lymphoid", "myeloid"]:
    print(f"  cos({k} topic, true {k} program) = {cosine(res_tree.beta[k].to_numpy(), programs[k]):.3f}")
print(f"  cos(T_cell topic, lymphoid program) = "
      f"{cosine(res_tree.beta['T_cell'].to_numpy(), programs['lymphoid']):.2f}  (was absorbed above)")
is_myeloid = adata.obs["cell_type"].isin(LINEAGE["myeloid"]).to_numpy()
print(f"  myeloid cells never use 'lymphoid': {(res_tree.theta.loc[is_myeloid, 'lymphoid'] == 0).all()}")
print("\nmean topic usage by cell type (lineage geometry)")
print(res_tree.theta.groupby(res_tree.cell_types.values, observed=True).mean().round(2).to_string())

# Every fit writes its geometry as topics.csv, which activity= reads back.
same = schlda.geometry(adata, "cell_type", activity=OUT / "lineage" / "topics.csv")
assert same.to_frame().equals(g.to_frame())

# ── 3d. Samples and the likelihood trace ──────────────────────────────────
# By default only the posterior means are kept. save_samples=True also writes
# the kept draws of the count tables to samples/ (memmaps); track_likelihood
# records the collapsed joint log-likelihood log P(w, z) every report_every
# sweeps in likelihood.csv, burn-in included. likelihood_curve draws it — from
# likelihood.csv when the run tracked it, otherwise from samples/ (then only
# the kept, post-burn-in draws are available).

res_s = schlda.fit_hlda(adata, "cell_type", activity=OUT / "activity.csv", output_dir=OUT / "lineage_samples",
                        save_samples=True, track_likelihood=True, report_every=50, verbose=False, **FIT)
lik = schlda.likelihood_curve(OUT / "lineage_samples")
print(f"\nlog P(w, z): {lik.loglik_joint.iloc[0]:.0f} at sweep 0 -> {lik.loglik_joint.iloc[-1]:.0f} at sweep "
      f"{int(lik.sweep.iloc[-1])}  (likelihood.csv, likelihood.png)")
assert np.array_equal(res_s.theta.to_numpy(), res_tree.theta.to_numpy())   # saving changes nothing about the fit

# ── 4. Inspect ─────────────────────────────────────────────────────────────
# beta is genes x topics: the top genes of a topic say what it represents.

print("\ntop 5 genes per topic (lineage geometry)")
for topic in res_tree.beta.columns:
    top = res_tree.beta[topic].nlargest(5)
    print(f"  {topic:>9}: {', '.join(top.index)}")

# ── 5. Did it recover the planted stress program? ──────────────────────────
# The shared topic should match the stress program in gene space, and its
# per-cell weight should track the strength each cell was simulated with.

print(f"\ncos(stress genes, true stress program) = {cosine(res_tree.beta['stress'].to_numpy(), programs['stress']):.3f}")
print(f"corr(stress per-cell weight, true weight) = "
      f"{np.corrcoef(res_tree.theta['stress'], true_stress)[0, 1]:.3f}")
print("(both near 1.0 means the shared topic found what was planted)")

# ── 6. Plot ────────────────────────────────────────────────────────────────
# Every fit already wrote theta_heatmap.png, activity_theta_heatmap.png and
# structure_plot.png next to its estimates. The same functions can be called
# with other settings. structure_plot: one stacked bar per cell, grouped by
# cell type, sorted within each group by activity weight. theta_heatmap: the
# same matrix as topics x cells. Activity topics are drawn in strong colours,
# identity topics muted.

schlda.structure_plot(res_tree, title="HLDA: lineage geometry", save=OUT / "structure.png")
schlda.theta_heatmap(res_tree, cells_per_group=5, title="Topic usage (meta-cells of 5)", save=OUT / "heatmap.png")

# Both return (fig, ax), so anything not exposed as an argument can be set
# afterwards:
fig, ax = schlda.structure_plot(res_tree, sort_by="stress", legend=False)
ax.set_ylabel("stress program usage")
fig.savefig(OUT / "structure_custom.png", dpi=150, bbox_inches="tight")

# ── 6b. Compare fits on the same cells ─────────────────────────────────────
# structure_plot takes a theta from any method. Here the third panel is the
# simulation truth, but it could just as well be an LDA or NMF fit. cell_order
# is computed once and passed to every plot, so column i is the same cell in
# each panel and differences between the panels are real differences.

truth = pd.DataFrame(0.0, index=adata.obs_names, columns=res_tree.topics.topic.tolist())   # same topic order
for i, (name, w) in enumerate(zip(labels, true_w)):
    truth.loc[adata.obs_names[i], [name, branch_of[name], "stress"]] = w

order = schlda.cell_order(res_tree)
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
schlda.structure_plot(res, order=order, ax=axes[0], title="default geometry (V1)", legend=False)
schlda.structure_plot(res_tree, order=order, ax=axes[1], title="lineage geometry", legend=False)
schlda.structure_plot(truth, cell_types=adata.obs["cell_type"], activity_topics=res_tree.activity_topics,
                      order=order, ax=axes[2], title="simulation truth")
fig.savefig(OUT / "structure_compare.png", dpi=150, bbox_inches="tight")

# ── 7. What ended up on disk, and on the object ────────────────────────────
print(f"\nwritten to {OUT.resolve()}")
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(OUT)}  ({path.stat().st_size:,} bytes)")

print("\nthe AnnData was annotated in place (last fit, key_added='schlda'):")
print(f"  adata.obsm['schlda_theta']  {adata.obsm['schlda_theta'].shape}")
print(f"  adata.varm['schlda_beta']   {adata.varm['schlda_beta'].shape}")
print(f"  adata.uns['schlda']         {sorted(adata.uns['schlda'])}")
print("\nso adata.write_h5ad(...) carries the fit with it.")
