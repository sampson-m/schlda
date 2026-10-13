"""End-to-end HLDA vignette on simulated data.

Runs in well under a minute and needs no input files: it simulates four cell
types plus one shared activity program, fits HLDA, checks that the fit
recovered the program that was planted, and shows how to read a fit when
deciding on a topic geometry.

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
# One topic per cell type is created automatically from the annotation.
# n_activity_topics is the only structural choice you make.
#
# n_loops/burn_in are kept short here so the vignette finishes quickly. For
# real data the defaults (3000/1000/30) are a more sensible starting point.
# Every chain starts from a uniform random assignment, so on a new dataset or
# geometry compare two or three seeds: a split found by some seeds and not
# others means the data admit more than one solution.

res = schlda.fit_hlda(
    adata,
    "cell_type",
    n_activity_topics=1,
    layer="counts",
    output_dir=OUT,
    n_loops=600,
    burn_in=300,
    thin=10,
    seed=0,
)

print(f"\n{res}")
print(f"identity topics: {res.identity_topics}")
print(f"activity topics: {res.activity_topics}\n")

# ── 3b. The same fit on two threads ───────────────────────────────────────
# n_threads splits the cells into blocks swept concurrently (1 = the exact
# serial sampler). alpha_identity / alpha_activity set the fixed theta prior
# (defaults 1.0 / 0.1); the concentrations are the same for every cell.

res_par = schlda.fit_hlda(
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
    alpha_identity=1.0,
)
print(f"\ntwo-thread fit: {res_par}  ({res_par.params['seconds_per_sweep']:.3f} s/sweep)")

# ── 3c. A custom topic geometry ───────────────────────────────────────────
# schlda.geometry gives every cell type one topic. topics(node, n) sets how
# many topics a node carries; group(name, members) adds a group of cell types
# whose topics every member may use ("all" or a list). Preview with
# describe(), then pass it as geometry=.

geo = schlda.geometry(adata, "cell_type").topics("T_cell", 2).group("stress") \
    .group("lymphoid", ["T_cell", "B_cell", "NK_cell"])
print(geo.describe())
res_geo = schlda.fit_hlda(adata, "cell_type", geometry=geo, layer="counts", output_dir=OUT / "geometry",
                          n_loops=600, burn_in=300, thin=10, seed=0, verbose=False)
print(f"\ncustom geometry: {res_geo}")
print(f"cos(stress topic, true stress program) = "
      f"{float(res_geo.beta['stress'] @ programs['stress'] / np.linalg.norm(res_geo.beta['stress']) / np.linalg.norm(programs['stress'])):.3f}")
print(f"Monocyte cells never use 'lymphoid': "
      f"{(res_geo.theta.loc[adata.obs.cell_type == 'Monocyte', 'lymphoid'] == 0).all()}\n")

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── 3d. When a type's cells abandon their own topic ───────────────────────
# A cell type that hides a substate (two expression states under one label)
# is the usual reason a cell ends up with theta near 0 on its own topic: a
# shared topic takes the substate's profile, and that substate's cells are
# then explained entirely by the shared topic. The theta prior cannot prevent
# it at realistic depth: with alpha pseudo-counts against the L_i tokens a
# cell actually has, E[theta_own | data] ~ (alpha + n_own) / (sum(alpha) + L_i),
# and with L_i in the hundreds and n_own ~ 0 no reasonable alpha moves it off
# zero. The remedy is geometry — give the type a second topic. Simulated
# below: type A has a 40 % substate, and all three types carry a cross-type
# program.

def simulate_substate(rng, n_genes=300):
    def program(block):
        p = rng.gamma(0.3, 1.0, n_genes) + 1e-3
        p[block] *= 60
        return p / p.sum()
    prog = {k: program(slice(50 * i, 50 * (i + 1))) for i, k in enumerate(["A", "A_sub", "B", "C", "shared"])}
    X, lab = [], []
    for name, n in [("A", 300), ("B", 150), ("C", 150)]:
        for _ in range(n):
            base = prog["A_sub"] if name == "A" and rng.random() < 0.4 else prog[name]
            w = rng.beta(2, 4) if rng.random() < 0.3 else 0.0
            X.append(rng.multinomial(rng.integers(700, 1400), (1 - w) * base + w * prog["shared"]))
            lab.append(name)
    a = ad.AnnData(sp.csr_matrix(np.array(X, dtype=np.float32)))
    a.obs_names = [f"s{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["cell_type"] = pd.Categorical(lab, categories=["A", "B", "C"])
    return a, prog

def identity_share(res, labels):
    """Each cell's total theta on its own type's topic(s)."""
    own = pd.Series(0.0, index=res.theta.index)
    for _, row in res.topics[res.topics.topic.isin(res.identity_topics)].iterrows():
        m = labels == row.node
        own[m] += res.theta.loc[m, row.topic]
    return own

sub, sub_prog = simulate_substate(np.random.default_rng(1))
labels_sub = sub.obs["cell_type"].to_numpy()
sub_kw = dict(n_loops=400, burn_in=200, thin=10, seed=0, verbose=False)

res_def = schlda.fit_hlda(sub, "cell_type", 2, output_dir=OUT / "substate_default", **sub_kw)
g_split = schlda.geometry(sub, "cell_type").topics("A", 2).group("V1")
res_split = schlda.fit_hlda(sub, "cell_type", geometry=g_split, output_dir=OUT / "substate_split", **sub_kw)

print("\nfraction of cells with identity theta < 0.01, by type")
tab = pd.DataFrame({
    "default (A, B, C, V1, V2)": (identity_share(res_def, labels_sub) < 0.01).groupby(labels_sub).mean(),
    "split A  (A_1, A_2, B, C, V1)": (identity_share(res_split, labels_sub) < 0.01).groupby(labels_sub).mean(),
})
print(tab.round(2).to_string())
for name, r in [("default", res_def), ("split", res_split)]:
    for v in r.activity_topics:
        b = r.beta[v].to_numpy()
        print(f"{name:>7} {v}: cos with shared program {cosine(b, sub_prog['shared']):.2f}, "
              f"A main state {cosine(b, sub_prog['A']):.2f}, A substate {cosine(b, sub_prog['A_sub']):.2f}")
print("(default: one activity topic is one of A's two states, and the A cells in that state have no\n"
      " identity; after the split both states are A topics and V1 is the cross-type program)")

# ── 3e. Reading a default fit to decide on a geometry ─────────────────────
# There is no rule for the number of topics a type needs. What has worked is
# to fit the default geometry and look at what the shared topics became: for
# each shared topic V, its mean usage per cell type and its cos^2 similarity
# to each type's profile.
#
#   V used by one or two types and close to one of their profiles
#       -> a substate of that type:          g.topics(type, 2)
#   V used by one lineage only (several related types)
#       -> a lineage program:                g.group(name, [those types])
#   cells of a type with own-topic share near 0, loading on such a V
#       -> the first case, seen from the cells
#   V used by many types and resembling none of them
#       -> a cross-type program: what shared topics are for; keep it
#
# After adding a topic, check that the new topics of the type did not absorb
# the program (their cos^2 with its genes should stay low).

B = res_def.beta
use = res_def.theta.groupby(labels_sub)[res_def.activity_topics].mean()
cos2 = pd.DataFrame(
    (B[res_def.identity_topics].T @ B[res_def.activity_topics]) ** 2
    / np.outer((B[res_def.identity_topics] ** 2).sum(), (B[res_def.activity_topics] ** 2).sum()),
    index=res_def.identity_topics, columns=res_def.activity_topics)
low = (identity_share(res_def, labels_sub) < 0.01).groupby(labels_sub).mean()
print("\ndefault fit on the substate data: mean shared-topic usage by type")
print(use.round(2).to_string())
print("cos^2 between each type's topic and each shared topic")
print(cos2.round(2).to_string())
print("fraction of cells with own-topic share < 0.01: " + ", ".join(f"{k} {v:.2f}" for k, v in low.items()))
print("(V1 is used by A alone and 60 % of A cells sit at zero on their own topic -> g.topics('A', 2);\n"
      " cos^2 is 0 here because the two A states share no genes — it is the signal when they overlap)")

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
schlda.structure_plot(res_par, order=order, ax=axes[1], title="two-thread fit", legend=False)
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
