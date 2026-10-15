"""End-to-end HLDA vignette on simulated data.

Runs in a few minutes and needs no input files. It simulates five cell types in
a lineage tree (lymphoid and myeloid branches), each branch with a shared
program, plus one stress program active in every type; fits the default and a
lineage geometry (3c); shows the theta prior options (3d); then two cases a
user meets in practice — a label hiding two states, which a second topic for
that label resolves (3e), and an activity program that modulates identity,
where the identity prior decides what the shared topic holds (3f); and plots.

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
# stress program is switched on by a third of the cells of every type. The
# lineage programs are what a tree geometry is for: a program shared by some
# types but not others.

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
    schlda.fit_hlda(adata, "cell_type", 1, output_dir=OUT / "not_created",
                    n_loops=2, burn_in=0, thin=1)
except ValueError as err:
    print(f"as expected, the normalised .X is rejected:\n  {err}\n")

# ── 3. Fit the default geometry ────────────────────────────────────────────
# One topic per cell type is created automatically from the annotation, plus
# n_activity_topics shared by every cell. n_loops/burn_in are kept short so the
# vignette finishes quickly; for real data the defaults (3000/1000/30) are a
# more sensible starting point. Every chain starts from a uniform random
# assignment, so on a new dataset compare two or three seeds.

FIT = dict(layer="counts", n_loops=600, burn_in=300, thin=10, seed=0)

res = schlda.fit_hlda(adata, "cell_type", n_activity_topics=1, output_dir=OUT, **FIT)

print(f"\n{res}")
print(f"identity topics: {res.identity_topics}")
print(f"activity topics: {res.activity_topics}")
print(f"cos(V1 genes, true stress program) = {cosine(res.beta['V1'].to_numpy(), programs['stress']):.3f}")
print("in the default geometry each type's topic absorbs its lineage program:")
for t in ["T_cell", "Monocyte"]:
    print(f"  cos({t} topic, {branch_of[t]} program) = {cosine(res.beta[t].to_numpy(), programs[branch_of[t]]):.2f}")

# ── 3b. The same fit on two threads ───────────────────────────────────────
# n_threads splits the cells into blocks swept concurrently (1 = the exact
# serial sampler). A run repeats exactly for a fixed seed and n_threads.

res_par = schlda.fit_hlda(adata, "cell_type", n_activity_topics=1, output_dir=OUT / "parallel",
                          n_threads=2, verbose=False, **FIT)
print(f"\ntwo-thread fit: {res_par}  ({res_par.params['seconds_per_sweep']:.3f} s/sweep)")

# ── 3c. A lineage geometry ────────────────────────────────────────────────
# schlda.geometry gives every cell type one topic. group(name, members) adds a
# node whose topics every member may use; topics(node, n) sets how many topics
# a node carries. Written out level by level, the tree above is:
#
#     every cell ──── stress
#     lymphoid   ──── T_cell, B_cell, NK_cell
#     myeloid    ──── Monocyte, Dendritic
#
# Groups may nest or overlap, so any DAG over the cell types is expressible.

g = schlda.geometry(adata, "cell_type")
g.group("stress", "all")                       # one topic every cell may use
for branch, members in LINEAGE.items():
    g.group(branch, members)                   # one topic per branch
print("\n" + g.describe())

res_tree = schlda.fit_hlda(adata, "cell_type", geometry=g, output_dir=OUT / "lineage", verbose=False, **FIT)
print(f"\nlineage geometry: {res_tree}")
for k in ["stress", "lymphoid", "myeloid"]:
    print(f"  cos({k} topic, true {k} program) = {cosine(res_tree.beta[k].to_numpy(), programs[k]):.3f}")
print(f"  cos(T_cell topic, lymphoid program) = "
      f"{cosine(res_tree.beta['T_cell'].to_numpy(), programs['lymphoid']):.2f}  (was absorbed above)")
is_myeloid = adata.obs["cell_type"].isin(LINEAGE["myeloid"]).to_numpy()
print(f"  myeloid cells never use 'lymphoid': {(res_tree.theta.loc[is_myeloid, 'lymphoid'] == 0).all()}")
print("\nmean topic usage by cell type (lineage geometry)")
print(res_tree.theta.groupby(res_tree.cell_types.values, observed=True).mean().round(2).to_string())

# The geometry is also a table: res_tree.topics / lineage/topics.csv, which
# geometry= reads back. The dict shortcuts build the same object:
same = schlda.geometry(adata, "cell_type", groups={"stress": "all", **LINEAGE})
assert same.to_frame().equals(g.to_frame())

# ── 3d. The theta prior ───────────────────────────────────────────────────
# A cell of type c has theta_i ~ Dirichlet(alpha_k : k in A(c)) over the topics
# it may use. alpha_identity (default 1.0, split over a type's own topics) and
# alpha_activity (default 0.1, each group topic) set it; g.alpha(topic, value)
# or alpha={node or topic: value} override single entries. Every value may be
#
#     a float                  the same concentration for every cell
#     one value per cell       an array, or a Series indexed by obs_names
#     schlda.depth(kappa)      kappa * L_i: pseudo-counts equal to the fraction
#                              kappa of the cell's own counts
#
# so the prior can be made to carry the same share of every cell's evidence
# whatever its depth. When every cell has the same depth this is a fixed alpha.
# Nothing else about the fit changes; res.topics / topics.csv record what was
# used, and a prior that varies by cell is written to alpha_cells.csv.

res_depth = schlda.fit_hlda(adata, "cell_type", geometry=g, output_dir=OUT / "lineage_depth",
                            alpha_identity=schlda.depth(0.001), alpha={"stress": 0.05}, verbose=False, **FIT)
print("\ntopics table with a depth-scaled identity prior:")
print(res_depth.topics.to_string(index=False))
alpha_cells = pd.read_csv(OUT / "lineage_depth" / "alpha_cells.csv", index_col=0)
L = np.asarray(adata.layers["counts"].sum(1)).ravel()
print(f"alpha on T_cell for the shallowest / median / deepest cell: "
      f"{alpha_cells['T_cell'].min():.2f} / {alpha_cells['T_cell'].median():.2f} / {alpha_cells['T_cell'].max():.2f}"
      f"   (L_i {int(L.min())} / {int(np.median(L))} / {int(L.max())}; always 0.1 % of L_i)")

# A full array works too: (K,) per topic or (n_cells, K) per cell, in the
# order of res.topics. This is the escape hatch for any prior not expressible
# above (and the form every other input is reduced to).
A = res_depth.topics.assign(alpha=[1.0] * 5 + [0.05, 0.1, 0.1]).alpha.to_numpy()
res_arr = schlda.fit_hlda(adata, "cell_type", geometry=g, output_dir=OUT / "lineage_array", alpha=A,
                          verbose=False, **FIT)
print(f"array prior recorded as: {res_arr.params['alpha']}, alpha column {res_arr.topics.alpha.tolist()}")


def identity_share(res, labels):
    """Each cell's theta on its own type's topic(s)."""
    own = pd.Series(0.0, index=res.theta.index)
    for _, row in res.topics[res.topics.topic.isin(res.identity_topics)].iterrows():
        m = labels == row.node
        own[m] += res.theta.loc[m, row.topic]
    return own.to_numpy()


def block_programs(rng, names, n_genes):
    out = {}
    for i, k in enumerate(names):
        p = rng.gamma(0.3, 1.0, n_genes) + 1e-3
        p[50 * i:50 * (i + 1)] *= 60
        out[k] = p / p.sum()
    return out


def build(rng, X, lab, types, prefix):
    a = ad.AnnData(sp.csr_matrix(np.array(X, dtype=np.float32)))
    a.obs_names = [f"{prefix}{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(a.n_vars)]
    a.obs["cell_type"] = pd.Categorical(lab, categories=types)
    return a


# ── 3e. A hidden state under one label: the geometry's job ────────────────
# Annotation is often coarser than the biology. "Monocyte" in a PBMC atlas
# covers CD14+ classical and FCGR3A+ non-classical monocytes, two programs
# with largely different marker genes; here 20 % of the monocytes are
# FCGR3A+. Two shared topics are requested for two cross-type programs: a
# stress program in 30 % of cells and a rare interferon response in 5 %.
#
# With one topic per label, the Monocyte topic goes to the majority state, and
# the sampler finds a better use for the second shared topic than the rare
# program: the FCGR3A+ state. Those cells are then explained entirely by a
# "shared" topic and have no weight on their own type (identity theta = 0),
# and the interferon program is lost. No prior changes this — the tokens
# follow the likelihood, and the Monocyte topic cannot represent FCGR3A+
# cells. A second topic for the label can: topics("Monocyte", 2), with the
# sampler deciding which cells use which. Annotating finer would do the same
# with the membership supplied by hand.

rng2 = np.random.default_rng(1)
PROG2 = block_programs(rng2, ["Mono_CD14", "Mono_FCGR3A", "T_cell", "B_cell", "stress", "IFN"], 350)
X2, lab2, state2 = [], [], []
for name, n in [("Monocyte", 300), ("T_cell", 150), ("B_cell", 150)]:
    for _ in range(n):
        fcgr3a = name == "Monocyte" and rng2.random() < 0.2
        base = PROG2["Mono_FCGR3A"] if fcgr3a else PROG2[name if name != "Monocyte" else "Mono_CD14"]
        w_s = rng2.beta(2, 4) if rng2.random() < 0.30 else 0.0
        w_i = rng2.beta(2, 4) * (1 - w_s) if rng2.random() < 0.05 else 0.0
        X2.append(rng2.multinomial(rng2.integers(700, 1400), (1 - w_s - w_i) * base + w_s * PROG2["stress"] + w_i * PROG2["IFN"]))
        lab2.append(name); state2.append("FCGR3A" if fcgr3a else "CD14" if name == "Monocyte" else name)
pbmc = build(rng2, X2, lab2, ["Monocyte", "T_cell", "B_cell"], "p")
lab2, state2 = np.array(lab2), np.array(state2)
SHORT = dict(n_loops=400, burn_in=200, thin=10, seed=0, verbose=False)

res_one = schlda.fit_hlda(pbmc, "cell_type", 2, output_dir=OUT / "pbmc_default", **SHORT)
res_prior = schlda.fit_hlda(pbmc, "cell_type", 2, output_dir=OUT / "pbmc_prior", alpha_identity=50.0, **SHORT)
g2 = schlda.geometry(pbmc, "cell_type", 2).topics("Monocyte", 2)
res_two = schlda.fit_hlda(pbmc, "cell_type", geometry=g2, output_dir=OUT / "pbmc_split", **SHORT)

print("\n20 % FCGR3A+ monocytes under the 'Monocyte' label, two shared topics for stress (30 %) and IFN (5 %):")
print(f"  {'geometry':<34} {'identity theta CD14+ / FCGR3A+':>31}   {'shared topics hold (cos)':<40}")
for name, r in [("Monocyte, V1, V2", res_one), ("same, alpha_identity = 50", res_prior), ("Monocyte_1, Monocyte_2, V1, V2", res_two)]:
    own = identity_share(r, lab2)
    held = []
    for v in r.activity_topics:
        k, c = max(((k, cosine(r.beta[v].to_numpy(), PROG2[k])) for k in ["stress", "IFN", "Mono_FCGR3A", "Mono_CD14"]), key=lambda t: t[1])
        held.append(f"{v} = {k} ({c:.2f})")
    print(f"  {name:<34} {own[state2 == 'CD14'].mean():>14.2f} / {own[state2 == 'FCGR3A'].mean():.2f}   {', '.join(held):<40}")
print("  (one topic per label: the second shared topic takes the hidden state, its cells lose their identity and the\n"
      "   rare program is lost; a stronger prior changes nothing; a second Monocyte topic recovers both)")

# ── 3f. A program that modulates identity: the prior's job ────────────────
# Real activity programs are modulations of a cell's own program: an
# interferon-stimulated monocyte is still a monocyte with a few hundred genes
# turned up. The shared topic then overlaps each type's topic on most genes,
# and for those genes a token could have come from either. The identity prior
# settles that tie: alpha_identity above alpha_activity says "a cell is first
# its type; the shared topic explains only what the type does not". With a
# symmetric prior the tie goes the other way and the shared topic becomes a
# generic baseline that every cell uses, while low-count carriers can lose
# their identity entirely.
#
# Simulated below with a common baseline, type programs = baseline x type
# DE, an interferon program = baseline x induced genes, 20 % carriers, and two
# sequencing platforms pooled (median counts ~170 and ~3,500). depth(kappa)
# gives every cell the same prior strength relative to its depth, so one value
# serves both platforms; a fixed alpha is relatively 20x stronger on the
# shallow platform than on the deep one. The strength is a trade-off: a small
# kappa leaves some baseline expression in the shared topic, a large one pulls
# carriers' ambiguous tokens back to their type; 0.5 % of each cell's counts
# (depth(0.005): alpha ~ 1 on the shallow platform, ~ 18 on the deep one) gives
# a shared topic that non-carriers do not use.

rng3 = np.random.default_rng(11)
G3 = 600
base = rng3.gamma(0.6, 1.0, G3) + 1e-3


def modulate(frac, loc, sd):
    f = np.ones(G3)
    m = rng3.random(G3) < frac
    f[m] = np.exp(rng3.normal(loc, sd, m.sum()))
    return f


PROG3 = {t: base * modulate(0.15, 0.0, 0.6) for t in ["Monocyte", "T_cell", "B_cell"]}
PROG3["IFN"] = base * modulate(0.10, 0.3, 0.4)
PROG3 = {k: v / v.sum() for k, v in PROG3.items()}
X3, lab3, w3, plat3 = [], [], [], []
for name in ["Monocyte", "T_cell", "B_cell"]:
    for _ in range(300):
        w = rng3.beta(2, 4) if rng3.random() < 0.2 else 0.0
        deep = rng3.random() < 0.5
        L3 = int(np.exp(rng3.uniform(*(np.log([2000, 6000]) if deep else np.log([100, 300])))))
        X3.append(rng3.multinomial(L3, (1 - w) * PROG3[name] + w * PROG3["IFN"]))
        lab3.append(name); w3.append(w); plat3.append("deep" if deep else "shallow")
ifn = build(rng3, X3, lab3, ["Monocyte", "T_cell", "B_cell"], "i")
lab3, w3, plat3 = np.array(lab3), np.array(w3), np.array(plat3)
carrier = w3 > 0
print(f"\ninterferon program as a modulation of identity: cos(beta_Monocyte, beta_IFN) = "
      f"{cosine(PROG3['Monocyte'], PROG3['IFN']):.2f}; {carrier.sum()} carriers")

print(f"{'prior':<24} {'non-carriers on V1':>18} {'carrier identity':>16} {'collapsed (shallow)':>20} {'carrier MAE shallow/deep':>25}")
ifn_fits = {}
for name, al in [("symmetric 0.1", 0.1), ("fixed 10", 10.0), ("depth(0.005)", schlda.depth(0.005))]:
    r = ifn_fits[name] = schlda.fit_hlda(ifn, "cell_type", 1, output_dir=OUT / f"ifn_{name.split(' ')[0].replace('(', '').replace(')', '')}",
                        alpha_identity=al, n_loops=500, burn_in=250, thin=10, seed=0, verbose=False)
    own = identity_share(r, lab3); v1 = r.theta["V1"].to_numpy()
    sh, dp = plat3 == "shallow", plat3 == "deep"
    print(f"{name:<24} {v1[~carrier].mean():>18.2f} {own[carrier].mean():>16.2f} {(own[carrier & sh] < 0.05).mean():>20.2f} "
          f"{np.abs(v1 - w3)[carrier & sh].mean():>12.2f} / {np.abs(v1 - w3)[carrier & dp].mean():.2f}")
print("  (symmetric: V1 is baseline expression used by everyone; identity prior: V1 is the induced program;\n"
      "   depth(0.005) applies it evenly across the two platforms.)")

# Structure plots of 3e and 3f on shared cell orders, for the eye.
order2 = schlda.cell_order(res_two)
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
schlda.structure_plot(res_one, order=order2, ax=axes[0], title="Monocyte + V1, V2: a shared topic holds the FCGR3A+ state, IFN is lost", legend=True)
schlda.structure_plot(res_two, order=order2, ax=axes[1], title="Monocyte_1, Monocyte_2 + V1, V2: both states are Monocyte, V2 = IFN", legend=True)
fig.savefig(OUT / "structure_substate.png", dpi=150, bbox_inches="tight")
order3 = schlda.cell_order(ifn_fits["depth(0.005)"])
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (name, r) in zip(axes, ifn_fits.items()):
    schlda.structure_plot(r, order=order3, ax=ax, title=f"alpha_identity = {name}", legend=ax is axes[0])
fig.savefig(OUT / "structure_prior.png", dpi=150, bbox_inches="tight")

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
# structure_plot: one stacked bar per cell, grouped by cell type, sorted within
# each group by activity weight. theta_heatmap: the same matrix as topics x
# cells, which stays readable when there are many topics. Topics carried by
# groups are drawn in strong colours, the cell types' own topics muted.

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
