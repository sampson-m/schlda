# schlda

Hierarchical LDA for annotated single-cell count data.

Every annotated cell type carries its own topic, and a configurable number of
topics (`V1 … Vn`) are shared by every cell. A cell's counts can only be
explained by its own type's topic or by the shared ones, so each type's topic
stays pinned to the annotation and the shared topics are left to pick up
programs that cut across cell types — cell cycle, interferon response, and so
on. Any other arrangement — several topics for one type, a topic shared by a
subset of types, a lineage tree — is a [topic geometry](#topic-geometry).

- [Install](#install)
- [Fit](#fit) · [Output directory](#output-directory) · [Initialisation](#initialisation) · [Sampler settings](#sampler-settings) · [Threads](#threads) · [Theta prior](#theta-prior)
- [Topic geometry](#topic-geometry)
- [Plot](#plot)
- [Example](#example) · [Cost](#cost) · [Development](#development) · [Citation](#citation)

## Install

```bash
pip install -e .
```

## Fit

```python
import scanpy as sc
import schlda

adata = sc.read_h5ad("pbmc.h5ad")

res = schlda.fit_hlda(
    adata,
    "cell_type",            # column in adata.obs holding the annotation
    n_activity_topics=2,    # shared activity topics V1, V2
    layer="counts",         # raw counts; defaults to adata.X
    output_dir="results/pbmc",
    seed=0,
)

res.theta              # cells x topics, rows sum to 1
res.beta               # genes x topics, columns sum to 1
res.topics             # topic / node / cell_types / alpha
res.identity_topics    # ['B', 'CD4T', ...]  the cell types' own topics
res.activity_topics    # ['V1', 'V2']        the shared ones
```

The counts must be raw integers. Normalised or log-transformed values are
rejected rather than silently rounded — point `layer` at the layer holding the
counts.

The fit also annotates the AnnData in place:

| location | contents |
| --- | --- |
| `adata.obsm["schlda_theta"]` | cells x topics |
| `adata.varm["schlda_beta"]` | genes x topics |
| `adata.uns["schlda"]` | topic names, topic types, run parameters |

Use `key_added` to fit several models into one object.

### Output directory

`output_dir` receives:

```
HLDA_beta.csv           genes x topics posterior mean
HLDA_theta.csv          cells x topics posterior mean
topics.csv              the geometry: topic, node, cell_types, alpha
alpha_cells.csv         cells x topics prior, only when it varies by cell
samples/A_chain.memmap  (n_save, n_genes, n_topics) int32 gene-topic counts
samples/D_chain.memmap  (n_save, n_cells, n_topics) int32 cell-topic counts
samples/samples_meta.json   shapes, topic names, run parameters
```

`topics.csv` can be passed back as `geometry=` to refit the same geometry.

Omitting `output_dir` writes to `./schlda_output` and warns. Reload the chains
for convergence diagnostics using the shapes in the sidecar:

```python
import json, numpy as np
meta = json.load(open("results/pbmc/samples/samples_meta.json"))
A = np.memmap("results/pbmc/samples/A_chain.memmap", dtype="int32",
              mode="r", shape=tuple(meta["A_chain"]["shape"]))
```

### Initialisation

Every token starts on a topic drawn uniformly from those its cell may use;
`seed` fixes the draw. `burn_in` sweeps are discarded before draws are kept.

### Sampler settings

| argument | default | meaning |
| --- | --- | --- |
| `n_loops` | 3000 | total Gibbs sweeps |
| `burn_in` | 1000 | sweeps discarded |
| `thin` | 30 | sweeps between kept draws |
| `n_threads` | 1 | threads per sweep (see [Threads](#threads)) |
| `alpha_beta` | 0.1 | Dirichlet concentration on gene distributions |
| `alpha_identity` | 1.0 | concentration on a cell type's own topics (see [Theta prior](#theta-prior)) |
| `alpha_activity` | 0.1 | concentration on each shared topic |
| `alpha` | `None` | per-node / per-topic overrides, or a full array |
| `seed` | `None` | seeds init and the chain |
| `report_every` | 100 | sweeps between progress lines |

Pass `seed` for reproducibility. A run repeats exactly for a fixed `seed` and
`n_threads`; changing `n_threads` changes the draws.

### Threads

With `n_threads=1` (default) the sampler is the exact collapsed Gibbs sampler:
every token is resampled in turn, conditional on all others. With
`n_threads=P` the cells are split into `P` contiguous blocks balanced by token
count; each block is swept by one thread that updates its own cells' topic
counts in place and works from a private copy of the gene-topic table, and the
per-block changes are merged after every sweep. This is the distributed scheme
of Newman et al. (2009): the within-sweep dependence between blocks is dropped,
which is negligible when every block holds many cells. `n_threads` is capped at
Numba's thread limit (`NUMBA_NUM_THREADS`); `res.params["seconds_per_sweep"]`
records the measured cost.

```python
res = schlda.fit_hlda(adata, "cell_type", n_activity_topics=2, layer="counts",
                      output_dir="results/pbmc", seed=0, n_threads=4)
```

### Theta prior

A cell `i` of type `c` may use the topics `A(c)`: the topics of `c` and of
every group containing `c`. Its topic usage has the Dirichlet prior

```
theta_i ~ Dirichlet(alpha_k : k in A(c))
alpha_k = alpha_identity / n(c)   for each of the n(c) topics of type c
alpha_k = alpha_activity          for a group's topic
```

Every concentration — `alpha_identity`, `alpha_activity`, a value set with
`g.alpha(topic, value)` or passed in the `alpha=` dict — takes one of three
forms:

| form | `alpha_ik` for cell `i`, topic `k` |
| --- | --- |
| a float `a` | `a` for every cell |
| one value per cell (array, or Series indexed by `adata.obs_names`) | that cell's value |
| `schlda.depth(kappa)` | `kappa * L_i`, with `L_i` the cell's total counts |

`depth(kappa)` scales the prior with sequencing depth: the pseudo-counts are
the fraction `kappa` of each cell's own counts (`depth(0.01)` is 1 % of every
cell's counts), so the prior carries the same share of every cell's evidence
whatever its depth, instead of a fixed number of pseudo-counts that is strong
for a shallow cell and negligible for a deep one. When every cell has the
same depth, `depth(kappa)` is the fixed value `kappa * L`.

`alpha=` overrides single entries: a dict keyed by node or topic, with values
of the same three forms (a cell type's value is split over its topics like
`alpha_identity`, a topic name sets that topic), or a full array — `(K,)` per
topic or `(n_cells, K)` per cell and topic, in the order of `res.topics` — that
replaces every other setting.

```python
schlda.fit_hlda(adata, "cell_type", 1, alpha_identity=5.0)
schlda.fit_hlda(adata, "cell_type", 1, alpha_identity=schlda.depth(0.01))
schlda.fit_hlda(adata, "cell_type", 2, alpha={"CD4T": 3.0, "V2": schlda.depth(0.001)})
g = schlda.geometry(adata, "cell_type", 2).alpha("V2", 0.01)
schlda.fit_hlda(adata, "cell_type", geometry=g)
```

`res.topics` / `topics.csv` record the `alpha` of every topic (`depth(kappa)`
written as such, a per-cell value as `per-cell`); a prior that varies by cell
is also written out as `alpha_cells.csv`, and `res.params["prior"]` is
`"fixed"` or `"per_cell"`.

The sampler is collapsed: each token's topic is drawn from

```
p(z = k | ...)  ∝  (alpha_beta + A_gk) / (G * alpha_beta + B_k)  *  (alpha_k + D_ik)
```

with `A` the gene x topic counts, `B` their column sums, `D` the cell x topic
counts (all excluding the token), and `theta`, `beta` the posterior means over
the retained sweeps.

## Topic geometry

A geometry is a set of **nodes**, each carrying one or more topics. The cell
types are nodes; a **group** is a node made of several cell types. A cell may
use the topics of its own type and of every group its type belongs to.
`n_activity_topics=m` is the geometry with one topic per type and a group `V`
of every type carrying `V1 … Vm`. Anything else starts from
`schlda.geometry(adata, "cell_type")` — one topic per type, no groups — and is
edited in place:

| method | effect |
| --- | --- |
| `g.topics(node, n)` / `g.topics(node, [names])` | the node carries `n` topics (`<node>` if 1, else `<node>_1 … <node>_n`) or the names given; `0` removes them |
| `g.group(name, members, n=1)` | a group of cell types (`"all"` or a list of two or more) carrying `n` topics, or `[names]` |
| `g.drop(topic)` | remove one topic |
| `g.alpha(topic, value)` | that topic's concentration (default by node, see [Theta prior](#theta-prior)) |

```python
g = schlda.geometry(adata, "cell_type")
g.topics("CD4T", 2)                                  # CD4T_1, CD4T_2
g.group("V", "all", 2)                               # V_1, V_2 for every cell
g.group("lymphoid", ["B", "CD4T", "CD8T", "NK"])     # one topic, lymphocytes only
g.group("T", ["CD4T", "CD8T"])                       # nested: a sub-branch
print(g.describe())
# B     (1,231 cells): B, V_1, V_2, lymphoid
# CD4T  (5,091 cells): CD4T_1, CD4T_2, V_1, V_2, lymphoid, T
# CD8T  (812 cells): CD8T, V_1, V_2, lymphoid, T
# NK    (402 cells): NK, V_1, V_2, lymphoid
# Mono  (2,940 cells): Mono, V_1, V_2
# groups:
#   V (V_1, V_2) <- all
#   lymphoid (lymphoid) <- B; CD4T; CD8T; NK
#   T (T) <- CD4T; CD8T

res = schlda.fit_hlda(adata, "cell_type", geometry=g, layer="counts",
                      output_dir="results/pbmc_geo", seed=0)
```

Groups may nest or overlap, so any DAG over cell types is expressible, each
level written out. A group of a single cell type is not a group — give that
type another topic with `g.topics(type, 2)`. `res.identity_topics` are the
topics carried by cell types and `res.activity_topics` those carried by
groups; the plots colour them accordingly.

The same geometry as a table, `g.to_frame()`:

| topic | node | cell_types | alpha |
| --- | --- | --- | --- |
| B | B | B | |
| CD4T_1 | CD4T | CD4T | |
| CD4T_2 | CD4T | CD4T | |
| CD8T | CD8T | CD8T | |
| NK | NK | NK | |
| Mono | Mono | Mono | |
| V_1 | V | all | |
| V_2 | V | all | |
| lymphoid | lymphoid | B; CD4T; CD8T; NK | |
| T | T | CD4T; CD8T | |

An empty `alpha` means the node's default. The table can be edited as a
DataFrame or CSV and passed as `geometry=` to `fit_hlda` (or
`table=` to `schlda.geometry`), and every fit writes its own as `topics.csv`
with the `alpha` values used. The dict shortcuts build the same thing:

```python
schlda.geometry(adata, "cell_type",
                topics={"CD4T": 2},
                groups={"V": ("all", 2), "lymphoid": ["B", "CD4T", "CD8T", "NK"]})
```

Topic order is the cell types' topics (annotation order), then each group's
in the order added. Topic and node names must be unique, every cell type needs
at least one topic it may use, and the geometry's cell types must be those of
the column the fit conditions on (`res.allowed` is the derived cell type x
topic mask).

`examples/vignette.py` fits a lineage tree — lymphoid and myeloid branches,
each with a shared topic, plus a topic for every cell (3c) — and a label
hiding two cell states, which a second topic for that label resolves (3e).

## Plot

```python
schlda.structure_plot(res, save="structure.pdf")
schlda.theta_heatmap(res, cells_per_group=10, save="heatmap.pdf")
```

`structure_plot` draws one stacked bar per cell, grouped by cell type.
`theta_heatmap` draws topics as rows and cells as columns, optionally averaging
runs of `cells_per_group` cells into meta-cells to stay legible on large
datasets. Both return `(fig, ax)`.

### Any theta, any method

Both plots take a cells x topics matrix from **any** method — an
`HLDAResult`, a fitted AnnData, a DataFrame, or a plain array — plus the
per-cell annotation. Only two things need saying for a non-HLDA fit: which
columns are activity topics, and the labels.

```python
theta_nmf = pd.read_csv("NMF_theta.csv", index_col=0)     # cells x topics
schlda.structure_plot(
    theta_nmf,
    cell_types=adata.obs["cell_type"],       # any array / Series / Categorical
    activity_topics=["topic_3"],             # names or integer positions
)
```

`activity_topics` defaults to the fit's own topic types (for an `HLDAResult`
or AnnData), else to columns named `V1`, `V2`, …. It sets the colours (muted
for identity, strong for activity), the stacking order, and the sort. Rows that
do not sum to 1 (NMF loadings, fastTopics `L`) are normalised with a warning.

### Cell ordering within a cell type

Let `A` be the activity topics, `a_i = Σ_{k∈A} θ_ik` the total activity weight
of cell `i`, and `k*_i = argmax_{k∈A} θ_ik` its dominant activity topic.

| `sort_by` | order within each group |
| --- | --- |
| `"dominant"` (default) | blocks by `k*_i` in `A` order, then a quiet block with `max_k θ_ik < min_activity` (0.05); each block by `a_i` descending |
| `"activity"` | `a_i` descending |
| `"embed"` | first principal component of centred `log θ`, signed so `a_i` rises left to right |
| a topic name | that topic descending |
| `None` | input order |

With one activity topic `"dominant"` and `"activity"` coincide. With several,
`"dominant"` keeps V1-driven and V2-driven cells in separate contiguous blocks
instead of interleaving them.

### Comparing fits on the same cells

Sorting each model by its own theta makes every panel look equally smooth and
hides where the models disagree. Compute the order once and pass it to every
plot, so column `i` is the same cell in every panel:

```python
order = schlda.cell_order(res)                         # from the HLDA fit

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, (name, theta, act) in zip(axes, [
    ("scHLDA", res.theta, res.activity_topics),
    ("LDA", theta_lda, ["topic_5"]),
    ("NMF", theta_nmf, ["topic_3"]),
]):
    schlda.structure_plot(theta, cell_types=labels, activity_topics=act,
                          order=order, ax=ax, title=name, legend=ax is axes[0])
```

`cell_order` takes the same inputs as the plots and returns integer row
positions; `order=` also accepts cell names (`theta.index[order]`) when the
thetas do not share a row order. Cells absent from `order` are not drawn, which
doubles as a way to subsample.

Other knobs: `topic_order`, `group_order`, `colors`, `max_cells_per_group`,
`min_activity`, `ax`, `figsize`, `title`, `legend`.

## Example

`examples/vignette.py` simulates five cell types in two lineage branches,
each branch with a shared program, plus a stress program active in every type;
fits the default geometry and the lineage tree (3c); shows the fixed,
depth-scaled and array forms of the theta prior (3d); then two situations met
in practice — a label that hides two cell states, resolved by giving that
label a second topic (3e), and an activity program that modulates the cell's
own program, where the identity prior decides what the shared topic holds and
`depth(kappa)` applies it evenly across sequencing depths (3f) — and draws the
fits next to the simulation truth on shared cell orders. It runs in a few
minutes:

```bash
python examples/vignette.py vignette_out
```

## Cost

The sampler works on one token per UMI, so memory scales with total counts, not
with cells x genes: roughly 16 bytes per UMI. A dataset with 50M total counts
needs about 800 MB for the token arrays. Runtime scales with tokens x sweeps.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

See `CITATION.cff`.
