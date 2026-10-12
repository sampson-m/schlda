# schlda

Hierarchical LDA for annotated single-cell count data.

Every annotated cell type gets its own **identity topic**, and a configurable
number of **activity topics** (`V1 … Vn`) are shared across all cells. A cell's
counts can only be explained by its own identity topic or by the shared activity
topics, so identity topics stay pinned to the annotation and the activity topics
are left to pick up programs that cut across cell types — cell cycle, interferon
response, and so on.

Other geometries — several identity topics for a type, or programs shared by
only some types — are built with `schlda.geometry` (see
[Topic geometry](#topic-geometry)).

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
res.topics             # topic / kind / cell_types / alpha
res.activity_topics    # ['V1', 'V2']
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
topics.csv              the geometry: topic, kind, cell_types, alpha
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

Every token starts on a topic drawn uniformly from those its cell may use
(`init="random"`, the only scheme). The chain, not the starting point, decides
the fit; `burn_in` is what absorbs the start. Compare two or three `seed`s on
a new dataset or geometry — a split that is found by some seeds and not others
is a sign the data admit more than one solution.

### Sampler settings

| argument | default | meaning |
| --- | --- | --- |
| `n_loops` | 3000 | total Gibbs sweeps |
| `burn_in` | 1000 | sweeps discarded |
| `thin` | 30 | sweeps between kept draws |
| `n_threads` | 1 | threads per sweep (see [Threads](#threads)) |
| `alpha_beta` | 0.1 | Dirichlet concentration on gene distributions |
| `alpha_identity` | 1.0 | concentration on a cell's identity (see [Theta prior](#theta-prior)) |
| `alpha_activity` | 0.1 | concentration on each activity topic |
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

A cell `i` of type `c` may use the topics `A(c)` its type is allowed (its
identity topic(s) and the activity topics it shares). Its topic usage has the
Dirichlet prior

```
theta_i ~ Dirichlet(alpha_k : k in A(c))
alpha_k = alpha_identity / n_identity(c)   for an identity topic of type c
alpha_k = alpha_activity                   for an activity topic
```

so the prior is part of the geometry: `res.topics` / `topics.csv` list the
`alpha` of every topic, and `g.alpha(name, value)` overrides one. With the
defaults a cell of a type with one identity topic and `m` activity topics has
prior mean identity share `1 / (1 + 0.1 m)`; `alpha_identity = 1` is the
smallest value whose density stays bounded at zero identity, so the prior
itself never pushes a cell off its identity topic. The prior counts as
`alpha` pseudo-tokens against the cell's `L_i` real ones, so at typical depths
it only matters for cells whose data leave the split undecided.

```python
schlda.fit_hlda(adata, "cell_type", 1, alpha_identity=5.0)              # lean harder on identity
g = schlda.geometry(adata, "cell_type", 2).alpha("V2", 0.01)             # one sparse activity topic
schlda.fit_hlda(adata, "cell_type", geometry=g)
```

Nothing is truncated: `theta` is still the pooled frequency of the retained
assignments, and the prior acts only through each token's conditional,

```
p(z = k | ...)  ∝  (alpha_beta + A_gk) / (G * alpha_beta + B_k)  *  (alpha_k + D_ik)
```

## Topic geometry

`n_activity_topics=m` gives every cell type one identity topic, named after the
type, plus `V1 … Vm` shared by every cell. Anything else starts from
`schlda.geometry(adata, "cell_type")` — one identity topic per cell type and
nothing shared — and is edited in place:

| method | effect |
| --- | --- |
| `g.split(type, n)` / `g.split(type, [names])` | `n` identity topics for `type` (`<type>_1 … <type>_n`), or the names given; `0` removes them |
| `g.share(name)` / `g.share(name, [types])` | an activity topic used by every type, or by the listed types only |
| `g.drop(name)` | remove a topic |
| `g.alpha(name, value)` | that topic's concentration (default by kind, see [Theta prior](#theta-prior)) |

```python
g = schlda.geometry(adata, "cell_type")
g.split("CD4T", 2)                                   # CD4T_1, CD4T_2
g.share("V1")                                        # every cell
g.share("lymphoid", ["B", "CD4T", "CD8T"])           # lymphocytes only
print(g.describe())
# B     (1,231 cells): B, V1, lymphoid
# CD4T  (5,091 cells): CD4T_1, CD4T_2, V1, lymphoid
# CD8T  (812 cells): CD8T, V1, lymphoid
# Mono  (2,940 cells): Mono, V1
# shared topics:
#   V1 <- all
#   lymphoid <- B, CD4T, CD8T

res = schlda.fit_hlda(adata, "cell_type", geometry=g, layer="counts",
                      output_dir="results/pbmc_geo", seed=0)
```

Every topic is `identity` (owned by one cell type) or `activity` (shared by
the types listed, possibly a single one). An activity topic restricted to one
type is still an activity topic: it takes `alpha_activity`, not a share of the
type's identity concentration.

The geometry is a table, `g.to_frame()`:

| topic | kind | cell_types | alpha |
| --- | --- | --- | --- |
| B | identity | B | |
| CD4T_1 | identity | CD4T | |
| CD4T_2 | identity | CD4T | |
| CD8T | identity | CD8T | |
| Mono | identity | Mono | |
| V1 | activity | all | |
| lymphoid | activity | B; CD4T; CD8T | |

An empty `alpha` means the kind's default. The table can be edited as a
DataFrame or CSV and passed as `geometry=` (`g.to_csv(path)`;
`schlda.geometry(adata, "cell_type", topics=path)` reads it back), and every
fit writes its own as `topics.csv` with the `alpha` values used. The dict
shortcuts `identity_topics={type: n or [names]}` and
`shared_topics={name: "all" or [types]}` build the same thing:
`schlda.geometry(adata, "cell_type", identity_topics={"CD4T": 2}, shared_topics={"V1": "all", "lymphoid": [...]})`.

Topic order is identity topics (cell type order, each type's together) then
activity topics in the order added. Topic names must be unique, every type
needs at least one topic, and the geometry's cell types must be those of the
column the fit conditions on (`res.allowed` is the derived cell type x topic
mask).

A program used at a near-constant share by every cell of the types that share
it cannot be split between that shared topic and their identity topics (the
likelihood is flat along the split). Shared topics are identified by
cell-to-cell variation in their use.

### Hierarchies

Sharing is per cell type, so a lineage tree is a set of nested shared topics:
a cell may use its own identity topic(s) plus every shared topic whose list
contains its type.

```python
g = schlda.geometry(adata, "cell_type")
g.share("V1")                                      # root: every cell
g.share("lymphoid", ["B", "CD4T", "CD8T", "NK"])    # one branch
g.share("T", ["CD4T", "CD8T"])                     # a sub-branch
g.share("myeloid", ["CD14Mono", "FCGR3AMono", "DC"])
g.split("CD4T", 2)                                 # leaves can still be split
print(g.describe())
# CD4T  (5,091 cells): CD4T_1, CD4T_2, V1, lymphoid, T
```

Groups may overlap (a type can sit in several), so any DAG over cell types is
expressible. Two things are not: a topic shared between sub-topics rather than
types (`AC_1` and `MES1_1` only), and inheritance — each level is written out.

### Choosing a geometry

There is no rule for the number of identity topics per type; the diagnostic
that has worked is to fit the default geometry and look at what the activity
topics became. For each activity topic `V`, compute its mean usage per cell
type and its similarity to each identity profile:

```python
res = schlda.fit_hlda(adata, "cell_type", 2, layer="counts", output_dir="results/default", seed=0)
B, th = res.beta, res.theta
use = th.groupby(adata.obs["cell_type"].values)[res.activity_topics].mean()      # types x activity topics
cos2 = (B[res.identity_topics].T @ B[res.activity_topics]) ** 2 / \
       np.outer((B[res.identity_topics] ** 2).sum(), (B[res.activity_topics] ** 2).sum())
```

| what you see | what it means | geometry change |
| --- | --- | --- |
| `V` is used by one or two types and looks like one of their identity profiles (high `cos2`) | `V` is a substate of that type, not a cross-type program | `g.split(type, 2)` and refit; `V` is freed for the program |
| `V` is used by one lineage only (several related types) | a lineage program | `g.share(name, [those types])` so it cannot leak into the others |
| cells of a type have identity share near 0 and load on such a `V` | same as the first row, seen from the cells | same |
| `V` is used by many types and resembles none of them | a cross-type program — what activity topics are for | keep |

After a split, check that the new identity topics did not take the program
(`cos2` of `type_1`, `type_2` with the program's genes should stay low); if one
did, the program is specific to that type and belongs to it as an identity
topic, or needs `share(name, [type])` to be kept apart from the identity.

### Cells with low identity usage

Some cells of a type can end up with identity share near zero: their counts
are explained by an activity topic alone. This is expected behaviour, not a
sampler failure, and it is decided by the likelihood. When a cell type hides
two expression states under one label, the identity topic can hold only one
profile; a shared topic is free to take the other, and every cell in that
state then fits the shared topic better than its own identity.

The theta prior cannot undo this at normal depth. With `alpha_identity` as
pseudo-counts against the `L_i` tokens the cell actually has, the posterior
share of an identity topic is roughly

```
E[theta_id | data] ≈ (alpha_identity + n_id) / (sum(alpha) + L_i)
```

where `n_id` is the number of tokens the likelihood puts on identity; with
`L_i` in the hundreds and `n_id ≈ 0`, no reasonable `alpha_identity` moves it
off zero (`alpha_identity = 1` only keeps the prior from *pushing* cells to
zero). Scaling `alpha` with `L_i` would move it, but is no longer a prior on
the generative model and was removed in 0.6.0.

The remedy is the geometry: give the type a second identity topic so both
states are identity, and the shared topic is freed for the cross-type program.
To check for it:

```python
own = sum(res.theta[t] * (adata.obs["cell_type"] == c).values
          for t, c in zip(res.topics.topic, res.topics.cell_types) if t in res.identity_topics)
(own < 0.01).groupby(adata.obs["cell_type"].values).mean()      # fraction of cells per type
```

A type with a non-trivial fraction here, whose activity topic resembles one
of its profiles ([Choosing a geometry](#choosing-a-geometry)), is a candidate
for `g.split(type, 2)`. Section 3d of `examples/vignette.py` simulates
a type with a hidden substate, shows the default geometry losing about half of
that type's cells to an activity topic, and the split recovering them with the
cross-type program intact.

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
instead of interleaving them. A small example with two activity topics and
`min_activity = 0.05`:

| cell | θ_V1 | θ_V2 | block | a_i |
| --- | --- | --- | --- | --- |
| c1 | 0.10 | 0.30 | V2 | 0.40 |
| c2 | 0.02 | 0.01 | quiet | 0.03 |
| c3 | 0.25 | 0.05 | V1 | 0.30 |

Order: c3, c1, c2.

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

`examples/vignette.py` simulates four cell types plus one planted activity
program, fits the model in under a minute, checks that the program was
recovered, draws the fit next to the simulation truth on a shared cell order,
and (section 3d) shows a hidden substate costing a type its identity and a
split geometry restoring it:

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
