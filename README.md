# schlda

Hierarchical LDA for annotated single-cell count data.

Every annotated cell type carries its own **identity topic**, and **activity
topics** are shared by the cell types that may use them — by default a
configurable number of topics (`V1 … Vm`) shared by every cell. A cell's counts
can only be explained by its own type's topic or by the activity topics its
type is given, so each identity topic stays pinned to the annotation and the
activity topics are left to pick up programs that cut across cell types — cell
cycle, interferon response, a lineage program, and so on.

- [Install](#install)
- [Fit](#fit) · [Output directory](#output-directory) · [Sampler settings](#sampler-settings) · [Threads](#threads) · [Theta prior](#theta-prior)
- [Activity topics](#activity-topics)
- [Samples and the likelihood trace](#samples-and-the-likelihood-trace)
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
    n_activity_topics=2,    # activity topics V1, V2 shared by every cell
    layer="counts",         # raw counts; defaults to adata.X
    output_dir="results/pbmc",
    seed=0,
)

res.theta              # cells x topics, rows sum to 1
res.beta               # genes x topics, columns sum to 1
res.topics             # topic / kind / cell_types / alpha
res.identity_topics    # ['B', 'CD4T', ...]  the cell types' own topics
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

A finished run leaves in `output_dir`:

```
theta.csv                   cells x topics posterior mean
beta.csv                    genes x topics posterior mean
topics.csv                  topic, kind, cell_types, alpha
params.json                 every setting of the run, the seed, seconds per sweep
log.txt                     everything the run printed, ending "exit 0" (or the traceback and "exit 1")
theta_heatmap.png           topics x cells
activity_theta_heatmap.png  the activity topics only
structure_plot.png          one stacked bar per cell
likelihood.csv              only with track_likelihood=True (see below)
samples/                    only with save_samples=True (see below)
```

`topics.csv` can be passed back as `activity=` to refit the same geometry.
Omitting `output_dir` writes to `./schlda_output` and warns; `plots=False`
skips the three figures.

### Sampler settings

| argument | default | meaning |
| --- | --- | --- |
| `n_loops` | 3000 | total Gibbs sweeps |
| `burn_in` | 1000 | sweeps discarded |
| `thin` | 30 | sweeps between kept draws |
| `n_threads` | 1 | threads per sweep (see [Threads](#threads)) |
| `alpha_beta` | 0.1 | Dirichlet concentration on gene distributions |
| `alpha_identity` | 1.0 | concentration on every identity topic |
| `alpha_activity` | 0.1 | concentration on every activity topic |
| `save_samples` | `False` | write the kept draws to `samples/` |
| `track_likelihood` | `False` | record log P(w, z) every `report_every` sweeps |
| `plots` | `True` | write the three figures |
| `seed` | `None` | seeds init and the chain |
| `report_every` | 100 | sweeps between progress lines |

Every token starts on a topic drawn uniformly from those its cell may use;
`seed` fixes the draw. A run repeats exactly for a fixed `seed` and
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

A cell `i` of type `c` may use the topics `A(c)`: the topic of `c` and the
activity topics `c` is given. Its topic usage has the Dirichlet prior

```
theta_i ~ Dirichlet(alpha_k : k in A(c))
alpha_k = alpha_identity   for the identity topic of c
alpha_k = alpha_activity   for each activity topic
```

Each is one positive number applied to all topics of its kind; `res.topics` /
`topics.csv` record the value on every topic.

```python
schlda.fit_hlda(adata, "cell_type", 1, alpha_identity=5.0, alpha_activity=0.05)
```

The sampler is collapsed: each token's topic is drawn from

```
p(z = k | ...)  ∝  (alpha_beta + A_gk) / (G * alpha_beta + B_k)  *  (alpha_k + D_ik)
```

with `A` the gene x topic counts, `B` their column sums, `D` the cell x topic
counts (all excluding the token), and `theta`, `beta` the posterior means over
the retained sweeps.

## Activity topics

`n_activity_topics=m` gives `V1 … Vm`, each shared by every cell type. To say
which activity topics each cell type may use, pass `activity=` instead: a CSV
file (or DataFrame) with one row per cell type and the topics it may use,
separated by `;` (blank for none):

```
cell_type,activity_topics
B,stress; lymphoid
CD4T,stress; lymphoid; T
CD8T,stress; lymphoid; T
NK,stress; lymphoid
Mono,stress
```

```python
res = schlda.fit_hlda(adata, "cell_type", activity="activity.csv",
                      layer="counts", output_dir="results/pbmc_tree", seed=0)
```

Every topic named anywhere in the file becomes an activity topic, shared by
exactly the cell types that list it; a cell type missing from the file has no
activity topic. The transposed table — one row per topic, with the cell types
that share it or `all` — is also accepted, and is the form every fit writes
out as `topics.csv`:

```
topic,cell_types
stress,all
lymphoid,B; CD4T; CD8T; NK
T,CD4T; CD8T
```

Either orientation may be given as a dict instead of a file
(`{"B": ["stress", "lymphoid"], ...}` or `{"stress": "all", "T": ["CD4T",
"CD8T"]}`). Topic names must differ from cell type names; the cell types must
be those of the annotation column. `schlda.geometry(adata, "cell_type",
activity=...)` builds the geometry without fitting, for a look first:

```python
g = schlda.geometry(adata, "cell_type", activity="activity.csv")
print(g.describe())
# B     (1,231 cells): B, stress, lymphoid
# CD4T  (5,091 cells): CD4T, stress, lymphoid, T
# CD8T  (812 cells): CD8T, stress, lymphoid, T
# NK    (402 cells): NK, stress, lymphoid
# Mono  (2,940 cells): Mono, stress
g.to_frame()          # the topics table; g.allowed_frame() the cell type x topic mask
```

Topic order is the identity topics (annotation order), then the activity
topics in the order the file names them.

## Samples and the likelihood trace

By default only the posterior means are kept. `save_samples=True` also writes
the kept draws of the count tables to `samples/`:

```
samples/A_chain.memmap      (n_save, n_genes, n_topics) int32 gene-topic counts
samples/D_chain.memmap      (n_save, n_cells, n_topics) int32 cell-topic counts
samples/samples_meta.json   shapes, topic names and types, run parameters
samples/allowed.npy, cell_node.npy   the cell type x topic mask and each cell's type
```

```python
import json, numpy as np
meta = json.load(open("results/pbmc/samples/samples_meta.json"))
A = np.memmap("results/pbmc/samples/A_chain.memmap", dtype="int32",
              mode="r", shape=tuple(meta["A_chain"]["shape"]))
```

`track_likelihood=True` records, every `report_every` sweeps from the first,
the collapsed joint log-likelihood of the tokens and their current assignments
with `theta` and `beta` integrated out,

```
log P(w, z) = sum_k [ lnΓ(G η) − lnΓ(G η + B_k) + sum_g (lnΓ(η + A_gk) − lnΓ(η)) ]
            + sum_i [ lnΓ(a_i) − lnΓ(a_i + L_i) + sum_k (lnΓ(alpha_k + D_ik) − lnΓ(alpha_k)) ]
```

(`η` = `alpha_beta`, `L_i` the cell's counts, `a_i` the sum of `alpha_k` over
the topics cell `i` may use), to `likelihood.csv` as `sweep, loglik_joint`. It
is the quantity the sampler moves on and is for watching one chain settle; it
is not the marginal likelihood `P(w)` and is not comparable across runs with
different priors or geometries. The cost is a histogram of the count tables,
negligible next to a sweep.

`likelihood_curve` draws it:

```python
schlda.likelihood_curve("results/pbmc")    # -> likelihood.csv, likelihood.png
```

It reads `likelihood.csv` when the run tracked it, and otherwise computes the
same quantity from `samples/` — then only the kept, post-burn-in draws are
available, so the curve starts after `burn_in`. A run with neither raises.

## Plot

```python
schlda.structure_plot(res, save="structure.pdf")
schlda.theta_heatmap(res, cells_per_group=10, save="heatmap.pdf")
```

`structure_plot` draws one stacked bar per cell, grouped by cell type.
`theta_heatmap` draws topics as rows and cells as columns, optionally averaging
runs of `cells_per_group` cells into meta-cells to stay legible on large
datasets. Both return `(fig, ax)`. Every fit writes both (and the activity
topics alone) to its output directory.

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
fits the default geometry and the lineage geometry from a flat file (3c);
saves samples and the likelihood trace (3d); and draws the fits next to the
simulation truth on shared cell orders. It runs in a couple of minutes:

```bash
python examples/vignette.py vignette_out
```

## Cost

The sampler works on one token per UMI, so memory scales with total counts, not
with cells x genes: roughly 16 bytes per UMI. A dataset with 50M total counts
needs about 800 MB for the token arrays. Runtime scales with tokens x sweeps;
the posterior means are accumulated as the chain runs, so a fit ends as soon as
the last sweep does.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

See `CITATION.cff`.
