# schlda

Hierarchical LDA for annotated single-cell count data.

Every annotated cell type gets its own **identity topic**, and a configurable
number of **activity topics** (`V1 … Vn`) are shared across all cells. A cell's
counts can only be explained by its own identity topic or by the shared activity
topics, so identity topics stay pinned to the annotation and the activity topics
are left to pick up programs that cut across cell types — cell cycle, interferon
response, and so on.

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
res.topics             # topic_id / topic_name / topic_type
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
topic_info.csv          topic_id, topic_name, topic_type
samples/A_chain.memmap  (n_save, n_genes, n_topics) int32 gene-topic counts
samples/D_chain.memmap  (n_save, n_cells, n_topics) int32 cell-topic counts
samples/samples_meta.json   shapes, topic names, run parameters
```

Omitting `output_dir` writes to `./schlda_output` and warns. Reload the chains
for convergence diagnostics using the shapes in the sidecar:

```python
import json, numpy as np
meta = json.load(open("results/pbmc/samples/samples_meta.json"))
A = np.memmap("results/pbmc/samples/A_chain.memmap", dtype="int32",
              mode="r", shape=tuple(meta["A_chain"]["shape"]))
```

### Initialisation

`init="data_driven"` (default) builds a provisional topic per identity from that
type's mean expression, then splits each cell's tokens between its identity
topic and the activity topics according to how well the cell correlates with its
own type. Cells that fit their annotation start out nearly pure; cells that fit
it poorly start with more weight on activity topics.

`init="random"` assigns each token uniformly over the topics its cell is allowed
to use.

### Sampler settings

| argument | default | meaning |
| --- | --- | --- |
| `n_loops` | 3000 | total Gibbs sweeps |
| `burn_in` | 1000 | sweeps discarded |
| `thin` | 30 | sweeps between kept draws |
| `alpha_beta` | 0.1 | Dirichlet concentration on gene distributions |
| `alpha_c` | 0.1 | Dirichlet concentration on activity topic usage |
| `alpha_c_identity` | `alpha_c` | concentration on identity topic usage |
| `seed` | `None` | seeds init and the chain |

Pass `seed` for reproducibility: the compiled kernel keeps its own RNG state, so
without it neither the initialisation nor the chain repeats.

## Parallel sampler and identity prior

`fit_hlda_parallel` takes every argument of `fit_hlda`, returns the same
`HLDAResult`, writes the same files and annotates the AnnData under the same
keys. It adds two things.

```python
res = schlda.fit_hlda_parallel(
    adata, "cell_type", n_activity_topics=2, layer="counts",
    output_dir="results/pbmc_par", seed=0,
    n_threads=4,            # cells are split into 4 blocks, one thread each
    prior_scale=0.1,        # depth-scaled theta prior (see below); None = fixed alpha_c
    identity_share=0.7,     # prior mean of each cell's identity share
)
```

### Threads

With `n_threads=1` the sampler is the exact collapsed Gibbs sampler. With
`n_threads=P` the cells are split into `P` contiguous blocks balanced by token
count; each block is swept by one thread that updates its own cells' topic
counts in place and works from a private copy of the gene-topic table, and the
per-block changes to the gene-topic table are merged after every sweep. This is
the distributed scheme of Newman et al. (2009): the within-sweep dependence
between blocks is dropped, which is negligible when every block holds many
cells. Every block reseeds from `(seed, sweep, block)`, so a run repeats
exactly for a fixed `seed` and `n_threads`; changing `n_threads` changes the
draws. `n_threads` is capped at Numba's thread limit (`NUMBA_NUM_THREADS`).

### Depth-scaled theta prior

The fixed prior of `fit_hlda` puts the same concentration `alpha_c` on every
topic, so with deep cells the prior is a few pseudo-counts against hundreds or
thousands of tokens and has no say in how much of a cell stays in its identity
topic. `prior_scale` ties the prior to each cell's depth. With `L_i` the number
of tokens (UMIs) in cell `i`, `m` activity topics, `s = identity_share` and
`kappa = prior_scale`,

```
alpha_i,identity = s * kappa * L_i
alpha_i,activity = (1 - s) * kappa * L_i / m        (each activity topic)
```

so the prior mean of the identity share is `s` at every depth and the prior
strength is `kappa * L_i` pseudo-counts. Example: `L_i = 1000`, `m = 2`,
`s = 0.7`, `kappa = 0.1` gives `alpha_i = (70, 15, 15)` on (identity, V1, V2).
A priori the identity share is `Beta(s * kappa * L_i, (1 - s) * kappa * L_i)`;
at `kappa = 0.1` and `L_i = 600` the prior probability of an identity share
below 0.4 is about 1e-6, so cells leave their identity topic only when their
counts insist. Nothing is truncated: the estimator is still the pooled
empirical frequency of the retained assignments, and the prior acts through the
token conditional, which gains a cell index on `alpha`. `prior_scale=None`
(default) reproduces the fixed prior; `prior_scale` and `alpha_c` /
`alpha_c_identity` cannot be combined.

| argument | default | meaning |
| --- | --- | --- |
| `n_threads` | 1 | threads, and cell blocks, per sweep |
| `prior_scale` | `None` | `kappa`: prior pseudo-counts per cell as a fraction of its tokens |
| `identity_share` | 0.7 | prior mean of the identity share under `prior_scale` |
| `report_every` | 100 | sweeps between progress lines |

`res.params["seconds_per_sweep"]` records the measured cost.

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
recovered, and draws the fit next to the simulation truth on a shared cell
order:

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
