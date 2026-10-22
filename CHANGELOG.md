# Changelog

## 1.0.0 — 2026-10-22

- Two kinds of topic only: one identity topic per annotated cell type, and
  activity topics shared by the cell types that may use them. The default is
  still `n_activity_topics=m` (`V1 .. Vm` for every cell); any other sharing
  is given as `activity=` — a CSV / DataFrame with columns `cell_type,
  activity_topics` (`;`-separated, blank for none) or `topic, cell_types`
  (`all` or `;`-separated), or a dict in either orientation. Removed:
  `schlda.geometry(...).topics / group / drop / alpha`, identity sub-topics,
  nested groups, `geometry=`, `TopicGeometry` (now `Geometry`, read-only).
- Two prior values only: `alpha_identity` (default 1) on every identity topic
  and `alpha_activity` (default 0.1) on every activity topic. Removed:
  `depth(kappa)`, per-cell arrays, the `alpha=` dict / array, `alpha_cells.csv`.
- A finished run writes `theta.csv`, `beta.csv` (were `HLDA_theta.csv`,
  `HLDA_beta.csv`), `topics.csv` (columns `topic, kind, cell_types, alpha`),
  `params.json`, `log.txt` (everything printed, ending `exit 0` or the
  traceback and `exit 1`), `theta_heatmap.png`, `activity_theta_heatmap.png`
  and `structure_plot.png` (`plots=False` skips them).
- Posterior means are accumulated during the run; the kept draws are written
  to `samples/` only with `save_samples=True` (default off). `samples/` gains
  `topic_types` in the sidecar plus `allowed.npy` and `cell_node.npy`.
- `track_likelihood=True` records the collapsed joint log-likelihood
  `log P(w, z)` every `report_every` sweeps in `likelihood.csv`;
  `schlda.likelihood_curve(output_dir)` draws `likelihood.png` from it, or
  computes it from `samples/` when the run did not track it.
- `init` is accepted only as `"random"` (kept for old call sites).
- Default fits reproduce 0.9.0 exactly (golden tests unchanged).
- README and vignette rewritten around these features.

## 0.9.0 — 2026-10-14

- Theta prior: every concentration (`alpha_identity`, `alpha_activity`,
  `g.alpha(topic, value)`, the values of the new `alpha=` dict) is a float,
  one value per cell (array or Series indexed by `obs_names`), or
  `schlda.depth(kappa)` — `kappa * L_i`, pseudo-counts equal to the fraction
  `kappa` of each cell's counts, so the prior carries the same share of every
  cell's evidence. This restores the depth-scaled identity prior removed in
  0.6.0 (`prior_scale_identity=kappa` is `depth(kappa)`), now available on
  every concentration.
- `fit_hlda(..., alpha=)`: a dict `{node or topic: value}` applied on top of
  the defaults (a cell type's value split over its topics, a topic name set
  directly), or a full `(K,)` / `(n_cells, K)` array in `res.topics` order
  (a DataFrame aligned by column) that replaces everything.
- `topics.csv` writes `depth(kappa)` / `per-cell` in the `alpha` column and
  reads `depth(kappa)` back; a prior that varies by cell is written to
  `alpha_cells.csv`; `res.params` gains `alpha` and `prior`
  (`"fixed"` / `"per_cell"`). `TopicGeometry.resolve_alpha` returns a list of
  specs rather than an array.
- Defaults unchanged: fits with float priors reproduce 0.8.0 exactly.
- Vignette rewritten: a lineage tree (lymphoid / myeloid branches with a
  shared topic each, plus a stress topic), the prior forms, a label hiding two
  cell states (resolved by a second topic for the label, not by the prior),
  and an activity program that modulates identity (where the identity prior
  decides what the shared topic holds, and `depth(kappa)` applies it evenly
  across sequencing depths). The README passage on identity collapse is gone.

## 0.8.0 — 2026-10-13

- Geometry is built from nodes, not labelled topics. A node is a cell type or
  a group of cell types; `g.topics(node, n)` sets how many topics a node
  carries and `g.group(name, members, n=1)` adds a group (`"all"` or a list
  of two or more types). `split` / `share` are gone: `g.split(t, n)` is
  `g.topics(t, n)`, `g.share(name, types)` is `g.group(name, types)`. A group
  of one cell type is rejected — give the type another topic instead.
- The `kind` column is gone. The table is `topic, node, cell_types, alpha`;
  "identity" and "activity" survive only as views (`res.identity_topics` =
  topics of cell types, `res.activity_topics` = topics of groups) used by the
  plots. Tables written by earlier versions still read.
- `schlda.geometry` shortcuts are `topics={node: n or [names]}` and
  `groups={name: members or (members, n or [names])}`; the table argument is
  `table=` (was `topics=`). `n_activity_topics=m` is a group `V` carrying
  `V1 .. Vm`.
- Removed the deprecated `alpha_c` / `alpha_c_identity` arguments.
- README rewritten as a description of the package; the diagnostics for
  choosing a geometry moved to `examples/vignette.py` section 3e.

## 0.7.0 — 2026-10-12

- Removed the data-driven initialiser (`init="data_driven"`): it seeded the
  shared topics with the tokens of a type's poorly-fitting cells and started a
  type's identity sub-topics identical, so it worked against `split`
  geometries. Every chain now starts from `init="random"` (the only accepted
  value; `"data_driven"` raises). This is how every fit in the paper was run.
## 0.6.0 — 2026-10-09

- One geometry object. `schlda.geometry(adata, "cell_type")` returns the
  default geometry (one identity topic per cell type, nothing shared), edited
  in place with `split`, `share`, `drop` and `alpha`; `fit_hlda(..., geometry=g)`
  fits it. `n_activity_topics=m` remains the shortcut for `V1 .. Vm`.
- The geometry is a table (`g.to_frame()` / `g.to_csv()`; columns `topic,
  kind, cell_types, alpha`) that `geometry=` also accepts as a DataFrame or CSV
  path; every fit writes it as `topics.csv` (replacing `topic_info.csv` and
  `allowed_topics.csv`). `res.topics` is this table.
- The theta prior lives in the geometry: `alpha_identity` (default **1.0**,
  split evenly over a type's identity topics) and `alpha_activity` (default
  0.1) fill the `alpha` column by kind, and `g.alpha(name, value)` overrides
  one topic. Default changed from 0.1 on every topic: pass `alpha_identity=0.1`
  to reproduce 0.4/0.5 fits.
- Removed: `prior_scale_identity` (depth-scaled prior), per-cell `alpha_c` /
  `alpha_c_identity`, `theta_prior.csv`, `topic_geometry`, `fit_hlda`'s
  `identity_topics` / `shared_topics` (now arguments of `schlda.geometry`),
  `HLDAResult.identity_share`. Scalar `alpha_c` / `alpha_c_identity` still
  work with a `DeprecationWarning`.
- Kinds are `identity` and `activity` only; an activity topic restricted to one
  cell type is activity and takes `alpha_activity`.
- README: "Cells with low identity usage" — why identity theta reaches zero,
  why the prior cannot fix it, and the split geometry that does; vignette
  section 3d demonstrates it on a simulated hidden substate. `init="random"`
  is recommended for split geometries (the data-driven initialiser seeds the
  shared topics with the substate).

## 0.5.0 — 2026-10-06

- Custom topic geometry: `fit_hlda(..., identity_topics=, shared_topics=)`.
  `identity_topics={type: n or [names]}` gives a type several identity topics
  (or none); `shared_topics={name: "all" or [types]}` lists the activity topics
  and the types that may use each, replacing `n_activity_topics`. Topic kinds
  are stated by the argument, never inferred.
- `topic_geometry(...)` previews a geometry without fitting (`describe()`,
  `allowed_frame()`).
- The identity concentration is split evenly over a type's identity topics.
- `allowed_topics.csv` (the derived cell type x topic mask) is written with
  every fit; `HLDAResult` gains `geometry`, `allowed` and `identity_share()`;
  `params["geometry"]` records `identity_topics` and `shared_topics`.
- The sampler kernel is unchanged. The default geometry reproduces v0.4.0 fits
  exactly at a fixed seed (`tests/data/golden_v040.npz`), also when given as
  the equivalent `identity_topics` / `shared_topics`.

## 0.4.0 — 2026-09-29

- One fitting function. `fit_hlda` now runs the blocked sampler: `n_threads=1`
  (default) is the exact collapsed Gibbs sampler, `n_threads > 1` sweeps cell
  blocks in parallel. The old serial kernel is removed; at a fixed seed
  `fit_hlda` draws differ from 0.3.x (same sampler, different random stream)
  and match 0.3.x `fit_hlda_parallel` exactly.
- `fit_hlda_parallel` is removed: pass `n_threads` to `fit_hlda`.
- `alpha_c` and `alpha_c_identity` accept a float or one value per cell (array
  in `adata` row order, or a Series aligned on `adata.obs_names`).
  `prior_scale_identity` is available in `fit_hlda`. Per-cell or depth-scaled
  concentrations are written to `theta_prior.csv`.
- The symmetric depth-scaled prior (`prior_scale`, `identity_share`) is
  removed.
- `params` always records `seed`, `n_threads`, `n_blocks`, `prior` and
  `seconds_per_sweep`.

## 0.3.1 — 2026-09-29

- `prior_scale_identity` (kappa): identity-only depth-scaled theta prior,
  alpha_i,identity = kappa * L_i while the activity topics keep `alpha_c`.
  Pulls each cell toward its identity topic without putting prior mass on the
  activity topics. Mutually exclusive with `prior_scale` and
  `alpha_c_identity`.

## 0.3.0 — 2026-09-24

- New `fit_hlda_parallel`: blocked collapsed Gibbs sampler (`n_threads` cells
  blocks swept in parallel, exact at 1; Newman et al. 2009 scheme above that)
  with the same result object, files and AnnData keys as `fit_hlda`.
- Depth-scaled theta prior: `prior_scale` (kappa) and `identity_share` give each
  cell the concentration kappa * L_i split identity_share : 1 - identity_share
  between its identity topic and the activity topics. `prior_concentrations`
  builds the cells x topics matrix.
- `fit_hlda`, `_gibbs` and `_init` are unchanged.

## 0.2.0 — 2026-09-10

- Package renamed `hlda` → `schlda`; AnnData keys are now `schlda_theta`,
  `schlda_beta`, `uns["schlda"]`.
- `structure_plot` / `theta_heatmap` take a theta from any method:
  new `activity_topics=` (names or positions); rows not summing to 1 are
  normalised with a warning.
- New `order=` argument (row positions or cell names) and `cell_order()` helper
  so several fits can be drawn on one shared cell order.
- Default within-group sort is now `"dominant"`: blocks by dominant activity
  topic, then a quiet block, each sorted by total activity. Unchanged for fits
  with a single activity topic. `"activity"` keeps the old rule; `"embed"`
  added.
- Fixed group tick labels sitting half a bar to the right.
- Vignette draws the fit next to the simulation truth on a shared order.
- Tests (`pytest`) and CI added.

## 0.1.0

- Initial release: `fit_hlda`, `HLDAResult`, `structure_plot`, `theta_heatmap`.
