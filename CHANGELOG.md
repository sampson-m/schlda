# Changelog

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
