# Changelog

## 0.5.0 — 2026-10-01

- Custom topic geometry: `fit_hlda(..., allowed=mask)` takes a cell type x
  topic boolean DataFrame saying which topics each type may use (identity
  sub-topics, programs shared by a subset of types, any other pattern). A
  topic allowed for exactly one type is an identity topic; any other is an
  activity topic. `n_activity_topics` (the default geometry) is unchanged.
- `allowed_topics(adata, key, m)` returns the default mask to edit;
  `topic_geometry(...)` previews a geometry without fitting (`describe()`,
  `allowed_frame()`).
- The identity concentration is split evenly over a type's identity topics.
- `allowed_topics.csv` is written with every fit; `HLDAResult` gains
  `geometry`, `allowed` and `identity_share()`.
- The sampler kernel is unchanged. The default geometry reproduces v0.4.0 fits
  exactly at a fixed seed (`tests/data/golden_v040.npz`), also when passed as
  the equivalent `allowed` mask.

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
