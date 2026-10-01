# Changelog

## 0.5.0 — 2026-10-01

- Topic geometry. A cell may use every topic on its path in a topic tree:
  its identity topic(s), the topics of the groups above it, and the activity
  topics on the root. New `fit_hlda` arguments:
  - `n_identity_topics`: int or `{cell_type: k}`, identity sub-topics
    (`type_1 .. type_k`);
  - `cell_type_key` may be a list of obs columns, coarse to fine; each coarse
    value becomes a group, and a cell missing a finer label sits at its coarser
    node;
  - `groups={group: [members]}` builds the same hierarchy without a column
    (members are types or groups, any depth); `n_group_topics` (int or dict,
    default 1) sets the topics each group owns;
  - `allowed`: a cell type x topic boolean DataFrame for any other geometry.
- `topic_geometry(...)` previews the geometry without fitting
  (`describe()`, `allowed_frame()`).
- Identity concentration (`alpha_c_identity` / `prior_scale_identity`) is
  split evenly over a type's identity sub-topics; group topics take `alpha_c`.
- `topic_info.csv` gains a `node` column and topic type `group`;
  `allowed_topics.csv` is written; `HLDAResult` gains `geometry`, `allowed`,
  `group_topics`, `topics_of()` and `identity_share()`.
- The sampler kernel is unchanged. The v0.4.0 geometry reproduces v0.4.0 fits
  exactly at a fixed seed (`tests/data/golden_v040.npz`).

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
