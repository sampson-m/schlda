# Changelog

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
