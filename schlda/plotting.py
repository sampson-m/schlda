"""Structure plot, topic-usage heatmap and cell ordering for fitted thetas."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

__all__ = ["structure_plot", "theta_heatmap", "cell_order"]

# Identity topics take the muted end of the palette; activity topics get strong
# colours so they read as the thing being looked for.
_IDENTITY_COLORS = [
    "#4575b4", "#1a9850", "#8c6bb1", "#01665e", "#bf812d", "#5aae61",
    "#3690c0", "#66c2a5", "#b8860b", "#7fbc41", "#35978f", "#762a83",
]
_ACTIVITY_COLORS = ["#d73027", "#fc8d59", "#e7298a", "#984ea3", "#ff7f00"]
_EPS = 1e-12


def structure_plot(
    theta,
    cell_types=None,
    *,
    activity_topics=None,
    key="schlda",
    topic_order=None,
    group_order=None,
    sort_by="auto",
    order=None,
    min_activity=0.05,
    colors=None,
    max_cells_per_group=None,
    ax=None,
    figsize=None,
    title=None,
    legend=True,
    save=None,
    dpi=150,
):
    """Stacked bar structure plot: one bar per cell, split by topic proportion.

    Parameters
    ----------
    theta
        Cells x topics matrix from any method: an :class:`~schlda.HLDAResult`,
        a fitted AnnData, a DataFrame, or a plain array. Rows that do not sum
        to 1 (e.g. NMF loadings) are normalised with a warning.
    cell_types
        Per-cell annotation used to group the bars. May be an array, a Series,
        or the name of an ``adata.obs`` column when ``theta`` is an AnnData.
        Taken from the fit automatically for an ``HLDAResult`` or AnnData. Pass
        ``False`` to draw every cell in one ungrouped block.
    activity_topics
        Which topics are activity topics: column names or integer positions.
        Defaults to the fit's own topic types, else to columns named ``V1``,
        ``V2``, ... Drives colours, the default ``topic_order`` and sorting.
    key
        ``uns``/``obsm`` key to read when ``theta`` is an AnnData.
    topic_order
        Topics to draw, bottom to top. Defaults to identity topics first, then
        activity topics.
    group_order
        Order of the cell type groups along the x axis. Defaults to first
        appearance for a categorical annotation, else sorted.
    sort_by
        How cells are ordered inside each group; see :func:`cell_order`.
    order
        Explicit cell order (row positions or cell names) that overrides
        ``sort_by``. Cells absent from ``order`` are not drawn. Compute it once
        with :func:`cell_order` and pass it to every model's plot so the same
        cell sits in the same column across models.
    min_activity
        Threshold below which a cell counts as quiet under ``sort_by="dominant"``.
    colors
        Mapping from topic name to colour, or a list in ``topic_order``.
    max_cells_per_group
        Randomly subsample groups larger than this. Only affects rendering
        cost; the plot stays proportional because bars are equal width.
    ax, figsize, title, legend
        Standard matplotlib controls. ``ax=None`` creates a new figure.
    save, dpi
        Path to write the figure to, and its resolution.

    Returns
    -------
    (fig, ax)
    """
    theta_df, groups, activity = _resolve(theta, cell_types, key, activity_topics)
    topic_order = _resolve_topic_order(theta_df, topic_order, activity)
    theta_df = theta_df[topic_order]

    idx, boundaries, group_labels = _order_cells(
        theta_df, groups, group_order, sort_by, order, activity, min_activity,
        max_cells_per_group,
    )
    values = theta_df.to_numpy()[idx].T  # topics x cells

    palette = _resolve_colors(colors, topic_order, activity)

    if ax is None:
        width = min(18.0, max(7.0, 1.6 * max(len(group_labels), 1) + 3.0))
        fig, ax = plt.subplots(figsize=figsize or (width, 4.2))
    else:
        fig = ax.figure

    ax.stackplot(
        np.arange(values.shape[1]),
        values,
        colors=[palette[t] for t in topic_order],
        labels=topic_order,
        linewidth=0,
    )

    prev = 0
    ticks = []
    for edge in boundaries:
        if edge < values.shape[1]:
            ax.axvline(edge - 0.5, color="white", linewidth=1.5, zorder=3)
        ticks.append((prev + edge - 1) / 2)
        prev = edge

    ax.set_xlim(0, max(values.shape[1] - 1, 1))
    ax.set_ylim(0, 1)
    # Group names go below the axis, rotated: narrow groups would otherwise
    # collide with their neighbours above the bars.
    if group_labels:
        ax.set_xticks(ticks)
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    ax.set_ylabel("Topic proportion", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    if legend:
        handles = [Patch(facecolor=palette[t], label=t) for t in topic_order]
        ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=8, frameon=False,
        )

    fig.tight_layout()
    _save(fig, save, dpi)
    return fig, ax


def theta_heatmap(
    theta,
    cell_types=None,
    *,
    activity_topics=None,
    key="schlda",
    topic_order=None,
    group_order=None,
    sort_by=None,
    order=None,
    min_activity=0.05,
    cells_per_group=1,
    cmap="viridis",
    vmax=None,
    ax=None,
    figsize=None,
    title=None,
    colorbar=True,
    save=None,
    dpi=150,
):
    """Topic usage heatmap: topics as rows, cells (or meta-cells) as columns.

    Same inputs as :func:`structure_plot`, plus:

    Parameters
    ----------
    cells_per_group
        Average consecutive runs of this many cells within each cell type into
        one column. Use it to keep very large datasets legible; 1 draws every
        cell.
    sort_by
        Defaults to ``None`` here — the heatmap is usually read for the raw
        pattern within a group, not a sorted gradient. Takes the same values as
        in :func:`structure_plot`.
    vmax
        Upper limit of the colour scale. Defaults to the largest value plotted,
        which is usually more readable than pinning it to 1.
    """
    theta_df, groups, activity = _resolve(theta, cell_types, key, activity_topics)
    topic_order = _resolve_topic_order(theta_df, topic_order, activity)
    theta_df = theta_df[topic_order]

    idx, boundaries, group_labels = _order_cells(
        theta_df, groups, group_order, sort_by, order, activity, min_activity, None
    )
    values = theta_df.to_numpy()[idx]

    cells_per_group = int(cells_per_group)
    if cells_per_group > 1:
        values, boundaries = _meta_cells(values, boundaries, cells_per_group)

    if ax is None:
        width = min(18.0, max(7.0, 1.5 * max(len(group_labels), 1) + 3.0))
        height = 0.35 * len(topic_order) + 2.0
        fig, ax = plt.subplots(figsize=figsize or (width, height))
    else:
        fig = ax.figure

    im = ax.imshow(
        values.T, aspect="auto", cmap=cmap, vmin=0,
        vmax=vmax if vmax is not None else max(values.max(), _EPS),
        interpolation="nearest",
    )

    prev = 0
    ticks, tick_labels = [], []
    for label, edge in zip(group_labels, boundaries):
        if edge < values.shape[0]:
            ax.axvline(edge - 0.5, color="white", linewidth=1.0, zorder=3)
        ticks.append((prev + edge) / 2 - 0.5)
        tick_labels.append(label)
        prev = edge

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(topic_order)))
    ax.set_yticklabels(topic_order, fontsize=9)
    if title:
        ax.set_title(title, fontsize=12)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label("Topic proportion", fontsize=9)

    fig.tight_layout()
    _save(fig, save, dpi)
    return fig, ax


def cell_order(
    theta,
    cell_types=None,
    *,
    activity_topics=None,
    key="schlda",
    group_order=None,
    sort_by="auto",
    min_activity=0.05,
):
    """Row positions of cells grouped by cell type and sorted within each group.

    Let ``A`` be the activity topics, ``a_i = sum_{k in A} theta_ik`` and
    ``k*_i = argmax_{k in A} theta_ik``. ``sort_by`` may be:

    ``"dominant"`` (the default under ``"auto"``)
        Blocks of cells by dominant activity topic in ``A`` order, then a
        quiet block of cells with ``max_k theta_ik < min_activity``; within
        each block sorted by ``a_i`` descending. Identical to ``"activity"``
        when there is one activity topic.
    ``"activity"``
        ``a_i`` descending.
    ``"embed"``
        First principal component of centred ``log theta`` within the group,
        signed so that ``a_i`` increases left to right.
    a topic name
        That topic descending.
    ``None``
        Input order.

    With no activity topics, ``"auto"`` sorts by the group's own identity
    topic when a topic of that name exists, else keeps the input order.

    Returns
    -------
    numpy.ndarray of int row positions, usable as ``order=`` in the plots.
    """
    theta_df, groups, activity = _resolve(theta, cell_types, key, activity_topics)
    idx, _, _ = _order_cells(
        theta_df, groups, group_order, sort_by, None, activity, min_activity, None
    )
    return idx


# ── input handling ─────────────────────────────────────────────────────────

def _resolve(theta, cell_types, key, activity_topics):
    """Normalise the accepted inputs to (theta DataFrame, labels, activity topics)."""
    activity = None
    groups = None

    if hasattr(theta, "theta") and hasattr(theta, "topics"):  # HLDAResult
        activity = theta.activity_topics
        groups = theta.cell_types
        theta_df = theta.theta
    elif hasattr(theta, "obsm"):  # AnnData
        obsm_key = f"{key}_theta"
        if obsm_key not in theta.obsm:
            raise KeyError(
                f"adata.obsm[{obsm_key!r}] not found — fit the model first, or "
                f"pass the theta DataFrame directly."
            )
        info = theta.uns.get(key, {})
        names = list(info.get("topic_names", []))
        values = np.asarray(theta.obsm[obsm_key])
        if len(names) != values.shape[1]:
            names = [f"Topic_{i}" for i in range(values.shape[1])]
        types = list(info.get("topic_types", []))
        if len(types) == len(names):
            activity = [n for n, t in zip(names, types) if t == "activity"]
        theta_df = pd.DataFrame(values, index=theta.obs_names, columns=names)
        if isinstance(cell_types, str):
            groups = theta.obs[cell_types]
            cell_types = None
        elif cell_types is None:
            obs_key = info.get("params", {}).get("cell_type_key")
            if obs_key is not None and obs_key in theta.obs:
                groups = theta.obs[obs_key]
    elif isinstance(theta, pd.DataFrame):
        theta_df = theta
    else:
        values = np.asarray(theta)
        if values.ndim != 2:
            raise ValueError(f"theta must be 2-D, got shape {values.shape}.")
        theta_df = pd.DataFrame(
            values, columns=[f"Topic_{i}" for i in range(values.shape[1])]
        )

    theta_df = _normalise_rows(theta_df)

    if cell_types is False:
        groups = None
    elif cell_types is not None:
        if isinstance(cell_types, str):
            raise TypeError(
                "cell_types may only be a column name when theta is an AnnData; "
                "otherwise pass the labels themselves."
            )
        groups = cell_types

    if groups is not None:
        # Reindex onto theta without going through a plain ndarray, which would
        # drop a Categorical's category order and silently re-sort the groups.
        values = groups.values if isinstance(groups, pd.Series) else np.asarray(groups)
        if len(values) != len(theta_df):
            raise ValueError(
                f"cell_types has {len(values)} entries but theta has "
                f"{len(theta_df)} rows."
            )
        groups = pd.Series(values, index=theta_df.index)

    if activity_topics is not None:
        activity = [
            theta_df.columns[t] if isinstance(t, (int, np.integer)) else t
            for t in activity_topics
        ]
        missing = [t for t in activity if t not in theta_df.columns]
        if missing:
            raise KeyError(f"activity_topics not in theta: {missing}")
    elif activity is None:
        activity = [c for c in theta_df.columns if _is_activity_name(c)]
    return theta_df, groups, list(activity)


def _normalise_rows(theta_df):
    values = theta_df.to_numpy(dtype=float)
    if (values < 0).any():
        raise ValueError("theta contains negative values.")
    sums = values.sum(axis=1)
    if np.allclose(sums, 1.0, atol=1e-3):
        return theta_df
    warnings.warn("theta rows do not sum to 1; normalising each row.", stacklevel=3)
    values = values / np.clip(sums, _EPS, None)[:, None]
    return pd.DataFrame(values, index=theta_df.index, columns=theta_df.columns)


def _is_activity_name(name):
    name = str(name)
    return len(name) > 1 and name[0] == "V" and name[1:].isdigit()


def _resolve_topic_order(theta_df, topic_order, activity):
    if topic_order is not None:
        missing = [t for t in topic_order if t not in theta_df.columns]
        if missing:
            raise KeyError(f"Topics not in theta: {missing}")
        return list(topic_order)
    identity = [c for c in theta_df.columns if c not in set(activity)]
    return identity + [c for c in theta_df.columns if c in set(activity)]


def _resolve_colors(colors, topic_order, activity):
    if isinstance(colors, dict):
        missing = [t for t in topic_order if t not in colors]
        if missing:
            raise KeyError(f"No colour given for topics: {missing}")
        return colors
    if colors is not None:
        colors = list(colors)
        if len(colors) < len(topic_order):
            raise ValueError(
                f"Got {len(colors)} colours for {len(topic_order)} topics."
            )
        return dict(zip(topic_order, colors))

    palette, n_id, n_act = {}, 0, 0
    for topic in topic_order:
        if topic in set(activity):
            palette[topic] = _ACTIVITY_COLORS[n_act % len(_ACTIVITY_COLORS)]
            n_act += 1
        else:
            palette[topic] = _IDENTITY_COLORS[n_id % len(_IDENTITY_COLORS)]
            n_id += 1
    return palette


# ── cell ordering ──────────────────────────────────────────────────────────

def _order_cells(theta_df, groups, group_order, sort_by, order, activity,
                 min_activity, max_per_group):
    """Row indices grouped by cell type, plus group right edges and labels."""
    n = len(theta_df)
    rank = _resolve_order(theta_df, order)

    def sort(idx, group):
        if rank is not None:
            idx = idx[~np.isnan(rank[idx])]
            return idx[np.argsort(rank[idx], kind="stable")]
        return _sort_within(theta_df, idx, sort_by, activity, group, min_activity)

    if groups is None:
        return sort(np.arange(n), None), [n], []

    if group_order is None:
        if isinstance(groups.dtype, pd.CategoricalDtype):
            used = set(groups.unique())
            group_order = [c for c in groups.cat.categories if c in used]
        else:
            group_order = sorted(groups.unique())
    else:
        unknown = [g for g in group_order if g not in set(groups.unique())]
        if unknown:
            raise ValueError(f"group_order names absent from cell_types: {unknown}")

    values = np.asarray(groups)
    rng = np.random.default_rng(0)
    out, boundaries, labels = [], [], []
    for group in group_order:
        idx = np.flatnonzero(values == group)
        if max_per_group is not None and idx.size > max_per_group:
            idx = np.sort(rng.choice(idx, size=max_per_group, replace=False))
        idx = sort(idx, group)
        if idx.size == 0:
            continue
        out.extend(idx.tolist())
        boundaries.append(len(out))
        labels.append(str(group))
    return np.asarray(out, dtype=int), boundaries, labels


def _resolve_order(theta_df, order):
    """Per-row rank from an explicit order (positions or cell names); NaN = drop."""
    if order is None:
        return None
    order = np.asarray(order)
    if order.dtype.kind in "iu":
        pos = order
    else:
        pos = theta_df.index.get_indexer(order)
        if (pos < 0).any():
            raise KeyError(f"order names not in theta: {list(order[pos < 0][:5])}")
    if pos.min() < 0 or pos.max() >= len(theta_df):
        raise IndexError("order positions out of range.")
    rank = np.full(len(theta_df), np.nan)
    rank[pos] = np.arange(len(pos))
    return rank


def _sort_within(theta_df, idx, sort_by, activity, group, min_activity):
    """Sort one group's rows; see cell_order for the rules."""
    if sort_by is None or idx.size == 0:
        return idx
    if sort_by == "auto":
        sort_by = "dominant" if activity else None
        if sort_by is None:
            if group is not None and str(group) in theta_df.columns:
                sort_by = str(group)
            else:
                return idx
    block = theta_df.iloc[idx]
    if sort_by in ("dominant", "activity", "embed") and not activity:
        raise ValueError(f"sort_by={sort_by!r} needs activity topics.")
    if sort_by in ("dominant", "activity"):
        act = block[activity].to_numpy()
        total = act.sum(axis=1)
        if sort_by == "activity" or len(activity) == 1:
            return idx[np.argsort(-total, kind="stable")]
        dom = np.where(act.max(axis=1) < min_activity, len(activity), act.argmax(axis=1))
        return idx[np.lexsort((-total, dom))]
    if sort_by == "embed":
        x = np.log(block.to_numpy() + 1e-6)
        x = x - x.mean(axis=0)
        if len(idx) < 2:
            return idx
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        score = x @ vt[0]
        total = block[activity].to_numpy().sum(axis=1)
        if total.std() > 0 and np.corrcoef(score, total)[0, 1] < 0:
            score = -score
        return idx[np.argsort(score, kind="stable")]
    if sort_by not in theta_df.columns:
        raise KeyError(f"sort_by={sort_by!r} is not a topic in theta.")
    return idx[np.argsort(-block[sort_by].to_numpy(), kind="stable")]


def _meta_cells(values, boundaries, cells_per_group):
    """Average consecutive runs of cells inside each group into single columns."""
    out, new_boundaries, start = [], [], 0
    for edge in boundaries:
        block = values[start:edge]
        for chunk_start in range(0, len(block), cells_per_group):
            chunk = block[chunk_start:chunk_start + cells_per_group]
            if len(chunk):
                out.append(chunk.mean(axis=0))
        new_boundaries.append(len(out))
        start = edge
    return np.asarray(out), new_boundaries


def _save(fig, save, dpi):
    if save is None:
        return
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
