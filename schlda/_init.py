"""Initial topic assignments for the Gibbs chain.

Two schemes are available:

``random``
    Every token draws uniformly from the topics its cell may use.

``data_driven``
    Each node with identity topics gets a provisional profile from the mean expression of its cells. A cell
    whose profile correlates well with it starts out heavily loaded on its identity topic(s); a cell that
    correlates poorly starts with more of its tokens on the topics it shares with other cells (group and
    activity topics). The correlation is mapped through a sigmoid centred at 0.7, so the split moves
    smoothly between roughly 20% and 90% identity. Tokens kept on identity are spread uniformly over the
    cell's identity sub-topics.
"""

from __future__ import annotations

import numpy as np


def identity_betas(counts, identity_ints, n_identity):
    """Mean expression profile per identity, normalised to a distribution."""
    n_genes = counts.shape[1]
    betas = np.zeros((n_identity, n_genes), dtype=np.float64)
    for k in range(n_identity):
        mask = identity_ints == k
        if not mask.any():
            betas[k] = 1.0 / n_genes
            continue
        mean_expr = np.asarray(counts[mask].mean(axis=0)).ravel() + 1e-10
        betas[k] = mean_expr / mean_expr.sum()
    return betas


def identity_correlations(counts, identity_ints, betas):
    """Pearson correlation of each cell's profile with its own identity topic.

    Correlation is scale invariant, so this is computed on raw counts and needs
    no per-cell normalisation. Everything is expressed in terms of row sums and
    row sums of squares, which keeps it vectorised over the sparse matrix.
    """
    n_cells, n_genes = counts.shape
    corr = np.zeros(n_cells, dtype=np.float64)

    counts = counts.tocsr()
    sum_x = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    sum_x2 = np.asarray(counts.multiply(counts).sum(axis=1)).ravel().astype(np.float64)

    for k in range(betas.shape[0]):
        mask = identity_ints == k
        if not mask.any():
            continue
        b = betas[k]
        sum_b = b.sum()
        sum_b2 = (b * b).sum()

        dot = counts[mask] @ b
        cov = dot - sum_x[mask] * sum_b / n_genes
        var_x = sum_x2[mask] - sum_x[mask] ** 2 / n_genes
        var_b = sum_b2 - sum_b**2 / n_genes

        denom = np.sqrt(np.maximum(var_x, 0.0) * max(var_b, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(denom > 0, cov / denom, 0.0)
        corr[mask] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    return corr


def identity_weights(corr):
    """Map correlation to a starting identity-topic share via a sigmoid."""
    clamped = np.clip(corr, 0.0, 1.0)
    return 0.2 + 0.7 / (1.0 + np.exp(-10.0 * (clamped - 0.7)))


def _pick(topics, offsets, nodes, rng):
    """One uniform draw per token from its node's row of a CSR topic table."""
    if nodes.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    start = offsets[nodes]
    width = offsets[nodes + 1] - start
    w = np.unique(width)
    if w.size == 1:
        j = np.zeros(nodes.shape[0], dtype=np.int64) if w[0] == 1 else rng.integers(0, w[0], size=nodes.shape[0])
    else:
        j = np.minimum(np.floor(rng.random(nodes.shape[0]) * width).astype(np.int64), width - 1)
    return topics[start + j].astype(np.int32)


def initial_assignments(counts, cell_node, cell_idx, node_rep, geometry, scheme, rng, verbose=True):
    """Return the initial topic of every token (``node_rep`` = node id per token)."""
    n_tokens = node_rep.shape[0]
    valid_t, valid_o = geometry.csr()
    is_id = geometry.identity_mask()

    if scheme == "random":
        z0 = _pick(valid_t, valid_o, node_rep, rng)
        if verbose:
            _report(z0, geometry)
        return z0

    if scheme != "data_driven":
        raise ValueError(f"Unknown init scheme {scheme!r}; expected 'data_driven' or 'random'.")

    n_nodes = len(geometry.nodes)
    betas = identity_betas(counts, cell_node, n_nodes)
    corr = identity_correlations(counts, cell_node, betas)
    weights = identity_weights(corr)

    if verbose:
        print(
            f"[schlda] init: correlation with own identity "
            f"mean={corr.mean():.3f} range=[{corr.min():.3f}, {corr.max():.3f}]"
        )
        print(
            f"[schlda] init: starting identity share "
            f"mean={weights.mean():.3f} range=[{weights.min():.3f}, {weights.max():.3f}]"
        )

    own_t, own_o = geometry.csr(is_id)
    sh_t, sh_o = geometry.csr(~is_id)
    own_w = np.diff(own_o)[node_rep]
    sh_w = np.diff(sh_o)[node_rep]
    move = np.zeros(n_tokens, dtype=bool)
    if (sh_w > 0).any():
        move = rng.random(n_tokens) > weights[cell_idx]
    move = (move | (own_w == 0)) & (sh_w > 0)
    z0 = np.empty(n_tokens, dtype=np.int32)
    if move.any():
        z0[move] = _pick(sh_t, sh_o, node_rep[move], rng)
    z0[~move] = _pick(own_t, own_o, node_rep[~move], rng)

    if verbose:
        _report(z0, geometry)
    return z0


def _report(z0, geometry):
    n_tokens = z0.shape[0]
    is_id = geometry.identity_mask()
    share = np.bincount(z0, minlength=geometry.n_topics) / n_tokens
    n_id = float(share[is_id].sum())
    print(f"[schlda] init: {n_tokens:,} tokens -> {n_id:.1%} identity, {1 - n_id:.1%} shared")
    for name, s, i in zip(geometry.topic_names, share, is_id):
        if not i:
            print(f"[schlda] init:   {name}: {s:.1%}")
