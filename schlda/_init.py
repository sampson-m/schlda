"""Initial topic assignments for the Gibbs chain.

Two schemes are available:

``random``
    Every token draws uniformly from the topics its cell's identity may use.

``data_driven``
    Each identity gets a provisional topic from the mean expression of its
    cells. A cell whose profile correlates well with that topic starts out
    heavily loaded on its identity topic; a cell that correlates poorly starts
    with more of its tokens pushed onto the shared activity topics. The
    correlation is mapped through a sigmoid centred at 0.7, so the split moves
    smoothly between roughly 20% and 90% identity.
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


def initial_assignments(
    counts,
    identity_ints,
    cell_idx,
    ident_rep,
    n_identity,
    n_activity,
    scheme,
    rng,
    verbose=True,
):
    """Return the initial topic of every token.

    Identity topic ``k`` has topic index ``k``; activity topic ``Vi`` has index
    ``n_identity + i - 1``.
    """
    n_tokens = ident_rep.shape[0]

    if scheme == "random":
        if n_activity == 0:
            z0 = ident_rep.astype(np.int32)
        else:
            pick = rng.integers(0, n_activity + 1, size=n_tokens)
            z0 = np.where(pick == 0, ident_rep, n_identity + pick - 1).astype(np.int32)
        if verbose:
            _report(z0, n_identity, n_activity)
        return z0

    if scheme != "data_driven":
        raise ValueError(f"Unknown init scheme {scheme!r}; expected 'data_driven' or 'random'.")

    betas = identity_betas(counts, identity_ints, n_identity)
    corr = identity_correlations(counts, identity_ints, betas)
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

    z0 = ident_rep.astype(np.int32)
    if n_activity > 0:
        to_activity = rng.random(n_tokens) > weights[cell_idx]
        n_move = int(to_activity.sum())
        if n_move:
            z0[to_activity] = (
                n_identity + rng.integers(0, n_activity, size=n_move)
            ).astype(np.int32)

    if verbose:
        _report(z0, n_identity, n_activity)
    return z0


def _report(z0, n_identity, n_activity):
    n_tokens = z0.shape[0]
    n_id = int((z0 < n_identity).sum())
    print(
        f"[schlda] init: {n_tokens:,} tokens -> "
        f"{n_id / n_tokens:.1%} identity, {1 - n_id / n_tokens:.1%} activity"
    )
    for i in range(n_activity):
        share = float((z0 == n_identity + i).sum()) / n_tokens
        print(f"[schlda] init:   V{i + 1}: {share:.1%}")
