"""Initial topic assignments for the Gibbs chain: every token draws uniformly from the topics its cell may use."""

from __future__ import annotations

import numpy as np


def _pick(topics, offsets, nodes, rng):
    """One uniform draw per token from its cell type's row of a CSR topic table."""
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


def initial_assignments(node_rep, geometry, rng, verbose=True):
    """Return the initial topic of every token (``node_rep`` = cell type index per token)."""
    valid_t, valid_o = geometry.csr()
    z0 = _pick(valid_t, valid_o, node_rep, rng)
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
