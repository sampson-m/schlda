"""Blocked collapsed Gibbs sweep (ported from hlda 0.1.0).

Cells are split into contiguous blocks; each block is swept by one thread with
a private copy of the gene-topic table ``A`` and the topic totals ``B`` while
the cell-topic table ``D`` (disjoint rows per block) is updated in place. The
per-block deltas are merged after the sweep. One block is the exact collapsed
Gibbs sampler; several blocks give the approximate distributed scheme of
Newman et al. (2009). Every block reseeds its thread's generator from
``seeds[p]``, so a run is reproducible for a fixed ``seed`` and block count.

``alpha_c`` is cells x topics: the concentration may differ per cell (the
depth-scaled prior of ``fit_hlda_parallel``).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def cell_offsets_from_idx(cell_idx, n_cells):
    """Token ranges per cell for tokens stored in cell order (as ``expand_tokens`` emits them)."""
    return np.searchsorted(cell_idx, np.arange(n_cells + 1), side="left").astype(np.int64)


def make_blocks(cell_offsets, n_blocks):
    """Cell boundaries (length <= n_blocks + 1) of contiguous blocks balanced by token count."""
    n_cells = cell_offsets.shape[0] - 1
    n_blocks = int(max(1, min(n_blocks, n_cells)))
    if n_blocks == 1:
        return np.array([0, n_cells], dtype=np.int64)
    targets = cell_offsets[-1] * np.arange(1, n_blocks) / n_blocks
    cuts = np.searchsorted(cell_offsets, targets)
    return np.unique(np.concatenate(([0], cuts, [n_cells]))).astype(np.int64)


def block_seeds(seed, sweep, n_blocks):
    ss = np.random.SeedSequence([int(seed), int(sweep)])
    return ss.generate_state(n_blocks, dtype=np.uint32).astype(np.int64)


def build_valid(n_identity, n_activity):
    """CSR-style permitted topics: identity k followed by every activity topic."""
    width = 1 + n_activity
    valid_topics = np.empty(n_identity * width, dtype=np.int32)
    valid_offsets = np.arange(0, (n_identity + 1) * width, width, dtype=np.int32)
    for k in range(n_identity):
        valid_topics[valid_offsets[k]] = k
        valid_topics[valid_offsets[k] + 1:valid_offsets[k] + width] = n_identity + np.arange(n_activity)
    return valid_topics, valid_offsets


@njit(parallel=True, cache=True)
def gibbs_sweep(cell_offsets, gene_idx, z, cell_ident, valid_topics, valid_offsets,
                A, B, D, alpha_beta, alpha_c, block_bounds, seeds, dA, dB):
    n_blocks = block_bounds.shape[0] - 1
    G = A.shape[0]
    K = A.shape[1]
    G_abeta = G * alpha_beta
    max_width = 1
    for k in range(valid_offsets.shape[0] - 1):
        width = valid_offsets[k + 1] - valid_offsets[k]
        if width > max_width:
            max_width = width

    for p in prange(n_blocks):
        np.random.seed(seeds[p])
        A_loc = A.copy()
        B_loc = B.copy()
        probs = np.empty(max_width, np.float64)
        for c in range(block_bounds[p], block_bounds[p + 1]):
            start = valid_offsets[cell_ident[c]]
            m = valid_offsets[cell_ident[c] + 1] - start
            for idx in range(cell_offsets[c], cell_offsets[c + 1]):
                g = gene_idx[idx]
                old = z[idx]
                A_loc[g, old] -= 1
                B_loc[old] -= 1
                D[c, old] -= 1
                total = 0.0
                for j in range(m):
                    t = valid_topics[start + j]
                    pr = ((alpha_beta + A_loc[g, t]) / (G_abeta + B_loc[t])) * (alpha_c[c, t] + D[c, t])
                    probs[j] = pr
                    total += pr
                r = np.random.rand() * total
                cum = 0.0
                new = valid_topics[start + m - 1]
                for j in range(m):
                    cum += probs[j]
                    if r < cum:
                        new = valid_topics[start + j]
                        break
                A_loc[g, new] += 1
                B_loc[new] += 1
                D[c, new] += 1
                z[idx] = new
        for g in range(G):
            for t in range(K):
                dA[p, g, t] = A_loc[g, t] - A[g, t]
        for t in range(K):
            dB[p, t] = B_loc[t] - B[t]

    for p in range(n_blocks):
        for g in range(G):
            for t in range(K):
                A[g, t] += dA[p, g, t]
        for t in range(K):
            B[t] += dB[p, t]
