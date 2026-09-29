"""Token arrays, count tables and posterior chains for the HLDA sampler.

The token-level representation is the long format used throughout: every UMI is
one token carrying (cell, gene, identity, topic). The sampling kernel itself is
in ``_gibbs_blocked``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def expand_tokens(counts, cell_identity_ints):
    """Expand a sparse count matrix into per-token (cell, gene, identity) arrays.

    `counts` is CSR with integer-valued data. Working from the sparse structure
    keeps the peak footprint at the token arrays themselves rather than a dense
    cells x genes copy.
    """
    coo = counts.tocoo()
    reps = coo.data.astype(np.int64)
    cell_idx = np.repeat(coo.row.astype(np.int32), reps)
    gene_idx = np.repeat(coo.col.astype(np.int32), reps)
    ident = cell_identity_ints[cell_idx]
    return cell_idx, gene_idx, ident


def compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, K):
    """Initial sufficient statistics: A (gene x topic), B (topic), D (cell x topic)."""
    A = np.bincount(gene_idx.astype(np.int64) * K + z, minlength=n_genes * K)
    D = np.bincount(cell_idx.astype(np.int64) * K + z, minlength=n_cells * K)
    B = np.bincount(z, minlength=K)
    return (
        A.reshape(n_genes, K).astype(np.int32),
        B.astype(np.int32),
        D.reshape(n_cells, K).astype(np.int32),
    )


def open_chains(sample_dir, n_save, n_genes, n_cells, K, mode="w+"):
    """Create (or reopen) the A/D chain memmaps under `sample_dir`."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    A_chain = np.memmap(
        str(sample_dir / "A_chain.memmap"), mode=mode, dtype=np.int32,
        shape=(n_save, n_genes, K),
    )
    D_chain = np.memmap(
        str(sample_dir / "D_chain.memmap"), mode=mode, dtype=np.int32,
        shape=(n_save, n_cells, K),
    )
    return A_chain, D_chain


def write_sample_metadata(sample_dir, n_save, n_genes, n_cells, K, topic_names, params):
    """Sidecar recording the memmap shapes, so readers need not re-derive them."""
    meta = {
        "dtype": "int32",
        "n_save": int(n_save),
        "n_genes": int(n_genes),
        "n_cells": int(n_cells),
        "n_topics": int(K),
        "A_chain": {"file": "A_chain.memmap", "shape": [int(n_save), int(n_genes), int(K)]},
        "D_chain": {"file": "D_chain.memmap", "shape": [int(n_save), int(n_cells), int(K)]},
        "topic_names": list(topic_names),
        "params": params,
    }
    with open(Path(sample_dir) / "samples_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def posterior_means(A_chain, D_chain, n_save, gene_names, cell_names, topic_names):
    """Posterior mean beta (genes x K) and theta (cells x K) from the saved chains."""
    beta_sum = np.zeros(A_chain.shape[1:], dtype=np.float64)
    theta_sum = np.zeros(D_chain.shape[1:], dtype=np.float64)
    for idx in range(n_save):
        beta_sum += A_chain[idx]
        theta_sum += D_chain[idx]

    # A topic that never wins a token, or a cell with no tokens, would give 0/0;
    # leave those as exact zeros instead of NaN.
    beta_norm = beta_sum.sum(axis=0, keepdims=True)
    theta_norm = theta_sum.sum(axis=1, keepdims=True)
    beta = beta_sum / np.where(beta_norm == 0, 1.0, beta_norm)
    theta = theta_sum / np.where(theta_norm == 0, 1.0, theta_norm)

    beta_df = pd.DataFrame(beta, index=pd.Index(gene_names), columns=list(topic_names))
    theta_df = pd.DataFrame(theta, index=pd.Index(cell_names), columns=list(topic_names))
    return beta_df, theta_df
