"""Collapsed Gibbs sampler for HLDA.

The token-level representation is the long format used throughout: every UMI is
one token carrying (cell, gene, identity, topic). The sampler resamples each
token's topic from the topics its cell's identity is allowed to use, which is
its own identity topic plus every shared activity topic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as TypedList


@njit(cache=True)
def _seed_numba_rng(seed):
    """Numba keeps its own RNG state; seeding it requires a call from njit code."""
    np.random.seed(seed)


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


def build_valid_list(topic_hierarchy):
    """Numba typed list mapping identity int -> array of topic ints it may use."""
    valid_list = TypedList()
    for ident_int in sorted(topic_hierarchy):
        valid_list.append(np.asarray(topic_hierarchy[ident_int], dtype=np.int32))
    return valid_list


@njit(cache=True)
def _gibbs_block(
    cell_idx_arr,
    gene_idx_arr,
    cell_identity_arr,
    z_arr,
    A,
    B,
    D,
    valid_list,
    alpha_beta,
    alpha_c,
    G,
    K,
    it_start,
    it_end,
    burn_in,
    thin,
    cnt_start,
    A_chain,
    D_chain,
):
    """Run Gibbs sweeps [it_start, it_end) and write thinned post-burn-in draws."""
    total_tokens = cell_idx_arr.shape[0]
    n_cells = D.shape[0]
    G_abeta = G * alpha_beta
    probs = np.empty(K, np.float64)  # reused across tokens
    cnt = cnt_start

    for it in range(it_start, it_end):
        for idx in range(total_tokens):
            c = cell_idx_arr[idx]
            g = gene_idx_arr[idx]
            ident = cell_identity_arr[idx]
            old_z = z_arr[idx]

            A[g, old_z] -= 1
            B[old_z] -= 1
            D[c, old_z] -= 1

            vt = valid_list[ident]
            m = vt.shape[0]
            total_p = 0.0
            for j in range(m):
                t = vt[j]
                p = ((alpha_beta + A[g, t]) / (G_abeta + B[t])) * (alpha_c[t] + D[c, t])
                probs[j] = p
                total_p += p

            r = np.random.rand() * total_p
            cum = 0.0
            new_z = vt[m - 1]
            for j in range(m):
                cum += probs[j]
                if r < cum:
                    new_z = vt[j]
                    break

            A[g, new_z] += 1
            B[new_z] += 1
            D[c, new_z] += 1
            z_arr[idx] = new_z

        if it >= burn_in and ((it - burn_in + 1) % thin == 0):
            for gg in range(G):
                for tt in range(K):
                    A_chain[cnt, gg, tt] = A[gg, tt]
            for cc in range(n_cells):
                for tt in range(K):
                    D_chain[cnt, cc, tt] = D[cc, tt]
            cnt += 1

    return cnt


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
