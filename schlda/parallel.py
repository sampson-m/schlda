"""Parallel (blocked) sampler with a depth-scaled identity prior.

``fit_hlda_parallel`` is ``fit_hlda`` with two additions and the same result
object, AnnData keys and output files:

* ``n_threads``: cells are split into that many contiguous blocks, each swept
  by one thread (``_gibbs_blocked``). 1 = the exact collapsed Gibbs sampler.
* ``identity_share`` / ``prior_scale``: per-cell Dirichlet concentration
  alpha_i = prior_scale * L_i (L_i = the cell's token count), split
  identity_share : (1 - identity_share) between the identity topic and the
  activity topics (equally). The prior mean of the identity share is then
  ``identity_share`` at every depth. Unset = the fixed ``alpha_c`` /
  ``alpha_c_identity`` prior of ``fit_hlda``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numba import config as numba_config, set_num_threads

from . import _gibbs, _gibbs_blocked, _init
from .model import (HLDAResult, _check_int, _get_counts, _get_labels,
                    _identity_order, _resolve_output_dir)

__all__ = ["fit_hlda_parallel", "prior_concentrations"]


def prior_concentrations(tokens_per_cell, n_identity, n_activity, *, identity_share=0.7,
                         prior_scale=0.1, alpha_c=None, alpha_c_identity=None):
    """Cells x topics concentration matrix. Depth-scaled when ``prior_scale`` is set, else fixed."""
    n_cells = len(tokens_per_cell)
    K = n_identity + n_activity
    alpha = np.empty((n_cells, K), dtype=np.float64)
    if prior_scale is not None:
        if not 0.0 < identity_share < 1.0:
            raise ValueError("identity_share must be in (0, 1).")
        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive.")
        total = prior_scale * np.asarray(tokens_per_cell, dtype=np.float64)
        alpha[:, :n_identity] = (identity_share * total)[:, None]
        alpha[:, n_identity:] = ((1.0 - identity_share) * total / max(n_activity, 1))[:, None]
    else:
        a = 0.1 if alpha_c is None else float(alpha_c)
        alpha[:, :] = a
        alpha[:, :n_identity] = a if alpha_c_identity is None else float(alpha_c_identity)
    return alpha


def fit_hlda_parallel(
    adata,
    cell_type_key,
    n_activity_topics,
    *,
    layer=None,
    output_dir=None,
    init="data_driven",
    n_loops=3000,
    burn_in=1000,
    thin=30,
    alpha_beta=0.1,
    alpha_c=None,
    alpha_c_identity=None,
    identity_share=0.7,
    prior_scale=None,
    n_threads=1,
    seed=None,
    key_added="schlda",
    verbose=True,
    report_every=100,
):
    """Fit HLDA by blocked collapsed Gibbs sampling. See the module docstring and ``fit_hlda``."""
    t_start = time.time()
    n_activity = _check_int(n_activity_topics, "n_activity_topics", minimum=0)
    n_loops = _check_int(n_loops, "n_loops", minimum=1)
    burn_in = _check_int(burn_in, "burn_in", minimum=0)
    thin = _check_int(thin, "thin", minimum=1)
    n_threads = _check_int(n_threads, "n_threads", minimum=1)
    n_save = (n_loops - burn_in) // thin
    if n_save < 1:
        raise ValueError(f"thin={thin} is larger than n_loops - burn_in = {n_loops - burn_in}.")
    if init not in ("data_driven", "random"):
        raise ValueError(f"init must be 'data_driven' or 'random', got {init!r}.")
    if prior_scale is not None and (alpha_c is not None or alpha_c_identity is not None):
        raise ValueError("Give either prior_scale (depth-scaled prior) or alpha_c/alpha_c_identity, not both.")

    counts = _get_counts(adata, layer)
    labels = _get_labels(adata, cell_type_key)
    identity_values = _identity_order(labels)
    identity_topics = [str(v) for v in identity_values]
    n_identity = len(identity_topics)
    topic_names = identity_topics + [f"V{i + 1}" for i in range(n_activity)]
    n_topics = len(topic_names)
    if len(set(topic_names)) != n_topics:
        raise ValueError(f"Topic names are not unique: {topic_names}.")

    output_dir = _resolve_output_dir(output_dir)
    sample_dir = output_dir / "samples"
    n_cells, n_genes = counts.shape
    identity_ints = labels.map({v: i for i, v in enumerate(identity_values)}).to_numpy(dtype=np.int32)

    max_threads = int(numba_config.NUMBA_NUM_THREADS)
    if n_threads > max_threads:
        if verbose:
            print(f"[schlda] n_threads={n_threads} exceeds Numba's limit {max_threads}; using {max_threads}")
        n_threads = max_threads
    set_num_threads(n_threads)

    if verbose:
        print(f"[schlda] {n_cells:,} cells x {n_genes:,} genes, {n_topics} topics "
              f"({n_identity} identity + {n_activity} activity), {n_threads} thread(s)")
        print(f"[schlda] output -> {output_dir}")

    chain_seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    np.random.seed(chain_seed)
    rng = np.random.default_rng(chain_seed)

    cell_idx, gene_idx, ident_rep = _gibbs.expand_tokens(counts, identity_ints)
    z = _init.initial_assignments(counts, identity_ints, cell_idx, ident_rep,
                                  n_identity, n_activity, init, rng, verbose=verbose)
    A, B, D = _gibbs.compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, n_topics)
    cell_offsets = _gibbs_blocked.cell_offsets_from_idx(cell_idx, n_cells)
    tokens_per_cell = np.diff(cell_offsets)
    alpha = prior_concentrations(tokens_per_cell, n_identity, n_activity, identity_share=identity_share,
                                 prior_scale=prior_scale, alpha_c=alpha_c, alpha_c_identity=alpha_c_identity)
    valid_topics, valid_offsets = _gibbs_blocked.build_valid(n_identity, n_activity)
    block_bounds = _gibbs_blocked.make_blocks(cell_offsets, n_threads)
    n_blocks = block_bounds.shape[0] - 1
    dA = np.zeros((n_blocks, n_genes, n_topics), dtype=np.int32)
    dB = np.zeros((n_blocks, n_topics), dtype=np.int32)
    gene_idx = np.ascontiguousarray(gene_idx.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))

    if verbose:
        print(f"[schlda] {cell_idx.shape[0]:,} tokens, {n_blocks} block(s)")
        if prior_scale is not None:
            print(f"[schlda] prior: alpha_i = {prior_scale} * L_i, identity share {identity_share} "
                  f"(median L_i {np.median(tokens_per_cell):.0f})")
        print(f"[schlda] Gibbs: n_loops={n_loops} burn_in={burn_in} thin={thin} ({n_save} draws kept)")

    A_chain, D_chain = _gibbs.open_chains(sample_dir, n_save, n_genes, n_cells, n_topics)
    t_gibbs = time.time()
    cnt = 0
    for sweep in range(n_loops):
        seeds = _gibbs_blocked.block_seeds(chain_seed, sweep, n_blocks)
        _gibbs_blocked.gibbs_sweep(cell_offsets, gene_idx, z, identity_ints, valid_topics, valid_offsets,
                                   A, B, D, np.float64(alpha_beta), alpha, block_bounds, seeds, dA, dB)
        if sweep >= burn_in and (sweep - burn_in + 1) % thin == 0:
            A_chain[cnt] = A
            D_chain[cnt] = D
            cnt += 1
        if verbose and (sweep + 1) % report_every == 0:
            elapsed = time.time() - t_gibbs
            print(f"[schlda]   sweep {sweep + 1}/{n_loops} ({elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / (sweep + 1) * (n_loops - sweep - 1) / 60:.1f} min left)", flush=True)
    if cnt != n_save:
        raise RuntimeError(f"Expected {n_save} saved draws, sampler wrote {cnt}.")
    A_chain.flush()
    D_chain.flush()
    seconds_per_sweep = (time.time() - t_gibbs) / n_loops

    beta, theta = _gibbs.posterior_means(A_chain, D_chain, n_save, adata.var_names, adata.obs_names, topic_names)
    topics = pd.DataFrame({"topic_id": range(n_topics), "topic_name": topic_names,
                           "topic_type": ["identity"] * n_identity + ["activity"] * n_activity})
    params = {
        "cell_type_key": str(cell_type_key), "layer": "X" if layer is None else str(layer),
        "n_activity_topics": int(n_activity), "init": init, "n_loops": int(n_loops), "burn_in": int(burn_in),
        "thin": int(thin), "n_save": int(n_save), "alpha_beta": float(alpha_beta),
        "prior": "depth_scaled" if prior_scale is not None else "fixed",
        "identity_share": float(identity_share) if prior_scale is not None else None,
        "prior_scale": None if prior_scale is None else float(prior_scale),
        "alpha_c": None if prior_scale is not None else float(alpha[0, -1] if n_activity else alpha[0, 0]),
        "alpha_c_identity": None if prior_scale is not None else float(alpha[0, 0]),
        "n_threads": int(n_threads), "n_blocks": int(n_blocks), "sampler": "blocked",
        "n_tokens": int(cell_idx.shape[0]), "seed": chain_seed, "seconds_per_sweep": float(seconds_per_sweep),
    }
    beta.to_csv(output_dir / "HLDA_beta.csv")
    theta.to_csv(output_dir / "HLDA_theta.csv")
    topics.to_csv(output_dir / "topic_info.csv", index=False)
    _gibbs.write_sample_metadata(sample_dir, n_save, n_genes, n_cells, n_topics, topic_names, params)
    adata.obsm[f"{key_added}_theta"] = theta.to_numpy()
    adata.varm[f"{key_added}_beta"] = beta.to_numpy()
    adata.uns[key_added] = {"topic_names": topic_names, "topic_types": params_types(n_identity, n_activity),
                            "params": params, "output_dir": str(output_dir)}
    if verbose:
        print(f"[schlda] done in {(time.time() - t_start) / 60:.1f} min ({seconds_per_sweep:.2f} s/sweep)")
    return HLDAResult(theta=theta, beta=beta, topics=topics, cell_types=labels, params=params, output_dir=output_dir)


def params_types(n_identity, n_activity):
    return ["identity"] * n_identity + ["activity"] * n_activity
