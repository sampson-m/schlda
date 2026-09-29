"""Hierarchical LDA for single-cell count data.

One identity topic per annotated cell type, plus a set of activity topics that
every cell may use. A cell's tokens are only ever assigned to its own identity
topic or to a shared activity topic, so identity topics stay pinned to the
annotation and the activity topics are left to absorb whatever varies across
cell types. ``fit_hlda`` samples with the blocked kernel in ``_gibbs_blocked``;
one thread is the exact collapsed Gibbs sampler.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from numba import config as numba_config, set_num_threads

from . import _gibbs, _gibbs_blocked, _init

__all__ = ["fit_hlda", "HLDAResult"]


@dataclass
class HLDAResult:
    """Fitted HLDA model.

    Attributes
    ----------
    theta
        Cells x topics posterior mean, rows summing to 1. Indexed by
        ``adata.obs_names``.
    beta
        Genes x topics posterior mean, columns summing to 1. Indexed by
        ``adata.var_names``.
    topics
        One row per topic with ``topic_id``, ``topic_name`` and ``topic_type``
        (``identity`` or ``activity``).
    cell_types
        The annotation the fit was conditioned on, aligned to ``theta``.
    params
        The settings the chain was run with.
    output_dir
        Where estimates and posterior samples were written.
    """

    theta: pd.DataFrame
    beta: pd.DataFrame
    topics: pd.DataFrame
    cell_types: pd.Series
    params: dict = field(default_factory=dict)
    output_dir: Path | None = None

    @property
    def identity_topics(self) -> list[str]:
        return self.topics.loc[self.topics.topic_type == "identity", "topic_name"].tolist()

    @property
    def activity_topics(self) -> list[str]:
        return self.topics.loc[self.topics.topic_type == "activity", "topic_name"].tolist()

    def __repr__(self) -> str:
        n_cells, n_topics = self.theta.shape
        return (
            f"HLDAResult({n_cells:,} cells x {self.beta.shape[0]:,} genes, "
            f"{n_topics} topics = {len(self.identity_topics)} identity + "
            f"{len(self.activity_topics)} activity)"
        )


def fit_hlda(
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
    n_threads=1,
    alpha_beta=0.1,
    alpha_c=0.1,
    alpha_c_identity=None,
    prior_scale_identity=None,
    seed=None,
    key_added="schlda",
    verbose=True,
    report_every=100,
):
    """Fit HLDA to raw counts in an AnnData by collapsed Gibbs sampling.

    Parameters
    ----------
    adata
        AnnData holding raw integer counts. ``adata.X`` is used unless ``layer``
        names a layer. Normalised or log-transformed values are rejected.
    cell_type_key
        Column in ``adata.obs`` giving each cell's type. Every distinct value
        becomes one identity topic, and a cell's tokens may only use that topic
        or the shared activity topics. No missing values allowed.
    n_activity_topics
        Number of shared activity topics, named ``V1 .. Vn``. May be 0, which
        reduces the fit to per-identity expression profiles.
    layer
        Layer to read counts from, e.g. ``"counts"``. Defaults to ``adata.X``.
    output_dir
        Where to write ``HLDA_beta.csv``, ``HLDA_theta.csv``, ``topic_info.csv``
        and the posterior samples under ``samples/``. Defaults to
        ``./schlda_output`` in the working directory, with a warning.
    init
        ``"data_driven"`` seeds the chain from per-identity mean expression;
        ``"random"`` assigns each token uniformly over its permitted topics.
    n_loops, burn_in, thin
        Gibbs sweeps in total, sweeps discarded, and the interval between kept
        draws. ``(n_loops - burn_in) // thin`` draws are saved.
    n_threads
        1 (default) runs the exact collapsed Gibbs sampler. ``P > 1`` splits the
        cells into ``P`` blocks swept in parallel, each against a private copy
        of the gene-topic table merged after every sweep (Newman et al. 2009).
    alpha_beta
        Dirichlet concentration on the gene distributions.
    alpha_c
        Dirichlet concentration on each activity topic: a float, or one value
        per cell (array in ``adata`` row order, or a Series aligned on
        ``adata.obs_names``).
    alpha_c_identity
        Concentration on the cell's identity topic, float or one value per
        cell. Defaults to ``alpha_c``.
    prior_scale_identity
        ``kappa``: identity concentration ``kappa * L_i``, with ``L_i`` the
        cell's UMI count, so the prior carries the same share of every cell's
        evidence. Replaces ``alpha_c_identity``.
    seed
        Seeds initialisation and the chain. A run repeats exactly for a fixed
        ``seed`` and ``n_threads``.
    key_added
        Results are written to ``adata.obsm[f"{key_added}_theta"]``,
        ``adata.varm[f"{key_added}_beta"]`` and ``adata.uns[key_added]``.
    verbose, report_every
        Print progress every ``report_every`` sweeps.

    Returns
    -------
    HLDAResult
    """
    t_start = time.time()

    n_activity = _check_int(n_activity_topics, "n_activity_topics", minimum=0)
    n_loops = _check_int(n_loops, "n_loops", minimum=1)
    burn_in = _check_int(burn_in, "burn_in", minimum=0)
    thin = _check_int(thin, "thin", minimum=1)
    n_threads = _check_int(n_threads, "n_threads", minimum=1)
    report_every = _check_int(report_every, "report_every", minimum=1)
    if burn_in >= n_loops:
        raise ValueError(f"burn_in ({burn_in}) must be less than n_loops ({n_loops}).")
    n_save = (n_loops - burn_in) // thin
    if n_save < 1:
        raise ValueError(
            f"thin ({thin}) is larger than the post-burn-in run "
            f"({n_loops - burn_in} sweeps), so no draws would be kept."
        )
    if init not in ("data_driven", "random"):
        raise ValueError(f"init must be 'data_driven' or 'random', got {init!r}.")
    if prior_scale_identity is not None:
        if alpha_c_identity is not None:
            raise ValueError("prior_scale_identity replaces alpha_c_identity; give one of them.")
        if not (np.isscalar(prior_scale_identity) and np.isfinite(prior_scale_identity)
                and prior_scale_identity > 0):
            raise ValueError(f"prior_scale_identity must be a positive number, got {prior_scale_identity!r}.")

    counts = _get_counts(adata, layer)
    labels = _get_labels(adata, cell_type_key)
    identity_values = _identity_order(labels)
    identity_topics = [str(v) for v in identity_values]
    n_identity = len(identity_topics)
    topic_names = identity_topics + [f"V{i + 1}" for i in range(n_activity)]
    n_topics = len(topic_names)
    if len(set(topic_names)) != n_topics:
        raise ValueError(
            f"Topic names are not unique: {topic_names}. Cell type labels must "
            f"not collide with the activity topic names V1..V{n_activity}."
        )
    act_cells = _per_cell(alpha_c, "alpha_c", adata.obs_names)
    id_cells = None if alpha_c_identity is None else _per_cell(alpha_c_identity, "alpha_c_identity",
                                                               adata.obs_names)

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
        print(
            f"[schlda] {n_cells:,} cells x {n_genes:,} genes, "
            f"{n_topics} topics ({n_identity} identity + {n_activity} activity), {n_threads} thread(s)"
        )
        print(f"[schlda] output -> {output_dir}")

    chain_seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    np.random.seed(chain_seed)
    rng = np.random.default_rng(chain_seed)

    cell_idx, gene_idx, ident_rep = _gibbs.expand_tokens(counts, identity_ints)
    z = _init.initial_assignments(
        counts, identity_ints, cell_idx, ident_rep,
        n_identity, n_activity, init, rng, verbose=verbose,
    )
    A, B, D = _gibbs.compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, n_topics)
    cell_offsets = _gibbs_blocked.cell_offsets_from_idx(cell_idx, n_cells)
    tokens_per_cell = np.diff(cell_offsets)

    if prior_scale_identity is not None:
        id_cells = float(prior_scale_identity) * tokens_per_cell.astype(np.float64)
        prior = "identity_depth_scaled"
    else:
        id_cells = act_cells if id_cells is None else id_cells
        scalar = np.isscalar(alpha_c) and (alpha_c_identity is None or np.isscalar(alpha_c_identity))
        prior = "fixed" if scalar else "per_cell"
    alpha = np.empty((n_cells, n_topics), dtype=np.float64)
    alpha[:, n_identity:] = act_cells[:, None]
    alpha[:, :n_identity] = id_cells[:, None]

    valid_topics, valid_offsets = _gibbs_blocked.build_valid(n_identity, n_activity)
    block_bounds = _gibbs_blocked.make_blocks(cell_offsets, n_threads)
    n_blocks = block_bounds.shape[0] - 1
    dA = np.zeros((n_blocks, n_genes, n_topics), dtype=np.int32)
    dB = np.zeros((n_blocks, n_topics), dtype=np.int32)
    gene_idx = np.ascontiguousarray(gene_idx.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))

    if verbose:
        print(f"[schlda] {cell_idx.shape[0]:,} tokens, {n_blocks} block(s)")
        if prior == "identity_depth_scaled":
            print(f"[schlda] prior: alpha_identity = {prior_scale_identity} * L_i "
                  f"(median L_i {np.median(tokens_per_cell):.0f})")
        elif prior == "per_cell":
            print(f"[schlda] prior: per-cell alpha (identity median {np.median(id_cells):.3g}, "
                  f"activity median {np.median(act_cells):.3g})")
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

    beta, theta = _gibbs.posterior_means(
        A_chain, D_chain, n_save, adata.var_names, adata.obs_names, topic_names
    )
    topic_types = ["identity"] * n_identity + ["activity"] * n_activity
    topics = pd.DataFrame({"topic_id": range(n_topics), "topic_name": topic_names, "topic_type": topic_types})

    def _record(x):
        return float(x) if x is not None and np.isscalar(x) else ("per_cell" if x is not None else None)

    params = {
        "cell_type_key": str(cell_type_key),
        "layer": "X" if layer is None else str(layer),
        "n_activity_topics": int(n_activity),
        "init": init,
        "n_loops": int(n_loops),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "n_save": int(n_save),
        "alpha_beta": float(alpha_beta),
        "prior": prior,
        "alpha_c": _record(alpha_c),
        "alpha_c_identity": (None if prior_scale_identity is not None else
                             _record(alpha_c if alpha_c_identity is None else alpha_c_identity)),
        "prior_scale_identity": None if prior_scale_identity is None else float(prior_scale_identity),
        "n_threads": int(n_threads),
        "n_blocks": int(n_blocks),
        "n_tokens": int(cell_idx.shape[0]),
        "seed": chain_seed,
        "seconds_per_sweep": float(seconds_per_sweep),
    }

    beta.to_csv(output_dir / "HLDA_beta.csv")
    theta.to_csv(output_dir / "HLDA_theta.csv")
    topics.to_csv(output_dir / "topic_info.csv", index=False)
    if prior != "fixed":
        pd.DataFrame({"identity": id_cells, "activity": act_cells},
                     index=adata.obs_names).to_csv(output_dir / "theta_prior.csv")
    _gibbs.write_sample_metadata(sample_dir, n_save, n_genes, n_cells, n_topics, topic_names, params)

    adata.obsm[f"{key_added}_theta"] = theta.to_numpy()
    adata.varm[f"{key_added}_beta"] = beta.to_numpy()
    adata.uns[key_added] = {
        "topic_names": topic_names,
        "topic_types": topic_types,
        "params": params,
        "output_dir": str(output_dir),
    }

    if verbose:
        print(f"[schlda] done in {(time.time() - t_start) / 60:.1f} min ({seconds_per_sweep:.2f} s/sweep)")

    return HLDAResult(
        theta=theta,
        beta=beta,
        topics=topics,
        cell_types=labels,
        params=params,
        output_dir=output_dir,
    )


def _per_cell(value, name, obs_names):
    """A positive float, or one value per cell (array in row order / Series aligned on obs_names)."""
    n = len(obs_names)
    if np.isscalar(value):
        out = np.full(n, float(value), dtype=np.float64)
    elif isinstance(value, pd.Series):
        aligned = value.reindex(obs_names)
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(f"{name}: {missing} cell(s) missing from the Series index or NaN.")
        out = aligned.to_numpy(dtype=np.float64)
    else:
        out = np.asarray(value, dtype=np.float64).ravel()
        if out.shape[0] != n:
            raise ValueError(f"{name} must be a number or have one value per cell ({n}), got {out.shape[0]}.")
    if not (np.all(np.isfinite(out)) and np.all(out > 0)):
        raise ValueError(f"{name} must be positive and finite.")
    return out


def _check_int(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _get_counts(adata, layer):
    """Validate and return raw counts as CSR."""
    source = "adata.X" if layer is None else f"adata.layers[{layer!r}]"
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer {layer!r} not found. Available layers: {list(adata.layers)}"
            )
        X = adata.layers[layer]

    if X is None:
        raise ValueError(f"{source} is None; HLDA needs raw counts.")

    # csr_matrix() shares arrays with an existing CSR input, so nothing here may
    # modify X in place: assigning X.data below rebinds our own reference and
    # leaves the caller's matrix untouched. Stored explicit zeros are harmless —
    # they expand to zero tokens.
    X = sp.csr_matrix(X) if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    data = X.data

    if data.size and data.min() < 0:
        raise ValueError(f"{source} contains negative values; HLDA needs raw counts.")
    if data.size and not np.all(np.mod(data, 1) == 0):
        raise ValueError(
            f"{source} contains non-integer values, so it looks normalised or "
            f"log-transformed. HLDA needs raw counts — pass "
            f"layer='counts' (or whichever layer holds them)."
        )

    X.data = data.astype(np.int64)

    empty = np.asarray(X.sum(axis=1)).ravel() == 0
    if empty.any():
        raise ValueError(
            f"{int(empty.sum())} cell(s) have zero total counts and carry no "
            f"tokens. Filter them first, e.g. "
            f"sc.pp.filter_cells(adata, min_counts=1)."
        )

    dead_genes = int((np.asarray(X.sum(axis=0)).ravel() == 0).sum())
    if dead_genes:
        warnings.warn(
            f"{dead_genes} gene(s) have zero counts across all cells; their beta "
            f"entries will be exactly zero.",
            stacklevel=3,
        )
    return X


def _get_labels(adata, cell_type_key):
    if cell_type_key not in adata.obs:
        raise KeyError(
            f"{cell_type_key!r} not in adata.obs. Available columns: "
            f"{list(adata.obs.columns)}"
        )
    labels = adata.obs[cell_type_key]
    if labels.isna().any():
        raise ValueError(
            f"adata.obs[{cell_type_key!r}] has {int(labels.isna().sum())} missing "
            f"value(s). Every cell needs an identity to condition on."
        )
    return labels


def _identity_order(labels):
    """Identity topics in category order when available, else sorted."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        used = set(labels.unique())
        return [c for c in labels.cat.categories if c in used]
    return sorted(labels.unique())


def _resolve_output_dir(output_dir):
    if output_dir is None:
        output_dir = Path.cwd() / "schlda_output"
        warnings.warn(
            f"No output_dir given; writing estimates and posterior samples to "
            f"{output_dir}. Pass output_dir to choose the location.",
            stacklevel=3,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
