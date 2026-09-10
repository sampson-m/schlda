"""Hierarchical LDA for single-cell count data.

One identity topic per annotated cell type, plus a set of activity topics that
every cell may use. A cell's tokens are only ever assigned to its own identity
topic or to a shared activity topic, so identity topics stay pinned to the
annotation and the activity topics are left to absorb whatever varies across
cell types.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _gibbs, _init

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
    alpha_beta=0.1,
    alpha_c=0.1,
    alpha_c_identity=None,
    seed=None,
    key_added="schlda",
    verbose=True,
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
    alpha_beta
        Dirichlet concentration on the gene distributions.
    alpha_c
        Dirichlet concentration on activity topic usage.
    alpha_c_identity
        Concentration on identity topic usage. Defaults to ``alpha_c``. Setting
        it apart from ``alpha_c`` biases how readily tokens leave the identity
        topic.
    seed
        Seeds initialisation and the Gibbs chain. Without it neither is
        reproducible, since the compiled kernel keeps its own RNG state.
    key_added
        Results are written to ``adata.obsm[f"{key_added}_theta"]``,
        ``adata.varm[f"{key_added}_beta"]`` and ``adata.uns[key_added]``.
    verbose
        Print progress.

    Returns
    -------
    HLDAResult
    """
    t_start = time.time()

    n_activity = _check_int(n_activity_topics, "n_activity_topics", minimum=0)
    n_loops = _check_int(n_loops, "n_loops", minimum=1)
    burn_in = _check_int(burn_in, "burn_in", minimum=0)
    thin = _check_int(thin, "thin", minimum=1)
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

    output_dir = _resolve_output_dir(output_dir)
    sample_dir = output_dir / "samples"

    n_cells, n_genes = counts.shape
    label_to_int = {value: i for i, value in enumerate(identity_values)}
    identity_ints = labels.map(label_to_int).to_numpy(dtype=np.int32)

    if verbose:
        print(
            f"[schlda] {n_cells:,} cells x {n_genes:,} genes, "
            f"{n_topics} topics ({n_identity} identity + {n_activity} activity)"
        )
        print(f"[schlda] output -> {output_dir}")

    if seed is not None:
        np.random.seed(seed)  # the numba kernel draws from numpy's global state
        _gibbs._seed_numba_rng(seed)
    rng = np.random.default_rng(seed)

    cell_idx, gene_idx, ident_rep = _gibbs.expand_tokens(counts, identity_ints)
    if verbose:
        print(f"[schlda] {cell_idx.shape[0]:,} tokens")

    z = _init.initial_assignments(
        counts, identity_ints, cell_idx, ident_rep,
        n_identity, n_activity, init, rng, verbose=verbose,
    )

    A, B, D = _gibbs.compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, n_topics)

    alpha_c_id = alpha_c if alpha_c_identity is None else alpha_c_identity
    alpha_c_arr = np.full(n_topics, float(alpha_c), dtype=np.float64)
    alpha_c_arr[:n_identity] = float(alpha_c_id)

    topic_hierarchy = {
        k: [k] + [n_identity + i for i in range(n_activity)] for k in range(n_identity)
    }
    valid_list = _gibbs.build_valid_list(topic_hierarchy)

    A_chain, D_chain = _gibbs.open_chains(sample_dir, n_save, n_genes, n_cells, n_topics)

    if verbose:
        print(
            f"[schlda] Gibbs: n_loops={n_loops} burn_in={burn_in} thin={thin} "
            f"({n_save} draws kept)"
        )
    t_gibbs = time.time()
    cnt = 0
    for block_start in range(0, n_loops, _BLOCK):
        block_end = min(block_start + _BLOCK, n_loops)
        cnt = _gibbs._gibbs_block(
            cell_idx, gene_idx, ident_rep, z,
            A, B, D, valid_list,
            np.float64(alpha_beta), alpha_c_arr,
            np.int32(n_genes), np.int32(n_topics),
            np.int32(block_start), np.int32(block_end),
            np.int32(burn_in), np.int32(thin), np.int32(cnt),
            A_chain, D_chain,
        )
        if verbose:
            elapsed = time.time() - t_gibbs
            rate = elapsed / block_end
            print(
                f"[schlda]   sweep {block_end}/{n_loops} "
                f"({elapsed / 60:.1f} min elapsed, "
                f"~{rate * (n_loops - block_end) / 60:.1f} min left)",
                flush=True,
            )

    if cnt != n_save:
        raise RuntimeError(f"Expected {n_save} saved draws, sampler wrote {cnt}.")
    A_chain.flush()
    D_chain.flush()

    beta, theta = _gibbs.posterior_means(
        A_chain, D_chain, n_save, adata.var_names, adata.obs_names, topic_names
    )

    topics = pd.DataFrame({
        "topic_id": range(n_topics),
        "topic_name": topic_names,
        "topic_type": ["identity"] * n_identity + ["activity"] * n_activity,
    })

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
        "alpha_c": float(alpha_c),
        "alpha_c_identity": float(alpha_c_id),
        "n_tokens": int(cell_idx.shape[0]),
    }
    if seed is not None:
        params["seed"] = int(seed)

    beta.to_csv(output_dir / "HLDA_beta.csv")
    theta.to_csv(output_dir / "HLDA_theta.csv")
    topics.to_csv(output_dir / "topic_info.csv", index=False)
    _gibbs.write_sample_metadata(
        sample_dir, n_save, n_genes, n_cells, n_topics, topic_names, params
    )

    adata.obsm[f"{key_added}_theta"] = theta.to_numpy()
    adata.varm[f"{key_added}_beta"] = beta.to_numpy()
    adata.uns[key_added] = {
        "topic_names": topic_names,
        "topic_types": ["identity"] * n_identity + ["activity"] * n_activity,
        "params": params,
        "output_dir": str(output_dir),
    }

    if verbose:
        print(f"[schlda] done in {(time.time() - t_start) / 60:.1f} min")

    return HLDAResult(
        theta=theta,
        beta=beta,
        topics=topics,
        cell_types=labels,
        params=params,
        output_dir=output_dir,
    )


_BLOCK = 100  # Gibbs sweeps between progress reports


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
