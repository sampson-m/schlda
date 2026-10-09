"""Hierarchical LDA for single-cell count data.

By default one identity topic per annotated cell type, plus a set of activity
topics that every cell may use; ``geometry(adata, key)`` builds any other
geometry (``_geometry``). A cell's tokens are only ever assigned to the topics
its type may use.
``fit_hlda`` samples with the blocked kernel in ``_gibbs_blocked``; one thread
is the exact collapsed Gibbs sampler.
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

from . import _geometry, _gibbs, _gibbs_blocked, _init

__all__ = ["fit_hlda", "HLDAResult", "geometry"]


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
        The geometry table: ``topic``, ``kind`` (``identity`` or ``activity``),
        ``cell_types`` and the ``alpha`` used.
    cell_types
        The annotation the fit was conditioned on, aligned to ``theta``.
    params
        The settings the chain was run with.
    output_dir
        Where estimates and posterior samples were written.
    geometry
        The topic geometry the fit used; ``geometry.describe()`` prints it.
    """

    theta: pd.DataFrame
    beta: pd.DataFrame
    topics: pd.DataFrame
    cell_types: pd.Series
    params: dict = field(default_factory=dict)
    output_dir: Path | None = None
    geometry: _geometry.TopicGeometry | None = None

    @property
    def identity_topics(self) -> list[str]:
        return self.topics.loc[self.topics.kind == "identity", "topic"].tolist()

    @property
    def activity_topics(self) -> list[str]:
        return self.topics.loc[self.topics.kind == "activity", "topic"].tolist()

    @property
    def allowed(self) -> pd.DataFrame:
        """Cell type x topic permission mask derived from the geometry."""
        return self.geometry.allowed_frame()

    def __repr__(self) -> str:
        n_cells, n_topics = self.theta.shape
        return (
            f"HLDAResult({n_cells:,} cells x {self.beta.shape[0]:,} genes, "
            f"{n_topics} topics = {len(self.identity_topics)} identity + "
            f"{len(self.activity_topics)} activity)"
        )


def geometry(adata, cell_type_key, n_activity_topics=None, *, identity_topics=None, shared_topics=None, topics=None):
    """A topic geometry for the cell types in ``adata.obs[cell_type_key]``.

    With no other argument: one identity topic per cell type and no shared
    topic. Edit it with ``g.split(type, n)``, ``g.share(name, types)``,
    ``g.drop(name)``, ``g.alpha(name, value)``; ``g.describe()`` prints it,
    ``g.to_frame()`` / ``g.to_csv()`` give the table that ``topics=`` (a
    DataFrame or CSV path) reads back. Shortcuts: ``n_activity_topics=m``
    adds ``V1 .. Vm`` for every type; ``identity_topics={type: n or [names]}``
    and ``shared_topics={name: "all" or [types]}`` as in ``split`` / ``share``.
    """
    obs = adata if isinstance(adata, pd.DataFrame) else adata.obs
    return _geometry.build(_geometry.get_labels(obs, cell_type_key), n_activity_topics,
                           identity_topics, shared_topics, topics)


def fit_hlda(
    adata,
    cell_type_key,
    n_activity_topics=None,
    *,
    geometry=None,
    layer=None,
    output_dir=None,
    init="data_driven",
    n_loops=3000,
    burn_in=1000,
    thin=30,
    n_threads=1,
    alpha_beta=0.1,
    alpha_identity=1.0,
    alpha_activity=0.1,
    seed=None,
    key_added="schlda",
    verbose=True,
    report_every=100,
    **legacy,
):
    """Fit HLDA to raw counts in an AnnData by collapsed Gibbs sampling.

    Parameters
    ----------
    adata
        AnnData holding raw integer counts. ``adata.X`` is used unless ``layer``
        names a layer. Normalised or log-transformed values are rejected.
    cell_type_key
        Column in ``adata.obs`` giving each cell's type. No missing values.
    n_activity_topics
        Number of activity topics ``V1 .. Vm`` (may be 0) shared by every
        cell, on top of one identity topic per cell type. A cell's tokens may
        only use its own identity topic or the activity topics.
    geometry
        Any other geometry, replacing ``n_activity_topics``: a
        ``TopicGeometry`` from ``schlda.geometry``, or its table (DataFrame or
        CSV path with columns ``topic, kind, cell_types, alpha``). Its cell
        types must be those of ``adata.obs[cell_type_key]``.
    layer
        Layer to read counts from, e.g. ``"counts"``. Defaults to ``adata.X``.
    output_dir
        Where to write ``HLDA_beta.csv``, ``HLDA_theta.csv``, ``topics.csv``
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
    alpha_identity, alpha_activity
        Dirichlet concentration on a cell's identity (split evenly over its
        type's identity topics when it has several) and on each activity
        topic. A topic whose ``alpha`` is set in the geometry keeps that value.
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
    alpha_identity, alpha_activity = _legacy(legacy, alpha_identity, alpha_activity)
    alpha_identity, alpha_activity = _geometry._alpha(alpha_identity), _geometry._alpha(alpha_activity)
    alpha_beta = _geometry._alpha(alpha_beta)

    counts = _get_counts(adata, layer)
    labels = _geometry.get_labels(adata.obs, cell_type_key)
    if geometry is None:
        geom = _geometry.build(labels, n_activity_topics)
    else:
        if n_activity_topics is not None:
            raise ValueError("geometry replaces n_activity_topics; give one of them.")
        g = geometry if isinstance(geometry, _geometry.TopicGeometry) else \
            _geometry.TopicGeometry.from_frame(geometry, _geometry.identity_order(labels))
        geom = g.bind(labels)
    topic_names = list(geom.topic_names)
    topic_types = list(geom.topic_types)
    n_topics = len(topic_names)
    is_id = geom.identity_mask()
    n_identity = int(is_id.sum())
    n_activity = topic_types.count("activity")
    alpha_topic = geom.resolve_alpha(alpha_identity, alpha_activity)

    output_dir = _resolve_output_dir(output_dir)
    sample_dir = output_dir / "samples"
    n_cells, n_genes = counts.shape
    identity_ints = geom.cell_node

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
        if geometry is not None:
            print(geom.describe())
        print(f"[schlda] output -> {output_dir}")

    chain_seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    np.random.seed(chain_seed)
    rng = np.random.default_rng(chain_seed)

    cell_idx, gene_idx, ident_rep = _gibbs.expand_tokens(counts, identity_ints)
    z = _init.initial_assignments(counts, identity_ints, cell_idx, ident_rep, geom, init, rng, verbose=verbose)
    A, B, D = _gibbs.compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, n_topics)
    cell_offsets = _gibbs_blocked.cell_offsets_from_idx(cell_idx, n_cells)

    alpha = np.ascontiguousarray(np.broadcast_to(alpha_topic, (n_cells, n_topics)))

    valid_topics, valid_offsets = geom.csr()
    block_bounds = _gibbs_blocked.make_blocks(cell_offsets, n_threads)
    n_blocks = block_bounds.shape[0] - 1
    dA = np.zeros((n_blocks, n_genes, n_topics), dtype=np.int32)
    dB = np.zeros((n_blocks, n_topics), dtype=np.int32)
    gene_idx = np.ascontiguousarray(gene_idx.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))

    if verbose:
        print(f"[schlda] {cell_idx.shape[0]:,} tokens, {n_blocks} block(s)")
        print(f"[schlda] prior: alpha_identity={alpha_identity:g} alpha_activity={alpha_activity:g}")
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
    topics = geom.to_frame(alpha_identity, alpha_activity)

    params = {
        "cell_type_key": str(cell_type_key),
        "geometry": topics.to_dict(orient="list"),
        "layer": "X" if layer is None else str(layer),
        "n_activity_topics": int(n_activity),
        "init": init,
        "n_loops": int(n_loops),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "n_save": int(n_save),
        "alpha_beta": float(alpha_beta),
        "alpha_identity": alpha_identity,
        "alpha_activity": alpha_activity,
        "n_threads": int(n_threads),
        "n_blocks": int(n_blocks),
        "n_tokens": int(cell_idx.shape[0]),
        "seed": chain_seed,
        "seconds_per_sweep": float(seconds_per_sweep),
    }

    beta.to_csv(output_dir / "HLDA_beta.csv")
    theta.to_csv(output_dir / "HLDA_theta.csv")
    topics.to_csv(output_dir / "topics.csv", index=False)
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
        geometry=geom,
    )


def _legacy(kw, alpha_identity, alpha_activity):
    """Map the 0.4/0.5 prior arguments; per-cell and depth-scaled priors are gone."""
    for k in ("alpha_c_identity", "prior_scale_identity"):
        if k in kw and (k == "prior_scale_identity" or not np.isscalar(kw[k])):
            raise TypeError(f"{k} was removed in 0.6.0: the prior is a fixed alpha_identity / alpha_activity "
                            f"(per topic via geometry.alpha).")
    if "alpha_c" in kw:
        warnings.warn("alpha_c is now alpha_activity", DeprecationWarning, stacklevel=3)
        alpha_activity = kw.pop("alpha_c")
    if "alpha_c_identity" in kw:
        warnings.warn("alpha_c_identity is now alpha_identity", DeprecationWarning, stacklevel=3)
        alpha_identity = kw.pop("alpha_c_identity")
    if kw:
        raise TypeError(f"fit_hlda() got unexpected keyword argument(s) {sorted(kw)}")
    return alpha_identity, alpha_activity


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
