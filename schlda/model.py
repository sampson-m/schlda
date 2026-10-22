"""Hierarchical LDA for single-cell count data.

Every annotated cell type carries its own topic, and activity topics are shared by the cell types that may use
them (``_geometry``). ``fit_hlda`` samples with the blocked kernel in ``_gibbs_blocked``; one thread is the exact
collapsed Gibbs sampler.
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from numba import config as numba_config, set_num_threads

from . import _geometry, _gibbs, _gibbs_blocked, _init, _likelihood

__all__ = ["fit_hlda", "HLDAResult", "geometry"]


@dataclass
class HLDAResult:
    """Fitted HLDA model.

    Attributes
    ----------
    theta
        Cells x topics posterior mean, rows summing to 1; indexed by ``adata.obs_names``.
    beta
        Genes x topics posterior mean, columns summing to 1; indexed by ``adata.var_names``.
    topics
        The table ``topic, kind, cell_types, alpha``.
    cell_types
        The annotation the fit was conditioned on, aligned to ``theta``.
    params
        The settings the chain was run with.
    output_dir
        Where the estimates were written.
    geometry
        The topic geometry; ``geometry.describe()`` prints it.
    """

    theta: pd.DataFrame
    beta: pd.DataFrame
    topics: pd.DataFrame
    cell_types: pd.Series
    params: dict = field(default_factory=dict)
    output_dir: Path | None = None
    geometry: _geometry.Geometry | None = None

    @property
    def identity_topics(self) -> list[str]:
        return self.geometry.identity_topics

    @property
    def activity_topics(self) -> list[str]:
        return self.geometry.activity_topics

    @property
    def allowed(self) -> pd.DataFrame:
        """Cell type x topic permission mask."""
        return self.geometry.allowed_frame()

    def __repr__(self) -> str:
        n_cells, n_topics = self.theta.shape
        return (f"HLDAResult({n_cells:,} cells x {self.beta.shape[0]:,} genes, {n_topics} topics = "
                f"{len(self.identity_topics)} identity + {len(self.activity_topics)} activity)")


def geometry(adata, cell_type_key, n_activity_topics=None, *, activity=None):
    """The topic geometry a fit would use, for inspection: ``g.describe()``, ``g.to_frame()``, ``g.to_csv()``."""
    obs = adata if isinstance(adata, pd.DataFrame) else adata.obs
    return _geometry.build(_geometry.get_labels(obs, cell_type_key), n_activity_topics, activity)


class _Log:
    def __init__(self, path, verbose):
        self.fh, self.verbose = open(path, "w"), verbose

    def __call__(self, msg):
        if self.verbose:
            print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self, status, err=None):
        if err is not None:
            self.fh.write(err)
        self.fh.write(f"exit {status}\n")
        self.fh.close()


def fit_hlda(
    adata,
    cell_type_key,
    n_activity_topics=None,
    *,
    activity=None,
    layer=None,
    output_dir=None,
    n_loops=3000,
    burn_in=1000,
    thin=30,
    n_threads=1,
    alpha_beta=0.1,
    alpha_identity=1.0,
    alpha_activity=0.1,
    save_samples=False,
    track_likelihood=False,
    plots=True,
    seed=None,
    key_added="schlda",
    verbose=True,
    report_every=100,
    init="random",
):
    """Fit HLDA to raw counts in an AnnData by collapsed Gibbs sampling.

    Parameters
    ----------
    adata
        AnnData holding raw integer counts. ``adata.X`` is used unless ``layer`` names a layer. Normalised or
        log-transformed values are rejected.
    cell_type_key
        Column in ``adata.obs`` giving each cell's type. No missing values.
    n_activity_topics
        Number of activity topics ``V1 .. Vm`` (may be 0) shared by every cell type, on top of one identity topic
        per cell type.
    activity
        Instead of ``n_activity_topics``, which activity topics each cell type may use: a CSV path or DataFrame
        with columns ``cell_type, activity_topics`` (topics ``;``-separated, blank for none) or ``topic,
        cell_types`` (``all`` or ``;``-separated types), or a dict in either orientation.
    layer
        Layer to read counts from, e.g. ``"counts"``. Defaults to ``adata.X``.
    output_dir
        Receives ``theta.csv``, ``beta.csv``, ``topics.csv``, ``params.json``, ``log.txt``, the plots, and
        ``samples/`` when ``save_samples``. Defaults to ``./schlda_output`` with a warning.
    n_loops, burn_in, thin
        Gibbs sweeps in total, sweeps discarded, and the interval between kept draws; ``(n_loops - burn_in) //
        thin`` draws enter the posterior means.
    n_threads
        1 runs the exact collapsed Gibbs sampler. ``P > 1`` splits the cells into ``P`` blocks swept in parallel
        against private copies of the gene-topic table, merged after every sweep (Newman et al. 2009).
    alpha_beta
        Dirichlet concentration on the gene distributions.
    alpha_identity, alpha_activity
        Dirichlet concentration on every identity topic and on every activity topic.
    save_samples
        Write the kept draws of the gene-topic and cell-topic count tables to ``samples/`` (memmaps).
    track_likelihood
        Record the collapsed joint log-likelihood every ``report_every`` sweeps in ``likelihood.csv``.
    plots
        Write ``theta_heatmap.png``, ``activity_theta_heatmap.png`` and ``structure_plot.png``.
    seed
        Seeds initialisation and the chain. A run repeats exactly for a fixed ``seed`` and ``n_threads``.
    key_added
        Results are written to ``adata.obsm[f"{key_added}_theta"]``, ``adata.varm[f"{key_added}_beta"]`` and
        ``adata.uns[key_added]``.
    verbose, report_every
        Print progress every ``report_every`` sweeps; everything printed also goes to ``log.txt``.

    Returns
    -------
    HLDAResult
    """
    from . import __version__

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
        raise ValueError(f"thin ({thin}) is larger than the post-burn-in run ({n_loops - burn_in} sweeps), "
                         f"so no draws would be kept.")
    if init != "random":
        raise ValueError(f"init must be 'random', got {init!r}.")
    alpha_identity = _geometry._alpha(alpha_identity, "alpha_identity")
    alpha_activity = _geometry._alpha(alpha_activity, "alpha_activity")
    alpha_beta = _geometry._alpha(alpha_beta, "alpha_beta")

    counts = _get_counts(adata, layer)
    labels = _geometry.get_labels(adata.obs, cell_type_key)
    geom = _geometry.build(labels, n_activity_topics, activity)
    topic_names, topic_types = geom.topic_names, geom.topic_types
    n_topics, n_identity, n_activity = geom.n_topics, len(geom.cell_types), len(geom.activity)
    alpha = geom.alphas(alpha_identity, alpha_activity)
    topics = geom.to_frame(alpha_identity, alpha_activity)

    output_dir = _resolve_output_dir(output_dir)
    n_cells, n_genes = counts.shape
    max_threads = int(numba_config.NUMBA_NUM_THREADS)
    capped = n_threads > max_threads
    n_threads = min(n_threads, max_threads)
    set_num_threads(n_threads)
    chain_seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)

    log = _Log(output_dir / "log.txt", verbose)
    try:
        log(f"[schlda {__version__}] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"[schlda] {n_cells:,} cells x {n_genes:,} genes, {n_topics} topics ({n_identity} identity + "
            f"{n_activity} activity), {n_threads} thread(s)" + (f" (capped at Numba's limit {max_threads})" if capped else ""))
        log(geom.describe())
        log(f"[schlda] prior: alpha_identity={alpha_identity:g} alpha_activity={alpha_activity:g} alpha_beta={alpha_beta:g}")
        log(f"[schlda] Gibbs: n_loops={n_loops} burn_in={burn_in} thin={thin} ({n_save} draws kept), seed={chain_seed}")
        log(f"[schlda] output -> {output_dir}")

        np.random.seed(chain_seed)
        rng = np.random.default_rng(chain_seed)
        cell_idx, gene_idx, ident_rep = _gibbs.expand_tokens(counts, geom.cell_node)
        z = _init.initial_assignments(ident_rep, geom, rng, log=log)
        A, B, D = _gibbs.compute_count_tables(cell_idx, gene_idx, z, n_cells, n_genes, n_topics)
        cell_offsets = _gibbs_blocked.cell_offsets_from_idx(cell_idx, n_cells)
        alpha_cells = np.ascontiguousarray(np.broadcast_to(alpha, (n_cells, n_topics)))
        valid_topics, valid_offsets = geom.csr()
        block_bounds = _gibbs_blocked.make_blocks(cell_offsets, n_threads)
        n_blocks = block_bounds.shape[0] - 1
        dA = np.zeros((n_blocks, n_genes, n_topics), dtype=np.int32)
        dB = np.zeros((n_blocks, n_topics), dtype=np.int32)
        gene_idx = np.ascontiguousarray(gene_idx.astype(np.int32))
        z = np.ascontiguousarray(z.astype(np.int32))
        log(f"[schlda] {cell_idx.shape[0]:,} tokens, {n_blocks} block(s)")

        beta_sum = np.zeros((n_genes, n_topics), dtype=np.float64)
        theta_sum = np.zeros((n_cells, n_topics), dtype=np.float64)
        if save_samples:
            sample_dir = output_dir / "samples"
            A_chain, D_chain = _gibbs.open_chains(sample_dir, n_save, n_genes, n_cells, n_topics)
        if track_likelihood:
            const = _likelihood.cell_constant(alpha, geom.allowed[geom.cell_node], np.diff(cell_offsets))
            lik_fh = open(output_dir / "likelihood.csv", "w")
            lik_fh.write("sweep,loglik_joint\n")

            def record(sweep):
                ll = _likelihood.joint_loglik(A, D, alpha_beta, alpha, const)
                lik_fh.write(f"{sweep},{ll:.6f}\n")
                lik_fh.flush()
                return ll
            log(f"[schlda]   sweep 0: log P(w,z) = {record(0):.1f}")

        t_gibbs = time.time()
        cnt = 0
        for sweep in range(n_loops):
            seeds = _gibbs_blocked.block_seeds(chain_seed, sweep, n_blocks)
            _gibbs_blocked.gibbs_sweep(cell_offsets, gene_idx, z, geom.cell_node, valid_topics, valid_offsets,
                                       A, B, D, np.float64(alpha_beta), alpha_cells, block_bounds, seeds, dA, dB)
            if sweep >= burn_in and (sweep - burn_in + 1) % thin == 0:
                beta_sum += A
                theta_sum += D
                if save_samples:
                    A_chain[cnt] = A
                    D_chain[cnt] = D
                cnt += 1
            if (sweep + 1) % report_every == 0:
                elapsed = time.time() - t_gibbs
                msg = (f"[schlda]   sweep {sweep + 1}/{n_loops} ({elapsed / 60:.1f} min elapsed, "
                       f"~{elapsed / (sweep + 1) * (n_loops - sweep - 1) / 60:.1f} min left)")
                if track_likelihood:
                    msg += f", log P(w,z) = {record(sweep + 1):.1f}"
                log(msg)
        if cnt != n_save:
            raise RuntimeError(f"Expected {n_save} saved draws, sampler wrote {cnt}.")
        seconds_per_sweep = (time.time() - t_gibbs) / n_loops
        if track_likelihood:
            lik_fh.close()

        beta = pd.DataFrame(beta_sum / np.where(beta_sum.sum(0) == 0, 1.0, beta_sum.sum(0)),
                            index=pd.Index(adata.var_names), columns=topic_names)
        theta = pd.DataFrame(theta_sum / theta_sum.sum(1, keepdims=True),
                             index=pd.Index(adata.obs_names), columns=topic_names)

        params = {
            "schlda_version": __version__,
            "cell_type_key": str(cell_type_key),
            "geometry": topics.to_dict(orient="list"),
            "layer": "X" if layer is None else str(layer),
            "n_activity_topics": int(n_activity),
            "n_loops": int(n_loops), "burn_in": int(burn_in), "thin": int(thin), "n_save": int(n_save),
            "alpha_beta": float(alpha_beta), "alpha_identity": float(alpha_identity), "alpha_activity": float(alpha_activity),
            "n_threads": int(n_threads), "n_blocks": int(n_blocks), "n_tokens": int(cell_idx.shape[0]),
            "seed": chain_seed, "save_samples": bool(save_samples), "track_likelihood": bool(track_likelihood),
            "seconds_per_sweep": float(seconds_per_sweep),
        }
        beta.to_csv(output_dir / "beta.csv")
        theta.to_csv(output_dir / "theta.csv")
        topics.to_csv(output_dir / "topics.csv", index=False)
        json.dump(params, open(output_dir / "params.json", "w"), indent=2)
        if save_samples:
            A_chain.flush()
            D_chain.flush()
            np.save(sample_dir / "allowed.npy", geom.allowed)
            np.save(sample_dir / "cell_node.npy", geom.cell_node)
            _gibbs.write_sample_metadata(sample_dir, n_save, n_genes, n_cells, n_topics, topic_names, topic_types, params)

        adata.obsm[f"{key_added}_theta"] = theta.to_numpy()
        adata.varm[f"{key_added}_beta"] = beta.to_numpy()
        adata.uns[key_added] = {"topic_names": topic_names, "topic_types": topic_types, "params": params,
                                "output_dir": str(output_dir)}
        res = HLDAResult(theta=theta, beta=beta, topics=topics, cell_types=labels, params=params,
                         output_dir=output_dir, geometry=geom)
        if plots:
            _write_plots(res, output_dir, log)
        log(f"[schlda] done in {(time.time() - t_start) / 60:.1f} min ({seconds_per_sweep:.2f} s/sweep)")
    except BaseException:
        log.close(1, traceback.format_exc())
        raise
    log.close(0)
    return res


def _write_plots(res, out, log):
    from . import plotting
    import matplotlib.pyplot as plt
    n = res.theta.shape[0]
    per = max(1, int(round(n / 2000)))
    try:
        fig, _ = plotting.theta_heatmap(res, cells_per_group=per, save=out / "theta_heatmap.png")
        plt.close(fig)
        if res.activity_topics:
            fig, _ = plotting.theta_heatmap(res, topic_order=res.activity_topics, cells_per_group=per,
                                            save=out / "activity_theta_heatmap.png")
            plt.close(fig)
        fig, _ = plotting.structure_plot(res, max_cells_per_group=2000, save=out / "structure_plot.png")
        plt.close(fig)
    except Exception as err:   # a failed plot must not lose the fit
        log(f"[schlda] plots skipped: {err!r}")


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
            raise KeyError(f"Layer {layer!r} not found. Available layers: {list(adata.layers)}")
        X = adata.layers[layer]
    if X is None:
        raise ValueError(f"{source} is None; HLDA needs raw counts.")
    X = sp.csr_matrix(X) if sp.issparse(X) else sp.csr_matrix(np.asarray(X))   # never modified in place
    data = X.data
    if data.size and data.min() < 0:
        raise ValueError(f"{source} contains negative values; HLDA needs raw counts.")
    if data.size and not np.all(np.mod(data, 1) == 0):
        raise ValueError(f"{source} contains non-integer values, so it looks normalised or log-transformed. "
                         f"HLDA needs raw counts — pass layer='counts' (or whichever layer holds them).")
    X.data = data.astype(np.int64)
    empty = np.asarray(X.sum(axis=1)).ravel() == 0
    if empty.any():
        raise ValueError(f"{int(empty.sum())} cell(s) have zero total counts and carry no tokens. "
                         f"Filter them first, e.g. sc.pp.filter_cells(adata, min_counts=1).")
    dead = int((np.asarray(X.sum(axis=0)).ravel() == 0).sum())
    if dead:
        warnings.warn(f"{dead} gene(s) have zero counts across all cells; their beta entries will be exactly zero.",
                      stacklevel=3)
    return X


def _resolve_output_dir(output_dir):
    if output_dir is None:
        output_dir = Path.cwd() / "schlda_output"
        warnings.warn(f"No output_dir given; writing to {output_dir}. Pass output_dir to choose the location.",
                      stacklevel=3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
