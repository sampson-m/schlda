"""Collapsed joint log-likelihood log P(w, z | alpha, eta) of the tokens and their topic assignments, with theta and
beta integrated out:

    sum_k [ lnG(G eta) - lnG(G eta + B_k) + sum_g (lnG(eta + A_gk) - lnG(eta)) ]
  + sum_i [ lnG(a_i) - lnG(a_i + L_i)     + sum_k (lnG(alpha_k + D_ik) - lnG(alpha_k)) ]

A: genes x topics counts, B = A.sum(0), D: cells x topics counts, L_i = D[i].sum(), a_i = the sum of alpha_k over
the topics cell i may use. The gene term is evaluated directly; the cell term by a lookup over the integer counts
of each topic, which is the same sum with terms grouped by value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln


def cell_constant(alpha, allowed_rows, L):
    """sum_i lnG(a_i) - lnG(a_i + L_i); constant over sweeps."""
    a = (allowed_rows * alpha[None, :]).sum(1)
    return float((gammaln(a) - gammaln(a + L)).sum())


def joint_loglik(A, D, alpha_beta, alpha, const):
    G = A.shape[0]
    ll = const + float(gammaln(alpha_beta + A).sum() - G * gammaln(alpha_beta) * A.shape[1]
                       + (gammaln(G * alpha_beta) - gammaln(G * alpha_beta + A.sum(0))).sum())
    for k in range(D.shape[1]):
        c = np.bincount(D[:, k])
        ll += float(c @ (gammaln(alpha[k] + np.arange(c.size)) - gammaln(alpha[k])))
    return ll


def from_samples(sample_dir):
    """One log-likelihood per saved draw, labelled by the sweep it was taken after."""
    sample_dir = Path(sample_dir)
    meta = json.load(open(sample_dir / "samples_meta.json"))
    p = meta["params"]
    A = np.memmap(sample_dir / "A_chain.memmap", dtype=np.int32, mode="r", shape=tuple(meta["A_chain"]["shape"]))
    D = np.memmap(sample_dir / "D_chain.memmap", dtype=np.int32, mode="r", shape=tuple(meta["D_chain"]["shape"]))
    allowed = np.load(sample_dir / "allowed.npy")
    cell_node = np.load(sample_dir / "cell_node.npy")
    kinds = np.array(meta["topic_types"])
    alpha = np.where(kinds == "identity", p["alpha_identity"], p["alpha_activity"]).astype(np.float64)
    L = np.asarray(D[0]).sum(1)
    const = cell_constant(alpha, allowed[cell_node], L)
    rows = [(p["burn_in"] + p["thin"] * (j + 1), joint_loglik(np.asarray(A[j]), np.asarray(D[j]), p["alpha_beta"], alpha, const))
            for j in range(meta["n_save"])]
    return pd.DataFrame(rows, columns=["sweep", "loglik_joint"])


def likelihood_curve(output_dir, save=True, ax=None):
    """Log-likelihood trace of a finished fit: ``likelihood.csv`` when the run tracked it, else computed from
    ``samples/``. Writes ``likelihood.csv`` and ``likelihood.png`` next to the fit; returns the DataFrame."""
    out = Path(output_dir)
    csv, samples = out / "likelihood.csv", out / "samples"
    if csv.exists():
        df = pd.read_csv(csv)
    elif (samples / "samples_meta.json").exists():
        df = from_samples(samples)
        if save:
            df.to_csv(csv, index=False)
    else:
        raise FileNotFoundError(f"{out}: no likelihood.csv (fit with track_likelihood=True) and no samples/ "
                                f"(fit with save_samples=True).")
    params = json.load(open(out / "params.json")) if (out / "params.json").exists() else {}
    import matplotlib.pyplot as plt
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 3.2))
    ax.plot(df.sweep, df.loglik_joint, "-", lw=1.2, color="#4575b4")
    if "burn_in" in params and df.sweep.min() < params["burn_in"]:
        ax.axvline(params["burn_in"], color="grey", ls="--", lw=1, label="burn-in")
        ax.legend(frameon=False)
    ax.set_xlabel("sweep"); ax.set_ylabel("log P(w, z)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "likelihood.png", dpi=150)
        plt.close(fig)
    return df
