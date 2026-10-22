import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest
from scipy.special import gammaln

import schlda

TYPES = ["A", "B", "C"]
SHORT = dict(n_loops=60, burn_in=30, thin=5, seed=7, verbose=False, plots=False)
FIT = dict(n_loops=200, burn_in=100, thin=10, seed=3, verbose=False, plots=False)


def simulate(seed=0, n_per=60, n_genes=120):
    rng = np.random.default_rng(seed)
    prog = {}
    for i, name in enumerate(TYPES + ["act"]):
        p = rng.gamma(0.3, 1.0, n_genes) + 1e-3
        p[i * 25:(i + 1) * 25] *= 40
        prog[name] = p / p.sum()
    X, labels, w_true = [], [], []
    for name in TYPES:
        for _ in range(n_per):
            w = rng.beta(2, 4) if rng.random() < 0.4 else 0.0
            X.append(rng.multinomial(rng.integers(300, 600), (1 - w) * prog[name] + w * prog["act"]))
            labels.append(name)
            w_true.append(w)
    a = ad.AnnData(sp.csr_matrix(np.array(X, dtype=np.float32)))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["cell_type"] = pd.Categorical(labels, categories=TYPES)
    return a, prog["act"], np.array(w_true)


def cos(u, v):
    return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


def ident(res, a):
    return np.array([res.theta.iloc[i][t] for i, t in enumerate(a.obs["cell_type"].astype(str))])


@pytest.mark.parametrize("n_threads", [1, 3])
def test_fit_recovers_program(tmp_path, n_threads):
    a, act, w = simulate()
    out = tmp_path / f"t{n_threads}"
    res = schlda.fit_hlda(a, "cell_type", 1, output_dir=out, n_loops=300, burn_in=150, thin=10, seed=1,
                          n_threads=n_threads, verbose=False)
    assert res.activity_topics == ["V1"] and res.identity_topics == TYPES and res.theta.shape == (a.n_obs, 4)
    assert np.allclose(res.theta.sum(1), 1) and np.allclose(res.beta.sum(0), 1)
    assert cos(res.beta["V1"].to_numpy(), act) > 0.9
    assert np.corrcoef(res.theta["V1"], w)[0, 1] > 0.8
    for t in TYPES:  # a cell never uses another type's identity topic
        other = [u for u in TYPES if u != t]
        assert (res.theta.loc[a.obs["cell_type"] == t, other].to_numpy() == 0).all()
    assert res.params["alpha_identity"] == 1.0 and res.params["alpha_activity"] == 0.1
    for f in ["theta.csv", "beta.csv", "topics.csv", "params.json", "log.txt",
              "theta_heatmap.png", "activity_theta_heatmap.png", "structure_plot.png"]:
        assert (out / f).exists(), f
    assert not (out / "samples").exists() and not (out / "likelihood.csv").exists()
    assert res.topics.columns.tolist() == ["topic", "kind", "cell_types", "alpha"]
    assert res.topics.alpha.tolist() == [1.0, 1.0, 1.0, 0.1]
    assert a.obsm["schlda_theta"].shape == (a.n_obs, 4) and a.uns["schlda"]["topic_types"] == ["identity"] * 3 + ["activity"]
    log = (out / "log.txt").read_text()
    assert log.startswith("[schlda 1.") and log.rstrip().endswith("exit 0") and "sweep 300/300" in log
    assert json.load(open(out / "params.json"))["seed"] == 1
    assert np.allclose(pd.read_csv(out / "theta.csv", index_col=0).to_numpy(), res.theta.to_numpy())


def test_reproducible(tmp_path):
    a, _, _ = simulate()
    r1 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r1", n_threads=2, **SHORT)
    r2 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r2", n_threads=2, **SHORT)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())
    assert r1.params["seed"] == 7


def test_alpha_identity_moves_identity_share(tmp_path):
    a, act, _ = simulate()
    lo = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "lo", alpha_identity=0.1, **FIT)
    hi = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "hi", alpha_identity=50.0, alpha_activity=0.5, **FIT)
    assert ident(hi, a).mean() > ident(lo, a).mean()
    assert hi.params["alpha_identity"] == 50.0 and hi.params["alpha_activity"] == 0.5
    assert cos(lo.beta["V1"].to_numpy(), act) > 0.9
    assert pd.read_csv(tmp_path / "hi" / "topics.csv").alpha.tolist() == [50, 50, 50, 0.5]


def test_samples_and_likelihood(tmp_path):
    a, _, _ = simulate()
    out = tmp_path / "s"
    res = schlda.fit_hlda(a, "cell_type", 2, output_dir=out, save_samples=True, track_likelihood=True,
                          report_every=10, **SHORT)
    meta = json.load(open(out / "samples" / "samples_meta.json"))
    assert meta["n_save"] == 6 and meta["topic_types"] == ["identity"] * 3 + ["activity"] * 2
    tracked = pd.read_csv(out / "likelihood.csv")
    assert tracked.sweep.tolist() == list(range(0, 61, 10)) and np.isfinite(tracked.loglik_joint).all()
    assert tracked.loglik_joint.iloc[-1] > tracked.loglik_joint.iloc[0]
    curve = schlda.likelihood_curve(out)
    assert curve.equals(tracked) and (out / "likelihood.png").exists()
    (out / "likelihood.csv").unlink()
    from_samples = schlda.likelihood_curve(out)
    assert from_samples.sweep.tolist() == [35, 40, 45, 50, 55, 60]
    m = tracked.merge(from_samples, on="sweep")
    assert np.allclose(m.loglik_joint_x, m.loglik_joint_y, atol=1e-4)
    # the lookup form equals the direct gammaln sum
    A = np.memmap(out / "samples" / "A_chain.memmap", dtype=np.int32, mode="r", shape=tuple(meta["A_chain"]["shape"]))[0]
    D = np.memmap(out / "samples" / "D_chain.memmap", dtype=np.int32, mode="r", shape=tuple(meta["D_chain"]["shape"]))[0]
    eta, al, G = 0.1, np.array([1.0] * 3 + [0.1] * 2), A.shape[0]
    allowed = np.load(out / "samples" / "allowed.npy")[np.load(out / "samples" / "cell_node.npy")]
    ai, L = (allowed * al).sum(1), D.sum(1)
    direct = ((gammaln(eta + A) - gammaln(eta)).sum() + (gammaln(G * eta) - gammaln(G * eta + A.sum(0))).sum()
              + (gammaln(ai) - gammaln(ai + L)).sum() + (gammaln(al + D) - gammaln(al)).sum())
    assert np.isclose(direct, from_samples.loglik_joint.iloc[0], rtol=1e-12)
    assert res.params["save_samples"] and res.params["track_likelihood"]


def test_likelihood_curve_needs_samples_or_csv(tmp_path):
    a, _, _ = simulate()
    schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **SHORT)
    with pytest.raises(FileNotFoundError):
        schlda.likelihood_curve(tmp_path / "x")


def test_rejected_before_writing(tmp_path):
    a, _, _ = simulate()
    with pytest.raises(ValueError, match="unknown cell type"):
        schlda.fit_hlda(a, "cell_type", activity={"V1": ["A", "Z"]}, output_dir=tmp_path / "x", **SHORT)
    assert not (tmp_path / "x").exists()


def test_removed_arguments(tmp_path):
    a, _, _ = simulate()
    for kw in (dict(prior_scale_identity=0.1), dict(alpha_c=0.3), dict(alpha={"A": 1.0}), dict(geometry=None), dict(bogus=1)):
        with pytest.raises(TypeError):
            schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **SHORT, **kw)
    with pytest.raises(ValueError):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", init="data_driven", **SHORT)


@pytest.mark.parametrize("kw", [
    dict(alpha_activity=-1.0),
    dict(alpha_identity=0.0),
    dict(alpha_identity=np.nan),
    dict(alpha_identity=np.ones(5)),
    dict(alpha_beta=np.ones(5)),
    dict(n_activity_topics=-1),
    dict(burn_in=100),
])
def test_value_errors(tmp_path, kw):
    a, _, _ = simulate()
    with pytest.raises(ValueError):
        schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / "x", **{**dict(n_activity_topics=1), **SHORT, **kw})


def test_counts_validation(tmp_path):
    a, _, _ = simulate()
    a.layers["counts"] = a.X.copy()
    a.X = sp.csr_matrix(a.X.multiply(1e4 / np.asarray(a.X.sum(1))))
    with pytest.raises(ValueError, match="non-integer"):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **SHORT)
    res = schlda.fit_hlda(a, "cell_type", 1, layer="counts", output_dir=tmp_path / "y", **SHORT)
    assert res.params["layer"] == "counts"
