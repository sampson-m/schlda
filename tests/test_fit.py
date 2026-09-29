import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

import schlda

TYPES = ["A", "B", "C"]
SHORT = dict(n_loops=60, burn_in=30, thin=5, seed=7, init="random", verbose=False)
FIT = dict(n_loops=200, burn_in=100, thin=10, seed=3, init="random", verbose=False)


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
    res = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / f"t{n_threads}", n_loops=300,
                          burn_in=150, thin=10, seed=1, n_threads=n_threads, verbose=False)
    assert res.activity_topics == ["V1"] and res.theta.shape == (a.n_obs, 4)
    assert np.allclose(res.theta.sum(1), 1) and np.allclose(res.beta.sum(0), 1)
    assert cos(res.beta["V1"].to_numpy(), act) > 0.9
    assert np.corrcoef(res.theta["V1"], w)[0, 1] > 0.8
    for t in TYPES:  # a cell never uses another type's identity topic
        other = [u for u in TYPES if u != t]
        assert (res.theta.loc[a.obs["cell_type"] == t, other].to_numpy() == 0).all()
    assert res.params["prior"] == "fixed" and res.params["n_threads"] >= 1
    assert (tmp_path / f"t{n_threads}" / "HLDA_beta.csv").exists()
    assert not (tmp_path / f"t{n_threads}" / "theta_prior.csv").exists()
    assert a.obsm["schlda_theta"].shape == (a.n_obs, 4)


def test_reproducible(tmp_path):
    a, _, _ = simulate()
    r1 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r1", n_threads=2, **SHORT)
    r2 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r2", n_threads=2, **SHORT)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())
    assert r1.params["seed"] == 7


def test_constant_vector_equals_scalar(tmp_path):
    a, _, _ = simulate()
    s = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "s", alpha_c=0.3, alpha_c_identity=2.0, **SHORT)
    v = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "v", alpha_c=np.full(a.n_obs, 0.3),
                        alpha_c_identity=pd.Series(2.0, index=a.obs_names[::-1]), **SHORT)
    assert np.array_equal(s.theta.to_numpy(), v.theta.to_numpy())
    assert s.params["prior"] == "fixed" and s.params["alpha_c_identity"] == 2.0
    assert v.params["prior"] == "per_cell" and v.params["alpha_c"] == "per_cell"
    tp = pd.read_csv(tmp_path / "v" / "theta_prior.csv", index_col=0)
    assert list(tp.columns) == ["identity", "activity"] and np.allclose(tp.identity, 2.0) and np.allclose(tp.activity, 0.3)


def test_vector_identity_with_scalar_activity(tmp_path):
    a, act, _ = simulate()
    vec = np.where(a.obs["cell_type"] == "A", 50.0, 0.1)
    r = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "v", alpha_c=1.0, alpha_c_identity=vec, **FIT)
    base = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "b", alpha_c=1.0, **FIT)
    in_a = (a.obs["cell_type"] == "A").to_numpy()
    assert ident(r, a)[in_a].mean() > ident(base, a)[in_a].mean()
    tp = pd.read_csv(tmp_path / "v" / "theta_prior.csv", index_col=0)
    assert np.allclose(tp.identity.to_numpy(), vec) and np.allclose(tp.activity, 1.0)


def test_identity_depth_scaled_prior(tmp_path):
    a, act, _ = simulate()
    fixed = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "f", **FIT)
    for n_threads in (1, 2):
        r = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / f"k{n_threads}", prior_scale_identity=0.1,
                            n_threads=n_threads, **FIT)
        assert ident(r, a).mean() > ident(fixed, a).mean()
        assert r.params["prior"] == "identity_depth_scaled" and r.params["prior_scale_identity"] == 0.1
        assert r.params["alpha_c_identity"] is None and r.params["alpha_c"] == 0.1
        assert cos(r.beta["V1"].to_numpy(), act) > 0.9
    tp = pd.read_csv(tmp_path / "k1" / "theta_prior.csv", index_col=0)
    L = np.asarray(a.X.sum(1)).ravel()
    assert np.allclose(tp.identity.to_numpy(), 0.1 * L)


@pytest.mark.parametrize("kw", [
    dict(alpha_c=np.ones(5)),
    dict(alpha_c_identity=np.r_[np.ones(179), 0.0]),
    dict(alpha_c=np.r_[np.ones(179), np.nan]),
    dict(alpha_c_identity=pd.Series(1.0, index=["c0", "c1"])),
    dict(alpha_c=-1.0),
    dict(prior_scale_identity=0.1, alpha_c_identity=1.0),
    dict(prior_scale_identity=0.0),
])
def test_prior_errors(tmp_path, kw):
    a, _, _ = simulate()
    with pytest.raises(ValueError):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **{**SHORT, **kw})
