import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

import schlda
from schlda.parallel import prior_concentrations

TYPES = ["A", "B", "C"]


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


def test_prior_concentrations():
    L = np.array([100.0, 200.0])
    a = prior_concentrations(L, 2, 2, identity_share=0.7, prior_scale=0.1)
    assert np.allclose(a[:, :2], [[7, 7], [14, 14]]) and np.allclose(a[:, 2:], [[1.5, 1.5], [3, 3]])
    assert np.allclose(a[0, 0] / (a[0, 0] + a[0, 2:].sum()), 0.7)
    f = prior_concentrations(L, 2, 1, prior_scale=None, alpha_c=0.1, alpha_c_identity=1.0)
    assert np.allclose(f, [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]])


@pytest.mark.parametrize("n_threads", [1, 3])
def test_fit_recovers_program(tmp_path, n_threads):
    a, act, w = simulate()
    res = schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / f"t{n_threads}", n_loops=300,
                                   burn_in=150, thin=10, seed=1, n_threads=n_threads, verbose=False)
    assert res.activity_topics == ["V1"] and res.theta.shape == (a.n_obs, 4)
    assert np.allclose(res.theta.sum(1), 1) and np.allclose(res.beta.sum(0), 1)
    assert cos(res.beta["V1"].to_numpy(), act) > 0.9
    assert np.corrcoef(res.theta["V1"], w)[0, 1] > 0.8
    for t in TYPES:  # a cell never uses another type's identity topic
        other = [u for u in TYPES if u != t]
        assert (res.theta.loc[a.obs["cell_type"] == t, other].to_numpy() == 0).all()
    assert res.params["n_blocks"] == min(n_threads, res.params["n_threads"]) or res.params["n_blocks"] >= 1
    assert (tmp_path / f"t{n_threads}" / "HLDA_beta.csv").exists()
    assert a.obsm["schlda_theta"].shape == (a.n_obs, 4)


def test_reproducible(tmp_path):
    a, _, _ = simulate()
    kw = dict(n_loops=40, burn_in=20, thin=5, seed=7, n_threads=2, verbose=False)
    r1 = schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / "r1", **kw)
    r2 = schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / "r2", **kw)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())


def test_depth_scaled_prior_raises_identity(tmp_path):
    a, _, _ = simulate()
    kw = dict(n_loops=200, burn_in=100, thin=10, seed=3, init="random", verbose=False)
    fixed = schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / "f", alpha_c=0.1, **kw)
    scaled = schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / "s", prior_scale=5.0,
                                      identity_share=0.95, **kw)

    def ident(res):
        return np.array([res.theta.iloc[i][t] for i, t in enumerate(a.obs["cell_type"].astype(str))])
    assert ident(scaled).mean() > ident(fixed).mean()
    assert (ident(scaled) < 0.4).mean() <= (ident(fixed) < 0.4).mean()
    assert scaled.params["prior"] == "depth_scaled" and fixed.params["prior"] == "fixed"
    with pytest.raises(ValueError):
        schlda.fit_hlda_parallel(a, "cell_type", 1, output_dir=tmp_path / "x", prior_scale=0.1, alpha_c=0.1, **kw)
