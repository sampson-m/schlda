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
    assert res.params["alpha_identity"] == 1.0 and res.params["n_threads"] >= 1
    assert (tmp_path / f"t{n_threads}" / "HLDA_beta.csv").exists()
    assert res.topics.columns.tolist() == ["topic", "node", "cell_types", "alpha"]
    assert a.obsm["schlda_theta"].shape == (a.n_obs, 4)


def test_reproducible(tmp_path):
    a, _, _ = simulate()
    r1 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r1", n_threads=2, **SHORT)
    r2 = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "r2", n_threads=2, **SHORT)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())
    assert r1.params["seed"] == 7


def test_alpha_identity_moves_identity_share(tmp_path):
    a, act, _ = simulate()
    lo = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "lo", alpha_identity=0.1, **FIT)
    hi = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "hi", alpha_identity=50.0, **FIT)
    assert ident(hi, a).mean() > ident(lo, a).mean()
    assert hi.params["alpha_identity"] == 50.0 and hi.params["alpha_activity"] == 0.1
    assert cos(lo.beta["V1"].to_numpy(), act) > 0.9
    assert pd.read_csv(tmp_path / "hi" / "topics.csv").alpha.tolist() == [50, 50, 50, 0.1]


def test_removed_arguments(tmp_path):
    a, _, _ = simulate()
    for kw in (dict(prior_scale_identity=0.1), dict(alpha_c=0.3), dict(alpha_c_identity=2.0), dict(bogus=1)):
        with pytest.raises(TypeError):
            schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **SHORT, **kw)


@pytest.mark.parametrize("kw", [
    dict(alpha_activity=-1.0),
    dict(alpha_identity=0.0),
    dict(alpha_identity=np.nan),
    dict(alpha_beta=np.ones(5)),
])
def test_prior_errors(tmp_path, kw):
    a, _, _ = simulate()
    with pytest.raises(ValueError):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **{**SHORT, **kw})


def _L(a):
    return np.asarray(a.X.sum(1)).ravel().astype(float)


def test_depth_equals_explicit_per_cell(tmp_path):
    a, _, _ = simulate()
    L = _L(a)
    d = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "d", alpha_identity=schlda.depth(0.005), **SHORT)
    e = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "e", alpha_identity=0.005 * L, **SHORT)
    assert np.array_equal(d.theta.to_numpy(), e.theta.to_numpy())
    assert d.params["alpha_identity"] == "depth(0.005)" and d.params["prior"] == "per_cell"
    assert d.topics.alpha.tolist() == ["depth(0.005)"] * 3 + [0.1]
    ac = pd.read_csv(tmp_path / "d" / "alpha_cells.csv", index_col=0)
    assert ac.shape == (a.n_obs, 4) and np.allclose(ac["A"], 0.005 * L) and np.allclose(ac["V1"], 0.1)
    assert e.params["prior"] == "per_cell" and e.params["alpha_identity"] == "per-cell"


def test_series_alpha_aligned_by_name(tmp_path):
    a, _, _ = simulate()
    L = _L(a)
    s = pd.Series(0.005 * L, index=a.obs_names).iloc[::-1]          # reversed order, aligned by name
    d = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "d", alpha_identity=schlda.depth(0.005), **SHORT)
    e = schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "e", alpha_identity=s, **SHORT)
    assert np.array_equal(d.theta.to_numpy(), e.theta.to_numpy())


def test_alpha_dict_matches_geometry_and_floats(tmp_path):
    a, _, _ = simulate()
    g = schlda.geometry(a, "cell_type", 2).topics("A", 2)
    base = schlda.fit_hlda(a, "cell_type", geometry=g, output_dir=tmp_path / "0", **SHORT)
    byg = schlda.fit_hlda(a, "cell_type", geometry=schlda.geometry(a, "cell_type", 2).topics("A", 2).alpha("V2", 0.3),
                          output_dir=tmp_path / "1", **SHORT)
    byd = schlda.fit_hlda(a, "cell_type", geometry=g, alpha={"V2": 0.3}, output_dir=tmp_path / "2", **SHORT)
    assert np.array_equal(byg.theta.to_numpy(), byd.theta.to_numpy())
    assert not np.array_equal(base.theta.to_numpy(), byd.theta.to_numpy())
    node = schlda.fit_hlda(a, "cell_type", geometry=g, alpha={"A": 3.0, "V": 0.3}, output_dir=tmp_path / "3", **SHORT)
    assert node.topics.alpha.tolist() == [1.5, 1.5, 1.0, 1.0, 0.3, 0.3]        # cell type split, group not
    assert node.params["alpha"] == {"A": 3.0, "V": 0.3} and node.params["prior"] == "fixed"
    arr = schlda.fit_hlda(a, "cell_type", geometry=g, alpha=np.array([1.5, 1.5, 1.0, 1.0, 0.3, 0.3]),
                          output_dir=tmp_path / "4", **SHORT)
    assert np.array_equal(arr.theta.to_numpy(), node.theta.to_numpy()) and arr.params["alpha"] == "array"
    full = np.tile([1.5, 1.5, 1.0, 1.0, 0.3, 0.3], (a.n_obs, 1))
    arr2 = schlda.fit_hlda(a, "cell_type", geometry=g, alpha=pd.DataFrame(full, index=a.obs_names, columns=g.topic_names),
                           output_dir=tmp_path / "5", **SHORT)
    assert np.array_equal(arr2.theta.to_numpy(), node.theta.to_numpy())


def test_depth_in_dict_and_table_roundtrip(tmp_path):
    a, _, _ = simulate()
    g = schlda.geometry(a, "cell_type", 1).alpha("V1", schlda.depth(0.0005))
    f = g.to_frame()
    assert f.alpha.tolist()[-1] == "depth(0.0005)" and f.alpha.isna().sum() == 3
    h = schlda.geometry(a, "cell_type", table=f)
    assert h.resolve_alpha(1.0, 0.1) == [1.0, 1.0, 1.0, schlda.depth(0.0005)]
    res = schlda.fit_hlda(a, "cell_type", 1, alpha={"B": schlda.depth(0.01)}, output_dir=tmp_path / "x", **SHORT)
    ac = pd.read_csv(tmp_path / "x" / "alpha_cells.csv", index_col=0)
    L = _L(a)
    assert np.allclose(ac["B"], 0.01 * L) and np.allclose(ac["A"], 1.0)
    assert res.topics.alpha.tolist() == [1.0, "depth(0.01)", 1.0, 0.1]
    with pytest.raises(ValueError, match="per-cell"):
        schlda.geometry(a, "cell_type", table=f.assign(alpha=["per-cell", None, None, None]))


@pytest.mark.parametrize("kw", [
    dict(alpha_identity=-np.ones(180)),
    dict(alpha_identity=np.ones(10)),
    dict(alpha=np.ones(3)),
    dict(alpha=np.zeros((180, 4))),
    dict(alpha={"A": -1.0}),
])
def test_alpha_value_errors(tmp_path, kw):
    a, _, _ = simulate()
    with pytest.raises(ValueError):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", **{**SHORT, **kw})


def test_depth_rejects_nonpositive():
    for v in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            schlda.depth(v)


def test_alpha_key_error(tmp_path):
    a, _, _ = simulate()
    with pytest.raises(KeyError):
        schlda.fit_hlda(a, "cell_type", 1, output_dir=tmp_path / "x", alpha={"nope": 1.0}, **SHORT)
