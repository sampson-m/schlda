from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

import schlda
from make_golden import CASES, RUN
from test_fit import simulate, cos

GOLD = np.load(Path(__file__).parent / "data" / "golden_v040.npz")
FIT = dict(n_loops=300, burn_in=150, thin=10, seed=2, init="random", verbose=False)
FLAT_MASK = pd.DataFrame([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], index=["C", "A", "B"],
                         columns=["A", "B", "C", "V1"]).astype(bool)


def equivalent_calls(a, m):
    """Ways of asking for the v0.4.0 geometry; every one must reproduce it exactly."""
    yield "positional", dict(n_activity_topics=m)
    yield "default_mask", dict(allowed=schlda.allowed_topics(a, "cell_type", m))
    if m == 1:
        yield "row_order", dict(allowed=FLAT_MASK)
        yield "int_mask", dict(allowed=FLAT_MASK.astype(int))


@pytest.mark.parametrize("case", list(CASES))
def test_reproduces_v040(tmp_path, case):
    kw = dict(CASES[case])
    m = kw.pop("n_activity_topics")
    a0, _, _ = simulate()
    for label, geo in equivalent_calls(a0, m):
        a, _, _ = simulate()
        res = schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / label, **geo, **RUN, **kw)
        assert np.array_equal(res.theta.to_numpy(), GOLD[f"{case}_theta"]), label
        assert np.array_equal(res.beta.to_numpy(), GOLD[f"{case}_beta"]), label


# geometry only

def obs(labels):
    return ad.AnnData(sp.csr_matrix(np.ones((len(labels), 3))),
                      obs=pd.DataFrame({"cell_type": labels}, index=[f"c{i}" for i in range(len(labels))]))


def test_default_geometry():
    a = obs(["B", "A", "B"])
    g = schlda.topic_geometry(a, "cell_type", 2)
    assert g.kind == "default" and g.cell_types == ["A", "B"] and g.topic_names == ["A", "B", "V1", "V2"]
    assert g.allowed.tolist() == [[True, False, True, True], [False, True, True, True]]
    assert g.cell_node.tolist() == [1, 0, 1]
    assert schlda.allowed_topics(a, "cell_type", 2).equals(g.allowed_frame())


def test_custom_geometry():
    a = obs(["B", "CD4T", "CD8T", "Mono"])
    mask = schlda.allowed_topics(a, "cell_type", 1)
    mask["lymphoid"] = mask.index.isin(["B", "CD4T", "CD8T"])
    mask = mask.rename(columns={"CD4T": "CD4T_1"}).assign(CD4T_2=mask.index == "CD4T")
    g = schlda.topic_geometry(a, "cell_type", allowed=mask)
    types = dict(zip(g.topic_names, g.topic_types))
    assert g.kind == "custom" and g.topic_names == ["B", "CD4T_1", "CD8T", "Mono", "V1", "lymphoid", "CD4T_2"]
    assert types["CD4T_2"] == "identity" and types["lymphoid"] == "activity" and types["V1"] == "activity"
    assert g.allowed_frame().loc["Mono"].sum() == 2
    text = g.describe()
    assert "CD4T  (1 cells): CD4T_1, V1, lymphoid, CD4T_2" in text and "lymphoid <- B, CD4T, CD8T" in text
    assert "V1 <- all" in text


@pytest.mark.parametrize("mask", [
    FLAT_MASK.drop("C"),
    FLAT_MASK.assign(A=False),
    FLAT_MASK.assign(V2=False),
    FLAT_MASK.astype(float) * 0.5,
    FLAT_MASK.rename(columns={"V1": "A"}),
    FLAT_MASK.loc[:, ["A", "B", "C"]].assign(C=False),
    FLAT_MASK.values,
])
def test_mask_errors(mask):
    with pytest.raises((ValueError, TypeError)):
        schlda.topic_geometry(obs(["A", "B", "C"]), "cell_type", allowed=mask)


def test_argument_errors():
    a = obs(["A", "B", "C"])
    with pytest.raises(TypeError):
        schlda.topic_geometry(a, "cell_type")
    with pytest.raises(ValueError):
        schlda.topic_geometry(a, "cell_type", 1, allowed=FLAT_MASK)
    with pytest.raises(TypeError):
        schlda.topic_geometry(a, ["cell_type"], 1)
    with pytest.raises(ValueError):
        schlda.topic_geometry(obs(["A", None]), "cell_type", 1)
    with pytest.raises(ValueError):
        schlda.topic_geometry(obs(["A", "V1"]), "cell_type", 1)


def test_extra_mask_rows_are_ignored():
    g = schlda.topic_geometry(obs(["A", "B", "C"]), "cell_type",
                              allowed=pd.concat([FLAT_MASK, pd.DataFrame([[0, 0, 0, 1]], index=["Z"],
                                                                         columns=FLAT_MASK.columns).astype(bool)]))
    assert g.cell_types == ["A", "B", "C"]


# fits

def simulate_lineage(seed=0, n_per=60, n_genes=150):
    """Types A, B, C, D; P1 is used by A and B cells only, P0 by every cell."""
    rng = np.random.default_rng(seed)
    names = ["A", "B", "C", "D", "P1", "P0"]
    prog = {}
    for i, name in enumerate(names):
        p = rng.gamma(0.3, 1.0, n_genes) + 1e-3
        p[i * 25:(i + 1) * 25] *= 40
        prog[name] = p / p.sum()
    X, labels = [], []
    for t in names[:4]:
        for _ in range(n_per):
            w1 = rng.beta(2, 3) if t in "AB" and rng.random() < 0.5 else 0.0
            w0 = rng.beta(2, 4) if rng.random() < 0.4 else 0.0
            mix = (1 - w1 - w0 * (1 - w1)) * prog[t] + w1 * prog["P1"] + w0 * (1 - w1) * prog["P0"]
            X.append(rng.multinomial(rng.integers(300, 600), mix))
            labels.append(t)
    a = ad.AnnData(sp.csr_matrix(np.array(X, dtype=np.float32)))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["cell_type"] = pd.Categorical(labels, categories=names[:4])
    return a, prog


def lineage_mask(a):
    m = schlda.allowed_topics(a, "cell_type", 1)
    m["AB"] = m.index.isin(["A", "B"])
    return m


def assert_mask_respected(res):
    m = res.allowed.to_numpy()
    node = res.geometry.cell_node
    for i, t in enumerate(res.geometry.cell_types):
        assert (res.theta.to_numpy()[node == i][:, ~m[i]] == 0).all(), t


def test_shared_subset_topic_recovers_lineage_program(tmp_path):
    a, prog = simulate_lineage()
    res = schlda.fit_hlda(a, "cell_type", allowed=lineage_mask(a), output_dir=tmp_path, **FIT)
    assert res.activity_topics == ["V1", "AB"]
    assert cos(res.beta["AB"].to_numpy(), prog["P1"]) > 0.9
    assert cos(res.beta["V1"].to_numpy(), prog["P0"]) > 0.9
    assert (res.theta.loc[a.obs.cell_type.isin(["C", "D"]), "AB"] == 0).all()
    assert_mask_respected(res)
    saved = pd.read_csv(tmp_path / "allowed_topics.csv", index_col=0)
    assert saved.astype(bool).equals(res.allowed.rename_axis(saved.index.name))
    assert res.params["geometry"]["allowed"]["C"] == ["C", "V1"]


def test_identity_subtopics(tmp_path):
    rng = np.random.default_rng(4)
    a, act, _ = simulate()
    X = a.X.toarray()
    is_a = (a.obs.cell_type == "A").to_numpy()
    sub = np.zeros(a.n_obs, bool)
    sub[np.where(is_a)[0][::2]] = True
    p = np.full(a.n_vars, 1e-3)
    p[100:120] = 1.0
    X[sub] = X[sub] + rng.multinomial(300, p / p.sum(), size=sub.sum())
    a.X = sp.csr_matrix(X)
    mask = schlda.allowed_topics(a, "cell_type", 1).rename(columns={"A": "A_1"})
    mask.insert(1, "A_2", mask.index == "A")
    res = schlda.fit_hlda(a, "cell_type", allowed=mask, output_dir=tmp_path, **FIT)
    assert res.identity_topics == ["A_1", "A_2", "B", "C"]
    tA = res.theta.loc[is_a, ["A_1", "A_2"]].to_numpy()
    assert np.abs(tA[sub[is_a]].mean(0) - tA[~sub[is_a]].mean(0)).max() > 0.3
    assert cos(res.beta["V1"].to_numpy(), act) > 0.9
    assert np.allclose(res.identity_share() + res.theta["V1"], 1)
    assert_mask_respected(res)


def test_custom_reproducible_threads_and_prior(tmp_path):
    a, _ = simulate_lineage()
    kw = dict(allowed=lineage_mask(a), prior_scale_identity=0.05, n_threads=2,
              **{**FIT, "n_loops": 60, "burn_in": 30, "thin": 5, "init": "data_driven"})
    r1 = schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / "1", **kw)
    r2 = schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / "2", **kw)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())
    tp = pd.read_csv(tmp_path / "1" / "theta_prior.csv", index_col=0)
    assert np.allclose(tp.identity, 0.05 * np.asarray(a.X.sum(1)).ravel())
    assert_mask_respected(r1)


def test_plots_and_h5ad_with_custom_mask(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    a, _ = simulate_lineage()
    res = schlda.fit_hlda(a, "cell_type", allowed=lineage_mask(a), output_dir=tmp_path,
                          **{**FIT, "n_loops": 40, "burn_in": 20, "thin": 5})
    schlda.structure_plot(res, save=tmp_path / "s.png")
    schlda.theta_heatmap(res, save=tmp_path / "h.png")
    a.write_h5ad(tmp_path / "fit.h5ad")
    assert ad.read_h5ad(tmp_path / "fit.h5ad").uns["schlda"]["topic_types"].tolist() == res.topics.topic_type.tolist()
