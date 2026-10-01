import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

import schlda
from make_golden import CASES, RUN
from test_fit import simulate, cos

GOLD = None
FIT = dict(n_loops=300, burn_in=150, thin=10, seed=2, init="random", verbose=False)
FLAT_MASK = pd.DataFrame([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], index=["A", "B", "C"],
                         columns=["A", "B", "C", "V1"]).astype(bool)


def golden(name):
    global GOLD
    if GOLD is None:
        from pathlib import Path
        GOLD = np.load(Path(__file__).parent / "data" / "golden_v040.npz")
    return GOLD[f"{name}_theta"], GOLD[f"{name}_beta"]


def equivalent_calls(m):
    """Ways of asking for the v0.4.0 geometry; every one must reproduce it exactly."""
    yield "positional", dict(n_activity_topics=m)
    yield "dict_counts", dict(n_activity_topics=m, n_identity_topics={"B": 1})
    yield "empty_groups", dict(n_activity_topics=m, groups={})
    yield "topicless_group", dict(n_activity_topics=m, groups={"g": ["A", "B"]}, n_group_topics=0)
    if m == 1:
        yield "mask", dict(allowed=FLAT_MASK)


@pytest.mark.parametrize("case", list(CASES))
def test_reproduces_v040(tmp_path, case):
    kw = dict(CASES[case])
    m = kw.pop("n_activity_topics")
    theta0, beta0 = golden(case)
    for label, geo in equivalent_calls(m):
        a, _, _ = simulate()
        res = schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / label, **geo, **RUN, **kw)
        assert np.array_equal(res.theta.to_numpy(), theta0), label
        assert np.array_equal(res.beta.to_numpy(), beta0), label


def test_reproduces_v040_two_columns(tmp_path):
    a, _, _ = simulate()
    a.obs["lineage"] = pd.Categorical(np.where(a.obs.cell_type == "C", "g2", "g1"))
    kw = dict(CASES["data_m2_t2_kappa"])
    m = kw.pop("n_activity_topics")
    res = schlda.fit_hlda(a, ["lineage", "cell_type"], m, n_group_topics=0, output_dir=tmp_path, **RUN, **kw)
    theta0, beta0 = golden("data_m2_t2_kappa")
    assert np.array_equal(res.theta.to_numpy(), theta0) and np.array_equal(res.beta.to_numpy(), beta0)


# geometry only

def obs(labels, **cols):
    return ad.AnnData(sp.csr_matrix(np.ones((len(labels), 3))),
                      obs=pd.DataFrame({"cell_type": labels, **cols}, index=[f"c{i}" for i in range(len(labels))]))


def test_flat_geometry():
    g = schlda.topic_geometry(obs(["B", "A", "B"]), "cell_type", 2)
    assert g.kind == "flat" and g.nodes == ["A", "B"] and g.topic_names == ["A", "B", "V1", "V2"]
    assert g.allowed.tolist() == [[True, False, True, True], [False, True, True, True]]
    assert g.cell_node.tolist() == [1, 0, 1]


def test_groups_tree():
    a = obs(["B", "CD4T", "CD8T", "Mono", "DC"])
    g = schlda.topic_geometry(a, "cell_type", 1, groups={"lymphoid": ["B", "T"], "T": ["CD4T", "CD8T"],
                                                          "myeloid": ["Mono", "DC"]},
                              n_identity_topics={"CD4T": 2}, n_group_topics={"T": 0})
    m = g.allowed_frame()
    assert g.topic_names == ["B", "CD4T_1", "CD4T_2", "CD8T", "DC", "Mono", "lymphoid", "myeloid", "V1"]
    assert m.loc["CD4T"][m.loc["CD4T"]].index.tolist() == ["CD4T_1", "CD4T_2", "lymphoid", "V1"]
    assert m.loc["DC"][m.loc["DC"]].index.tolist() == ["DC", "myeloid", "V1"]
    assert g.parent["CD4T"] == "T" and g.parent["T"] == "lymphoid" and g.parent["lymphoid"] is None
    assert dict(zip(g.topic_names, g.topic_types))["lymphoid"] == "group"
    assert "└── " in g.describe() and "CD4T: CD4T_1, CD4T_2" in g.describe()


def test_columns_tree_and_coarse_cells():
    a = obs(["CD4T", None, "B", "Mono", None], lineage=["lymphoid", "lymphoid", "lymphoid", "myeloid", "Mega"])
    g = schlda.topic_geometry(a, ["lineage", "cell_type"], 1)
    m = g.allowed_frame()
    assert g.nodes == ["Mega", "B", "CD4T", "Mono", "lymphoid", "myeloid"]  # Mega has no subtypes: a leaf
    assert m.loc["Mega"][m.loc["Mega"]].index.tolist() == ["Mega", "V1"]
    assert m.loc["Mono"][m.loc["Mono"]].index.tolist() == ["Mono", "myeloid", "V1"]
    assert m.loc["lymphoid"][m.loc["lymphoid"]].index.tolist() == ["lymphoid", "V1"]
    assert g.nodes[g.cell_node[1]] == "lymphoid" and g.labels.tolist()[1] == "lymphoid"


def test_coarse_label_in_single_column():
    g = schlda.topic_geometry(obs(["A", "B", "g"]), "cell_type", 0, groups={"g": ["A", "B"]})
    assert g.allowed_frame().loc["g"].tolist() == [False, False, True]


def test_unused_members_are_pruned():
    g = schlda.topic_geometry(obs(["A", "B"]), "cell_type", 1, groups={"g": ["A", "Z"], "h": ["Y"]})
    assert g.nodes == ["A", "B", "g"] and "Z" not in g.topic_names and "h" not in g.topic_names


@pytest.mark.parametrize("kw, err", [
    (dict(n_identity_topics={"Q": 2}), ValueError),
    (dict(groups={"g": ["A"], "h": ["A"]}), ValueError),
    (dict(groups={"g": ["h"], "h": ["g", "A"]}), ValueError),
    (dict(groups={"g": "g"}), ValueError),
    (dict(groups={"V1": ["A"]}), ValueError),
    (dict(groups={"root": ["A"]}), ValueError),
    (dict(n_identity_topics={"A": 0}), ValueError),
    (dict(n_identity_topics=1.5), ValueError),
    (dict(allowed=FLAT_MASK), ValueError),
])
def test_geometry_errors(kw, err):
    a = obs(["A", "B", "C"])
    with pytest.raises(err):
        schlda.topic_geometry(a, "cell_type", 0 if "n_identity_topics" in kw else 1, **kw)


@pytest.mark.parametrize("mask", [
    FLAT_MASK.drop("C"),
    FLAT_MASK.assign(A=False),
    FLAT_MASK.assign(V2=False),
    FLAT_MASK.astype(float) * 0.5,
])
def test_mask_errors(mask):
    with pytest.raises(ValueError):
        schlda.topic_geometry(obs(["A", "B", "C"]), "cell_type", allowed=mask)


def test_missing_and_ambiguous_annotation():
    with pytest.raises(ValueError):
        schlda.topic_geometry(obs(["A", None]), "cell_type", 1)
    with pytest.raises(ValueError):  # finer level set below a missing coarse one
        schlda.topic_geometry(obs(["A", "B"], lineage=["g", None]), ["lineage", "cell_type"], 1)
    with pytest.raises(ValueError):  # same type under two lineages
        schlda.topic_geometry(obs(["A", "A"], lineage=["g", "h"]), ["lineage", "cell_type"], 1)
    with pytest.raises(ValueError):
        schlda.topic_geometry(obs(["A", "B"], lineage=["g", "g"]), ["lineage", "cell_type"], 1, groups={"x": ["g"]})
    with pytest.raises(TypeError):
        schlda.topic_geometry(obs(["A", "B"]), "cell_type")


# fits

def simulate_lineage(seed=0, n_per=60, n_genes=150):
    """Types A, B in group g1 and C, D in g2; P1 is used by g1 cells only, P0 by every cell."""
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


def assert_mask_respected(res):
    m = res.allowed
    node = res.geometry.cell_node
    for i, v in enumerate(res.geometry.nodes):
        cells = node == i
        if cells.any():
            assert (res.theta.to_numpy()[cells][:, ~m.loc[v].to_numpy()] == 0).all(), v


def test_group_topic_recovers_lineage_program(tmp_path):
    a, prog = simulate_lineage()
    res = schlda.fit_hlda(a, "cell_type", 1, groups={"g1": ["A", "B"], "g2": ["C", "D"]},
                          n_group_topics={"g2": 0}, output_dir=tmp_path, **FIT)
    assert res.group_topics == ["g1"] and res.topics_of("g1") == ["g1"]
    assert cos(res.beta["g1"].to_numpy(), prog["P1"]) > 0.9
    assert cos(res.beta["V1"].to_numpy(), prog["P0"]) > 0.9
    assert_mask_respected(res)
    assert (res.theta.loc[a.obs.cell_type.isin(["C", "D"]), "g1"] == 0).all()
    assert (tmp_path / "allowed_topics.csv").exists() and res.params["geometry"]["kind"] == "tree"
    assert "group" in pd.read_csv(tmp_path / "topic_info.csv").topic_type.tolist()


def test_identity_subtopics(tmp_path):
    rng = np.random.default_rng(4)
    a, act, _ = simulate()
    X = a.X.toarray()
    sub = np.zeros(a.n_obs, bool)
    is_a = (a.obs.cell_type == "A").to_numpy()
    sub[np.where(is_a)[0][::2]] = True
    p = np.full(a.n_vars, 1e-3)
    p[100:120] = 1.0
    p /= p.sum()
    X[sub] = X[sub] + rng.multinomial(300, p, size=sub.sum())
    a.X = sp.csr_matrix(X)
    res = schlda.fit_hlda(a, "cell_type", 1, n_identity_topics={"A": 2}, output_dir=tmp_path, **FIT)
    assert res.identity_topics == ["A_1", "A_2", "B", "C"]
    tA = res.theta.loc[is_a, ["A_1", "A_2"]].to_numpy()
    split = np.abs(tA[sub[is_a]].mean(0) - tA[~sub[is_a]].mean(0)).max()
    assert split > 0.3
    assert cos(res.beta["V1"].to_numpy(), act) > 0.9
    assert np.allclose(res.identity_share() + res.theta["V1"], 1)
    assert_mask_respected(res)


def test_mask_any_geometry(tmp_path):
    a, _, _ = simulate()
    mask = FLAT_MASK.assign(AB=[True, True, False])
    res = schlda.fit_hlda(a, "cell_type", allowed=mask, output_dir=tmp_path, **FIT)
    assert res.topics.set_index("topic_name").topic_type.to_dict() == \
        {"A": "identity", "B": "identity", "C": "identity", "V1": "activity", "AB": "activity"}
    assert (res.theta.loc[a.obs.cell_type == "C", "AB"] == 0).all()
    assert_mask_respected(res)


def test_coarse_cells_fit(tmp_path):
    a, _, _ = simulate()
    lab = a.obs.cell_type.astype(str).to_numpy()
    lab[:10] = "g"
    a.obs["cell_type"] = lab
    res = schlda.fit_hlda(a, "cell_type", 1, groups={"g": ["A", "B"]}, output_dir=tmp_path,
                          **{**FIT, "init": "data_driven"})
    assert (res.theta.iloc[:10][["A", "B", "C"]] == 0).all().all()
    assert_mask_respected(res)


def test_tree_reproducible_and_threads(tmp_path):
    a, _ = simulate_lineage()
    kw = dict(groups={"g1": ["A", "B"]}, n_identity_topics={"C": 2}, prior_scale_identity=0.05, n_threads=2,
              **{**FIT, "n_loops": 60, "burn_in": 30, "thin": 5, "init": "data_driven"})
    r1 = schlda.fit_hlda(a, "cell_type", 2, output_dir=tmp_path / "1", **kw)
    r2 = schlda.fit_hlda(a, "cell_type", 2, output_dir=tmp_path / "2", **kw)
    assert np.array_equal(r1.theta.to_numpy(), r2.theta.to_numpy())
    tp = pd.read_csv(tmp_path / "1" / "theta_prior.csv", index_col=0)
    assert np.allclose(tp.identity, 0.05 * np.asarray(a.X.sum(1)).ravel())
    assert_mask_respected(r1)


def test_plots_with_group_topics(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    a, _ = simulate_lineage()
    res = schlda.fit_hlda(a, "cell_type", 1, groups={"g1": ["A", "B"]}, output_dir=tmp_path,
                          **{**FIT, "n_loops": 40, "burn_in": 20, "thin": 5})
    schlda.structure_plot(res, save=tmp_path / "s.png")
    schlda.theta_heatmap(res, save=tmp_path / "h.png")
    a.write_h5ad(tmp_path / "fit.h5ad")
    assert ad.read_h5ad(tmp_path / "fit.h5ad").uns["schlda"]["topic_types"].tolist() == res.topics.topic_type.tolist()
