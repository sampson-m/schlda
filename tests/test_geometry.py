from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import schlda
from make_golden import CASES, RUN
from test_fit import simulate, cos

GOLD = np.load(Path(__file__).parent / "data" / "golden_v040.npz")
SHORT = dict(n_loops=60, burn_in=30, thin=5, seed=7, verbose=False, plots=False)
TYPES = ["A", "B", "C"]


def equivalent_calls(m):
    """Ways of asking for the v0.4.0 geometry (one identity topic per type, V1..Vm for all); all must match."""
    names = [f"V{i + 1}" for i in range(m)]
    yield "positional", dict(n_activity_topics=m)
    if m:
        yield "by_type", dict(activity={t: names for t in TYPES})
        yield "by_topic", dict(activity={k: "all" for k in names})
        yield "table_by_type", dict(activity=pd.DataFrame({"cell_type": TYPES, "activity_topics": ["; ".join(names)] * 3}))
        yield "table_by_topic", dict(activity=pd.DataFrame({"topic": names, "cell_types": ["all"] * m}))


@pytest.mark.parametrize("case", list(CASES))
def test_reproduces_v040(tmp_path, case):
    kw = dict(CASES[case])
    m = kw.pop("n_activity_topics")
    for label, geo in equivalent_calls(m):
        a, _, _ = simulate()
        res = schlda.fit_hlda(a, "cell_type", output_dir=tmp_path / label, **geo, **RUN, **kw)
        assert np.array_equal(res.theta.to_numpy(), GOLD[f"{case}_theta"]), label
        assert np.array_equal(res.beta.to_numpy(), GOLD[f"{case}_beta"]), label


def test_default_geometry():
    a, _, _ = simulate()
    g = schlda.geometry(a, "cell_type", 2)
    assert g.topic_names == TYPES + ["V1", "V2"] and g.identity_topics == TYPES and g.activity_topics == ["V1", "V2"]
    assert g.allowed.tolist() == [[1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]]
    assert g.to_frame().cell_types.tolist() == TYPES + ["all", "all"]
    assert schlda.geometry(a, "cell_type").n_topics == 3
    assert "A  (60 cells): A, V1, V2" in g.describe()


def test_flat_file_orientations(tmp_path):
    a, _, _ = simulate()
    by_type = pd.DataFrame({"cell_type": TYPES, "activity_topics": ["V1; V2", "V1", ""]})
    by_topic = pd.DataFrame({"topic": ["V1", "V2"], "cell_types": ["A; B", "A"]})
    by_type.to_csv(tmp_path / "t.csv", index=False)
    by_topic.to_csv(tmp_path / "k.csv", index=False)
    frames = [schlda.geometry(a, "cell_type", activity=x).to_frame() for x in
              (by_type, by_topic, tmp_path / "t.csv", str(tmp_path / "k.csv"),
               {"A": ["V1", "V2"], "B": ["V1"], "C": []}, {"V1": ["A", "B"], "V2": ["A"]})]
    for f in frames[1:]:
        assert f.equals(frames[0])
    assert frames[0].cell_types.tolist() == TYPES + ["A; B", "A"]
    g = schlda.geometry(a, "cell_type", activity=by_type)
    assert g.allowed.tolist() == [[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]
    assert g.activity_of("C") == []
    g2 = schlda.geometry(a, "cell_type", activity=by_type.iloc[:1])     # a type left out gets no activity topic
    assert g2.allowed.tolist() == [[1, 0, 0, 1, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    g3 = schlda.geometry(a, "cell_type", activity={"A": "V1, V2", "B": "V1"})
    assert g3.to_frame().equals(frames[0])


def test_fit_with_flat_file_restricts_cells(tmp_path):
    a, act, _ = simulate()
    res = schlda.fit_hlda(a, "cell_type", activity={"V1": ["A", "B"], "V2": ["A"]}, output_dir=tmp_path / "f", **SHORT)
    assert res.activity_topics == ["V1", "V2"]
    th = res.theta
    assert (th.loc[a.obs.cell_type == "C", ["V1", "V2"]].to_numpy() == 0).all()
    assert (th.loc[a.obs.cell_type == "B", "V2"] == 0).all()
    assert th.loc[a.obs.cell_type == "A", ["V1", "V2"]].to_numpy().sum() > 0
    assert res.allowed.loc["C"].tolist() == [False, False, True, False, False]
    again = schlda.fit_hlda(a, "cell_type", activity=tmp_path / "f" / "topics.csv", output_dir=tmp_path / "g", **SHORT)
    assert np.array_equal(again.theta.to_numpy(), th.to_numpy())
    assert schlda.geometry(a, "cell_type", activity=res.topics).to_frame().equals(res.geometry.to_frame())


def test_shared_subset_topic_recovers_restricted_program(tmp_path):
    a, act, w = simulate()
    rng = np.random.default_rng(5)
    X = a.X.toarray()
    extra = rng.gamma(0.3, 1.0, a.n_vars) + 1e-3
    extra[100:] *= 40
    extra /= extra.sum()
    ab = (a.obs.cell_type != "C").to_numpy()
    X[ab] += rng.multinomial(150, extra, size=int(ab.sum()))
    a.X = sp.csr_matrix(X.astype(np.float32))
    res = schlda.fit_hlda(a, "cell_type", activity={"V1": "all", "AB": ["A", "B"]}, output_dir=tmp_path / "s",
                          n_loops=300, burn_in=150, thin=10, seed=2, verbose=False, plots=False)
    assert cos(res.beta["AB"].to_numpy(), extra) > 0.9 and cos(res.beta["V1"].to_numpy(), act) > 0.9


@pytest.mark.parametrize("activity, err", [
    ({"A": ["V1"], "V2": ["B"]}, "keys must all be"),
    ({"V1": ["A", "Z"]}, "unknown cell type"),
    ({"V1": []}, "shared by no cell type"),
    ({"B": ["A"]}, "has the name of a cell type"),
    (pd.DataFrame({"x": [1]}), "needs columns"),
    (pd.DataFrame({"topic": ["V1", "V1"], "cell_types": ["all", "A"]}), "listed twice"),
    (pd.DataFrame({"cell_type": ["A", "A"], "activity_topics": ["V1", "V1"]}), "twice"),
    (pd.DataFrame({"cell_type": ["Q"], "activity_topics": ["V1"]}), "not in the data"),
])
def test_activity_errors(activity, err):
    a, _, _ = simulate()
    with pytest.raises(ValueError, match=err):
        schlda.geometry(a, "cell_type", activity=activity)


def test_argument_errors(tmp_path):
    a, _, _ = simulate()
    with pytest.raises(ValueError, match="replaces"):
        schlda.fit_hlda(a, "cell_type", 1, activity={"V1": "all"}, output_dir=tmp_path / "x", **SHORT)
    with pytest.raises(TypeError):
        schlda.geometry(a, "cell_type", activity=3)
    with pytest.raises(TypeError):
        schlda.geometry(a, "cell_type", 1.5)
    with pytest.raises(KeyError):
        schlda.geometry(a, "nope")
    a.obs["cell_type"] = a.obs["cell_type"].astype(object)
    a.obs.loc[a.obs_names[0], "cell_type"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        schlda.geometry(a, "cell_type")


def test_plots_and_h5ad_with_restricted_geometry(tmp_path):
    a, _, _ = simulate()
    res = schlda.fit_hlda(a, "cell_type", activity={"V1": ["A", "B"]}, output_dir=tmp_path / "p",
                          n_loops=60, burn_in=30, thin=5, seed=7, verbose=False)
    assert (tmp_path / "p" / "structure_plot.png").exists()
    schlda.structure_plot(a)
    assert a.uns["schlda"]["params"]["geometry"]["cell_types"] == TYPES + ["A; B"]
    assert schlda.cell_order(res).shape == (a.n_obs,)
