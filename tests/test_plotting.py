import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import schlda
from schlda.plotting import _sort_within

CELLS = ["c0", "c1", "c2", "c3", "c4", "c5"]


@pytest.fixture
def theta():
    return pd.DataFrame(
        [[0.60, 0.00, 0.10, 0.30],
         [0.97, 0.00, 0.02, 0.01],
         [0.70, 0.00, 0.25, 0.05],
         [0.00, 0.50, 0.50, 0.00],
         [0.00, 0.90, 0.04, 0.06],
         [0.00, 0.20, 0.10, 0.70]],
        index=CELLS, columns=["B", "T", "V1", "V2"],
    )


LABELS = np.array(["B", "B", "B", "T", "T", "T"])


def test_dominant_order_blocks_then_weight(theta):
    order = schlda.cell_order(theta, LABELS)
    # B: c2 (V1 block), c0 (V2 block), c1 (quiet); T: c3 (V1), c5 then c4 (V2 block)
    assert theta.index[order].tolist() == ["c2", "c0", "c1", "c3", "c5", "c4"]


def test_single_activity_reduces_to_activity_sort(theta):
    idx = np.arange(3)
    a = _sort_within(theta, idx, "dominant", ["V1"], "B", 0.05)
    b = _sort_within(theta, idx, "activity", ["V1"], "B", 0.05)
    assert a.tolist() == b.tolist() == [2, 0, 1]


def test_generic_theta_with_activity_by_index_and_normalisation(theta):
    L = theta.to_numpy() * 3.0  # rows sum to 3, like NMF loadings
    with pytest.warns(UserWarning, match="normalising"):
        order = schlda.cell_order(L, LABELS, activity_topics=[2, 3])
    assert order.tolist() == schlda.cell_order(theta, LABELS).tolist()


def test_explicit_order_shared_across_thetas(theta):
    order = schlda.cell_order(theta, LABELS)
    other = theta.iloc[:, ::-1]  # same cells, different topic layout
    fig, ax = schlda.structure_plot(other, LABELS, activity_topics=["V1", "V2"], order=order)
    assert ax.get_xlim()[1] == len(order) - 1
    fig, ax = schlda.structure_plot(theta, LABELS, order=theta.index[order[:4]])
    assert ax.get_xlim()[1] == 3  # cells outside `order` are dropped
    matplotlib.pyplot.close("all")


def test_embed_sort_runs(theta):
    order = schlda.cell_order(theta, LABELS, sort_by="embed")
    assert sorted(order.tolist()) == list(range(6))


def test_errors(theta):
    with pytest.raises(ValueError, match="entries"):
        schlda.cell_order(theta, LABELS[:5])
    with pytest.raises(KeyError):
        schlda.cell_order(theta, LABELS, activity_topics=["V9"])
    with pytest.raises(KeyError):
        schlda.structure_plot(theta, LABELS, order=["nope"])
    identity_only = theta[["B", "T"]].div(theta[["B", "T"]].sum(axis=1), axis=0)
    with pytest.raises(ValueError, match="activity"):
        schlda.cell_order(identity_only, LABELS, sort_by="dominant")


def test_heatmap_meta_cells(theta):
    fig, ax = schlda.theta_heatmap(theta, LABELS, cells_per_group=2)
    assert ax.images[0].get_array().shape == (4, 4)  # topics x meta-cells
    matplotlib.pyplot.close("all")
