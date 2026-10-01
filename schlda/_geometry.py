"""Topic geometry: which topics each cell type may use.

Default: one identity topic per cell type plus ``m`` activity topics ``V1 .. Vm`` shared by every cell.
Custom: ``allowed``, a cell type x topic boolean DataFrame. A topic allowed for exactly one cell type is
an identity topic of that type; any other is an activity (shared) topic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TopicGeometry:
    """Cell types, topics and the type x topic permission mask of a fit (``schlda.topic_geometry``)."""

    kind: str
    cell_types: list
    topic_names: list
    topic_types: list
    allowed: np.ndarray
    cell_node: np.ndarray
    labels: pd.Series

    @property
    def n_topics(self):
        return len(self.topic_names)

    def allowed_frame(self):
        return pd.DataFrame(self.allowed, index=pd.Index(self.cell_types, name="cell_type"),
                            columns=self.topic_names)

    def topics_frame(self):
        return pd.DataFrame({"topic_id": range(self.n_topics), "topic_name": self.topic_names,
                             "topic_type": self.topic_types})

    def csr(self, mask=None):
        """Permitted topics per cell type as (topics, offsets); ``mask`` restricts the columns."""
        m = self.allowed if mask is None else self.allowed & mask[None, :]
        offsets = np.concatenate([[0], np.cumsum(m.sum(1))]).astype(np.int32)
        return np.nonzero(m)[1].astype(np.int32), offsets

    def identity_mask(self):
        return np.array([t == "identity" for t in self.topic_types])

    def describe(self):
        n = np.bincount(self.cell_node, minlength=len(self.cell_types))
        names = np.array(self.topic_names)
        w = max(len(t) for t in self.cell_types)
        lines = [f"{t:<{w}}  ({c:,} cells): {', '.join(names[row])}"
                 for t, c, row in zip(self.cell_types, n, self.allowed)]
        shared = [f"  {k} <- {'all' if col.all() else ', '.join(np.array(self.cell_types)[col])}"
                  for k, kind, col in zip(self.topic_names, self.topic_types, self.allowed.T) if kind == "activity"]
        return "\n".join(lines + (["shared topics:"] + shared if shared else []))

    def record(self):
        names = np.array(self.topic_names)
        return {"kind": self.kind, "allowed": {t: names[row].tolist() for t, row in zip(self.cell_types, self.allowed)}}


def identity_order(labels):
    """Values in category order when categorical, else sorted."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        used = set(labels.unique())
        return [c for c in labels.cat.categories if c in used]
    return sorted(labels.unique())


def _labels(obs, cell_type_key):
    if not isinstance(cell_type_key, str):
        raise TypeError("cell_type_key must be the name of one adata.obs column.")
    if cell_type_key not in obs:
        raise KeyError(f"{cell_type_key!r} not in adata.obs. Available columns: {list(obs.columns)}")
    lab = obs[cell_type_key]
    if lab.isna().any():
        raise ValueError(f"adata.obs[{cell_type_key!r}] has {int(lab.isna().sum())} missing value(s). "
                         f"Every cell needs an identity to condition on.")
    return lab


def default_allowed(cell_types, n_activity_topics):
    """The default mask: identity topic per type (named after it) and V1..Vm shared by all."""
    types = [str(t) for t in cell_types]
    m = pd.DataFrame(np.eye(len(types), dtype=bool), index=pd.Index(types, name="cell_type"), columns=types)
    for i in range(n_activity_topics):
        m[f"V{i + 1}"] = True
    return m


def build_geometry(obs, cell_type_key, n_activity_topics=None, allowed=None):
    lab = _labels(obs, cell_type_key)
    types = [str(v) for v in identity_order(lab)]
    if allowed is None:
        if n_activity_topics is None:
            raise TypeError("Give n_activity_topics (default geometry) or allowed (custom geometry).")
        if isinstance(n_activity_topics, (bool, np.bool_)) or not isinstance(n_activity_topics, (int, np.integer)):
            raise TypeError(f"n_activity_topics must be an int, got {type(n_activity_topics).__name__}.")
        if n_activity_topics < 0:
            raise ValueError(f"n_activity_topics must be >= 0, got {n_activity_topics}.")
        n_act = int(n_activity_topics)
        names = types + [f"V{i + 1}" for i in range(n_act)]
        if len(set(names)) != len(names):
            raise ValueError(f"Topic names are not unique: {names}. Cell type labels must "
                             f"not collide with the activity topic names V1..V{n_act}.")
        m, kind = default_allowed(types, n_act), "default"
        topic_types = ["identity"] * len(types) + ["activity"] * n_act
    else:
        if n_activity_topics is not None:
            raise ValueError("allowed replaces n_activity_topics; give one of them.")
        m, kind = _check_mask(allowed, types), "custom"
        topic_types = ["identity" if r == 1 else "activity" for r in m.sum(0)]
    idx = {v: i for i, v in enumerate(types)}
    cell_node = np.array([idx[str(v)] for v in lab.to_numpy()], dtype=np.int32)
    return TopicGeometry(kind, types, list(m.columns), topic_types, m.to_numpy(), cell_node, lab)


def _check_mask(allowed, types):
    if not isinstance(allowed, pd.DataFrame):
        raise TypeError("allowed must be a DataFrame: one row per cell type, one boolean column per topic.")
    m = allowed.copy()
    m.index, m.columns = m.index.map(str), m.columns.map(str)
    if m.index.duplicated().any() or m.columns.duplicated().any():
        raise ValueError("allowed has duplicated cell type or topic names.")
    if not all(pd.api.types.is_bool_dtype(t) or pd.api.types.is_integer_dtype(t) for t in m.dtypes) \
            or not m.isin([0, 1]).all().all():
        raise ValueError("allowed must hold booleans (or 0/1).")
    missing = sorted(set(types) - set(m.index))
    if missing:
        raise ValueError(f"allowed has no row for cell type(s) {missing}.")
    m = m.loc[types].astype(bool)
    if not m.any(axis=0).all():
        raise ValueError(f"Topic(s) {list(m.columns[~m.any(axis=0)])} are allowed for no cell type present.")
    if not m.any(axis=1).all():
        raise ValueError(f"Cell type(s) {list(m.index[~m.any(axis=1)])} may use no topic.")
    return m
