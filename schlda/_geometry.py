"""Topic geometry: which topics each cell type may use.

Default: one identity topic per cell type plus ``m`` activity topics ``V1 .. Vm`` shared by every cell.
Custom: ``identity_topics`` (identity topics per cell type) and/or ``shared_topics`` (each shared topic and
the cell types that may use it). Topic kinds are stated, never inferred; the permission mask is derived.
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
        names, types = np.array(self.topic_names), np.array(self.cell_types)
        is_id = self.identity_mask()
        return {"kind": self.kind,
                "identity_topics": {t: names[row & is_id].tolist() for t, row in zip(self.cell_types, self.allowed)},
                "shared_topics": {k: types[col].tolist() for k, col, i in zip(names, self.allowed.T, is_id) if not i},
                "allowed": {t: names[row].tolist() for t, row in zip(self.cell_types, self.allowed)}}


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


def _count(v, what):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {type(v).__name__}.")
    if v < 0:
        raise ValueError(f"{what} must be >= 0, got {v}.")
    return int(v)


def _identity_names(types, spec):
    """Identity topic names per cell type: one named after the type unless ``spec`` says otherwise."""
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError("identity_topics must be a dict {cell type: number of topics or list of names}.")
    spec = {str(k): v for k, v in spec.items()}
    unknown = sorted(set(spec) - set(types))
    if unknown:
        raise ValueError(f"identity_topics names cell type(s) {unknown} not in the data. Cell types: {types}")
    out = {}
    for t in types:
        v = spec.get(t, 1)
        if isinstance(v, (list, tuple)):
            if not all(isinstance(x, str) for x in v):
                raise TypeError(f"identity_topics[{t!r}]: topic names must be strings.")
            out[t] = list(v)
        else:
            n = _count(v, f"identity_topics[{t!r}]")
            out[t] = [t] if n == 1 else [f"{t}_{j + 1}" for j in range(n)]
    return out


def _shared(types, n_activity_topics, spec):
    """Shared topic name -> cell types that may use it."""
    if spec is None:
        if n_activity_topics is None:
            raise TypeError("Give n_activity_topics or shared_topics.")
        return {f"V{i + 1}": list(types) for i in range(_count(n_activity_topics, "n_activity_topics"))}
    if n_activity_topics is not None:
        raise ValueError("shared_topics replaces n_activity_topics; give one of them.")
    if not isinstance(spec, dict):
        raise TypeError('shared_topics must be a dict {topic name: "all" or list of cell types}.')
    out = {}
    for name, members in spec.items():
        if isinstance(members, str):
            if members != "all":
                raise ValueError(f'shared_topics[{name!r}] must be "all" or a list of cell types '
                                 f'(for one type write [{members!r}]).')
            members = list(types)
        elif isinstance(members, (list, tuple)):
            members = [str(m) for m in members]
            if not members or len(set(members)) != len(members):
                raise ValueError(f"shared_topics[{name!r}] must list each cell type once, at least one.")
            unknown = sorted(set(members) - set(types))
            if unknown:
                raise ValueError(f"shared_topics[{name!r}] names cell type(s) {unknown} not in the data. "
                                 f"Cell types: {types}")
        else:
            raise TypeError(f'shared_topics[{name!r}] must be "all" or a list of cell types.')
        out[str(name)] = members
    return out


def build_geometry(obs, cell_type_key, n_activity_topics=None, identity_topics=None, shared_topics=None):
    lab = _labels(obs, cell_type_key)
    types = [str(v) for v in identity_order(lab)]
    own = _identity_names(types, identity_topics)
    shared = _shared(types, n_activity_topics, shared_topics)
    names = [k for t in types for k in own[t]] + list(shared)
    if len(set(names)) != len(names):
        dup = sorted({k for k in names if names.count(k) > 1})
        raise ValueError(f"Topic names {dup} are used more than once (cell type labels name their identity "
                         f"topics, so they must not collide with shared topic names).")
    allowed = np.zeros((len(types), len(names)), dtype=bool)
    col = 0
    for i, t in enumerate(types):
        allowed[i, col:col + len(own[t])] = True
        col += len(own[t])
    for members in shared.values():
        allowed[:, col] = [t in members for t in types]
        col += 1
    empty = [t for t, row in zip(types, allowed) if not row.any()]
    if empty:
        raise ValueError(f"Cell type(s) {empty} have no identity topic and no shared topic.")
    n_id = sum(len(v) for v in own.values())
    topic_types = ["identity"] * n_id + ["activity"] * len(shared)
    kind = "default" if identity_topics is None and shared_topics is None else "custom"
    idx = {v: i for i, v in enumerate(types)}
    cell_node = np.array([idx[str(v)] for v in lab.to_numpy()], dtype=np.int32)
    return TopicGeometry(kind, types, names, topic_types, allowed, cell_node, lab)
