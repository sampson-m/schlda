"""Topic geometry: one identity topic per annotated cell type, plus activity topics each shared by a set of cell
types. ``n_activity_topics=m`` gives ``V1 .. Vm`` shared by every type; ``activity=`` gives the sharing explicitly
as a table with columns ``cell_type, activity_topics`` (the topics each type may use, ``;``-separated) or
``topic, cell_types`` (the types each topic is shared by, or ``all``), a CSV path to such a table, or a dict in
either orientation. A cell's tokens may only use its own type's topic and the activity topics its type is given.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "; "


class Geometry:
    def __init__(self, cell_types, activity=None, labels=None):
        self.cell_types = [str(t) for t in cell_types]
        self.activity = {str(k): [str(t) for t in v] for k, v in (activity or {}).items()}
        self.labels = labels
        for k, v in self.activity.items():
            if k in self.cell_types:
                raise ValueError(f"activity topic {k!r} has the name of a cell type.")
            if not v:
                raise ValueError(f"activity topic {k!r} is shared by no cell type.")
            bad = [t for t in v if t not in self.cell_types]
            if bad:
                raise ValueError(f"activity topic {k!r}: unknown cell type(s) {bad}. Cell types: {self.cell_types}")
            if len(set(v)) != len(v):
                raise ValueError(f"activity topic {k!r}: each cell type once.")

    @property
    def identity_topics(self):
        return list(self.cell_types)

    @property
    def activity_topics(self):
        return list(self.activity)

    @property
    def topic_names(self):
        return self.cell_types + list(self.activity)

    @property
    def topic_types(self):
        return ["identity"] * len(self.cell_types) + ["activity"] * len(self.activity)

    @property
    def n_topics(self):
        return len(self.cell_types) + len(self.activity)

    def identity_mask(self):
        return np.array(self.topic_types) == "identity"

    def activity_of(self, cell_type):
        return [k for k, v in self.activity.items() if cell_type in v]

    @property
    def allowed(self):
        n = len(self.cell_types)
        idx = {t: i for i, t in enumerate(self.cell_types)}
        m = np.zeros((n, self.n_topics), dtype=bool)
        m[np.arange(n), np.arange(n)] = True
        for j, v in enumerate(self.activity.values()):
            m[[idx[t] for t in v], n + j] = True
        return m

    def allowed_frame(self):
        return pd.DataFrame(self.allowed, index=pd.Index(self.cell_types, name="cell_type"), columns=self.topic_names)

    def csr(self):
        m = self.allowed
        offsets = np.concatenate([[0], np.cumsum(m.sum(1))]).astype(np.int32)
        return np.nonzero(m)[1].astype(np.int32), offsets

    @property
    def cell_node(self):
        idx = {t: i for i, t in enumerate(self.cell_types)}
        return np.array([idx[str(v)] for v in self.labels.to_numpy()], dtype=np.int32)

    def alphas(self, alpha_identity, alpha_activity):
        return np.array([alpha_identity] * len(self.cell_types) + [alpha_activity] * len(self.activity), dtype=np.float64)

    def _members_str(self, k):
        return "all" if len(self.activity[k]) == len(self.cell_types) else SEP.join(self.activity[k])

    def to_frame(self, alpha_identity=None, alpha_activity=None):
        """The table ``topic, kind, cell_types, alpha``."""
        f = pd.DataFrame({"topic": self.topic_names, "kind": self.topic_types,
                          "cell_types": self.cell_types + [self._members_str(k) for k in self.activity]})
        if alpha_identity is not None:
            f["alpha"] = self.alphas(alpha_identity, alpha_activity)
        return f

    def to_csv(self, path, **kw):
        self.to_frame(**kw).to_csv(path, index=False)

    def bind(self, labels):
        types = [str(v) for v in identity_order(labels)]
        if set(types) != set(self.cell_types):
            raise ValueError(f"geometry cell types {self.cell_types} differ from the data's {types}.")
        return Geometry(self.cell_types, self.activity, labels)

    def describe(self):
        n = (np.bincount(self.cell_node, minlength=len(self.cell_types)) if self.labels is not None
             else [None] * len(self.cell_types))
        w = max(len(t) for t in self.cell_types)
        return "\n".join(f"{t:<{w}}" + (f"  ({c:,} cells)" if c is not None else "") +
                         f": {', '.join([t] + self.activity_of(t))}" for t, c in zip(self.cell_types, n))

    def __repr__(self):
        return f"Geometry({len(self.cell_types)} identity + {len(self.activity)} activity topics)"


def identity_order(labels):
    if isinstance(labels.dtype, pd.CategoricalDtype):
        used = set(labels.unique())
        return [c for c in labels.cat.categories if c in used]
    return sorted(labels.unique())


def get_labels(obs, cell_type_key):
    if not isinstance(cell_type_key, str):
        raise TypeError("cell_type_key must be the name of one adata.obs column.")
    if cell_type_key not in obs:
        raise KeyError(f"{cell_type_key!r} not in adata.obs. Available columns: {list(obs.columns)}")
    lab = obs[cell_type_key]
    if lab.isna().any():
        raise ValueError(f"adata.obs[{cell_type_key!r}] has {int(lab.isna().sum())} missing value(s). "
                         f"Every cell needs an identity to condition on.")
    return lab


def build(labels, n_activity_topics=None, activity=None):
    types = [str(v) for v in identity_order(labels)]
    if activity is not None:
        if n_activity_topics is not None:
            raise ValueError("activity replaces n_activity_topics; give one of them.")
        return Geometry(types, parse_activity(activity, types), labels)
    m = _count(n_activity_topics or 0, "n_activity_topics")
    return Geometry(types, {f"V{i + 1}": list(types) for i in range(m)}, labels)


def _split(s):
    return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip()]


def _blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().lower() in ("", "nan")


def parse_activity(spec, types):
    """``{topic: [cell types]}`` in topic order from a table, CSV path or dict in either orientation."""
    if isinstance(spec, (str, Path)):
        spec = pd.read_csv(spec)
    if isinstance(spec, dict):
        keys = [str(k) for k in spec]
        by_type = [k in types for k in keys]
        if all(by_type):
            spec = pd.DataFrame({"cell_type": keys, "activity_topics": [SEP.join(_list(v)) for v in spec.values()]})
        elif not any(by_type):
            spec = pd.DataFrame({"topic": keys, "cell_types": ["all" if isinstance(v, str) and v == "all" else SEP.join(_list(v))
                                                                for v in spec.values()]})
        else:
            raise ValueError("activity dict: keys must all be cell types (type -> topics) or all topic names (topic -> cell types).")
    if not isinstance(spec, pd.DataFrame):
        raise TypeError("activity must be a DataFrame, a CSV path or a dict.")
    cols = set(spec.columns)
    out = {}
    if {"cell_type", "activity_topics"} <= cols:
        for r in spec.itertuples(index=False):
            t = str(r.cell_type).strip()
            if t not in types:
                raise ValueError(f"activity table: cell type {t!r} not in the data. Cell types: {types}")
            for k in ([] if _blank(r.activity_topics) else _split(r.activity_topics)):
                out.setdefault(k, [])
                if t in out[k]:
                    raise ValueError(f"activity table: {t!r} lists topic {k!r} twice.")
                out[k].append(t)
        for k in out:
            out[k] = [t for t in types if t in out[k]]
    elif {"topic", "cell_types"} <= cols:
        for r in spec.itertuples(index=False):
            k = str(r.topic).strip()
            if k in types or ("kind" in cols and str(getattr(r, "kind")) == "identity"):
                continue                                   # identity rows of a fit's topics.csv
            if k in out:
                raise ValueError(f"activity table: topic {k!r} listed twice.")
            m = str(r.cell_types).strip()
            out[k] = list(types) if m == "all" else _split(m)
    else:
        raise ValueError("activity table needs columns 'cell_type, activity_topics' or 'topic, cell_types'; "
                         f"got {list(spec.columns)}.")
    return out


def _list(v):
    if isinstance(v, str):
        return _split(v)
    return [str(t) for t in v]


def _count(v, what):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {type(v).__name__}.")
    if v < 0:
        raise ValueError(f"{what} must be >= 0, got {v}.")
    return int(v)


def _alpha(v, what="alpha"):
    if not (np.isscalar(v) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) and float(v) > 0):
        raise ValueError(f"{what} must be a positive number, got {v!r}.")
    return float(v)
