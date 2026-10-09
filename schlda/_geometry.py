"""Topic geometry: which topics each cell type may use, and the Dirichlet concentration on each.

``geometry(adata, key)`` gives the default (one identity topic per cell type, no shared topic); ``split``,
``share``, ``drop`` and ``alpha`` edit it, ``to_frame`` / ``from_frame`` round-trip it as a table with columns
``topic, kind, cell_types, alpha``. Kinds are ``identity`` (one owning type) and ``activity`` (listed types or
``all``); a type-restricted activity topic is still activity. The permission mask is derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("identity", "activity")
SEP = "; "


@dataclass
class Topic:
    name: str
    kind: str
    cell_types: list
    alpha: float | None = None


@dataclass
class TopicGeometry:
    cell_types: list
    topics: list = field(default_factory=list)
    labels: pd.Series | None = None

    # ── building ──────────────────────────────────────────────────────────
    def split(self, cell_type, n_or_names):
        """Replace ``cell_type``'s identity topics by ``n`` (named ``<type>_1 ..``) or the given names; 0 removes them."""
        t = self._type(cell_type)
        if isinstance(n_or_names, (list, tuple)):
            names = list(n_or_names)
        else:
            n = _count(n_or_names, "n")
            names = [t] if n == 1 else [f"{t}_{j + 1}" for j in range(n)]
        own = [i for i, x in enumerate(self.topics) if x.kind == "identity" and x.cell_types == [t]]
        pos = own[0] if own else len(self.topics)
        keep = [x for i, x in enumerate(self.topics) if i not in own]
        return self._commit(keep[:pos] + [Topic(_name(k), "identity", [t]) for k in names] + keep[pos:])

    def share(self, name, cell_types="all", alpha=None):
        """Add an activity topic used by every type (``"all"``) or by the listed types."""
        members = list(self.cell_types) if isinstance(cell_types, str) and cell_types == "all" else \
            [self._type(t) for t in _list(cell_types, f"share({name!r})")]
        if len(set(members)) != len(members):
            raise ValueError(f"share({name!r}): each cell type once.")
        return self._commit(self.topics + [Topic(_name(name), "activity", members, None if alpha is None else _alpha(alpha))])

    def drop(self, name):
        return self._commit([x for x in self.topics if x is not self._topic(name)])

    def alpha(self, name, value):
        """Set one topic's concentration (``None`` restores the kind default)."""
        self._topic(name).alpha = None if value is None else _alpha(value)
        return self

    # ── views ─────────────────────────────────────────────────────────────
    @property
    def topic_names(self):
        return [x.name for x in self.topics]

    @property
    def topic_types(self):
        return [x.kind for x in self.topics]

    @property
    def n_topics(self):
        return len(self.topics)

    @property
    def allowed(self):
        idx = {t: i for i, t in enumerate(self.cell_types)}
        m = np.zeros((len(self.cell_types), self.n_topics), dtype=bool)
        for j, x in enumerate(self.topics):
            m[[idx[t] for t in x.cell_types], j] = True
        return m

    @property
    def cell_node(self):
        idx = {t: i for i, t in enumerate(self.cell_types)}
        return np.array([idx[str(v)] for v in self.labels.to_numpy()], dtype=np.int32)

    def identity_mask(self):
        return np.array([x.kind == "identity" for x in self.topics])

    def allowed_frame(self):
        return pd.DataFrame(self.allowed, index=pd.Index(self.cell_types, name="cell_type"), columns=self.topic_names)

    def csr(self, mask=None):
        """Permitted topics per cell type as (topics, offsets); ``mask`` restricts the columns."""
        m = self.allowed if mask is None else self.allowed & mask[None, :]
        offsets = np.concatenate([[0], np.cumsum(m.sum(1))]).astype(np.int32)
        return np.nonzero(m)[1].astype(np.int32), offsets

    def resolve_alpha(self, alpha_identity, alpha_activity):
        """Concentration per topic: identity ``alpha_identity / n_sub(type)``, activity ``alpha_activity``, unless set."""
        n_sub = {t: sum(x.kind == "identity" and x.cell_types == [t] for x in self.topics) for t in self.cell_types}
        return np.array([x.alpha if x.alpha is not None else
                         alpha_identity / n_sub[x.cell_types[0]] if x.kind == "identity" else alpha_activity
                         for x in self.topics], dtype=np.float64)

    def to_frame(self, alpha_identity=None, alpha_activity=None):
        """The table ``topic, kind, cell_types, alpha``; alpha filled in when both defaults are given."""
        a = (self.resolve_alpha(alpha_identity, alpha_activity) if alpha_identity is not None
             else [x.alpha for x in self.topics])
        return pd.DataFrame({"topic": self.topic_names, "kind": self.topic_types,
                             "cell_types": ["all" if x.kind == "activity" and len(x.cell_types) == len(self.cell_types)
                                            else SEP.join(x.cell_types) for x in self.topics],
                             "alpha": a}).astype({"alpha": float})

    def to_csv(self, path, **kw):
        self.to_frame(**kw).to_csv(path, index=False)

    @classmethod
    def from_frame(cls, frame, cell_types, labels=None):
        if isinstance(frame, (str, Path)):
            frame = pd.read_csv(frame)
        need = {"topic", "kind", "cell_types"} - set(frame.columns)
        if need:
            raise ValueError(f"geometry table needs columns topic, kind, cell_types (missing {sorted(need)}).")
        g = cls([str(t) for t in cell_types], [], labels)
        for r in frame.itertuples(index=False):
            kind = str(r.kind)
            if kind not in KINDS:
                raise ValueError(f"topic {r.topic!r}: kind must be one of {KINDS}, got {kind!r}.")
            members = str(r.cell_types).strip()
            members = list(g.cell_types) if members == "all" else [g._type(t.strip()) for t in members.split(";")]
            a = getattr(r, "alpha", None)
            a = None if a is None or (isinstance(a, float) and np.isnan(a)) else _alpha(a)
            if kind == "identity" and len(members) != 1:
                raise ValueError(f"identity topic {r.topic!r} must belong to exactly one cell type.")
            g.topics.append(Topic(_name(r.topic), kind, members, a))
        return g._commit(g.topics)

    def bind(self, labels):
        """A copy conditioned on ``labels`` (the per-cell annotation of the data being fitted)."""
        types = [str(v) for v in identity_order(labels)]
        if set(types) != set(self.cell_types):
            raise ValueError(f"geometry cell types {self.cell_types} differ from the data's {types}.")
        return TopicGeometry(list(self.cell_types), [Topic(x.name, x.kind, list(x.cell_types), x.alpha)
                                                     for x in self.topics], labels)

    def describe(self):
        n = (np.bincount(self.cell_node, minlength=len(self.cell_types)) if self.labels is not None
             else [None] * len(self.cell_types))
        w = max(len(t) for t in self.cell_types)
        lines = [f"{t:<{w}}" + (f"  ({c:,} cells)" if c is not None else "") + f": {', '.join(np.array(self.topic_names)[row])}"
                 for t, c, row in zip(self.cell_types, n, self.allowed)]
        shared = [f"  {x.name} <- {'all' if len(x.cell_types) == len(self.cell_types) else ', '.join(x.cell_types)}"
                  for x in self.topics if x.kind == "activity"]
        return "\n".join(lines + (["shared topics:"] + shared if shared else []))

    def record(self):
        return self.to_frame().to_dict(orient="records")

    # ── internals ─────────────────────────────────────────────────────────
    def _type(self, t):
        t = str(t)
        if t not in self.cell_types:
            raise ValueError(f"cell type {t!r} not in the data. Cell types: {self.cell_types}")
        return t

    def _topic(self, name):
        for x in self.topics:
            if x.name == str(name):
                return x
        raise KeyError(f"no topic {name!r}. Topics: {self.topic_names}")

    def _commit(self, topics):
        """Validate ``topics`` and adopt them; on error the geometry is unchanged."""
        names = [x.name for x in topics]
        if len(set(names)) != len(names):
            dup = sorted({k for k in names if names.count(k) > 1})
            raise ValueError(f"Topic names {dup} are used more than once.")
        used = {t for x in topics for t in x.cell_types}
        empty = [t for t in self.cell_types if t not in used]
        if empty:
            raise ValueError(f"Cell type(s) {empty} have no topic.")
        self.topics = topics
        return self


def identity_order(labels):
    """Values in category order when categorical, else sorted."""
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


def build(labels, n_activity_topics=None, identity_topics=None, shared_topics=None, topics=None):
    """Default geometry for ``labels``, then the shortcuts: ``n_activity_topics`` (V1..Vm for all),
    ``identity_topics`` {type: n or [names]}, ``shared_topics`` {name: "all" or [types]}, or a full table."""
    types = [str(v) for v in identity_order(labels)]
    if topics is not None:
        if any(v is not None for v in (n_activity_topics, identity_topics, shared_topics)):
            raise ValueError("topics replaces n_activity_topics / identity_topics / shared_topics.")
        return TopicGeometry.from_frame(topics, types, labels)
    g = TopicGeometry(types, [Topic(t, "identity", [t]) for t in types], labels)
    for t, v in _dict(identity_topics, "identity_topics").items():
        g.split(t, v)
    if shared_topics is not None and n_activity_topics is not None:
        raise ValueError("shared_topics replaces n_activity_topics; give one of them.")
    for i in range(_count(n_activity_topics or 0, "n_activity_topics")):
        g.share(f"V{i + 1}")
    for k, v in _dict(shared_topics, "shared_topics").items():
        g.share(k, v)
    return g


def _dict(spec, what):
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise TypeError(f"{what} must be a dict.")
    return spec


def _list(v, what):
    if isinstance(v, str) or not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f'{what}: cell types must be "all" or a non-empty list of cell types.')
    return [str(t) for t in v]


def _name(k):
    if not isinstance(k, str) or not k:
        raise TypeError(f"topic names must be non-empty strings, got {k!r}.")
    return k


def _count(v, what):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {type(v).__name__}.")
    if v < 0:
        raise ValueError(f"{what} must be >= 0, got {v}.")
    return int(v)


def _alpha(v):
    if not (np.isscalar(v) and np.isfinite(v) and float(v) > 0):
        raise ValueError(f"alpha must be a positive number, got {v!r}.")
    return float(v)
