"""Topic geometry: the nodes (cell types and groups of cell types), the topics each node carries, and the
Dirichlet concentration on each topic.

``geometry(adata, key)`` gives the default: every cell type is a node with one topic and there are no groups.
``topics(node, n)`` sets how many topics a node carries, ``group(name, members, n)`` adds a group node, ``drop``
removes a topic and ``alpha`` sets one topic's concentration. A cell may use the topics of its own type and of
every group containing its type. ``to_frame`` / ``from_frame`` round-trip the table ``topic, node, cell_types,
alpha``. Identity topics are those of a cell type node; activity topics those of a group — a view, not a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "; "


@dataclass
class Topic:
    name: str
    node: str
    alpha: float | None = None


class TopicGeometry:
    def __init__(self, cell_types, groups=None, topics=None, labels=None):
        self.cell_types = [str(t) for t in cell_types]
        self.groups = dict(groups or {})
        self._topics = list(topics or [])
        self.labels = labels

    # ── building ──────────────────────────────────────────────────────────
    def topics_of(self, node):
        return [x.name for x in self._topics if x.node == node]

    def topics(self, node, n_or_names):
        """Set the topics a node carries: ``n`` (named ``<node>`` or ``<node>_1 ..``) or a list of names; 0 removes them."""
        return self._set_topics(self._node(node), n_or_names)

    def group(self, name, members="all", n_or_names=1, alpha=None):
        """Add a group of cell types (``"all"`` or a list of at least two) carrying ``n`` topics."""
        name = _name(name)
        if name in self.cell_types or name in self.groups:
            raise ValueError(f"group {name!r}: a node of that name exists.")
        m = list(self.cell_types) if isinstance(members, str) and members == "all" else \
            [self._type(t) for t in _list(members, f"group({name!r})")]
        if len(set(m)) != len(m):
            raise ValueError(f"group({name!r}): each cell type once.")
        if len(m) == 1:
            raise ValueError(f"group({name!r}): a group of one cell type is that type's own topic; "
                             f"use g.topics({m[0]!r}, n) instead.")
        g = TopicGeometry(self.cell_types, {**self.groups, name: m}, self._topics, self.labels)
        g._set_topics(name, n_or_names, alpha)
        return self._adopt(g)

    def drop(self, name):
        return self._commit(self.groups, [x for x in self._topics if x is not self._topic(name)])

    def alpha(self, name, value):
        """Set one topic's concentration (``None`` restores the node default)."""
        self._topic(name).alpha = None if value is None else _alpha(value)
        return self

    def _set_topics(self, node, n_or_names, alpha=None):
        if isinstance(n_or_names, (list, tuple)):
            names = [_name(k) for k in n_or_names]
        else:
            n = _count(n_or_names, "n")
            names = [node] if n == 1 else [f"{node}_{j + 1}" for j in range(n)]
        own = [i for i, x in enumerate(self._topics) if x.node == node]
        pos = own[0] if own else self._insert_at(node)
        keep = [x for i, x in enumerate(self._topics) if i not in own]
        a = None if alpha is None else _alpha(alpha)
        return self._commit(self.groups, keep[:pos] + [Topic(k, node, a) for k in names] + keep[pos:])

    def _insert_at(self, node):
        order = self.cell_types + list(self.groups)
        later = {n for n in order[order.index(node) + 1:]}
        for i, x in enumerate(self._topics):
            if x.node in later:
                return i
        return len(self._topics)

    # ── views ─────────────────────────────────────────────────────────────
    @property
    def nodes(self):
        return self.cell_types + list(self.groups)

    def members(self, node):
        return [node] if node in self.cell_types else list(self.groups[node])

    @property
    def topic_names(self):
        return [x.name for x in self._topics]

    @property
    def topic_nodes(self):
        return [x.node for x in self._topics]

    @property
    def topic_types(self):
        return ["identity" if x.node in self.cell_types else "activity" for x in self._topics]

    @property
    def identity_topics(self):
        return [x.name for x in self._topics if x.node in self.cell_types]

    @property
    def activity_topics(self):
        return [x.name for x in self._topics if x.node not in self.cell_types]

    @property
    def n_topics(self):
        return len(self._topics)

    @property
    def allowed(self):
        idx = {t: i for i, t in enumerate(self.cell_types)}
        m = np.zeros((len(self.cell_types), self.n_topics), dtype=bool)
        for j, x in enumerate(self._topics):
            m[[idx[t] for t in self.members(x.node)], j] = True
        return m

    @property
    def cell_node(self):
        idx = {t: i for i, t in enumerate(self.cell_types)}
        return np.array([idx[str(v)] for v in self.labels.to_numpy()], dtype=np.int32)

    def identity_mask(self):
        return np.array([x.node in self.cell_types for x in self._topics])

    def allowed_frame(self):
        return pd.DataFrame(self.allowed, index=pd.Index(self.cell_types, name="cell_type"), columns=self.topic_names)

    def csr(self, mask=None):
        """Permitted topics per cell type as (topics, offsets); ``mask`` restricts the columns."""
        m = self.allowed if mask is None else self.allowed & mask[None, :]
        offsets = np.concatenate([[0], np.cumsum(m.sum(1))]).astype(np.int32)
        return np.nonzero(m)[1].astype(np.int32), offsets

    def resolve_alpha(self, alpha_identity, alpha_activity):
        """Concentration per topic: ``alpha_identity / n`` over a cell type's ``n`` topics, ``alpha_activity`` on a
        group's topics, unless set on the topic."""
        n = {node: len(self.topics_of(node)) for node in self.cell_types}
        return np.array([x.alpha if x.alpha is not None else
                         alpha_identity / n[x.node] if x.node in self.cell_types else alpha_activity
                         for x in self._topics], dtype=np.float64)

    def to_frame(self, alpha_identity=None, alpha_activity=None):
        """The table ``topic, node, cell_types, alpha``; alpha filled in when both defaults are given."""
        a = (self.resolve_alpha(alpha_identity, alpha_activity) if alpha_identity is not None
             else [x.alpha for x in self._topics])
        return pd.DataFrame({"topic": self.topic_names, "node": self.topic_nodes,
                             "cell_types": [self._members_str(x.node) for x in self._topics],
                             "alpha": a}).astype({"alpha": float})

    def _members_str(self, node):
        m = self.members(node)
        return "all" if node not in self.cell_types and len(m) == len(self.cell_types) else SEP.join(m)

    def to_csv(self, path, **kw):
        self.to_frame(**kw).to_csv(path, index=False)

    @classmethod
    def from_frame(cls, frame, cell_types, labels=None):
        if isinstance(frame, (str, Path)):
            frame = pd.read_csv(frame)
        if "node" not in frame.columns and "kind" in frame.columns:   # tables written before 0.8.0
            frame = frame.assign(node=np.where(frame.kind == "identity", frame.cell_types, frame.topic))
        need = {"topic", "node", "cell_types"} - set(frame.columns)
        if need:
            raise ValueError(f"geometry table needs columns topic, node, cell_types (missing {sorted(need)}).")
        g = cls(cell_types, {}, [], labels)
        for r in frame.itertuples(index=False):
            node, mem = _name(r.node), str(r.cell_types).strip()
            m = list(g.cell_types) if mem == "all" else [g._type(t.strip()) for t in mem.split(";")]
            if node in g.cell_types:
                if m != [node]:
                    raise ValueError(f"topic {r.topic!r}: node {node!r} is a cell type, so cell_types must be {node!r}.")
            elif node in g.groups:
                if m != g.groups[node]:
                    raise ValueError(f"group {node!r} is listed with two different cell_types.")
            else:
                if len(m) < 2 or len(set(m)) != len(m):
                    raise ValueError(f"group {node!r}: cell_types must list at least two distinct cell types.")
                g.groups[node] = m
            a = getattr(r, "alpha", None)
            a = None if a is None or (isinstance(a, float) and np.isnan(a)) else _alpha(a)
            g._topics.append(Topic(_name(r.topic), node, a))
        return g._commit(g.groups, g._topics)

    def bind(self, labels):
        """A copy conditioned on ``labels`` (the per-cell annotation of the data being fitted)."""
        types = [str(v) for v in identity_order(labels)]
        if set(types) != set(self.cell_types):
            raise ValueError(f"geometry cell types {self.cell_types} differ from the data's {types}.")
        return TopicGeometry(self.cell_types, {k: list(v) for k, v in self.groups.items()},
                             [Topic(x.name, x.node, x.alpha) for x in self._topics], labels)

    def describe(self):
        n = (np.bincount(self.cell_node, minlength=len(self.cell_types)) if self.labels is not None
             else [None] * len(self.cell_types))
        w = max(len(t) for t in self.cell_types)
        lines = [f"{t:<{w}}" + (f"  ({c:,} cells)" if c is not None else "") + f": {', '.join(np.array(self.topic_names)[row])}"
                 for t, c, row in zip(self.cell_types, n, self.allowed)]
        groups = [f"  {k} ({', '.join(self.topics_of(k))}) <- {self._members_str(k)}" for k in self.groups]
        return "\n".join(lines + (["groups:"] + groups if groups else []))

    def record(self):
        return self.to_frame().to_dict(orient="records")

    # ── internals ─────────────────────────────────────────────────────────
    def _type(self, t):
        t = str(t)
        if t not in self.cell_types:
            raise ValueError(f"cell type {t!r} not in the data. Cell types: {self.cell_types}")
        return t

    def _node(self, node):
        node = str(node)
        if node not in self.cell_types and node not in self.groups:
            raise ValueError(f"no node {node!r}. Nodes: {self.nodes}")
        return node

    def _topic(self, name):
        for x in self._topics:
            if x.name == str(name):
                return x
        raise KeyError(f"no topic {name!r}. Topics: {self.topic_names}")

    def _commit(self, groups, topics):
        """Validate and adopt; on error the geometry is unchanged."""
        names = [x.name for x in topics]
        if len(set(names)) != len(names):
            dup = sorted({k for k in names if names.count(k) > 1})
            raise ValueError(f"Topic names {dup} are used more than once.")
        used = {t for x in topics for t in ([x.node] if x.node in self.cell_types else groups[x.node])}
        empty = [t for t in self.cell_types if t not in used]
        if empty:
            raise ValueError(f"Cell type(s) {empty} have no topic.")
        self.groups = {k: v for k, v in groups.items() if any(x.node == k for x in topics)}
        self._topics = topics
        return self

    def _adopt(self, other):
        self.groups, self._topics = other.groups, other._topics
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


def build(labels, n_activity_topics=None, topics=None, groups=None, table=None):
    """Default geometry for ``labels``, then: ``n_activity_topics`` (group ``V`` of all types with ``V1 .. Vm``),
    ``topics`` {node: n or [names]}, ``groups`` {name: members or (members, n or [names])}, or a full table."""
    types = [str(v) for v in identity_order(labels)]
    if table is not None:
        if any(v is not None for v in (n_activity_topics, topics, groups)):
            raise ValueError("table replaces n_activity_topics / topics / groups.")
        return TopicGeometry.from_frame(table, types, labels)
    g = TopicGeometry(types, {}, [Topic(t, t) for t in types], labels)
    if groups is not None and n_activity_topics is not None:
        raise ValueError("groups replaces n_activity_topics; give one of them.")
    m = _count(n_activity_topics or 0, "n_activity_topics")
    if m:
        g.group("V", "all", [f"V{i + 1}" for i in range(m)])
    for k, v in _dict(groups, "groups").items():
        members, n = v if isinstance(v, tuple) else (v, 1)
        g.group(k, members, n)
    for k, v in _dict(topics, "topics").items():
        g.topics(k, v)
    return g


def _dict(spec, what):
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise TypeError(f"{what} must be a dict.")
    return spec


def _list(v, what):
    if isinstance(v, str) or not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f'{what}: cell types must be "all" or a list of cell types.')
    return [str(t) for t in v]


def _name(k):
    if not isinstance(k, str) or not k:
        raise TypeError(f"names must be non-empty strings, got {k!r}.")
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
