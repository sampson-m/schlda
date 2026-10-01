"""Topic geometry: which topics each cell may use.

Nodes form a tree under an implicit root. The root holds the activity topics ``V1 .. Vm``, every other
node may hold topics of its own, and a cell may use every topic on the path from the root to its node.
Leaves hold identity topics, internal nodes hold group topics. ``allowed`` (a labels x topics mask)
replaces the tree and permits any pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = "root"


@dataclass
class TopicGeometry:
    """Nodes, topics and the node x topic permission mask of a fit (``schlda.topic_geometry``)."""

    kind: str
    nodes: list
    parent: dict
    topic_names: list
    topic_types: list
    topic_nodes: list
    allowed: np.ndarray
    cell_node: np.ndarray
    labels: pd.Series

    @property
    def n_topics(self):
        return len(self.topic_names)

    def allowed_frame(self):
        return pd.DataFrame(self.allowed, index=pd.Index(self.nodes, name="node"), columns=self.topic_names)

    def topics_frame(self):
        return pd.DataFrame({"topic_id": range(self.n_topics), "topic_name": self.topic_names,
                             "topic_type": self.topic_types, "node": self.topic_nodes})

    def csr(self, mask=None):
        """Permitted topics per node as (topics, offsets); ``mask`` restricts the columns."""
        m = self.allowed if mask is None else self.allowed & mask[None, :]
        offsets = np.concatenate([[0], np.cumsum(m.sum(1))]).astype(np.int32)
        return np.nonzero(m)[1].astype(np.int32), offsets

    def identity_mask(self):
        return np.array([t == "identity" for t in self.topic_types])

    def n_cells(self):
        return np.bincount(self.cell_node, minlength=len(self.nodes))

    def describe(self):
        n = dict(zip(self.nodes, self.n_cells()))
        own = {}
        for name, node in zip(self.topic_names, self.topic_nodes):
            own.setdefault(node, []).append(name)
        if self.kind == "mask":
            return self.allowed_frame().astype(int).to_string()
        children = {}
        for v in self.nodes:
            children.setdefault(self.parent[v] or ROOT, []).append(v)
        lines = [f"{ROOT}: {', '.join(own.get(ROOT, [])) or '-'}"]

        def walk(v, prefix, last):
            cells = f"  ({n[v]:,} cells)" if n[v] else ""
            lines.append(f"{prefix}{'└── ' if last else '├── '}{v}: {', '.join(own.get(v, [])) or '-'}{cells}")
            kids = children.get(v, [])
            for i, c in enumerate(kids):
                walk(c, prefix + ("    " if last else "│   "), i == len(kids) - 1)

        top = children.get(ROOT, [])
        for i, v in enumerate(top):
            walk(v, "", i == len(top) - 1)
        return "\n".join(lines)

    def record(self):
        return {"kind": self.kind, "parent": {v: self.parent[v] or ROOT for v in self.nodes},
                "topics": {t: v or "" for t, v in zip(self.topic_names, self.topic_nodes)}}


def identity_order(labels):
    """Values in category order when categorical, else sorted."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        used = set(labels.dropna().unique())
        return [c for c in labels.cat.categories if c in used]
    return sorted(labels.dropna().unique())


def _column(obs, key):
    if key not in obs:
        raise KeyError(f"{key!r} not in adata.obs. Available columns: {list(obs.columns)}")
    return obs[key]


def _count(spec, name, nodes, default):
    if isinstance(spec, dict):
        extra = set(map(str, spec)) - set(nodes)
        if extra:
            raise ValueError(f"{name} names unknown nodes: {sorted(extra)}. Valid: {list(nodes)}")
        out = {v: spec.get(v, default) for v in nodes}
    else:
        out = {v: spec for v in nodes}
    for v, k in out.items():
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError(f"{name}[{v!r}] must be a non-negative int, got {k!r}.")
    return {v: int(k) for v, k in out.items()}


def _names(node, k):
    return [node] if k == 1 else [f"{node}_{j + 1}" for j in range(k)]


def _check_unique(names, n_activity):
    dup = sorted({t for t in names if names.count(t) > 1})
    if dup:
        raise ValueError(
            f"Topic names are not unique: {dup}. Cell type and group names must not collide with each other, "
            f"with sub-topic names (name_1, name_2, ...) or with the activity topic names V1..V{n_activity}.")


def _paths(obs, cell_type_key, groups):
    """Per-cell node names, the parent map and the node order (leaves first is decided later)."""
    keys = [cell_type_key] if isinstance(cell_type_key, str) else list(cell_type_key)
    if not keys:
        raise ValueError("cell_type_key must name at least one obs column.")
    cols = [_column(obs, k) for k in keys]
    if len(keys) == 1 and cols[0].isna().any():
        raise ValueError(
            f"adata.obs[{keys[0]!r}] has {int(cols[0].isna().sum())} missing value(s). "
            f"Every cell needs an identity to condition on.")
    if len(keys) > 1 and groups is not None:
        raise ValueError("Give the hierarchy either as several cell_type_key columns or as groups, not both.")
    values = np.empty((len(obs), len(keys)), dtype=object)
    for j, c in enumerate(cols):
        values[:, j] = [None if pd.isna(x) else str(x) for x in c.to_numpy()]
    present = np.array([[x is not None for x in row] for row in values]) if len(obs) else np.zeros((0, len(keys)), bool)
    if not present[:, 0].all():
        raise ValueError(f"adata.obs[{keys[0]!r}] (the coarsest level) has missing values; every cell needs one.")
    gaps = present[:, 1:] & ~present[:, :-1]
    if gaps.any():
        raise ValueError("A finer annotation level is set where a coarser one is missing; "
                         "missing values may only appear at the end of a cell's path.")
    depth = present.sum(1)
    cell = values[np.arange(len(obs)), depth - 1]

    parent, order, level = {}, [], {}
    for j, c in enumerate(cols):
        for v in map(str, identity_order(c)):
            if v in level and level[v] != j:
                raise ValueError(f"{v!r} appears at two annotation levels ({keys[level[v]]!r}, {keys[j]!r}).")
            level[v] = j
            if v not in order:
                order.append(v)
    for row, d in zip(values, depth):
        for j in range(d):
            v, p = row[j], (row[j - 1] if j else None)
            if parent.setdefault(v, p) != p:
                raise ValueError(f"{v!r} has two parents ({parent[v]!r}, {p!r}); node names must be unique.")
    if groups is not None:
        if not isinstance(groups, dict):
            raise TypeError("groups must be a dict {group: [members]}.")
        for g, members in groups.items():
            g = str(g)
            if isinstance(members, str):
                members = [members]
            for m in map(str, members):
                if m == g:
                    raise ValueError(f"group {g!r} lists itself.")
                if parent.get(m) not in (None, g):
                    raise ValueError(f"{m!r} is in two groups ({parent[m]!r}, {g!r}).")
                parent[m] = g
            parent.setdefault(g, None)
            if g not in order:
                order.append(g)
        for v in list(parent):
            seen = set()
            while v is not None:
                if v in seen:
                    raise ValueError(f"groups contain a cycle through {v!r}.")
                seen.add(v)
                v = parent.get(v)
    if ROOT in parent:
        raise ValueError(f"{ROOT!r} is reserved for the root node; rename that label or group.")
    return cell, parent, order


def build_geometry(obs, cell_type_key, n_activity_topics=None, n_identity_topics=1, groups=None, n_group_topics=1,
                   allowed=None):
    if allowed is not None:
        return _mask_geometry(obs, cell_type_key, allowed, n_activity_topics, n_identity_topics, groups)
    if n_activity_topics is None:
        raise TypeError("n_activity_topics is required unless allowed is given.")
    if isinstance(n_activity_topics, (bool, np.bool_)) or not isinstance(n_activity_topics, (int, np.integer)):
        raise TypeError(f"n_activity_topics must be an int, got {type(n_activity_topics).__name__}.")
    if n_activity_topics < 0:
        raise ValueError(f"n_activity_topics must be >= 0, got {n_activity_topics}.")
    n_act = int(n_activity_topics)

    cell, parent, order = _paths(obs, cell_type_key, groups)
    occupied = set(cell)
    while True:
        has_child = {p for v, p in parent.items() if p is not None}
        dead = [v for v in parent if v not in has_child and v not in occupied]
        if not dead:
            break
        for v in dead:
            del parent[v]
    has_child = {p for p in parent.values() if p is not None}
    leaves = [v for v in order if v in parent and v not in has_child]
    internal = [v for v in order if v in parent and v in has_child]
    nodes = leaves + internal

    k_id = _count(n_identity_topics, "n_identity_topics", leaves, 1)
    k_grp = _count(n_group_topics, "n_group_topics", internal, 1)
    names, types, owner = [], [], []
    for v in leaves:
        names += _names(v, k_id[v]); types += ["identity"] * k_id[v]; owner += [v] * k_id[v]
    for v in internal:
        names += _names(v, k_grp[v]); types += ["group"] * k_grp[v]; owner += [v] * k_grp[v]
    names += [f"V{i + 1}" for i in range(n_act)]; types += ["activity"] * n_act; owner += [ROOT] * n_act
    _check_unique(names, n_act)

    owner_arr = np.array(owner, dtype=object)
    allowed_m = np.zeros((len(nodes), len(names)), dtype=bool)
    for i, v in enumerate(nodes):
        path, u = {ROOT}, v
        while u is not None:
            path.add(u); u = parent[u]
        allowed_m[i] = [o in path for o in owner_arr]
    idx = {v: i for i, v in enumerate(nodes)}
    cell_node = np.array([idx[v] for v in cell], dtype=np.int32)
    empty = [nodes[i] for i in np.unique(cell_node) if not allowed_m[i].any()]
    if empty:
        raise ValueError(f"Cells at {empty} have no topic to use: give them an identity topic, a group topic "
                         f"above them, or activity topics.")
    flat = groups is None and isinstance(cell_type_key, str) and all(k == 1 for k in k_id.values())
    labels = (obs[cell_type_key] if isinstance(cell_type_key, str) else
              pd.Series(pd.Categorical(cell, categories=[v for v in nodes if v in occupied]), index=obs.index))
    return TopicGeometry("flat" if flat else "tree", nodes, parent, names, types, owner, allowed_m, cell_node,
                         labels)


def _mask_geometry(obs, cell_type_key, allowed, n_activity_topics, n_identity_topics, groups):
    if n_activity_topics is not None or groups is not None or n_identity_topics != 1:
        raise ValueError("allowed replaces n_activity_topics, n_identity_topics and groups; give only allowed.")
    if not isinstance(cell_type_key, str):
        raise ValueError("allowed takes a single cell_type_key column.")
    if not isinstance(allowed, pd.DataFrame):
        raise TypeError("allowed must be a DataFrame: one row per cell type, one boolean column per topic.")
    lab = _column(obs, cell_type_key)
    if lab.isna().any():
        raise ValueError(f"adata.obs[{cell_type_key!r}] has {int(lab.isna().sum())} missing value(s). "
                         f"Every cell needs an identity to condition on.")
    m = allowed.copy()
    m.index, m.columns = m.index.map(str), m.columns.map(str)
    if m.index.duplicated().any() or m.columns.duplicated().any():
        raise ValueError("allowed has duplicated row or column names.")
    if not all(pd.api.types.is_bool_dtype(t) or pd.api.types.is_integer_dtype(t) for t in m.dtypes) \
            or not m.isin([0, 1]).all().all():
        raise ValueError("allowed must hold booleans (or 0/1).")
    used = [str(v) for v in identity_order(lab)]
    missing = sorted(set(used) - set(m.index))
    if missing:
        raise ValueError(f"allowed has no row for cell type(s) {missing}.")
    m = m.loc[used].astype(bool)
    if not m.any(axis=0).all():
        raise ValueError(f"Topic(s) {list(m.columns[~m.any(axis=0)])} are allowed for no cell type present.")
    if not m.any(axis=1).all():
        raise ValueError(f"Cell type(s) {list(m.index[~m.any(axis=1)])} may use no topic.")
    rows = m.sum(0)
    types = ["identity" if r == 1 else "activity" for r in rows]
    owner = [m.index[m[c].to_numpy()][0] if r == 1 else None for c, r in zip(m.columns, rows)]
    idx = {v: i for i, v in enumerate(m.index)}
    cell_node = np.array([idx[str(v)] for v in lab.to_numpy()], dtype=np.int32)
    return TopicGeometry("mask", list(m.index), {v: None for v in m.index}, list(m.columns), types, owner,
                         m.to_numpy(), cell_node, lab)
