"""Finding columns and values the user refers to.

Users rarely type a column or value exactly ("cali", "revenue", "tech"). Two
indexes help the agent map their words onto the data:

- ``ValueIndex``: every distinct value of every text column, searched with exact,
  substring and fuzzy (difflib) matching. This is what makes large categorical
  columns (thousands of cities or products) usable without sending them to the model.
- ``ColumnIndex``: column names, matched fuzzily and, optionally, by embedding
  similarity (``semantic_search: true`` in config.yaml) so "revenue" finds "Sales".
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import DatasetSchema

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass
class Match:
    column: str
    value: str
    score: float
    rows: int


@dataclass
class ValueIndex:
    values: dict[str, pd.Series] = field(default_factory=dict)  # column -> value counts

    @classmethod
    def build(cls, df: pd.DataFrame, schema: DatasetSchema, max_unique: int = 200_000) -> ValueIndex:
        index = cls()
        for col in schema.columns:
            if col.semantic_type in ("categorical", "identifier", "text", "boolean") and col.unique <= max_unique:
                index.values[col.name] = df[col.name].astype("string").value_counts(dropna=True)
        return index

    def search(self, term: str, column: str | None = None, limit: int = 10) -> list[Match]:
        term_cf = term.strip().casefold()
        if not term_cf:
            return []
        matches: list[Match] = []
        columns = [column] if column else list(self.values)
        for col in columns:
            counts = self.values.get(col)
            if counts is None:
                continue
            folded = counts.index.to_series().str.casefold()
            exact = counts[(folded == term_cf).values]
            sub = counts[folded.str.contains(term_cf, regex=False).values & (folded != term_cf).values]
            for v, n in exact.items():
                matches.append(Match(col, str(v), 1.0, int(n)))
            for v, n in sub.head(limit).items():
                matches.append(Match(col, str(v), 0.8, int(n)))
            if exact.empty and sub.empty:
                close = difflib.get_close_matches(term_cf, folded.tolist()[:20000], n=3, cutoff=0.75)
                for c in close:
                    original = counts.index[(folded == c).values][0]
                    ratio = difflib.SequenceMatcher(None, term_cf, c).ratio()
                    matches.append(Match(col, str(original), round(0.7 * ratio, 3), int(counts[original])))
        matches.sort(key=lambda m: (-m.score, -m.rows))
        return matches[:limit]


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


@dataclass
class ColumnIndex:
    schema: DatasetSchema
    embed: EmbedFn | None = None
    _vectors: np.ndarray | None = None

    def _descriptions(self) -> list[str]:
        out = []
        for c in self.schema.columns:
            examples = ", ".join(v.value for v in c.top_values[:5]) or ", ".join(c.sample_values[:3])
            out.append(f"{c.name} ({c.semantic_type} {c.role}); examples: {examples}")
        return out

    def search(self, phrase: str, limit: int = 5) -> list[tuple[str, float]]:
        names = self.schema.names()
        scores: dict[str, float] = {}
        p = phrase.casefold()
        for name in names:
            n = name.casefold()
            s = difflib.SequenceMatcher(None, p, n).ratio()
            if p in n or n in p:
                s = max(s, 0.9)
            scores[name] = s
        if self.embed is not None:
            if self._vectors is None:
                self._vectors = np.array(self.embed(self._descriptions()), dtype=float)
            q = np.array(self.embed([phrase]), dtype=float)
            sims = _cosine(q, self._vectors)[0]
            for name, sim in zip(names, sims):
                scores[name] = max(scores[name], float(sim))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(n, round(s, 3)) for n, s in ranked[:limit]]


def openai_embedder(model: str) -> EmbedFn:
    """Embedding function backed by the OpenAI API (used only when semantic_search is on)."""
    from openai import OpenAI

    client = OpenAI()

    def embed(texts: Sequence[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in response.data]

    return embed
