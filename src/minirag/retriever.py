"""A minimal BM25 retriever over a folder of Markdown documents.

This is the system under test for the retrieval tests. It is intentionally
small and deterministic: no model downloads, no GPU, no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it
    me my of on or the this to was we what when where which who why will with
    you your""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, and strip a plural 's'."""
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


@dataclass(frozen=True)
class Hit:
    source: str  # file name of the document the chunk comes from
    text: str    # chunk text
    score: float # BM25 score (higher is better)


class Retriever:
    def __init__(self, corpus_dir: str | Path):
        self.corpus_dir = Path(corpus_dir)
        self._chunks: list[tuple[str, str]] = []  # (source, text)
        for path in sorted(self.corpus_dir.glob("*.md")):
            self._chunks.extend(self._split(path))
        if not self._chunks:
            raise ValueError(f"No Markdown documents found in {self.corpus_dir}")
        self._bm25 = BM25Okapi([tokenize(text) for _, text in self._chunks])

    @staticmethod
    def _split(path: Path) -> list[tuple[str, str]]:
        """One chunk per paragraph, each prefixed with the document title."""
        blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n")]
        title = blocks[0].lstrip("# ").strip()
        return [(path.name, f"{title}. {body}") for body in blocks[1:] if body]

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        """Return up to top_k chunks, best first. Chunks with no term overlap are dropped."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            Hit(self._chunks[i][0], self._chunks[i][1], float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] > 0
        ]
