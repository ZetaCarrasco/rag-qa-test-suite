"""Retrieval metrics used by the quality gate."""
from __future__ import annotations

from .retriever import Retriever


def hit_rate_at_k(retriever: Retriever, golden: list[dict], k: int) -> float:
    """Share of questions whose expected source appears in the top-k results."""
    hits = 0
    for item in golden:
        sources = [h.source for h in retriever.retrieve(item["question"], top_k=k)]
        hits += item["expected_source"] in sources
    return hits / len(golden)
