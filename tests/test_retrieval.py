"""Retrieval quality tests driven by a golden set of questions."""
import pytest

from conftest import load_golden
from minirag.metrics import hit_rate_at_k

GOLDEN = load_golden()
HIT_RATE_AT_3_THRESHOLD = 0.9


@pytest.mark.parametrize(
    "item", GOLDEN["in_scope"], ids=[i["question"] for i in GOLDEN["in_scope"]]
)
def test_expected_document_is_ranked_first(retriever, item):
    hits = retriever.retrieve(item["question"], top_k=3)
    assert hits, "no results returned for an in-scope question"
    assert hits[0].source == item["expected_source"]


def test_hit_rate_at_3_quality_gate(retriever, golden):
    """Aggregate gate: fails if retrieval quality drops below the threshold."""
    rate = hit_rate_at_k(retriever, golden["in_scope"], k=3)
    assert rate >= HIT_RATE_AT_3_THRESHOLD, f"hit@3 dropped to {rate:.2f}"


@pytest.mark.parametrize("question", GOLDEN["out_of_scope"])
def test_out_of_scope_question_returns_no_results(retriever, question):
    assert retriever.retrieve(question) == []


def test_question_about_two_topics_returns_both_documents(retriever):
    hits = retriever.retrieve("laptop replacement and vacation days", top_k=2)
    assert {h.source for h in hits} == {"equipment-policy.md", "vacation-policy.md"}


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: BM25 is lexical and cannot match 'food' to 'meals'.",
)
def test_semantic_paraphrase_is_not_supported_yet(retriever):
    hits = retriever.retrieve("How much can I spend on food while travelling?", top_k=3)
    assert hits and hits[0].source == "expense-policy.md"
