"""Behaviour and edge-case tests for the retriever."""
import pytest

from minirag.retriever import Retriever


@pytest.mark.parametrize("top_k", [1, 2, 3, 6])
def test_returns_at_most_top_k_results(retriever, top_k):
    assert len(retriever.retrieve("policy employees company", top_k=top_k)) <= top_k


def test_results_are_sorted_by_score(retriever):
    scores = [h.score for h in retriever.retrieve("employees company days", top_k=6)]
    assert scores == sorted(scores, reverse=True)


def test_all_returned_scores_are_positive(retriever):
    assert all(h.score > 0 for h in retriever.retrieve("employees company days", top_k=6))


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.retrieve("vacation", top_k=top_k)


@pytest.mark.parametrize("query", ["", "   ", "???", "the of and"])
def test_empty_or_meaningless_query_returns_nothing(retriever, query):
    assert retriever.retrieve(query) == []


def test_same_query_gives_same_results(retriever):
    assert retriever.retrieve("vacation days") == retriever.retrieve("vacation days")


def test_case_and_punctuation_do_not_change_the_top_result(retriever):
    plain = retriever.retrieve("vacation days")[0].source
    noisy = retriever.retrieve("VACATION DAYS!!!")[0].source
    assert plain == noisy


def test_non_english_query_does_not_crash(retriever):
    assert isinstance(retriever.retrieve("¿Cuántos días de vacaciones tengo?"), list)


def test_very_long_query_does_not_crash(retriever):
    assert isinstance(retriever.retrieve("vacation " * 5000), list)


def test_empty_corpus_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        Retriever(tmp_path)
