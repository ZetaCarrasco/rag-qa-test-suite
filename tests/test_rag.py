"""Tests for answer generation, using a fake LLM (no network, no API key)."""
import pytest

from helpers import load_golden
from minirag.rag import REFUSAL, answer_question, build_prompt

OUT_OF_SCOPE = load_golden()["out_of_scope"]


class FakeLLM:
    """Records every prompt it receives and returns a canned answer."""

    def __init__(self, reply="  Twenty-five days.  "):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_question_is_refused_without_calling_the_llm(retriever, question):
    llm = FakeLLM()
    result = answer_question(question, retriever, llm)
    assert result.answer == REFUSAL
    assert result.contexts == []
    assert llm.prompts == []


def test_prompt_contains_the_question_and_the_retrieved_context(retriever):
    llm = FakeLLM()
    answer_question("How many vacation days do I get per year?", retriever, llm)
    assert len(llm.prompts) == 1
    assert "How many vacation days do I get per year?" in llm.prompts[0]
    assert "25 paid vacation days" in llm.prompts[0]


def test_prompt_tells_the_model_to_use_only_the_context(retriever):
    hits = retriever.retrieve("vacation days")
    assert "ONLY the context" in build_prompt("vacation days", hits)


def test_answer_exposes_contexts_and_sources_for_evaluation(retriever):
    result = answer_question("When is my laptop replaced?", retriever, FakeLLM())
    assert result.sources[0] == "equipment-policy.md"
    assert len(result.contexts) == len(result.sources) >= 1


def test_llm_output_is_stripped(retriever):
    result = answer_question("vacation days", retriever, FakeLLM("  hello \n"))
    assert result.answer == "hello"
