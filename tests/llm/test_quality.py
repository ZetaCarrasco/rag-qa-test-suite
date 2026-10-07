"""End-to-end answer quality, scored by an LLM judge (DeepEval).

Run on demand:  pytest -m llm
Costs a few cents per run: 6 questions x 2 metrics.
"""
import pytest

# The fast CI installs only requirements.txt: skip this module if the LLM extras are missing.
pytest.importorskip("deepeval")
pytest.importorskip("openai")

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from helpers import load_golden
from judge import Judge
from minirag.llm import make_llm
from minirag.rag import answer_question

pytestmark = pytest.mark.llm

THRESHOLD = 0.7
QUESTIONS = [i["question"] for i in load_golden()["in_scope"] if i.get("llm_eval")]


@pytest.fixture(scope="module")
def llm():
    return make_llm()


@pytest.fixture(scope="module")
def judge():
    return Judge()


@pytest.mark.parametrize("question", QUESTIONS)
def test_answer_is_faithful_and_relevant(retriever, llm, judge, question):
    result = answer_question(question, retriever, llm)
    test_case = LLMTestCase(
        input=question,
        actual_output=result.answer,
        retrieval_context=result.contexts,
    )
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=False, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=False, async_mode=False),
    ]
    assert_test(test_case, metrics)
