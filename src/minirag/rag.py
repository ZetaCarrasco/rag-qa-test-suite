"""Answer generation on top of the retriever.

The LLM is injected as a plain function (prompt -> text), so the logic can be
tested without any network call or API key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .retriever import Hit, Retriever

REFUSAL = "I don't know based on the available company documents."

PROMPT_TEMPLATE = """You are an assistant for company employees.
Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(h.text for h in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: Callable[[str], str],
    top_k: int = 3,
) -> RagAnswer:
    hits = retriever.retrieve(question, top_k=top_k)
    if not hits:  # nothing relevant was retrieved: refuse without calling the LLM
        return RagAnswer(question, REFUSAL, [], [])
    text = llm(build_prompt(question, hits)).strip()
    return RagAnswer(question, text, [h.text for h in hits], [h.source for h in hits])
