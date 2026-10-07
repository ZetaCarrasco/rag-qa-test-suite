"""LLM access through any OpenAI-compatible API, configured with env vars.

LLM_API_KEY   required
LLM_BASE_URL  default: https://api.deepseek.com
LLM_MODEL     default: deepseek-chat
"""
from __future__ import annotations

import os
from typing import Callable

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def settings() -> dict:
    return {
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }


def make_llm() -> Callable[[str], str]:
    from openai import OpenAI  # imported here so the fast tests don't need it

    cfg = settings()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""

    return call
