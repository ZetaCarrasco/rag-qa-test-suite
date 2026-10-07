"""LLM judge for DeepEval, backed by any OpenAI-compatible API."""
import asyncio

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

from minirag.llm import settings


class Judge(DeepEvalBaseLLM):
    def __init__(self):
        cfg = settings()
        self.model_name = cfg["model"]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or "{}"

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self):
        return self.model_name
