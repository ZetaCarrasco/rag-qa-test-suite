import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


@pytest.fixture(scope="session", autouse=True)
def require_llm_key():
    """Skip locally when there is no key; fail loudly in CI (a skip would hide a broken secret)."""
    if os.getenv("LLM_API_KEY"):
        return
    if os.getenv("CI"):
        pytest.fail("LLM_API_KEY is not set: configure it as a GitHub Actions secret.")
    pytest.skip("LLM_API_KEY not set")
