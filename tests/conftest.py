from pathlib import Path

import pytest

from helpers import load_golden
from minirag.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    return Retriever(ROOT / "corpus")


@pytest.fixture(scope="session")
def golden() -> dict:
    return load_golden()
