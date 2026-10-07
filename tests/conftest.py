import json
from pathlib import Path

import pytest

from minirag.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    return Retriever(ROOT / "corpus")


@pytest.fixture(scope="session")
def golden() -> dict:
    return load_golden()
