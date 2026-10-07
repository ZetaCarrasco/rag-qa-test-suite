import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
