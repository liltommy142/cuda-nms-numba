"""Structural contracts for the checked-in Google Colab workflow."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "collab" / "gpu_test_colab.ipynb"


def test_colab_gpu_notebook_contains_complete_v1_v2_workflow() -> None:
    """Keep the one-click seminar workflow complete and deliberately V3-free."""
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert payload["nbformat"] == 4
    for marker in (
        "COMMIT=main",
        "numba-cuda[cu13]",
        "nms-cu13-venv",
        "tests/test_correctness.py -q -rs -k \"cpu or gpu_v1 or gpu_v2\"",
        "tests/baseline tests/common tests/compat tests/v1",
        "benchmarks/run_all.py",
        "benchmarks/run_v2_batch.py",
        "files.download",
        "--max-candidates 11000",
    ):
        assert marker in source
    assert "gpu_v3" not in source
