from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "seminar_3"


def test_final_report_executes_against_submission_evidence():
    notebook = nbformat.read(SUBMISSION / "FINAL_REPORT.ipynb", as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    output = "\n".join(
        item.get("text", "")
        for cell in executed.cells
        for item in cell.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    assert "GPU V1/V2 parity: PASS" in output
    assert "Evidence source commit:" in output
    assert "Batch-32 target status: MISSED (<5 ms/batch)" in output


def test_submission_reproduction_commands_match_saved_evidence():
    readme = (SUBMISSION / "README.md").read_text(encoding="utf-8")
    notebook = nbformat.read(SUBMISSION / "FINAL_REPORT.ipynb", as_version=4)
    notebook_source = "\n".join(
        "\n".join(cell.source)
        if isinstance(cell.source, list)
        else cell.source
        for cell in notebook.cells
    )

    for text in (readme, notebook_source):
        assert "benchmarks\\run_v2_batch.py" in text
        assert "benchmarks\\run_batch_v2.py" not in text
        assert "--versions cpu v1 v2" in text
        assert "--seed 0" in text
        assert "--batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0" in text
        assert "<5 ms/batch" in text
        assert "per_image_ms < 5" not in text
