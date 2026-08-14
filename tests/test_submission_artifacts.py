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
    assert "Batch-32 target status:" in output
