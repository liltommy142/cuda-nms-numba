import hashlib
from pathlib import Path
import subprocess

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "seminar_3"


def _checksum_payload(path: Path) -> bytes:
    """Return Git's canonical blob bytes when available, raw bytes otherwise.

    The hand-in ZIP is a Git archive and therefore contains Git blob bytes.
    On Windows, a working tree may have CRLF checkout conversion, so hashing
    raw working-tree text would not reliably describe the archive payload.
    """
    if not (ROOT / ".git").exists():
        return path.read_bytes()

    relative = path.relative_to(ROOT).as_posix()
    object_id = subprocess.run(
        ["git", "hash-object", f"--path={relative}", str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


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


def test_submission_checksums_match_files():
    lines = (SUBMISSION / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert lines
    relatives = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert len(expected) == 64
        assert expected == expected.lower()
        assert relative != "SHA256SUMS.txt"
        payload = _checksum_payload(SUBMISSION / relative)
        assert hashlib.sha256(payload).hexdigest() == expected
        relatives.append(relative)
    assert relatives == sorted(relatives)


def test_manifest_names_every_required_deliverable():
    manifest = (SUBMISSION / "SUBMISSION_MANIFEST.txt").read_text(encoding="utf-8")
    for name in (
        "FINAL_REPORT.ipynb",
        "README.md",
        "TEAM_PLAN.md",
        "evidence/pytest_cuda.txt",
        "evidence/benchmark_v1_v2.json",
        "evidence/batch32_v2.json",
        "evidence/environment.txt",
    ):
        assert name in manifest
