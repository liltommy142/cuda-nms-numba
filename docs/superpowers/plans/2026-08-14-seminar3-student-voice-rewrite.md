# Seminar 3 Student-Voice Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Rewrite reader-facing Seminar 3 materials in simple Vietnamese student voice while code, commands, evidence, metrics, and package integrity remain unchanged.

**Architecture:** Treat JSON/TXT evidence, commands, benchmark configuration, and machine-readable notebook strings as immutable interfaces. Add regression tests first, rewrite only human text, execute notebooks, regenerate canonical Git-blob checksums, archive the committed allowlist, validate a fresh extraction, and safely sync the existing extracted copy.

**Tech Stack:** Python 3.11, pytest, nbformat, nbclient, Jupyter, PowerShell, Git, SHA-256.

## Global Constraints

- Change only prose, Markdown, titles, and human-facing notebook labels. Source code, commands, raw JSON/TXT evidence, benchmark configuration/counts, GPU information, and metrics must not change. `SHA256SUMS.txt` is derived metadata and must be regenerated after its covered docs/notebooks change; do not treat its old digests as immutable evidence.
- Use short Vietnamese sentences and “nhóm em”; introduce CUDA, NMS, CPU, GPU, benchmark, and notebook plainly for new readers.
- Do not use “primary path”, “submission contract”, “provenance”, “đảm bảo tối ưu”, or equivalent corporate/overstated phrases.
- Everywhere it appears, state honestly that <5 ms/batch is **MISSED**; never turn 29.599 ms/image into a target claim.
- Preserve exactly: GPU V1/V2 parity: PASS; Suppression exercised: PASS; Evidence source commit:; Batch-32 target status: MISSED (<5 ms/batch).
- Preserve benchmarks\run_v2_batch.py, --versions cpu v1 v2, --seed 0, and --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0; never use run_batch_v2.py.
- Preserve the evidence/source commit text 7ee76cd5f6e12b87ddee247d58c9fd6ac866245b, package base commit text 378cae1389de06aca9a1b92214798fd0fa5f0370, RTX 4060 Ti facts, and every measured value. These are reader-facing facts; they are distinct from the regenerated SHA-256 digest values.
- Preserve current TEAM_PLAN: Phung Quoc Tuan has combined V1/V2 GPU, repository, CPU, initial-test, and proposal work; Le Quang Tan’s responsibility cell is blank. Do not infer work for Le Quang Tan.
- Create no presentation artifact.
- Archive scope/prefix remains Group11_Seminar3_CUDA_NMS/ plus README.md, requirements.txt, requirements-cuda13.txt, src, benchmarks, tests, submission/seminar_3, and docs/superpowers/specs/2026-08-14-seminar3-submission-design.md.
- SHA256SUMS contains canonical Git-blob SHA-256 values for exact ordered CHECKSUM_PATHS and never hashes itself.

---

## File Structure

- tests/test_submission_artifacts.py — voice/interface regression tests.
- README.md; src/readme.md; src/baseline/explain.md; src/common/explain.md; src/v1/explain.md; src/v2/explain.md — rewritten reader guides.
- src/gpu_v1.ipynb; src/gpu_v2.ipynb; src/gpu_v3.ipynb; submission/seminar_3/FINAL_REPORT.ipynb — rewritten Markdown/labels and executed notebooks.
- submission/seminar_3/README.md; TEAM_PLAN.md; SUBMISSION_MANIFEST.txt; docs/superpowers/specs/2026-08-14-seminar3-submission-design.md — rewritten hand-in/docs.
- submission/seminar_3/SHA256SUMS.txt — final canonical checksums.
- submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip and its existing inner extraction — validated package and safe synchronization target.

### Task 1: Add student-voice and immutable-interface tests

**Files:**
- Modify: tests/test_submission_artifacts.py
- Test: tests/test_submission_artifacts.py

**Interfaces:**
- Consumes: ROOT, SUBMISSION, CHECKSUM_PATHS, docs, and notebook JSON.
- Produces: per-file regression tests which prose/notebook tasks must satisfy while existing tests remain the authority for machine commands and notebook markers.

- [ ] **Step 1: Write the failing tests**

Append this exact code, preserving all existing helpers/assertions:

~~~python
import json

READER_DOCS = (
    ROOT / "README.md", ROOT / "src" / "readme.md",
    ROOT / "src" / "baseline" / "explain.md", ROOT / "src" / "common" / "explain.md",
    ROOT / "src" / "v1" / "explain.md", ROOT / "src" / "v2" / "explain.md",
    SUBMISSION / "README.md", SUBMISSION / "TEAM_PLAN.md",
    SUBMISSION / "SUBMISSION_MANIFEST.txt",
    ROOT / "docs" / "superpowers" / "specs" / "2026-08-14-seminar3-submission-design.md",
)
NOTEBOOKS = (ROOT / "src" / "gpu_v1.ipynb", ROOT / "src" / "gpu_v2.ipynb",
             ROOT / "src" / "gpu_v3.ipynb", SUBMISSION / "FINAL_REPORT.ipynb")
BANNED_READER_PHRASES = ("primary path", "submission contract", "provenance", "đảm bảo tối ưu")

def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        items.append("".join(source) if isinstance(source, list) else source)
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            items.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(items)

def test_each_reader_facing_doc_uses_student_voice_and_no_banned_phrase():
    for path in READER_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "nhóm em" in text, path
        for phrase in BANNED_READER_PHRASES:
            assert phrase not in text.lower(), (path, phrase)

def test_each_notebook_uses_student_voice_and_no_banned_phrase():
    for path in NOTEBOOKS:
        text = _notebook_text(path)
        assert "nhóm em" in text, path
        for phrase in BANNED_READER_PHRASES:
            assert phrase not in text.lower(), (path, phrase)

def test_team_plan_preserves_recorded_assignments():
    text = (SUBMISSION / "TEAM_PLAN.md").read_text(encoding="utf-8")
    assert "Phung Quoc Tuan" in text
    for concept in ("V1/V2", "GPU", "CPU", "repository", "proposal"):
        assert concept in text
    assert "| Le Quang Tan | 22127378 | |" in text
~~~

- [ ] **Step 1a: Capture the evidence immutability preflight**

Run before changing any prose/notebook file:

~~~powershell
git diff --exit-code -- submission\seminar_3\evidence
if ($LASTEXITCODE -ne 0) { throw 'Raw evidence already differs; stop and preserve or resolve it before this rewrite.' }
git rev-parse HEAD
~~~

Expected: no evidence diff and the printed baseline commit is `13e3fc4` (or the current committed base if the plan is rebased). This check distinguishes immutable evidence from the SHA256SUMS file that Task 5 will intentionally regenerate.

- [ ] **Step 2: Run RED**

Run:

~~~powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
~~~

Expected: the three new tests FAIL and identify each document/notebook still missing “nhóm em” or containing a banned phrase; existing machine-marker, command, checksum, and manifest tests remain active.

- [ ] **Step 3: Commit the test checkpoint**

~~~powershell
git add tests\test_submission_artifacts.py
git commit -m "test: cover Seminar 3 student-facing wording"
~~~

### Task 2: Rewrite root/source documents

**Files:**
- Modify: README.md; src/readme.md; src/baseline/explain.md; src/common/explain.md; src/v1/explain.md; src/v2/explain.md
- Test: tests/test_submission_artifacts.py

**Interfaces:**
- Consumes: Task 1 test contract and current code blocks/contracts.
- Produces: Vietnamese guides that keep all command/API/shape strings usable.

- [ ] **Step 1: Rewrite prose only**

Use “nhóm em” in each document. Keep code blocks, commands, filenames, API names, shapes, values, existing submission link, NMS-only boundary, and historical T4 warning verbatim. Explain: NMS removes overlapping boxes; CPU is reference; different class_ids never suppress one another; V1 builds dense CUDA pair IoU then host greedy resolution; V2 uses SoA/packed masks, partitions classes, and keeps host greedy resolution; V3 is Matrix NMS and not hard-NMS parity. Do not claim production optimality or guaranteed optimization.

- [ ] **Step 2: Verify GREEN**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q -k "each_reader_facing_doc_uses_student_voice"
git diff --check
git diff -- README.md src/readme.md src/baseline/explain.md src/common/explain.md src/v1/explain.md src/v2/explain.md
~~~

Expected: PASS, no whitespace errors, and no changed commands/configuration/metrics.

- [ ] **Step 3: Commit**

~~~powershell
git add README.md src\readme.md src\baseline\explain.md src\common\explain.md src\v1\explain.md src\v2\explain.md
git commit -m "docs: rewrite NMS guides in student voice"
~~~

### Task 3: Rewrite submission documents and packaged design

**Files:**
- Modify: submission/seminar_3/README.md; submission/seminar_3/TEAM_PLAN.md; submission/seminar_3/SUBMISSION_MANIFEST.txt; docs/superpowers/specs/2026-08-14-seminar3-submission-design.md
- Test: tests/test_submission_artifacts.py

**Interfaces:**
- Consumes: Task 1 contracts and current evidence/assignment facts.
- Produces: Vietnamese hand-in documents satisfying existing manifest/reproduction tests.

- [ ] **Step 1: Preserve README facts while rewriting**

Keep table numbers exactly: 100 = 1.378/2.467/5.065/0.56×/0.27×; 1,000 = 16.259/4.501/7.625/3.61×/2.13×; 10,000 = 308.965/113.991/33.327/2.71×/9.27×; batch = 947.180 ms total, 29.599 ms/image. Keep the five-command PowerShell block byte-for-byte. Explain in Vietnamese: synthetic NMS-only, no inference/preprocessing/loading; V3 separate; YOLO checkpoint absent; <5 ms/batch MISSED on RTX 4060 Ti.

- [ ] **Step 2: Make TEAM_PLAN fully Vietnamese without changing ownership**

Translate every heading, table description, and checklist into Vietnamese, including the responsibility wording. The Phung Quoc Tuan cell must still say in Vietnamese that Tuan owns the combined V1/V2 GPU implementation, GPU tests, proposal finalization, repository setup, CPU baseline, initial tests, and proposal problem/background/risk work. Keep the Le Quang Tan row’s responsibility cell blank exactly: `| Le Quang Tan | 22127378 | |`. Keep shared work at team level; add no hours, commits, or individual accomplishments.

- [ ] **Step 3: Rewrite manifest/design while preserving interfaces**

Keep every filename, two fixed commit lines, literal final metadata commit, target/result values, checksum instructions, scope/limitations, and raw-data invariants. Use Vietnamese simple wording; replace provenance with “thông tin chạy đã lưu”.

- [ ] **Step 4: Verify and commit**

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q -k "each_reader_facing_doc_uses_student_voice or team_plan_preserves or submission_reproduction_commands_match_saved_evidence or manifest_names"
git add submission\seminar_3\README.md submission\seminar_3\TEAM_PLAN.md submission\seminar_3\SUBMISSION_MANIFEST.txt docs\superpowers\specs\2026-08-14-seminar3-submission-design.md
git commit -m "docs: rewrite Seminar 3 hand-in in student voice"
~~~

Expected: PASS; fixed commits and blank Le Quang Tan cell remain.

### Task 4: Rewrite and execute all four notebooks

**Files:**
- Modify: src/gpu_v1.ipynb; src/gpu_v2.ipynb; src/gpu_v3.ipynb; submission/seminar_3/FINAL_REPORT.ipynb
- Test: tests/test_submission_artifacts.py

**Interfaces:**
- Consumes: Task 1 test contract, original code cells, commands, and raw evidence.
- Produces: valid executed notebooks with Vietnamese human text and unchanged runtime behavior.

- [ ] **Step 1: Rewrite only titles, Markdown, comments, and human labels**

Use “nhóm em” in each. Do not change imports, URLs, %cd, !git clone, !nvidia-smi, assertions, callable names, command arguments, JSON names, or files.download. State: V1 checks IoU; V2 batch 32 still resolves greedily on CPU and cannot claim target success; V3 soft Matrix NMS uses matrix_nms_reference; FINAL_REPORT is CPU/V1/V2 hard-NMS evidence. Preserve exactly:

~~~python
print('Suppression exercised: PASS')
print('GPU V1/V2 parity: PASS')
target_status = 'MET' if batch_ms < 5 else 'MISSED'
print(f'Batch-32 target status: {target_status} (<5 ms/batch)')
~~~

- [ ] **Step 2: Verify source interfaces**

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q -k "each_notebook_uses_student_voice"
~~~

Expected: PASS.

- [ ] **Step 3: Execute all four notebooks in place**

~~~powershell
@'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
for relative in ("src/gpu_v1.ipynb", "src/gpu_v2.ipynb", "src/gpu_v3.ipynb", "submission/seminar_3/FINAL_REPORT.ipynb"):
    path = Path(relative)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbformat.write(notebook, path)
    print(f"executed: {relative}")
'@ | & .\.venv\Scripts\python.exe
~~~

Expected: four executed lines/no errors. CUDA absence is a real blocker; never fabricate successful output or substitute new evidence facts.

- [ ] **Step 4: Validate and commit**

~~~powershell
& .\.venv\Scripts\python.exe -c "import json; from pathlib import Path; [json.loads(Path(p).read_text(encoding='utf-8')) for p in ('src/gpu_v1.ipynb','src/gpu_v2.ipynb','src/gpu_v3.ipynb','submission/seminar_3/FINAL_REPORT.ipynb')]; print('notebook JSON: PASS')"
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
git add src\gpu_v1.ipynb src\gpu_v2.ipynb src\gpu_v3.ipynb submission\seminar_3\FINAL_REPORT.ipynb
git commit -m "docs: rewrite Seminar 3 notebooks in student voice"
~~~

Expected: valid JSON and all focused checks PASS.

### Task 5: Regenerate canonical checksums and run verification

**Files:**
- Modify: submission/seminar_3/SHA256SUMS.txt
- Test: tests/test_submission_artifacts.py; tests/

**Interfaces:**
- Consumes: final prose/notebooks and exact ordered CHECKSUM_PATHS.
- Produces: ten lower-case SHA-256 entries calculated from canonical Git blobs.

- [ ] **Step 1: Generate exact allowlist checksums**

~~~powershell
@'
from hashlib import sha256
from pathlib import Path
import subprocess
root = Path.cwd(); submission = root / "submission" / "seminar_3"
paths = ("FINAL_REPORT.ipynb", "README.md", "SUBMISSION_MANIFEST.txt", "TEAM_PLAN.md", "evidence/batch32_v2.json", "evidence/batch32_v2.txt", "evidence/benchmark_v1_v2.json", "evidence/benchmark_v1_v2.txt", "evidence/environment.txt", "evidence/pytest_cuda.txt")
lines = []
for relative in paths:
    path = submission / relative
    if not path.is_file(): raise FileNotFoundError(path)
    repo_path = path.relative_to(root).as_posix()
    oid = subprocess.run(["git", "hash-object", "-w", f"--path={repo_path}", str(path)], check=True, capture_output=True, text=True).stdout.strip()
    blob = subprocess.run(["git", "cat-file", "blob", oid], check=True, capture_output=True).stdout
    lines.append(f"{sha256(blob).hexdigest()}  {relative}")
(submission / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("canonical SHA256SUMS: PASS")
'@ | & .\.venv\Scripts\python.exe
~~~

- [ ] **Step 2: Run focused/full tests and commit**

~~~powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
& .\.venv\Scripts\python.exe -m pytest tests -q -rs
git diff --check
git add submission\seminar_3\SHA256SUMS.txt
git commit -m "build: refresh Seminar 3 canonical checksums"
~~~

Expected: all PASS; no raw evidence changes.

- [ ] **Step 3: Prove raw evidence was not changed across the rewrite commits**

Run:

~~~powershell
git diff --exit-code 13e3fc4..HEAD -- submission\seminar_3\evidence
if ($LASTEXITCODE -ne 0) { throw 'The student-voice rewrite changed immutable raw evidence.' }
~~~

Expected: exit code 0. If execution begins from a rebased base, substitute the Task 1a recorded baseline commit for `13e3fc4`; do not substitute a commit created during this rewrite.

### Task 6: Archive, fresh-validate, and safely sync

**Files:**
- Create (ignored): submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip
- Modify (verified paths only): submission/seminar_3/output/Group11_Seminar3_CUDA_NMS/Group11_Seminar3_CUDA_NMS/

**Interfaces:**
- Consumes: clean committed HEAD and archive scope/prefix.
- Produces: fresh tested archive and destination sync that never deletes unknown user files.

- [ ] **Step 1: Build only clean tracked archive**

~~~powershell
if (git status --porcelain) { throw 'Commit or explicitly preserve every tracked change before packaging.' }
New-Item -ItemType Directory -Force submission\seminar_3\output | Out-Null
git archive --format=zip --prefix=Group11_Seminar3_CUDA_NMS/ --output=submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip HEAD README.md requirements.txt requirements-cuda13.txt src benchmarks tests submission/seminar_3 docs/superpowers/specs/2026-08-14-seminar3-submission-design.md
~~~

- [ ] **Step 2: Extract fresh and verify package**

~~~powershell
$zip = (Resolve-Path submission\seminar_3\output\Group11_Seminar3_CUDA_NMS.zip).Path
$testPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$extract = Join-Path ([System.IO.Path]::GetTempPath()) ('seminar3-voice-' + [guid]::NewGuid())
Expand-Archive -LiteralPath $zip -DestinationPath $extract
$bundle = Join-Path $extract 'Group11_Seminar3_CUDA_NMS'
foreach ($item in @('README.md','src/gpu_v1.ipynb','src/gpu_v2.ipynb','src/gpu_v3.ipynb','submission/seminar_3/FINAL_REPORT.ipynb','submission/seminar_3/SHA256SUMS.txt','tests/test_submission_artifacts.py','docs/superpowers/specs/2026-08-14-seminar3-submission-design.md')) { if (-not (Test-Path (Join-Path $bundle $item))) { throw "missing ZIP entry: $item" } }
$env:PYTHONPATH = Join-Path $bundle 'src'
Push-Location $bundle
& $testPython -m pytest tests\test_submission_artifacts.py -q
$exitCode = $LASTEXITCODE
Pop-Location
if ($exitCode -ne 0) { exit $exitCode }
~~~

Expected: fresh extraction artifact tests PASS.

- [ ] **Step 3: Reject unmanaged paths, copy without delete, compare hashes**

~~~powershell
$destinationInput = Read-Host 'Paste the verified absolute main-repository inner extraction path (ending in Group11_Seminar3_CUDA_NMS)'
if (-not [System.IO.Path]::IsPathFullyQualified($destinationInput)) { throw 'Destination must be an absolute path supplied for the user main repository.' }
$destination = (Resolve-Path -LiteralPath $destinationInput).Path
if (-not $destination.EndsWith('\submission\seminar_3\output\Group11_Seminar3_CUDA_NMS\Group11_Seminar3_CUDA_NMS', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Unexpected destination suffix: $destination" }
$sourceFiles = Get-ChildItem $bundle -File -Recurse | ForEach-Object { $_.FullName.Substring($bundle.Length).TrimStart('\','/') -replace '\\','/' } | Sort-Object
$destinationFiles = Get-ChildItem $destination -File -Recurse | ForEach-Object { $_.FullName.Substring($destination.Length).TrimStart('\','/') -replace '\\','/' } | Sort-Object
$unexpected = Compare-Object $sourceFiles $destinationFiles -PassThru | Where-Object { $_ -in $destinationFiles }
if ($unexpected) { throw ('Refusing to delete unmanaged destination files: ' + ($unexpected -join ', ')) }
robocopy $bundle $destination /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
if ($LASTEXITCODE -gt 7) { throw "robocopy failed: $LASTEXITCODE" }
$mismatches = foreach ($relative in $sourceFiles) {
  if ((Get-FileHash (Join-Path $bundle ($relative -replace '/','\')) -Algorithm SHA256).Hash -ne (Get-FileHash (Join-Path $destination ($relative -replace '/','\')) -Algorithm SHA256).Hash) { $relative }
}
if ($mismatches) { throw ('synchronized file mismatch: ' + ($mismatches -join ', ')) }
Write-Output 'fresh archive and extracted inner folder: PASS'
~~~

Expected: PASS. Extras abort before any copy; do not use /MIR, /PURGE, or recursive deletion.

- [ ] **Step 4: Record handoff**

~~~powershell
Get-Item $zip | Select-Object FullName,Length,LastWriteTime
git log -6 --oneline
git status --short --branch
~~~

Expected: non-empty archive and clean tracked tree except ignored output.

## Review Gates

- [ ] Compare all numbers, GPU, commits, commands, and paths with evidence files; raw JSON/TXT remains untouched.
- [ ] Confirm all four notebooks execute, remain valid JSON, retain PASS markers, and honestly show MISSED.
- [ ] Confirm SHA256SUMS has exactly ten ordered canonical entries.
- [ ] Confirm fresh-extract tests PASS and destination equality is proven or safely aborted before unknown-file deletion.

## Parallelization Recommendation

After Task 1, Task 2 source docs can proceed independently from Tasks 3–4 submission/notebook work. One integrator must review, checksum, and package only after all prose/notebook edits are committed.

## Execution Handoff

Plan complete and saved to docs/superpowers/plans/2026-08-14-seminar3-student-voice-rewrite.md. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
