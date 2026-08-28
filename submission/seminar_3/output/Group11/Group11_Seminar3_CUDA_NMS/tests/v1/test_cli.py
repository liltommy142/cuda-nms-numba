"""CPU-only checks for V1 CLI presentation logic."""

import numpy as np


def test_v1_cli_verify_prints_cpu_parity_without_cuda(monkeypatch):
    from v1 import cli

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    boxes = np.array([[0, 0, 2, 2]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)
    monkeypatch.setattr(cli, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(cli, "cuda", FakeCuda())
    monkeypatch.setattr(cli, "load_synthetic_candidates", lambda n, seed: (boxes, scores, class_ids))
    monkeypatch.setattr(cli, "run_gpu_v1", lambda *args: np.array([0], dtype=np.int64))
    monkeypatch.setattr(cli, "run_cpu", lambda *args: np.array([0], dtype=np.int64))
    monkeypatch.setattr("sys.argv", ["gpu_v1.py", "--verify"])

    cli.main()
