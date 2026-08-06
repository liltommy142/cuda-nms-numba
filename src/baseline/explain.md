# CPU baseline

The baseline is the reference implementation for greedy, class-aware hard NMS.
It is deliberately serial and provides the result that V1 and V2 must match.

| File | Responsibility |
|---|---|
| `core.py` | Vectorized one-to-many IoU, serial greedy resolution, torchvision verification, and CPU-only timing. |
| `yolov5_adapter.py` | Load matching YOLOv5 code/weights and convert raw pre-NMS output into the common candidate contract. |
| `cli.py` | Parse `cpu_baseline.py` arguments and present results. |

Flow: raw/synthetic candidates → validate → split by class → stable score order
→ greedy CPU suppression → original indices. The raw YOLO adapter deliberately
uses model output before YOLO's own NMS, so the project NMS is what is measured.

Use `cpu_baseline.py` for the public CLI/import path. New internal code should
import `baseline.core` or `baseline.yolov5_adapter` directly.
