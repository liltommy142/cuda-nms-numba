# Common candidate contract

`common` contains rules shared by every hard-NMS implementation. It has no
CUDA-version-specific logic.

| File | Responsibility |
|---|---|
| `candidates.py` | Validate candidates, enforce stable ordering, and generate deterministic synthetic inputs. |
| `oracle.py` | Run torchvision NMS per class as the correctness oracle. |

The canonical input for one image is:

```text
boxes      float32 (N, 4), xyxy, positive area
scores     float32 (N,)
class_ids  int32   (N,)
```

Every version returns original candidate indices, ordered by descending score
and then input index. A candidate may suppress another candidate only when
their class IDs match. `nms_common.py` re-exports these functions only for old
imports; it does not own the implementation.
