"""Unit tests for the canonical hard-NMS candidate contract."""

import numpy as np


def test_common_contract_has_stable_class_partitions():
    from common.candidates import stable_class_partitions

    scores = np.array([0.5, 0.9, 0.9], dtype=np.float32)
    class_ids = np.array([1, 0, 0], dtype=np.int32)

    partitions = stable_class_partitions(scores, class_ids)

    assert [partition.tolist() for partition in partitions] == [[1, 2], [0]]
