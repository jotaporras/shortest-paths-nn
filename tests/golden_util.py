"""Helpers for golden-file comparisons."""
import hashlib
import json

import numpy as np
import pytest

from conftest import GOLDEN_DIR


def array_digest(values) -> str:
    """SHA-256 of an integer array, so large index lists stay out of the JSON."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def check_golden(name: str, observed: dict, update: bool, float_keys=(), rel=1e-12):
    """Compare `observed` with tests/golden/<name>.json.

    Keys in `float_keys` are compared with a relative tolerance. All other
    keys must be equal.
    """
    path = GOLDEN_DIR / f"{name}.json"
    if update:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(observed, indent=1, sort_keys=True))
    golden = json.loads(path.read_text())
    assert set(observed) == set(golden)
    for key, value in observed.items():
        if key in float_keys:
            assert value == pytest.approx(golden[key], rel=rel), key
        else:
            assert value == golden[key], key
