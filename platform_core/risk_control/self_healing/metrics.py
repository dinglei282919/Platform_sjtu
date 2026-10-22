"""Output-wise metrics; preserve both paper and reference-code MAPE conventions."""
from __future__ import annotations

import numpy as np


def evaluate(actual, reference, time):
    """Eq. 33-35, using every supplied sample and absolute simulation time.

    The PDF's Eq. 34 divides by actual y; MATLAB divides by normal reference.
    ATAE is the discrete sum sum(t * abs(error)), NOT a time integral.
    """
    actual = np.asarray(actual, dtype=float)
    reference = np.broadcast_to(np.asarray(reference, dtype=float), actual.shape)
    time = np.asarray(time, dtype=float)
    if actual.ndim != 2 or time.shape != (len(actual),) or not len(actual):
        raise ValueError("actual must be nonempty (samples, outputs), time (samples,)")
    if not all(np.isfinite(a).all() for a in (actual, reference, time)):
        raise ValueError("metric inputs must be finite")
    error = np.abs(actual - reference)
    floor = 1e-12
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mape_actual_percent": (100 * np.mean(error / np.maximum(np.abs(actual), floor), axis=0)).tolist(),
        "mape_reference_percent": (100 * np.mean(error / np.maximum(np.abs(reference), floor), axis=0)).tolist(),
        "atae": np.sum(time[:, None] * error, axis=0).tolist(),
    }
