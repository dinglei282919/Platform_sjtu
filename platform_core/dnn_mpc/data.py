"""MATLAB arrays -> row-major samples; normalization fits training rows only."""
from io import BytesIO
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat


def load_dataset(source, layout="auto"):
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    stream = BytesIO(raw)
    # HDF5 permits a user block; MATLAB v7.3 commonly uses 512 bytes.
    # h5py.is_hdf5 expects a path, not a BytesIO object.
    offsets = [0] + [2 ** n for n in range(9, len(raw).bit_length())]
    if any(raw[o:o + 8] == b"\x89HDF\r\n\x1a\n" for o in offsets):
        with h5py.File(stream, "r") as f:
            # MATLAB v7.3 stores dimensions in reverse order in HDF5.
            arrays = {k: np.asarray(f[k]).T for k in ("X_data", "Y_data")}
    else:
        arrays = loadmat(stream)
    if not {"X_data", "Y_data"}.issubset(arrays):
        raise ValueError("MAT 文件必须含 X_data 和 Y_data。")
    x, y = arrays["X_data"], arrays["Y_data"]
    if any(a.ndim != 2 or np.iscomplexobj(a) for a in (x, y)):
        raise ValueError("X_data/Y_data 必须是二维实数数组。")
    candidates = []
    for transpose in (False, True):
        xx, yy = (x.T, y.T) if transpose else (x, y)
        if xx.shape[1] == 5 and yy.shape[1] == 4 and len(xx) == len(yy):
            candidates.append((transpose, xx, yy))
    if layout not in ("auto", "rows", "columns"):
        raise ValueError("未知样本排列方式。")
    if layout != "auto":
        candidates = [c for c in candidates if c[0] == (layout == "columns")]
    if len(candidates) != 1:
        raise ValueError(f"需要 N×5/N×4 或 5×N/4×N；实际 {x.shape}/{y.shape}。不自动丢弃第六维。")
    _, x, y = candidates[0]
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 20 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("至少需要 20 个样本，且数据不能含 NaN/Inf。")
    return x, y


def split_indices(n, seed=42, mode="sequential"):
    if n < 20:
        raise ValueError("样本数不足。")
    indices = np.arange(n)
    if mode == "random":
        indices = np.random.default_rng(seed).permutation(n)
    elif mode != "sequential":
        raise ValueError("未知划分方式。")
    return np.split(indices, [int(n * .7), int(n * .85)])


def fit_scaler(x):
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    return mean, np.where(std < 1e-12, 1.0, std)
