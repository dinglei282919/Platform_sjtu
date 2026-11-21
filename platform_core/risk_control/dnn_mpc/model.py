"""The same differentiable, scaled predictor is used by training and MPC."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from ..cstr import FEATURE_NAMES, PLANT_VERSION, SAMPLE_MIN, STATE_NAMES, UNITS

CONTRACT = dict(schema=2, plant=PLANT_VERSION, features=list(FEATURE_NAMES),
                inputs_units=list(UNITS), outputs=list(STATE_NAMES), output_units=list(UNITS[:3]),
                period=SAMPLE_MIN, target="next_state", hidden=[64, 64], activation="tanh")
DEFAULT_FILE = Path(__file__).resolve().parents[3] / "assets" / "cstr" / "default_model.npz"


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, 3))
        for name, length in (("xm", 7), ("xs", 7), ("ym", 3), ("ys", 3)):
            self.register_buffer(name, torch.ones(length))
        self.metadata = {}

    def forward(self, features):
        return self.net((features-self.xm)/self.xs)*self.ys + self.ym

    def freeze(self):
        return self.cpu().double().eval().requires_grad_(False)

    @property
    def version(self):
        return self.metadata["version"]


def arrays_and_version(model):
    arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    digest = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode())
    for key in sorted(arrays):
        digest.update(arrays[key].astype("<f8", copy=False).tobytes())
    return arrays, digest.hexdigest()[:20]


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, version = arrays_and_version(model)
    meta = {**model.metadata, **CONTRACT, "version": version}
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("wb") as out:
            np.savez_compressed(out, metadata=json.dumps(meta, allow_nan=False), **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    model.metadata = meta
    return version


def load(path=DEFAULT_FILE):
    model = Predictor().double()
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if any(meta.get(key) != value for key, value in CONTRACT.items()):
            raise ValueError("模型与三状态 CSTR 的变量、单位、结构或采样周期不匹配")
        state = {}
        for name, tensor in model.state_dict().items():
            array = data[name]
            if array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError(f"无效的 CSTR 模型张量 {name}")
            if name in ("xs", "ys") and np.any(array <= 0):
                raise ValueError("标准化尺度必须为正值")
            state[name] = torch.tensor(array, dtype=torch.float64)
    model.load_state_dict(state)
    model.metadata = meta
    if arrays_and_version(model)[1] != meta.get("version"):
        raise ValueError("模型校验失败")
    return model.freeze()
