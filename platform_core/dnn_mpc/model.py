"""Float64 inference and autograd retain useful gradients for SciPy."""
from io import BytesIO
import json
import numpy as np
import torch
from torch import nn


def network(hidden=(64, 64), activation="relu"):
    if len(hidden) != 2 or any(not 1 <= int(h) <= 512 for h in hidden):
        raise ValueError("必须设置两个隐藏层，每层 1–512 个节点。")
    if activation not in ("relu", "tanh"):
        raise ValueError("激活函数只能为 relu/tanh。")
    act = nn.ReLU if activation == "relu" else nn.Tanh
    return nn.Sequential(nn.Linear(5, hidden[0]), act(),
                         nn.Linear(hidden[0], hidden[1]), act(),
                         nn.Linear(hidden[1], 4)).double()


class Predictor:
    def __init__(self, net, scaler, metadata):
        self.net = net.eval()
        self.scaler = {k: np.asarray(v, dtype=np.float64) for k, v in scaler.items()}
        self.metadata = metadata
        for p in net.parameters():
            p.requires_grad_(False)

    def output_tensor(self, z):
        s = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in self.scaler.items()}
        return self.net((z - s["x_mean"]) / s["x_std"]) * s["y_std"] + s["y_mean"]

    def step_tensor(self, x, u):
        output = self.output_tensor(torch.cat((x, u.reshape(1))))
        mode = self.metadata["target_mode"]
        if mode == "next_state":
            return output
        if mode == "delta":
            return x + output
        if mode == "derivative":
            # Explicit Euler only. Must match the dataset's discretization.
            return x + self.metadata["dt"] * output
        raise ValueError("未知标签语义。")

    def step(self, x, u):
        with torch.no_grad():
            return self.step_tensor(torch.as_tensor(x, dtype=torch.float64),
                                    torch.tensor(float(u), dtype=torch.float64)).numpy()


def serialize(predictor):
    # NPZ uses numeric arrays + JSON, never executable pickle deserialization.
    arrays = dict(predictor.scaler)
    arrays.update({f"weight_{k}": v.detach().numpy()
                   for k, v in predictor.net.state_dict().items()})
    arrays["metadata"] = np.array(json.dumps(predictor.metadata, ensure_ascii=False))
    out = BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def deserialize(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        if meta.get("format_version") != 1 or meta.get("state_order") != ["x", "xdot", "theta", "thetadot"]:
            raise ValueError("模型格式或状态顺序不受支持。")
        if meta.get("target_mode") not in ("next_state", "delta", "derivative"):
            raise ValueError("模型标签语义无效。")
        if not np.isfinite(meta["dt"]) or meta["dt"] <= 0:
            raise ValueError("模型采样周期无效。")
        net = network(tuple(meta["hidden"]), meta["activation"])
        state = {}
        for k, v in net.state_dict().items():
            arr = z[f"weight_{k}"]
            if arr.shape != tuple(v.shape) or not np.isfinite(arr).all():
                raise ValueError("模型权重无效。")
            state[k] = torch.tensor(arr, dtype=torch.float64)
        net.load_state_dict(state)
        scaler = {}
        for key, size in (("x_mean", 5), ("x_std", 5), ("y_mean", 4), ("y_std", 4)):
            a = z[key]
            if a.shape != (size,) or not np.isfinite(a).all():
                raise ValueError("归一化参数无效。")
            if key.endswith("std") and np.any(a <= 0):
                raise ValueError("标准差必须为正。")
            scaler[key] = a.copy()
    return Predictor(net, scaler, meta)
