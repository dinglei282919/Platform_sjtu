from copy import deepcopy
import hashlib
import numpy as np
import torch
from .data import fit_scaler, split_indices
from .model import network, Predictor


def train(x, y, *, hidden=(64, 64), activation="relu", epochs=200,
          lr=1e-3, batch_size=128, seed=42, split="sequential",
          dt=.02, target_mode="next_state", callback=None, provenance="unverified MATLAB data"):
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("训练轮数、批大小、学习率必须为正。")
    if not np.isfinite(dt) or dt <= 0 or target_mode not in ("next_state", "delta", "derivative"):
        raise ValueError("采样周期或标签语义无效。")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tr, va, te = split_indices(len(x), seed, split)
    xm, xs = fit_scaler(x[tr])
    ym, ys = fit_scaler(y[tr])
    xt = torch.tensor((x - xm) / xs, dtype=torch.float64)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float64)
    net = network(hidden, activation)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history, best, best_state, best_epoch = [], float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        net.train()
        for indices in np.array_split(rng.permutation(tr), max(1, int(np.ceil(len(tr) / batch_size)))):
            optimizer.zero_grad()
            loss = ((net(xt[indices]) - yt[indices]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("训练发散，请检查数据或降低学习率。")
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_loss = float(((net(xt[tr]) - yt[tr]) ** 2).mean())
            val_loss = float(((net(xt[va]) - yt[va]) ** 2).mean())
            val_mse = float((((net(xt[va]) - yt[va]) * torch.tensor(ys)) ** 2).mean())
        if not np.isfinite([train_loss, val_loss, val_mse]).all():
            raise ValueError("训练或验证出现非有限值。")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mse_physical": val_mse,
                        "train_rmse": float(np.sqrt(train_loss)), "val_rmse": float(np.sqrt(val_loss))})
        if val_loss < best:
            best, best_state, best_epoch = val_loss, deepcopy(net.state_dict()), epoch
        if callback:
            callback(history)
    net.load_state_dict(best_state)
    with torch.no_grad():
        errors = (net(xt[te]).numpy() * ys + ym) - y[te]
    per_state = (errors ** 2).mean(axis=0)
    metrics = {"test_mse_per_output": per_state.tolist(),
               "test_rmse_per_output": np.sqrt(per_state).tolist(),
               "test_mse": float(per_state.mean()), "best_epoch": best_epoch}
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    meta = {"format_version": 1, "hidden": list(hidden), "activation": activation,
            "state_order": ["x", "xdot", "theta", "thetadot"], "target_mode": target_mode,
            "dt": dt, "seed": seed, "split": split, "split_sizes": list(map(len, (tr, va, te))),
            "normalization": "training-only mean/std, ddof=0, constant std=1",
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "data_sha256": digest, "provenance": provenance, "metrics": metrics,
            "train_input_min": x[tr].min(axis=0).tolist(),
            "train_input_max": x[tr].max(axis=0).tolist()}
    return Predictor(net, {"x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}, meta), history, metrics
