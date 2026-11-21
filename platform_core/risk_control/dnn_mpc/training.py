import copy
import time

import numpy as np
import torch
from .model import CONTRACT, Predictor, arrays_and_version


def device_for(choice):
    if choice not in ("auto", "cpu", "cuda"):
        raise ValueError("训练设备应为 auto、cpu 或 cuda")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA")
    return torch.device("cuda" if choice != "cpu" and torch.cuda.is_available() else "cpu")


def fit(train, valid, *, epochs=320, device="auto", seed=42, initial=None, lr=0.001,
        progress=None, cancelled=None):
    if not 1 <= epochs <= 2000 or not 0 < lr <= 0.05:
        raise ValueError("训练轮数或学习率越界")
    selected = device_for(device)
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = copy.deepcopy(initial) if initial is not None else Predictor()
    model = model.to(selected, torch.float32).requires_grad_(True)
    x, y = [torch.as_tensor(v, device=selected, dtype=torch.float32) for v in train.flat()]
    vx, vy = [torch.as_tensor(v, device=selected, dtype=torch.float32) for v in valid.flat()]
    if initial is None:
        for name, value in (("xm", x.mean(0)), ("xs", x.std(0).clamp_min(1e-7)),
                            ("ym", y.mean(0)), ("ys", y.std(0).clamp_min(1e-7))):
            getattr(model, name).copy_(value)
    x, y = (x-model.xm)/model.xs, (y-model.ym)/model.ys
    vx, vy = (vx-model.xm)/model.xs, (vy-model.ym)/model.ys
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*0.04)
    history, best, best_loss = [], None, float("inf")
    begin = time.perf_counter()
    for epoch in range(1, epochs+1):
        if cancelled is not None and cancelled():
            raise InterruptedError("任务已取消")
        indices = torch.as_tensor(rng.permutation(len(x)), device=selected)
        for batch in indices.split(384):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.net(x[batch])-y[batch]).square().mean()
            loss.backward()
            optimizer.step()
        schedule.step()
        with torch.no_grad():
            val_loss = (model.net(vx)-vy).square().mean().item()
            train_loss = (model.net(x)-y).square().mean().item()
        if not np.isfinite(val_loss + train_loss):
            raise ValueError("训练出现非有限误差")
        if val_loss < best_loss:
            best_loss = val_loss
            best = copy.deepcopy(model.state_dict())
        history.append([epoch, train_loss, val_loss])
        if progress is not None:
            progress(epoch, epochs, history[-1])
    model.load_state_dict(best)
    model.freeze()
    model.metadata = dict(CONTRACT, version=arrays_and_version(model)[1], seed=seed,
                          dataset=train.identifier, training_device=str(selected), epochs=epochs,
                          training_seconds=time.perf_counter()-begin, best_validation_loss=best_loss)
    return model, history


@torch.no_grad()
def errors(model, data, horizon=30):
    one_sum, rollout_sum = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    one_count = rollout_count = 0
    shortest = horizon
    for batch in data.batches():
        features = torch.as_tensor(batch.features, dtype=torch.float64)
        targets = torch.as_tensor(batch.targets, dtype=torch.float64)
        one_sum += (model(features)-targets).square().sum((0, 1))
        one_count += features.shape[0] * features.shape[1]
        h = min(horizon, features.shape[1])
        shortest = min(shortest, h)
        for start in range(0, features.shape[1]-h+1, h):
            x = features[:, start, :3]
            for k in range(start, start+h):
                x = model(torch.cat((x, features[:, k, 3:]), dim=-1))
                rollout_sum += (x-targets[:, k]).square().sum(0)
                rollout_count += x.shape[0]
    return {"one_rmse": (one_sum/one_count).sqrt().tolist(),
            "rollout_rmse": (rollout_sum/rollout_count).sqrt().tolist(), "horizon": shortest}
