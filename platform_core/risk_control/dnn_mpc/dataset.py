"""Whole trajectories for identification, validation and open-loop rollout checks."""
import csv
import hashlib
import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ..cstr import FEATURE_NAMES, INLET, NOMINAL, NOMINAL_FLOW, integrate


@dataclass
class Trajectories:
    features: np.ndarray  # [trajectory, interval, 7]
    targets: np.ndarray   # [trajectory, interval, 3]

    def __len__(self):
        return len(self.features)

    def batches(self):
        return (self,)

    def subset(self, indices):
        return Trajectories(self.features[indices], self.targets[indices])

    def flat(self):
        return self.features.reshape(-1, 7), self.targets.reshape(-1, 3)

    @property
    def identifier(self):
        digest = hashlib.sha256(self.features.tobytes())
        digest.update(self.targets.tobytes())
        return digest.hexdigest()[:20]


def generate(count=240, length=48, seed=42):
    """Vary the initial high-branch state, coolant excitation and measured inlets."""
    rng = np.random.default_rng(seed)
    features = np.empty((count, length, 7))
    targets = np.empty((count, length, 3))
    for i in range(count):
        scale = rng.choice([0.35, 0.7, 1.0])
        x = NOMINAL + scale * rng.uniform([-0.022, -5.5, -6.0], [0.022, 5.5, 6.0])
        d = INLET.copy()
        offset = 0.0
        for k in range(length):
            if k % 6 == 0:
                offset = rng.uniform(-43, 43) * scale
                d = INLET + scale * rng.uniform([-0.033, -2.0, -2.0], [0.033, 2.0, 2.0])
            u = np.clip(NOMINAL_FLOW + 4.0*(x[1]-NOMINAL[1]) + offset + rng.normal(0, 2), 95, 200)
            features[i, k] = np.r_[x, u, d]
            x = integrate(x, u, d)
            targets[i, k] = x
    if not np.isfinite(targets).all() or np.any(targets <= 0):
        raise ValueError("生成的轨迹越出物理域")
    return Trajectories(features, targets)


CSV_COLUMNS = ("trajectory", "step", *FEATURE_NAMES)
MAX_CSV_BYTES = 32 * 1024 * 1024


@dataclass
class ImportedTrajectories:
    # Each array contains consecutive measured states and inputs from one run.
    runs: list

    def __len__(self):
        return len(self.runs)

    def subset(self, indices):
        return ImportedTrajectories(self.runs[indices])

    def batches(self):
        by_length = defaultdict(list)
        for run in self.runs:
            by_length[len(run)].append(run)
        for group in by_length.values():
            rows = np.stack(group)
            yield Trajectories(rows[:, :-1, :], rows[:, 1:, :3])

    def flat(self):
        return (np.concatenate([run[:-1] for run in self.runs]),
                np.concatenate([run[1:, :3] for run in self.runs]))

    @property
    def identifier(self):
        digest = hashlib.sha256()
        for run in self.runs:
            digest.update(len(run).to_bytes(8, "little"))
            digest.update(run.tobytes())
        return digest.hexdigest()[:20]


def example_csv():
    data = generate(4, 8, seed=37)
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(CSV_COLUMNS)
    for i, (features, targets) in enumerate(zip(data.features, data.targets), 1):
        last = np.r_[targets[-1], features[-1, 3:]]
        for k, row in enumerate(np.vstack((features, last))):
            writer.writerow([i, k, *[format(v, ".12g") for v in row]])
    return text.getvalue()


def read_csv(content):
    if not content or len(content) > MAX_CSV_BYTES:
        raise ValueError("请选择 0–32 MiB 范围内的非空 CSV 文件")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(CSV_COLUMNS).issubset(fields):
            raise ValueError("CSV 需要表头 " + ",".join(CSV_COLUMNS))
        runs = defaultdict(dict)
        for row in reader:
            try:
                name = row["trajectory"].strip()
                step = int(row["step"])
                values = [float(row[k]) for k in FEATURE_NAMES]
                if None in row or not name or step < 0 or step in runs[name]:
                    raise ValueError
                if not all(np.isfinite(v) and v > 0 for v in values):
                    raise ValueError
                runs[name][step] = values
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"CSV 第 {reader.line_num} 行存在缺失值、无效数值或重复步号") from None
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("CSV 需使用 UTF-8 编码及逗号分隔") from exc
    if len(runs) < 3:
        raise ValueError("至少需要三条独立轨迹，分别用于训练、验证和测试")
    trajectories = []
    for name, steps in runs.items():
        first, last = min(steps), max(steps)
        if len(steps) < 2 or last - first + 1 != len(steps):
            raise ValueError(f"轨迹 {name} 至少需要两个连续采样点，步号不能缺失")
        trajectories.append(np.asarray([steps[k] for k in range(first, last + 1)], dtype=float))
    return ImportedTrajectories(trajectories)


def split(data):
    n = len(data)
    if n < 3:
        raise ValueError("划分训练、验证和测试集需要至少三条独立轨迹")
    a = min(int(0.7*n), n-2)
    b = min(max(a+1, int(0.85*n)), n-1)
    return data.subset(slice(None, a)), data.subset(slice(a, b)), data.subset(slice(b, None))
