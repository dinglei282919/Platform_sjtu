"""Generate deterministic easy/hard variants of the error-classification data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input_data"
TRAIN_SOURCE = DATA_DIR / "error_classification_train.csv"
TEST_SOURCE = DATA_DIR / "error_classification_test.csv"
FEATURE_COUNT = 600
ALLOWED_RAW_LABELS = {0, 1, 2, 3}

VARIANTS = {
    "easy": {
        "display_name": "数据集2",
        "center_scale": 2.0,
        "within_scale": 0.45,
        "noise_scale": 0.03,
        "seed": 20260715,
    },
    "hard": {
        "display_name": "数据集3",
        "center_scale": 0.08,
        "within_scale": 0.20,
        "noise_scale": 1.00,
        "seed": 20260716,
    },
}


def map_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw labels exactly as the platform training module does."""
    return np.where(raw_labels == 1, 0, np.where(raw_labels == 2, 1, 2))


def load_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"源数据集不存在: {path}")
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != FEATURE_COUNT + 1:
        raise ValueError(f"{path.name} 应为 601 列，实际为 {frame.shape[1]} 列")

    features = frame.iloc[:, :FEATURE_COUNT].to_numpy(dtype=np.float64)
    raw_labels_float = frame.iloc[:, FEATURE_COUNT].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(raw_labels_float).all():
        raise ValueError(f"{path.name} 包含 NaN 或 Inf")
    if not np.equal(raw_labels_float, np.floor(raw_labels_float)).all():
        raise ValueError(f"{path.name} 包含非整数标签")

    raw_labels = raw_labels_float.astype(np.int64)
    illegal = sorted(set(raw_labels.tolist()) - ALLOWED_RAW_LABELS)
    if illegal:
        raise ValueError(f"{path.name} 包含非法标签: {illegal}")
    return features, raw_labels


def transform(
    features: np.ndarray,
    mapped_labels: np.ndarray,
    global_mean: np.ndarray,
    feature_std: np.ndarray,
    class_centers: dict[int, np.ndarray],
    center_scale: float,
    within_scale: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.vstack([class_centers[int(label)] for label in mapped_labels])
    noise = rng.normal(size=features.shape) * feature_std * noise_scale
    return (
        global_mean
        + center_scale * (centers - global_mean)
        + within_scale * (features - centers)
        + noise
    )


def write_csv(path: Path, features: np.ndarray, raw_labels: np.ndarray) -> None:
    frame = pd.DataFrame(features)
    frame[FEATURE_COUNT] = raw_labels
    frame.to_csv(path, header=False, index=False, float_format="%.10g", lineterminator="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    train_x, train_raw_y = load_source(TRAIN_SOURCE)
    test_x, test_raw_y = load_source(TEST_SOURCE)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("训练集与测试集特征数不一致")

    train_y = map_labels(train_raw_y)
    test_y = map_labels(test_raw_y)
    global_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0, ddof=0)
    class_centers = {
        class_id: train_x[train_y == class_id].mean(axis=0)
        for class_id in range(3)
    }
    if any(not np.isfinite(center).all() for center in class_centers.values()):
        raise ValueError("训练集缺少映射后的类别，无法计算类别中心")

    metadata = {
        "purpose": "用于算法敏感性演示的确定性派生数据，不代表新增真实工业采样。",
        "sources": {
            "train": str(TRAIN_SOURCE.relative_to(BASE_DIR)).replace("\\", "/"),
            "test": str(TEST_SOURCE.relative_to(BASE_DIR)).replace("\\", "/"),
        },
        "feature_count": FEATURE_COUNT,
        "reshape": [60, 10],
        "statistics_fit_on": "training_set_only",
        "raw_label_mapping": {"1": 0, "2": 1, "0": 2, "3": 2},
        "source_shapes": {
            "train": [int(train_x.shape[0]), FEATURE_COUNT + 1],
            "test": [int(test_x.shape[0]), FEATURE_COUNT + 1],
        },
        "source_raw_label_counts": {
            "train": dict(sorted(Counter(map(int, train_raw_y)).items())),
            "test": dict(sorted(Counter(map(int, test_raw_y)).items())),
        },
        "variants": {},
    }

    for key, config in VARIANTS.items():
        rng = np.random.default_rng(config["seed"])
        transformed_train = transform(
            train_x, train_y, global_mean, feature_std, class_centers,
            config["center_scale"], config["within_scale"], config["noise_scale"], rng,
        )
        transformed_test = transform(
            test_x, test_y, global_mean, feature_std, class_centers,
            config["center_scale"], config["within_scale"], config["noise_scale"], rng,
        )
        if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_test).all():
            raise ValueError(f"{config['display_name']} 生成了 NaN 或 Inf")

        train_output = DATA_DIR / f"error_classification_{key}_train.csv"
        test_output = DATA_DIR / f"error_classification_{key}_test.csv"
        write_csv(train_output, transformed_train, train_raw_y)
        write_csv(test_output, transformed_test, test_raw_y)

        metadata["variants"][key] = {
            "display_name": config["display_name"],
            "seed": config["seed"],
            "formula": (
                "global_mean + center_scale * (class_center - global_mean) + "
                "within_scale * (sample - class_center) + "
                "noise_scale * feature_std * N(0, 1)"
            ),
            "parameters": {
                "center_scale": config["center_scale"],
                "within_scale": config["within_scale"],
                "noise_scale": config["noise_scale"],
            },
            "outputs": {
                "train": train_output.name,
                "test": test_output.name,
            },
            "shapes": {
                "train": [int(transformed_train.shape[0]), FEATURE_COUNT + 1],
                "test": [int(transformed_test.shape[0]), FEATURE_COUNT + 1],
            },
            "sha256": {
                "train": sha256(train_output),
                "test": sha256(test_output),
            },
        }

    metadata_path = DATA_DIR / "error_classification_variants_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 4 个派生 CSV 和元数据: {metadata_path}")


if __name__ == "__main__":
    main()
