"""Web task adapter for the existing 60x10 threat-classification model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIGS = {
    "original": {"name": "数据集1", "train": "error_classification_train.csv", "test": "error_classification_test.csv"},
    "easy": {"name": "数据集2", "train": "error_classification_easy_train.csv", "test": "error_classification_easy_test.csv"},
    "hard": {"name": "数据集3", "train": "error_classification_hard_train.csv", "test": "error_classification_hard_test.csv"},
}


def dataset_options() -> list[dict[str, str]]:
    return [{"id": key, "name": value["name"]} for key, value in DATASET_CONFIGS.items()]


def _load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"数据集文件不存在：{path.name}")
    data = pd.read_csv(path, header=None)
    if data.shape[1] != 601:
        raise ValueError(f"{path.name} 应为601列，实际为{data.shape[1]}列")
    features = data.iloc[:, :-1].to_numpy(dtype=np.float64)
    raw_labels = data.iloc[:, -1].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(raw_labels).all():
        raise ValueError(f"{path.name} 包含NaN或Inf")
    if not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError(f"{path.name} 包含非整数标签")
    raw_labels = raw_labels.astype(np.int64)
    if set(raw_labels.tolist()) - {0, 1, 2, 3}:
        raise ValueError(f"{path.name} 包含非法原始标签")
    labels = np.where(raw_labels == 1, 0, np.where(raw_labels == 2, 1, 2)).astype(np.int64)
    return torch.tensor(features.reshape(-1, 60, 10), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def train(config: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    from error_classification import DEFAULT_WEIGHT_DECAY, FCNClassifier, evaluate, seed_torch, train_one_epoch

    dataset_id = str(config.get("dataset", "original"))
    if dataset_id not in DATASET_CONFIGS:
        raise ValueError("未知数据集")
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 0.001))
    if not 1 <= epochs <= 500 or not 8 <= batch_size <= 256 or not 0.0001 <= learning_rate <= 0.1:
        raise ValueError("训练参数超出允许范围")
    definition = DATASET_CONFIGS[dataset_id]
    train_path = ROOT / "input_data" / definition["train"]
    test_path = ROOT / "input_data" / definition["test"]
    train_x, train_y = _load_dataset(train_path)
    test_x, test_y = _load_dataset(test_path)
    seed_torch(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    model = FCNClassifier(input_dim=10, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}
    best_accuracy = 0.0
    log("准备开始训练，使用设备: " + str(device))
    log(f"当前数据集: {definition['name']}")
    log(f"训练集路径: {train_path}")
    log(f"测试集路径: {test_path}")
    log(f"数据加载完成: 训练样本={len(train_x)}，测试样本={len(test_x)}，输入形状=({train_x.shape[1]}, {train_x.shape[2]})")
    for epoch in range(epochs):
        started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        for key, value in (("train_loss", train_loss), ("test_loss", test_loss), ("train_accuracy", train_accuracy), ("test_accuracy", test_accuracy)):
            history[key].append(float(value))
        best_accuracy = max(best_accuracy, test_accuracy)
        elapsed = time.perf_counter() - started
        log(f"Epoch [{epoch + 1:03d}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Time: {elapsed:.2f}s")
    log(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")
    predictions: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for values, labels in test_loader:
            output = model(values.to(device))
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    matrix = np.zeros((3, 3), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return {"dataset": definition["name"], "best_accuracy": float(best_accuracy), "history": history, "confusion_matrix": matrix.tolist(), "class_names": ["增加调整量", "减少调整量", "不调整"]}
