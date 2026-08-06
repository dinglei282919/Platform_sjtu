"""Shared implementation of the multi-criteria risk scoring rules."""

from __future__ import annotations

import random
from typing import Any


METRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "热电比": {"minimum": 0.45, "maximum": 0.75, "default": 0.60, "decimals": 3, "step": 0.001, "random": (0.50, 0.70)},
    "供电标煤耗": {"minimum": 150.0, "maximum": 250.0, "default": 200.0, "decimals": 2, "step": 0.1, "random": (190.0, 230.0)},
    "供热标煤耗": {"minimum": 30.0, "maximum": 45.0, "default": 38.0, "decimals": 2, "step": 0.1, "random": (36.0, 42.0)},
    "汽机负荷率": {"minimum": 0.60, "maximum": 1.00, "default": 0.80, "decimals": 3, "step": 0.001, "random": (0.65, 0.95)},
    "能量转换比": {"minimum": 0.40, "maximum": 1.00, "default": 0.60, "decimals": 3, "step": 0.001, "random": (0.45, 0.75)},
    "自发电占比": {"minimum": 0.50, "maximum": 0.90, "default": 0.70, "decimals": 3, "step": 0.001, "random": (0.55, 0.85)},
}


def default_values() -> dict[str, float]:
    return {name: float(config["default"]) for name, config in METRIC_CONFIGS.items()}


def random_values() -> dict[str, float]:
    return {
        name: round(random.uniform(*config["random"]), int(config["decimals"]))
        for name, config in METRIC_CONFIGS.items()
    }


def single_score(metric: str, value: float) -> float:
    score = 60.0
    if metric == "热电比":
        if 0.58 <= value <= 0.62:
            score = 100.0
        elif value < 0.58:
            score = 100.0 - 40.0 * ((0.58 - value) / 0.13)
        else:
            score = 100.0 - 40.0 * ((value - 0.62) / 0.13)
    elif metric == "供电标煤耗":
        score = 100.0 if value <= 200 else 100.0 - 40.0 * ((value - 200) / 50)
    elif metric == "供热标煤耗":
        score = 100.0 if value <= 38 else 100.0 - 40.0 * ((value - 38) / 7)
    elif metric == "汽机负荷率":
        score = 100.0 if value >= 0.8 else 100.0 - 40.0 * ((0.8 - value) / 0.2)
    elif metric == "能量转换比":
        score = 100.0 if value >= 0.6 else 100.0 - 40.0 * ((0.6 - value) / 0.2)
    elif metric == "自发电占比":
        if 0.65 <= value <= 0.75:
            score = 100.0
        elif value < 0.65:
            score = 100.0 - 40.0 * ((0.65 - value) / 0.15)
        else:
            score = 100.0 - 40.0 * ((value - 0.75) / 0.15)
    else:
        raise ValueError(f"未知指标：{metric}")
    return max(60.0, min(100.0, score))


def evaluate(values: dict[str, float], weights: dict[str, float]) -> dict[str, Any]:
    metric_names = list(METRIC_CONFIGS)
    normalized_values: dict[str, float] = {}
    raw_weights: list[float] = []
    scores: list[float] = []
    for metric in metric_names:
        if metric not in values:
            raise ValueError(f"缺少指标数值：{metric}")
        value = float(values[metric])
        config = METRIC_CONFIGS[metric]
        if not config["minimum"] <= value <= config["maximum"]:
            raise ValueError(f"{metric} 必须在 {config['minimum']}～{config['maximum']} 范围内")
        weight = float(weights.get(metric, 1.0))
        if not 0 <= weight <= 10:
            raise ValueError(f"{metric} 权重必须在 0～10 范围内")
        normalized_values[metric] = value
        raw_weights.append(weight)
        scores.append(single_score(metric, value))
    total_weight = sum(raw_weights)
    weighting_mode = "requested"
    if total_weight <= 0:
        raw_weights = [1.0] * len(metric_names)
        total_weight = float(len(metric_names))
        weighting_mode = "equal_fallback"
    normalized_weights = [weight / total_weight for weight in raw_weights]
    total = sum(score * weight for score, weight in zip(scores, normalized_weights))
    danger = max(0.0, min(40.0, 100.0 - total))
    items = [
        {
            "metric": metric,
            "value": normalized_values[metric],
            "score": round(score, 4),
            "weight": raw_weights[index],
            "normalized_weight": round(normalized_weights[index], 8),
        }
        for index, (metric, score) in enumerate(zip(metric_names, scores))
    ]
    return {
        "items": items,
        "total_score": round(total, 4),
        "danger_score": round(danger, 4),
        "weighting_mode": weighting_mode,
        "radar": {"indicators": metric_names, "scores": [round(score, 4) for score in scores]},
    }
