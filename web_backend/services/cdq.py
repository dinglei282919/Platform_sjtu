"""Adapter for the existing CDQ risk matching calculation."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_CV = [13.71, 4.8, 6.07, 16.18, 856.212, 156.0, 135.0]
CDQ_VECTOR_MIN = 0.0
CDQ_VECTOR_MAX = 999999.0
U_LABELS = [
    "装焦量", "空气导入量", "排焦量", "循环空气流量", "放散阀门开度", "氮气补充量", "锅炉过热蒸汽流量",
]
CV_LABELS = [
    "预存室料位", "气体成分H2", "气体成分CO", "气体成分CO2", "锅炉入口温度", "冷焦排出温度", "干熄炉入口温度",
]
DEFAULT_U_NOW = [100.0, 24578.0, 153.0, 243075.0, 24578.0, 50.0, 30.3]
DEFAULT_U_AFTER = [0.0, 24578.0, 120.0, 243075.0, 14578.0, 50.0, 30.3]


class CDQValidationError(ValueError):
    """Raised when a CDQ input is outside its configured physical bounds."""


def _vector_bounds() -> list[dict[str, float]]:
    return [{"min": CDQ_VECTOR_MIN, "max": CDQ_VECTOR_MAX} for _ in range(7)]


def _validate_vector(
    values: list[float] | None,
    name: str,
    fallback: list[float],
    bounds: list[dict[str, float]] | None = None,
) -> list[float]:
    current = fallback if values is None else values
    try:
        result = [float(value) for value in current]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须包含7个数值") from exc
    if len(result) != 7 or not np.isfinite(result).all():
        raise CDQValidationError(f"{name}必须包含7个有限数值")
    if bounds is not None:
        for index, value in enumerate(result):
            minimum = bounds[index]["min"]
            maximum = bounds[index]["max"]
            if value < minimum or value > maximum:
                raise CDQValidationError(
                    f"{name}第{index + 1}项当前值为{value:g}，超出参数边界。"
                )
    return result


def dataset_summary() -> dict[str, Any]:
    from cdq_risk_matching import CDQ_DATA_PATH, load_cdq_dataset

    data, headers, error = load_cdq_dataset()
    available = data is not None and len(data) >= 2
    vector_bounds = _vector_bounds()
    initial_u_now = data[0].tolist() if available else DEFAULT_U_NOW
    initial_u_after = data[1].tolist() if available else DEFAULT_U_AFTER
    return {
        "path": CDQ_DATA_PATH.name,
        "available": available,
        "samples": int(len(data)) if data is not None else 0,
        "headers": headers,
        "error": error,
        "u_labels": U_LABELS,
        "cv_labels": CV_LABELS,
        "initial_u_now": initial_u_now,
        "initial_u_after": initial_u_after,
        "default_cv": DEFAULT_CV,
        "default_step": 1.0,
        "default_horizon": 10,
        "default_sample_index": 0,
        "vector_bounds": {
            "u_now": vector_bounds,
            "u_after": vector_bounds,
            "cv": vector_bounds,
        },
        "vector_bounds_source": "fixed",
    }


def validate_request_vectors(
    cv: list[float] | None,
    u_now: list[float] | None,
    u_after: list[float] | None,
) -> None:
    """Validate request vectors before a route dispatches the algorithm."""
    vector_bounds = _vector_bounds()
    _validate_vector(cv, "CV特征向量", DEFAULT_CV, vector_bounds)
    _validate_vector(u_now, "U_now", DEFAULT_U_NOW, vector_bounds)
    _validate_vector(u_after, "U_after", DEFAULT_U_AFTER, vector_bounds)


def analyze(
    step: float,
    horizon: int,
    sample_index: int,
    cv: list[float] | None = None,
    u_now: list[float] | None = None,
    u_after: list[float] | None = None,
) -> dict[str, Any]:
    from cdq_risk_matching import CDQ_Model, Match_Risk_And_Generate_Scheme, extract_cdq_window, load_cdq_dataset

    if not 0.1 <= step <= 100:
        raise ValueError("时间步长必须在0.1～100之间")
    if not 1 <= horizon <= 500:
        raise ValueError("预测域必须在1～500之间")
    data, headers, error = load_cdq_dataset()
    available = data is not None and len(data) >= 2
    vector_bounds = _vector_bounds()
    current_cv = _validate_vector(
        cv,
        "CV特征向量",
        DEFAULT_CV,
        vector_bounds,
    )
    requested_u_now = _validate_vector(u_now, "U_now", DEFAULT_U_NOW, vector_bounds)
    requested_u_after = _validate_vector(u_after, "U_after", DEFAULT_U_AFTER, vector_bounds)
    bounded_index = max(0, min(int(sample_index), len(data) - 2)) if available else 0
    window = extract_cdq_window(data, bounded_index, horizon + 1) if available else None
    if available and window is None:
        raise RuntimeError("无法从真实数据中获取有效预测窗口")
    actual_u_now = window[0].tolist() if window is not None else requested_u_now
    actual_u_after = window[1].tolist() if window is not None else requested_u_after
    actual_u_now = _validate_vector(actual_u_now, "U_now", DEFAULT_U_NOW, vector_bounds)
    actual_u_after = _validate_vector(actual_u_after, "U_after", DEFAULT_U_AFTER, vector_bounds)
    state, x_update = CDQ_Model(
        actual_u_now,
        actual_u_after,
        current_cv,
        float(step),
        int(horizon),
        u_series=window,
    )
    if state != 1 or x_update is None:
        raise RuntimeError("CDQ物理演化模型计算失败")
    risks, schemes = Match_Risk_And_Generate_Scheme(x_update)
    return {
        "data_source": {
            "path": "cdq_data.xlsx",
            "available": available,
            "mode": "dataset" if available else "fallback",
            "samples": int(len(data)) if data is not None else 0,
            "headers": headers,
            "sample_index": bounded_index,
            "window_rows": int(window.shape[0]) if window is not None else 2,
            "error": error,
        },
        "inputs": {
            "step": float(step),
            "horizon": int(x_update.shape[0]),
            "cv": current_cv,
            "u_now": actual_u_now,
            "u_after": actual_u_after,
        },
        "series": {
            "steps": list(range(1, int(x_update.shape[0]) + 1)),
            "level": np.asarray(x_update[:, 0], dtype=float).tolist(),
            "h2": np.asarray(x_update[:, 1], dtype=float).tolist(),
            "co": np.asarray(x_update[:, 2], dtype=float).tolist(),
            "co2": np.asarray(x_update[:, 3], dtype=float).tolist(),
            "boiler_temperature": np.asarray(x_update[:, 4], dtype=float).tolist(),
            "coke_temperature": np.asarray(x_update[:, 5], dtype=float).tolist(),
        },
        "risks": risks,
        "schemes": schemes,
    }
