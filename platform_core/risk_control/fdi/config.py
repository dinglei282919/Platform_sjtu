"""Validated settings for the original PPTP experiment."""
import math

PLOT_GROUPS = (("tracking", "跟踪性能"), ("disturbance", "扰动观测"),
               ("state_control", "状态与控制"), ("gain_switching", "增益与切换"))


def defaults():
    return {"sim_time": 10.0}


def validate_config(config):
    unknown = set(config) - {"sim_time"}
    if unknown:
        raise ValueError("未知 FDI 参数：" + ", ".join(sorted(unknown)))
    value = config.get("sim_time", 10.0)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("仿真时长必须是 0.1～10 秒之间的数字。")
    value = float(value)
    if not math.isfinite(value) or not .1 <= value <= 10:
        raise ValueError("仿真时长必须在 0.1～10 秒之间。")
    return {"sim_time": value}
