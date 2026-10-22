"""Settings and attack schedule from replay.mdl and s_1/sr_1..sr_6.m."""
import math

ATTACKS = ((5., 9., 4.8), (12., 14., 2.5), (18., 21., 3.5),
           (27., 32., 5.5), (36., 39., 3.7), (42., 45., 3.))
PLOT_GROUPS = (("tracking", "跟踪性能"), ("disturbance", "扰动观测"),
               ("state_control", "状态与控制"), ("switching_replay", "切换与重放"))


def defaults():
    return {"sim_time": 50.0}


def validate_config(config):
    unknown = set(config) - {"sim_time"}
    if unknown:
        raise ValueError("未知重放仿真参数：" + ", ".join(sorted(unknown)))
    value = config.get("sim_time", 50.0)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("仿真时长必须是 0.1～50 秒之间的数字。")
    value = float(value)
    if not math.isfinite(value) or not .1 <= value <= 50:
        raise ValueError("仿真时长必须在 0.1～50 秒之间。")
    return {"sim_time": value}


def attack_index(t):
    for index, (start, end, _) in enumerate(ATTACKS, 1):
        # The original nested branches keep t=27 and t=36 in normal mode.
        if start <= t < end and not (t == start and start in (27., 36.)):
            return index
    return 0
