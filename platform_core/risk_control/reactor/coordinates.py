"""Paper equation (60) coordinates, without assigning unverified units."""
import math

G1, G2, FEED_RATE = .4, .5, .5


def validate_flows(r1, r2):
    for name, value, upper in (("R1", r1, 1.), ("R2", r2, .95)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name} must be finite in [0, {upper}]; R2=1 makes the paper transformation singular")


def scales(r2=.2):
    """xs = scale_s*x1, xd = scale_d*x2."""
    validate_flows(0., r2)
    return FEED_RATE * (1 - r2) / (G1 * G2), FEED_RATE / G2


def composition_series(data, config=None):
    """Always derive from the run's own parameters, including historical runs."""
    import numpy as np
    c = config or {}
    t = np.asarray(data["t"])
    r1 = np.asarray(data.get("r1", np.full(t.shape, c.get("r1", .3))))
    r2 = np.asarray(data.get("r2", np.full(t.shape, c.get("r2", .2))))
    ss = FEED_RATE * (1 - r2) / (G1 * G2)
    sd = FEED_RATE / G2
    result = dict(r1=r1, r2=r2, xs=ss * np.asarray(data["x1"]), xd=sd * np.asarray(data["x2"]))
    for old, new, scale in (("observer1", "observer_s", ss), ("observer2", "observer_d", sd)):
        if old in data:
            result[new] = scale * np.asarray(data[old])
    if "observer_s" in result:
        result["error_s"] = result["xs"] - result["observer_s"]
        result["error_d"] = result["xd"] - result["observer_d"]
    k = np.asarray(data.get("k", 1 + c.get("attack_scale", 1.) + 1.5 * c.get("attack_scale", 1.) * np.sin(c.get("attack_frequency", .25) * t + c.get("attack_phase", 0.))))
    result.update(k=k, attack_coefficient=k - 1, received_s=k * result["xs"])
    if "observer_d" in result:
        result["received_d"] = k * result["observer_d"]
    return result


FDI_DESCRIPTION = (
    "FDI 为反馈乘性篡改：Λ(t)=a[1+1.5 sin(ωt+φ)]，κ(t)=1+Λ(t)。默认 a=1、ω=0.25、φ=0 为论文算例。"
    "控制器接收 κx₁ 与 κx̂₂；观测器保留原始输出反馈，u 直接作用于对象。"
    "主界面显示反变换组分 xs、xd 及其估计，保留正负号，不推定浓度单位或百分比。"
    "最右侧罐仅为工艺示意，没有独立的第三个对象状态、液位或温度方程。"
    "G₁=0.4、G₂=0.5、F=0.5；R₁/R₂ 默认沿用 MATLAB 的 0.3/0.2，R₂=1 不可用。"
)
