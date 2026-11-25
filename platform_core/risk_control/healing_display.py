"""Shared labels, engineering units and completed-sample readouts."""
import math

from .healing import FAULT_PROFILE, GROUPS, NOMINAL, SCENARIOS, SIGNALS, TOLERANCE
from .presentation import tokens

STATUS = {"ready": "就绪", "running": "运行中", "paused": "已暂停", "ended": "已结束"}
HEADINGS = ("对照组", "联合容差内", "浓度 RMSE", "釜温 RMSE", "恢复起点", "流量限幅", "补偿 / 限幅")
COLORS = ("#84b8d9", "#f5bd73", "#6cd6ae", "#c6a6ff", "#889eb4", "#889eb4")
SPAN = 600
METHODS = {"scsc": "简单相关自愈（SCSC）", "ccsc": "典型相关自愈（CCSC）"}
RECOVERY_HELP = ("联合容差内 = 同时达标样本数 / 退化后该组已完成的采样数 × 100%。"
                 f"浓度与原健康值 {NOMINAL[0]:.3f} mol/L 相差不超过 {TOLERANCE[0]:g} mol/L，"
                 f"且釜温与 {NOMINAL[1]:.6f} K 相差不超过 {TOLERANCE[1]:g} K，才计为达标。")


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def effect_view(snapshot):
    trace, onset = snapshot["effect"], snapshot["fault_min"]
    signal, points = trace["signal"], trace["points"]
    name, unit = SIGNALS[signal]
    scenario = snapshot["config"]["scenario"]
    factor = FAULT_PROFILE[scenario][0]
    factor_name = "催化活性" if factor == "a" else "换热能力"
    remaining = snapshot["latest"]["groups"]["fault"][factor]
    if onset is None:
        phase = "本轮未施加退化。" if snapshot["status"] == "ended" else "健康基线 · 等待手动施加退化。"
        instruction = ("本轮已结束，复位可重新开始。" if snapshot["status"] == "ended"
                       else f"点击后{factor_name}将逐渐下降，正常 PI 组保持健康。")
    elif snapshot["status"] in ("paused", "ready") and snapshot["latest"]["t"] == onset:
        phase = f"已在 {onset:.1f} min 设定退化起点，继续推进后开始衰减。"
    else:
        phase = f"退化开始于 {onset:.1f} min　{factor_name}为健康工况的 {remaining:.2%}。"
    if onset is not None:
        instruction = f"本轮已施加{SCENARIOS[scenario]}。复位后可再次施加。"
    digits = 6 if signal == "C" else 4
    now = points[-1]
    methods = {}
    for column, method in enumerate(METHODS, 2):
        action = trace["first"][method]
        if action:
            channels = "、".join(SIGNALS[key][0] for key in action["channels"])
            step = f"{method.upper()} 于 {action['at']:.1f} min 作出退化后的首次入口修正，由{channels}越限触发。"
        elif onset is None:
            step = "自愈组从本轮开始监测。测量超出自定范围时，施加退化前也可能补偿。"
        elif snapshot["metrics"]["groups"][method]["updates"]:
            step = f"{method.upper()} 已触发，实际入口尚未改变。"
        else:
            step = f"{method.upper()} 在退化后尚未触发入口补偿。"
        difference = round(abs(now[1]) - abs(now[column]), digits)
        if difference:
            direction = "减小" if difference > 0 else "增大"
            comparison = f"自愈组的绝对偏差{direction} {abs(difference):.{digits}f} {unit}"
        else:
            comparison = "两组绝对偏差在当前显示精度下相同"
        values = [0., *(p[i] for p in points for i in (1, column))]
        low, high = min(values), max(values)
        margin = max((high-low)*.1, .00005 if signal == "C" else .01)
        rough_step = (high-low+2*margin)/6
        magnitude = 10**math.floor(math.log10(rough_step))
        tick = next(x*magnitude for x in (1, 2, 5, 10) if x*magnitude >= rough_step)
        methods[method] = dict(first_min=None if action is None else action["at"], step=step,
                               readout=f"仅 PI　{now[1]:+.{digits}f} {unit}　·　PI + {method.upper()}　"
                                       f"{now[column]:+.{digits}f} {unit}　·　{comparison}",
                               limits=[math.floor((low-margin)/tick)*tick, math.ceil((high+margin)/tick)*tick], tick=tick)
    return dict(phase=phase, instruction=instruction, title=f"{name}偏差 ({unit})", unit=unit, methods=methods,
                reference=f"零线是健康工况的{name}（{trace['reference']:.6g} {unit}），越接近零表示偏差越小。"
                          "两组都用 PI 调冷却流量，自愈组还可调整进料浓度、进料温度和冷却液入口温度。",
                domain=[points[0][0], max(points[0][0]+10., now[0])])


def view(snapshot):
    stats = snapshot["metrics"]
    rows = []
    for name, label in GROUPS.items():
        g = stats["groups"][name]
        rows.append([label, "—" if g["joint_percent"] is None else fmt(g["joint_percent"], 1)+" %", fmt(g["rmse"][0] if g["rmse"] else None, 6)+" mol/L",
                     fmt(g["rmse"][1] if g["rmse"] else None, 4)+" K",
                     fmt(g["sustained_since"], 1)+" min", fmt(g["upper_min"]+g["lower_min"], 1)+" min",
                     f"{g['updates']} / {g['limited']}"])
    latest = snapshot["latest"]
    timing = []
    for key, title in (("ordinary", "普通周期"), ("compensation", "补偿周期")):
        t = stats["timing"][key]
        timing.append(f"{title} {t['count']} 次 · P95 {fmt(t['p95_ms'], 3)} ms · 超时 {fmt(t['overrun_percent'], 1)} %")
    return dict(state="准备映射" if snapshot["status"]=="running" and snapshot["mapping"] is None else STATUS[snapshot["status"]],
                status=f"{SCENARIOS[snapshot['config']['scenario']]}　{latest['t']:.1f} min",
                speed=f"实际 {fmt(stats['actual_speed'], 0)}× / 设定 {snapshot['config']['speed']:g}×",
                timing=timing, headings=HEADINGS, recovery_help=f"{RECOVERY_HELP} 本轮已统计 {stats['post_fault_samples']} 个样本。",
                metrics=rows, effect=effect_view(snapshot),
                diagrams={key: tokens(row) for key, row in latest["groups"].items()})


def event_label(event):
    def vector(values):
        return " / ".join(f"{x:+.5g}" for x in values)
    clipped = [signal for signal, step, bound in zip(("Ci", "Ti", "Tci"), event["step_limited"], event["inlet_limited"])
               if step or bound]
    return (f"{event['at']:.1f} min · {event['group'].upper()} · {'、'.join(event['channels'])} 越限　"
            f"首个完成样本 {event['first_sample_min']:.1f} min　"
            f"请求 Δd {vector(event['requested'])}　实际 Δd {vector(event['applied'])}　"
            f"限幅 {'、'.join(clipped) or '无'}")
