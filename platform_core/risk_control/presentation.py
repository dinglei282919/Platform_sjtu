from functools import lru_cache
from pathlib import Path
import numpy as np
from .disturbance import PRESETS


@lru_cache(maxsize=1)
def template():
    return (Path(__file__).resolve().parents[2]/"assets/cstr/process.svg").read_text(encoding="utf-8")


def marker_tokens(marker):
    active = marker.get("active", False)
    return {"MEASUREMARK": "1" if marker.get("link") == "measurement" else "0",
            "EXECMARK": "1" if marker.get("link") == "execution" else "0",
            "OBSBRANCH": "1" if marker.get("link") == "observer" else "0",
            "OBSPREVIEW": "1" if marker.get("link") == "observer" and marker.get("preview") else "0",
            "OBSACTIVE": "1" if marker.get("link") == "observer" and active else "0",
            "SIGNALWIDTH": "5" if active else "3", "SIGNALDASH": "none" if active else "3 3",
            "SIGNALLABEL": marker.get("label", "")}


def tokens(sample):
    value = sample["T"]
    palette = np.array([[145, 197, 224], [244, 221, 186], [227, 141, 112]])
    channels = [int(round(np.interp(value, [424, 431, 438], channel))) for channel in palette.T]
    read = sample.get("flow_read_previous")
    return {**marker_tokens(sample.get("signal_marker") or {}),
            "FDISENSORS": "1" if sample.get("loop_side") or sample.get("show_sensors") else "0",
            "METERFLAG": "1" if sample.get("loop_side") == "measurement" else "0",
            "METERREAD": "—" if read is None else f"{read:.2f}",
            "COLOR": "#%02x%02x%02x" % tuple(channels), "TEMP": f"{value:.2f}",
            "JACKET": f"{sample['Tc']:.2f}", "FLOW": f"{sample['Qc']:.2f}",
            "C": "—" if sample.get("C") is None else f"{sample['C']:.4f}",
            "REF": f"{sample['T_ref']:.2f}", "CI": f"{sample['Ci']:.2f}",
            "TI": f"{sample['Ti']:.1f}", "TCI": f"{sample['Tci']:.1f}"}


def render(sample):
    svg = template()
    for key, value in tokens(sample).items():
        svg = svg.replace("@@"+key+"@@", value)
    return svg


def trial_status(trial):
    if not trial:
        return None
    labels = {"queued": "等待下一运行周期", "active": "扰动施加中", "recovery": "已恢复入口，观察响应",
              "complete": "试验完成", "cancelled": "已取消", "interrupted": "运行已结束"}
    phase = trial["phase"]
    status = labels[phase]
    if phase == "active" and trial["clear_requested"]:
        status = "等待下一周期撤除"
    elif phase in ("active", "recovery"):
        status += f" · 剩余 {trial['remaining']} 周期"
    since = []
    if trial["start_min"] is not None:
        since.append(f"{trial['start_min']:.1f} min 施加")
    if trial["restore_min"] is not None:
        since.append(f"{trial['restore_min']:.1f} min 撤除")
    metrics = ""
    if trial["closed_peak_K"] is not None:
        held = trial["held_peak_K"]
        comparison = f"{held:.2f} K" if held is not None else "不可用"
        metrics = f"峰值绝对温差　闭环 {trial['closed_peak_K']:.2f} K　恒流对照 {comparison}"
    return dict(status=f"{PRESETS[trial['preset']].label} · {status}",
                timing=" · ".join(since), metrics=metrics)


def trial_traces(rows, period):
    bands, groups = [], []
    previous = None
    for row in rows:
        adjacent = previous is not None and previous["seq"] + 1 == row["seq"]
        same = adjacent and previous.get("trial_id") == row.get("trial_id")
        if row.get("disturbance_phase") == "active":
            if same and previous.get("disturbance_phase") == "active":
                bands[-1][1] = row["t"]
            else:
                bands.append([max(0, row["t"]-period), row["t"]])
        if row.get("T_hold") is not None and row.get("Qc_hold") is not None:
            if not same or previous.get("T_hold") is None:
                anchor = [(previous["t"], previous["T"], row["Qc_hold"])] if adjacent else []
                groups.append(anchor)
            groups[-1].append((row["t"], row["T_hold"], row["Qc_hold"]))
        previous = row
    return bands, groups


def control_status(snapshot):
    sample = snapshot["latest"]
    active = sample.get("control_source") == "pi"
    if active:
        timing = f"PI 备用 · 控制决策 {sample['decision_ms']:.1f} ms"
    elif sample.get("solve_ms") is not None:
        timing = f"MPC 求解 {sample['solve_ms']:.1f} ms"
    else:
        timing = "等待首个控制周期"
    if sample.get("cycle_overrun"):
        timing += " · 整周期计算超时"
    alarm = snapshot.get("last_fallback")
    message = ""
    if alarm:
        prefix = "PI 已接管" if active else "MPC 已恢复 · 最近一次 PI 接管"
        message = (f"{prefix} · 第 {alarm['interval']} 周期 · {alarm['reason']} · "
                   f"实际流量 {alarm['Qc']:.2f} L/min · 连续 {alarm['streak']} 个周期")
    return dict(active=active, timing=timing, message=message)


def online_update_status(policy):
    phase = policy["phase"]
    labels = {"ended": "本次仿真已结束", "off": "自动更新已关闭", "waiting_run": "等待仿真继续采样",
              "queued_to_submit": "样本已就绪，等待提交", "queued": "已排队", "training": "正在微调预测模型",
              "validating": "正在验证候选模型", "awaiting_cycle": "验证通过，等待下一控制周期切换"}
    if phase == "collecting":
        status = f"积累新样本　{policy['new_intervals']} / {policy['batch_intervals']} 个区间"
    else:
        status = labels[phase]
    if policy["automatic"] is not None:
        status = ("自动更新" if policy["automatic"] else "手动更新") + " · " + status
    last = policy["last"]
    result = ""
    if last:
        outcomes = {"activated": "已启用", "rollback": "已回退至", "rejected": "候选未通过",
                    "failed": "更新失败", "cancelled": "更新已取消"}
        result = " · ".join(filter(None, (f"第 {last['seq']} 周期", outcomes[last["state"]],
                                          last.get("version"), last.get("detail"))))
    return {"status": status, "result": result}
