"""One set of labels, marker rules and sample readouts for both module-3 clients."""
from functools import lru_cache
from pathlib import Path

from .fdi import DEMO_VIEWS, GROUPS, signal_link
from . import cstr
from .fdi_control import DT
from .presentation import tokens


CHARTS = {
    "measurement": {"主测温、独立估计与 PI 反馈 (K)": {"Tm": "主釜温测量", "witness": "夹套估计", "feedback": "PI 采用值"}},
    "execution": {"夹套测量与预测 (K)": {"Tcm": "夹套测量", "prior_dnn_Tc": "上一拍对本拍的预测",
                                         "target_Tc": "下一拍夹套目标"}},
}
FLOW_CHART = {"Qc_cmd": "本地指令", "Qc": "实际流量 · 仿真评测"}
READINGS = {
    "T_ref": "釜温设定 (K)",
    "Tm": "釜温测量 (K)", "Tcm": "夹套测量 (K)", "witness": "夹套独立估计 (K)",
    "feedback": "有效反馈 (K)", "weight": "主测温权重", "residual": "一致性残差 (K)",
    "flow_read_previous": "上一拍流量计回读 (L/min)", "C_hat": "浓度估计 (mol/L)",
    "load_estimate": "釜侧热负荷估计 (K/min)", "load_used": "已采用釜侧估计 (K/min)",
    "prior_dnn_Tc": "本拍名义夹套预测 (K)", "jacket_innovation": "本拍夹套残差 (K)",
    "target_Tc": "下一拍夹套目标 (K)", "dnn_Tc": "下一拍名义夹套预测 (K)",
    "slope": "局部流量斜率 (K/(L/min))", "rbf": "RBF 估计 (K)", "robust": "有界鲁棒项 (K)",
    "inner_delta_q": "已采用夹套侧流量修正 (L/min)", "Qc_cmd": "最终本地指令 (L/min)",
    "T": "真实釜温 · 仿真评测 (K)", "Qc": "实际流量 · 仿真评测 (L/min)",
}
METRICS = {"event_intervals": "有效样本段区间数", "recovery_intervals": "有效恢复区间数",
           "iae_event": "样本段 IAE (K·min)", "iae_recovery": "恢复段 IAE (K·min)", "iae": "两段合计 IAE (K·min)",
           "event_peak": "样本段峰值温差 (K)", "recovery_peak": "恢复段峰值温差 (K)",
           "recovery_min": "恢复用时 (min)", "tv": "实际流量总变差 (L/min)",
           "saturated_min": "实际流量饱和时长 (min)", "concentration_peak": "浓度最大偏移 (mol/L)",
           "measurement": {"feedback_rmse": "反馈 RMSE (K)", "healthy_takeover_min": "健康段接管时长 (min)"},
           "execution": {"jacket_nominal_rmse": "名义夹套预测 RMSE (K)",
                         "jacket_adapted_rmse": "加入 RBF 后 RMSE (K)", "capacity_min": "能力受限时长 (min)"}}
EVENTS = {"begin_queued": "FDI 样本等待注入", "finish_queued": "结束样本段已排队",
          "segment_cancelled": "已取消排队样本", "segment_start": "样本段生效",
          "segment_end": "样本段结束", "feedback_source": "反馈切换", "feedback_unavailable": "分支暂停",
          "compensation_response": "补偿首次偏离配对健康轨迹", "horizon_reached": "恢复观察结束",
          "failed": "计算暂停", "pause": "暂停", "resume": "继续", "start": "开始", "end": "结束",
          "lease_paused": "连接超时暂停", "run_reset": "复位并保留记录"}
PHASES = {"healthy": "健康运行", "active": "样本段生效", "recovery": "恢复观察", "monitoring": "持续监测"}


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def preview_marker(side, scenario):
    link = signal_link(side, scenario)
    return dict(link=link, active=False, label="已选 · 待生效") if link else {}


def nominal_preview(side, scenario):
    return dict(seq=-1, loop_side=side, C=cstr.NOMINAL[0], T=cstr.NOMINAL[1], Tc=cstr.NOMINAL[2], Qc=cstr.NOMINAL_FLOW,
                T_ref=cstr.NOMINAL[1], Ci=cstr.INLET[0], Ti=cstr.INLET[1], Tci=cstr.INLET[2],
                signal_marker=preview_marker(side, scenario))


def diagram_sample(snapshot, row, name, *, live):
    group = row["groups"][name]
    if "T" not in group:
        return None
    active = group["valid"] and row["phase"] == "active"
    if live:
        preview = snapshot["pending"] == "begin" or (row["phase"] == "healthy" and snapshot["preview"])
    else:
        preview = row["preview"]
    marker = {}
    if row["link"] and (active or preview):
        side = "测量端" if row["link"] == "measurement" else "执行端"
        marker = dict(link=row["link"], active=active, label=side+" FDI 生效" if active else "已选 · 待生效")
    return dict(group, loop_side=snapshot["config"]["side"], signal_marker=marker,
                Qc=group["Qc"] if group["Qc"] is not None else group["previous_Qc"])


@lru_cache(maxsize=1)
def loop_template():
    return (Path(__file__).resolve().parents[2]/"assets/cstr/fdi-loops.svg").read_text(encoding="utf-8")


def loop_tokens(arm):
    return dict(MEASURELOOP="1" if arm in ("fixed","guard") else "0",
                EXECLOOP="1" if arm in ("full","bare","inner","outer") else "0",
                PILOOP="1" if arm in ("raw","pi") else "0",
                SELECTMODE="固定权重融合" if arm == "fixed" else "反馈校验与选择",
                OUTERMODE="采用热负荷估计" if arm in ("full","outer") else "釜侧补偿关闭",
                INNERMODE="采用等效温差补偿" if arm in ("full","inner") else "夹套侧补偿关闭")


def view(snapshot, *, live=True):
    side = snapshot["config"]["side"]
    groups = GROUPS[side]
    row = snapshot["latest"]
    diagrams, readings, sources = {}, {}, {}
    for name, group in (row["groups"].items() if row else ()):
        sample = diagram_sample(snapshot, row, name, live=live)
        if sample is not None:
            diagrams[name] = tokens(sample)
        readings[name] = [[label, number(group[key], 4)] for key, label in READINGS.items() if key in group]
        source = group.get("source", "无有效样本")
        if group.get("command_limited"):
            source += " · 本地指令已到限"
        if row["terminal"] and group["valid"]:
            source += " · 终点流量显示上一完成区间"
        if not group["valid"]:
            failure = group["failure"]
            source += f" · 在 {failure['seq']*DT:.1f} min 暂停，{failure['reason']}（保留最后有效样本）"
        sources[name] = source
    segment = snapshot["segment"]
    phase = PHASES[row["phase"]] if row else "准备观测器"
    if segment:
        phase = f"第 {segment} 次注入 · {phase}"
    if live and snapshot["pending"]:
        phase += " · 等待下一采样边界" + ("开始" if snapshot["pending"] == "begin" else "结束")
    cutoff = row["seq"] if row else -1
    events = []
    for event in snapshot["recent"]:
        if event["seq"] <= cutoff:
            sequence = f"第 {event['segment']} 次" if event.get("segment") else None
            detail = " · ".join(filter(None, (sequence, groups.get(event.get("group")), event.get("source"), event.get("reason"))))
            events.append(f"{event['t']:.1f} min　{EVENTS.get(event['event'], event['event'])}" + (" · "+detail if detail else ""))
    metrics = {k: v for k,v in METRICS.items() if isinstance(v, str)}
    metrics.update(METRICS[side])
    table = [[label, *[number(snapshot["metrics"][g][key], 0 if key.endswith("intervals") else 3)
                        for g in groups]] for key,label in metrics.items()]
    onset, stop = snapshot["start_seq"], snapshot["stop_seq"]
    first_response = (snapshot["metrics"]["guard"]["first_switch_min"] if side == "measurement" else
                      None if snapshot["response_seq"] is None else snapshot["response_seq"]*DT)
    times = [(onset*DT if onset is not None else None, "样本开始"),
             (first_response, "首次接管" if side == "measurement" else "补偿首次偏离健康对照"),
             (stop*DT if stop is not None else None, "样本结束")]
    timeline = "　".join(f"{label} {at:.1f} min" for at,label in times if at is not None and at <= cutoff*DT+1e-9)
    scored = snapshot["score_seq"]
    label = f"第 {segment} 次注入评测" if segment else "注入前基线"
    state = {"interrupted": " · 恢复观察被下次注入截短", "complete": " · 恢复观察完成",
             "ended": " · 试验已结束"}.get(snapshot["score_status"], "")
    metrics_title = f"{label} · 截至 {max(0, scored)*DT:.1f} min{state}"
    return dict(phase=phase, timeline=timeline, metrics_title=metrics_title, diagrams=diagrams, readings=readings, sources=sources,
                events=events, table=table)
