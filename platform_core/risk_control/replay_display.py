"""Presentation of the observer copy and the single valve command on both clients."""
from pathlib import Path

from . import cstr
from .replay import ARMS, DT, START, T_REF

TEMPLATE = (Path(__file__).resolve().parents[2] / "assets/cstr/replay-loop.svg").read_text(encoding="utf-8")
ACTIONS = {"current": "按本拍测温更新", "as_received": "按收到的测温更新", "hold": "保持上次可信估计",
           "fade": "减弱补偿", "local_update": "按本地釜温重估热负荷",
           "reanchor": "恢复首拍，对齐温度", "pi": "仅本地 PI", "local_missing": "本地测温缺失，分支停止"}
QUANTITIES = {"iae": "釜温 IAE (K·min)", "heat_rmse": "热负荷 RMSE (K/min)",
              "recovery_min": "观测恢复 (min)", "flow_tv": "流量总变差 (L/min)",
              "saturated_min": "流量限幅 (min)", "false_hold_rate": "正常通信误保持比例"}

COLORS = {"raw":"#ffbd72", "guard":"#79baff", "candidate":"#70dbc1", "continuous":"#fa8faf",
          "fresh":"#99dfa5", "pi":"#cda5fe"}
GUIDE = "观测支路可能收到旧测温，本地 PI 使用当前釜温。对比各组怎样估计热负荷、追加冷却流量。"
CASE_GUIDES = {"constant": "重放期间维持额外放热，可比较短时保持和两种本地重估的冷却流量。",
               "removed": "本段第 4 拍撤去额外放热。查看本地实测偏离预测后，各组何时减小附加冷却。"}


def chart(title, unit, prefix, groups, reference_key=None, reference_label=None):
    lines = [dict(key=prefix+g, arm=g, name=ARMS[g], color=COLORS[g], dashed=g in ("raw","pi"), optional=g in ("fresh","pi"))
             for g in groups]
    if reference_key:
        lines.append(dict(key=reference_key, name=reference_label, color="#c1c9d3", dashed=True, optional=False))
    return dict(title=title, unit=unit, lines=lines)


SENSORS = {"Tc": "夹套冷却液出口温度", "T": "釜内物料温度"}
OBSERVATIONS = {
    key: dict(title="测量影响 · " + name, unit="K", lines=[
        dict(key="local_"+key, name="本拍实测", color="#76d6be", dashed=False, optional=False),
        dict(key="received_"+key, name="观测器收到", color="#ffbd72", dashed=True, optional=False)])
    for key, name in SENSORS.items()}
CHARTS = [OBSERVATIONS["Tc"], chart("控制作用 · 下发冷却阀流量指令", "L/min", "u_", ARMS),
          chart("被控变量 · 真实釜温", "K", "T_", ARMS, "setpoint", "釜温设定")]
HEAT_CHART = chart("热负荷估计", "K/min", "w_", [g for g in ARMS if g != "pi"], "heat", "仿真施加值")


def trend(row, arm="guard"):
    groups = row["groups"]
    s = groups[arm]
    jacket = s.get("Tc_local_K", s.get("Tc_to_observer_K") if s.get("source_k") == row["seq"] else None)
    return {"t":row["t"], "setpoint":T_REF, "heat":groups["guard"].get("heat_true_K_min"),
            "local_T":s.get("T_local_K"), "local_Tc":jacket,
            "received_T":s.get("T_to_observer_K") if arm != "pi" else None,
            "received_Tc":s.get("Tc_to_observer_K") if arm != "pi" else None,
            **{"u_"+g:s.get("command_L_min") for g,s in groups.items()},
            **{"T_"+g:s.get("T_true_K") for g,s in groups.items()},
            **{"w_"+g:s.get("heat_est_K_min") for g,s in groups.items() if g != "pi"}}


def number(value, unit="", signed=False):
    if value is None:
        return "—"
    return (f"{value:+.3f}" if signed else f"{value:.3f}") + (" "+unit if unit else "")


def diagram_sample(row, arm, view):
    if row is None:
        return None
    group = row["groups"][arm]
    if not group["valid"]:
        return None
    marker = {} if arm == "pi" else dict(link="observer", label=view["link"],
                                         preview=view["preview"], active=view["active"])
    ci, ti, tci = cstr.INLET
    return dict(seq=row["seq"], C=group.get("C_true_mol_L"), T=group["T_true_K"], Tc=group["Tc_actual_K"],
                Qc=group["command_L_min"], T_ref=T_REF, Ci=ci, Ti=ti, Tci=tci,
                signal_marker=marker, show_sensors=True)


def display(row, arm, scheduled):
    sample = row["groups"][arm]
    seq = row["seq"]
    observer = arm != "pi"
    local_branch = arm in ("candidate", "continuous")
    marked = arm in ("raw", "guard") or local_branch
    active = marked and sample.get("delivery") == "record"
    preview = marked and scheduled and not active and ("pending" in row or seq < START)
    link = "重放生效" if active else "已选 · 待生效" if preview else "当前测温" if observer else "观测支路未接入"
    source = sample.get("source_k")
    base, comp, final = (sample.get(k) for k in ("PI_L_min", "comp_L_min", "command_L_min"))
    tokens = dict(LOCAL=number(sample.get("T_local_K"), "K"), BASE=number(base,"L/min"),
                  COMP=number(comp,"L/min",True), COMMAND=number(final,"L/min"), HEAT=number(sample.get("heat_est_K_min"),"K/min"),
                  LINK=link, COLOR="#ffbc71" if active or preview else "#8bb6ce",
                  LOCAL_BRANCH="1" if local_branch else "0",
                  LOCAL_PATH_COLOR="#70dbc1" if sample.get("local_updating") else "#8296a8",
                  WIDTH="4" if active else "2.5", DASH="8 4" if active else "none", OBSERVER="1" if observer else "0")
    svg = TEMPLATE
    for k,v in tokens.items():
        svg = svg.replace("@@"+k+"@@",v)
    mode = "—"
    if local_branch:
        if sample.get("local_triggered"):
            mode = "本拍触发，开始重估"
        elif sample.get("local_updating"):
            mode = "逐拍使用本地釜温更新"
        elif sample.get("local_residual_K") is not None:
            mode = "等待连续两拍同向越限"
        else:
            mode = ACTIONS[sample["status"]]
    streak = sample.get("local_streak")
    return dict(svg=svg, link=link, active=active, preview=preview,
                received=f"{seq*DT:.1f} min · #{seq}", source=f"{source*DT:.1f} min · #{source}" if observer and source is not None else "—",
                local=tokens["LOCAL"], observed=number(sample.get("T_to_observer_K"),"K") if observer else "—",
                jacket=number(sample.get("Tc_local_K"),"K"), jacket_observed=number(sample.get("Tc_to_observer_K"),"K") if observer else "—",
                action=ACTIONS[sample["status"]], base=tokens["BASE"], comp=tokens["COMP"], final=tokens["COMMAND"],
                request=number(sample.get("request_L_min"),"L/min"), weight=number(sample.get("weight")),
                local_prediction=number(sample.get("local_prediction_K"), "K"),
                local_residual=number(sample.get("local_residual_K"), "K", True),
                local_streak=f"{streak:+d}" if arm == "candidate" and streak else
                            ("0" if arm == "candidate" and sample.get("local_residual_K") is not None else "—"),
                local_mode=mode)
