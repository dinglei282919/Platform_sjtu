"""One configuration contract for the desktop and browser adapters."""
from dataclasses import asdict, fields
import math

from .simulation import ExperimentConfig, defaults as experiment_defaults

SCENARIOS = {"water": "三水箱出口堵塞", "cstr_htc": "CSTR 换热衰减", "cstr_catalyst": "CSTR 催化剂衰减"}
PLOT_GROUPS = (("outputs", "输出对比"), ("inputs", "输入设定值"))
NOTICE = "研究机理仿真；非论文逐点复现。CCSC 在换热故障下可能恶化 C/T；时间沿用模型单位。"

# Shared form metadata; the complete experiment remains configurable.
PARAMETERS = [
    {"key": "duration", "label": "仿真时长（模型时间）", "min": .001, "max": 5000, "step": .1},
    {"key": "fault_start", "label": "故障起点（模型时间）", "min": 0, "max": 5000, "step": .1},
    {"key": "dt", "label": "采样间隔", "min": .0001, "max": 10, "step": .1},
    {"key": "batch_duration", "label": "自愈批次时长", "min": .0001, "max": 5000, "step": .1},
    {"key": "training_batches", "label": "正常训练批数", "min": 8, "max": 1000, "step": 1, "integer": True},
    {"key": "training_hold", "label": "训练输入保持时长", "min": .0002, "max": 1000, "step": .1},
    {"key": "reference_mode", "label": "正常参考", "choices": {"constant": "固定平衡参考", "ar": "正常数据 AR（冻结）"}},
    {"key": "ccsc_centering", "label": "CCSC 中心化", "choices": {"reference": "正常参考（工程解释）", "batch": "当前批均值（净修正近零）"}},
    {"key": "scsc_gain", "label": "SCSC 有符号增益", "min": -10, "max": 10, "step": .01},
    {"key": "ccsc_gain", "label": "CCSC 有符号增益", "min": -10, "max": 10, "step": .001},
    {"key": "excitation", "label": "训练激励比例", "min": .000001, "max": .499999, "step": .001},
    {"key": "detection_fraction", "label": "输出越限比例", "min": .000001, "max": 1, "step": .001},
    {"key": "max_step_fraction", "label": "单批输入变化上限比例", "min": .000001, "max": 1, "step": .001},
    {"key": "measurement_noise", "label": "测量噪声比例", "min": 0, "max": 1, "step": .001},
    {"key": "seed", "label": "随机种子", "min": 0, "max": 2147483647, "step": 1, "integer": True},
    {"key": "cstr_input_mode", "label": "CSTR 自愈输入", "choices": {"paper": "Ci / Ti / Tci", "legacy": "Ci / Ti / Tci / Tsp"}},
]


def defaults(scenario="water"):
    return asdict(experiment_defaults(scenario))


def catalog():
    return {"scenarios": SCENARIOS, "defaults": {key: defaults(key) for key in SCENARIOS},
            "parameters": PARAMETERS, "notice": NOTICE}


def validate_config(values):
    if not isinstance(values, dict):
        raise ValueError("配置必须为对象。")
    known = {field.name for field in fields(ExperimentConfig)}
    if set(values) - known:
        raise ValueError("未知参数：" + ", ".join(sorted(set(values) - known)))
    scenario = values.get("scenario", "water")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError("未知自愈场景。")
    config = {**defaults(scenario), **values}
    for parameter in PARAMETERS:
        key, value = parameter["key"], config[parameter["key"]]
        if "choices" in parameter:
            if not isinstance(value, str) or value not in parameter["choices"]:
                raise ValueError(f"{parameter['label']}选项无效。")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not parameter["min"] <= value <= parameter["max"] or not math.isfinite(value)
              or (parameter.get("integer") and not isinstance(value, int))
              ):
            raise ValueError(f"{parameter['label']}必须在 {parameter['min']}～{parameter['max']} 之间" + ("，且为整数。" if parameter.get("integer") else "。"))
    experiment = ExperimentConfig(**config)
    experiment.validate()
    samples = config["duration"] / config["dt"]
    training_samples = config["training_batches"] * config["training_hold"] / config["dt"]
    internal_step = .02 if scenario == "water" else .01
    if samples + training_samples > 100000 or (4 * samples + training_samples) * math.ceil(config["dt"] / internal_step) > 2000000:
        raise ValueError("仿真规模过大，请减少时长、训练批数或增大采样间隔。")
    return config
