"""UI-independent defaults, paths and validation shared by CS and BS."""
from pathlib import Path
import json
import math
import hashlib
from platform_core.control_paths import CONTROL_OUTPUT_ROOT, module_output_dir

ROOT = Path(__file__).resolve().parents[2]
# Stable shared location for the published model, task lock and latest-result pointers.
DEFAULT_OUTPUT_DIR = CONTROL_OUTPUT_ROOT
DEFAULT_DATASET = ROOT / "input_data" / "dnn_mpc" / "cartpole_dataset.mat"
LATEST_TRAINING_STATE_PATH = DEFAULT_OUTPUT_DIR / "latest_training_state.json"
MPC_PLOT_GROUPS = (("dnn_mpc", "总览", "simulation_figure"),
                   ("mpc_states", "状态响应", "state_figure"),
                   ("mpc_control", "控制输入", "control_figure"))


def training_defaults():
    return dict(output_dir=str(module_output_dir("training", root=DEFAULT_OUTPUT_DIR)),
                model_path=str(DEFAULT_OUTPUT_DIR / "trained_model.npz"),
                dataset_path=str(DEFAULT_DATASET), epochs=200, lr=0.001,
                hidden_layers="64,64", sample_count=1000)


def mpc_defaults():
    result = dict(output_dir=str(module_output_dir("mpc", root=DEFAULT_OUTPUT_DIR)),
                  model_path=str(DEFAULT_OUTPUT_DIR / "trained_model.npz"),
                  sim_time=5.0, prediction_horizon=10)
    try:
        state = json.loads(LATEST_TRAINING_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("format_version") == 1 and Path(state["model_path"]).is_file():
            result.update(model_path=state["model_path"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def model_info(model_path):
    path = Path(model_path)
    if not path.is_absolute() or path.suffix.lower() != ".npz":
        raise ValueError("模型必须使用本机绝对路径和 .npz 扩展名。")
    if not path.is_file():
        return {"exists": False, "sha256": None}
    return {"exists": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_config(mode, config):
    if mode not in ("training", "mpc"):
        raise ValueError("未知 DNN-MPC 模块。")
    result = (training_defaults() if mode == "training" else mpc_defaults()) | config
    for key in ("output_dir", "model_path"):
        value = str(result[key]).strip()
        if not value or not Path(value).is_absolute():
            raise ValueError("输出目录和模型文件必须使用本机绝对路径。")
        result[key] = value
    if Path(result["model_path"]).suffix.lower() != ".npz":
        raise ValueError("模型必须是 Python 训练导出的 .npz 文件；不支持 MATLAB .mat 模型。")
    if mode == "training":
        samples = result["sample_count"]
        if isinstance(samples, bool) or not isinstance(samples, int) or not 100 <= samples <= 100000:
            raise ValueError("生成样本数必须是 100～100000 之间的整数。")
        if isinstance(result["epochs"], bool) or not isinstance(result["epochs"], int) or not 1 <= result["epochs"] <= 5000:
            raise ValueError("训练轮数必须是 1～5000 之间的整数。")
        if not math.isfinite(result["lr"]) or not 1e-6 <= result["lr"] <= 0.1:
            raise ValueError("学习率必须在 0.000001～0.1 之间。")
        try:
            layers = [int(part.strip()) for part in result["hidden_layers"].split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError("隐藏层填写两个整数，例如 64,64。") from exc
        if len(layers) != 2 or any(not 1 <= width <= 512 for width in layers):
            raise ValueError("必须设置两个隐藏层，每层 1～512 个节点。")
        result["hidden_layers"] = ",".join(map(str, layers))
        result["dataset_path"] = str(result.get("dataset_path", "")).strip()
        dataset = Path(result["dataset_path"])
        if result["dataset_path"] and (not dataset.is_absolute() or dataset.suffix.lower() != ".mat"):
            raise ValueError("训练数据必须使用本机绝对路径和 .mat 扩展名。")
        if Path(result["model_path"]).exists() and not Path(result["model_path"]).is_file():
            raise ValueError("模型保存位置必须是文件路径。")
    else:
        if not math.isfinite(result["sim_time"]) or not 0.2 <= result["sim_time"] <= 20:
            raise ValueError("仿真时长必须在 0.2～20 秒之间。")
        horizon = result["prediction_horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 60:
            raise ValueError("预测步长必须是 1～60 之间的整数。")
    return result
