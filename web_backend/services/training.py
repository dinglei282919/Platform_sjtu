"""Web adapter for the shared Python training implementation."""
from platform_core.risk_control.dnn_mpc.config import training_defaults

FIGURE_NAMES = ("training_rmse.svg",)


def default_config():
    return training_defaults()


def image_path(image_name, output_dir=None):
    from platform_core.risk_control.dnn_mpc.runner import artifact_path, latest_artifact_dir
    if image_name not in FIGURE_NAMES:
        raise FileNotFoundError("结果图片不存在。")
    return artifact_path("training", image_name, output_dir or latest_artifact_dir("training"))


def run(config, log, *, run_dir=None):
    from platform_core.risk_control.dnn_mpc.runner import run as execute
    return execute("training", config, log, run_dir=run_dir)
