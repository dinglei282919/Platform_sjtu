"""Web adapter for the shared Python mpc implementation."""
from platform_core.dnn_mpc.config import DEFAULT_OUTPUT_DIR, mpc_defaults

FIGURE_NAMES = ("dnn_mpc.svg",)


def default_config():
    return mpc_defaults()


def image_path(image_name, output_dir=None):
    from platform_core.dnn_mpc.runner import artifact_path, latest_artifact_dir
    if image_name not in FIGURE_NAMES:
        raise FileNotFoundError("结果图片不存在。")
    return artifact_path("mpc", image_name, output_dir or latest_artifact_dir("mpc"))


def run(config, log, *, run_dir=None):
    from platform_core.dnn_mpc.runner import run as execute
    return execute("mpc", config, log, run_dir=run_dir)
