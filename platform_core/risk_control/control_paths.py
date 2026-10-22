"""Output layout shared by the risk-control Python modules and both UIs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL_OUTPUT_ROOT = ROOT / "dnn_mpc" / "python_output"
MODULES = ("training", "mpc", "fdi", "replay")


def module_output_dir(module, *, root=CONTROL_OUTPUT_ROOT):
    if module not in MODULES:
        raise ValueError("未知风险管控模块：" + str(module))
    return Path(root) / module
