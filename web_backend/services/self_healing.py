"""Thin BS adapter; CS uses the same core runner directly."""
from concurrent.futures import CancelledError
from platform_core.risk_control.self_healing.config import catalog, validate_config


def default_config():
    return catalog()


def new_run_dir():
    from platform_core.risk_control.self_healing.runner import new_run_dir as create
    return create()


def is_artifact(name):
    from platform_core.risk_control.self_healing.runner import ARTIFACT_NAMES
    return name in ARTIFACT_NAMES


def artifact_path(directory, name):
    from platform_core.risk_control.self_healing.runner import artifact_path as resolve
    return resolve(directory, name)


def run(config, log, *, run_dir=None, cancelled=None):
    from platform_core.risk_control.self_healing.runner import run as execute
    from platform_core.risk_control.self_healing.simulation import SimulationCancelled
    try:
        return execute(config, log, run_dir=run_dir, cancelled=cancelled)
    except SimulationCancelled as exc:
        raise CancelledError(str(exc)) from exc
