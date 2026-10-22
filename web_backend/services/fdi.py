"""Web service adapter for the shared FDI simulation and its artifacts."""
from platform_core.risk_control.fdi.config import defaults
from concurrent.futures import CancelledError


def default_config():
    return defaults()


def new_run_dir():
    from platform_core.risk_control.fdi import runner
    return runner.new_run_dir()


def is_artifact(name):
    from platform_core.risk_control.fdi import runner
    return name in runner.ARTIFACT_NAMES


def artifact_path(directory, name):
    from platform_core.risk_control.fdi import runner
    return runner.artifact_path(directory, name)


def run(config, log, *, run_dir=None, cancelled=None):
    from platform_core.risk_control.fdi import runner
    from platform_core.risk_control.fdi.simulation import SimulationCancelled
    try:
        return runner.run(config, log, run_dir=run_dir, cancelled=cancelled)
    except SimulationCancelled as exc:
        raise CancelledError(str(exc)) from exc
