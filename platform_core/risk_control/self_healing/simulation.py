"""Normal-only identification, batch monitoring, and incremental self-healing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .algorithms import ARReference, CCSC, SCSC
from .metrics import evaluate


class SimulationCancelled(Exception):
    """Cooperative stop requested by either platform adapter."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise SimulationCancelled("自愈控制仿真已取消。")


@dataclass
class ExperimentConfig:
    scenario: str = "water"
    duration: float = 100.0
    dt: float = 0.1
    batch_duration: float = 0.1
    fault_start: float = 40.0
    training_batches: int = 100
    training_hold: float = 2.0
    excitation: float = 0.05
    seed: int = 42
    scsc_gain: float = -0.1
    ccsc_gain: float = -0.2
    detection_fraction: float = 0.005
    max_step_fraction: float = 0.01
    reference_mode: str = "constant"
    ccsc_centering: str = "reference"
    cstr_input_mode: str = "paper"
    measurement_noise: float = 0.0

    def validate(self):
        numeric = [self.duration, self.dt, self.batch_duration, self.fault_start,
                   self.training_hold, self.excitation, self.scsc_gain, self.ccsc_gain,
                   self.detection_fraction, self.max_step_fraction, self.measurement_noise]
        if not np.isfinite(numeric).all():
            raise ValueError("configuration numbers must be finite")
        if self.scenario not in ("water", "cstr_htc", "cstr_catalyst"):
            raise ValueError("unknown scenario")
        if not 0 <= self.fault_start < self.duration:
            raise ValueError("fault_start must be within [0, duration)")
        if min(self.dt, self.batch_duration, self.training_hold) <= 0:
            raise ValueError("time intervals must be positive")
        for value in (self.duration, self.batch_duration, self.training_hold):
            # A relative tolerance grows with the number of samples and can
            # accept a fractional sample at long horizons. Only allow float
            # roundoff here; rounding must not silently change requested times.
            if not np.isclose(value / self.dt, round(value / self.dt), rtol=0, atol=1e-8):
                raise ValueError("duration, batch_duration and training_hold must be multiples of dt")
        if self.batch_duration < self.dt or self.training_hold < 2 * self.dt:
            raise ValueError("batch_duration >= dt and training_hold >= 2*dt required")
        if self.batch_duration > self.duration:
            raise ValueError("batch_duration must not exceed duration")
        if not isinstance(self.training_batches, int) or isinstance(self.training_batches, bool) or self.training_batches < 8:
            raise ValueError("training_batches must be an integer >= 8")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 < self.excitation < 0.5:
            raise ValueError("excitation must be between 0 and 0.5")
        if self.detection_fraction <= 0 or self.max_step_fraction <= 0 or self.measurement_noise < 0:
            raise ValueError("monitor/step fractions must be positive; noise nonnegative")
        if self.reference_mode not in ("constant", "ar") or self.ccsc_centering not in ("reference", "batch"):
            raise ValueError("unsupported reference/CCSC centering mode")
        if self.cstr_input_mode not in ("paper", "legacy"):
            raise ValueError("unsupported CSTR input mode")


def defaults(scenario):
    if scenario == "water":
        return ExperimentConfig()
    if scenario in ("cstr_htc", "cstr_catalyst"):
        return ExperimentConfig(scenario=scenario, duration=1200.0, dt=1.0,
                                batch_duration=10.0, fault_start=200.0,
                                training_hold=10.0, scsc_gain=-0.3, ccsc_gain=-0.3,
                                excitation=0.005, max_step_fraction=0.005)
    raise ValueError(f"unknown scenario: {scenario}")


def make_plant(config, normal=False):
    if config.scenario == "water":
        from .water import WaterPlant
        return WaterPlant(dt=config.dt, fault="none" if normal else "blockage",
                          fault_start=config.fault_start)
    from .cstr import CSTRPlant
    return CSTRPlant(dt=config.dt, fault="none" if normal else config.scenario.removeprefix("cstr_"),
                     fault_start=config.fault_start, input_mode=config.cstr_input_mode)


def collect_training(config, *, cancelled=None, on_progress=None):
    """Excite only a fault-free plant; keep settled half of each hold.

    Simulation training and evaluation are independent trajectories. Training
    uses no fault samples or derivatives of the process model.
    """
    plant = make_plant(config, normal=True)
    rng = np.random.default_rng(config.seed)
    nominal = np.asarray(plant.nominal_input)
    lower, upper = plant.input_bounds
    n_hold = round(config.training_hold / config.dt)
    xs, ys, batch_means = [], [], []
    for episode in range(config.training_batches):
        command = np.clip(nominal * (1 + config.excitation * rng.uniform(-1, 1, len(nominal))), lower, upper)
        samples = []
        for j in range(n_hold):
            _check_cancelled(cancelled)
            y = plant.step(command, (episode * n_hold + j) * config.dt)
            if j >= n_hold // 2:
                xs.append(command.copy())
                ys.append(y.copy())
                samples.append(y.copy())
        batch_means.append(np.mean(samples, axis=0))
        if on_progress:
            on_progress(int(20 * (episode + 1) / config.training_batches), "正在收集正常训练数据")
    return np.asarray(xs), np.asarray(ys), np.asarray(batch_means)


def run_experiment(config, progress=None, *, cancelled=None, on_progress=None):
    config.validate()
    _check_cancelled(cancelled)
    if progress:
        progress("collecting fault-free training trajectories")
    x_train, y_train, batch_means = collect_training(config, cancelled=cancelled, on_progress=on_progress)
    plant = make_plant(config, normal=True)
    reference = np.asarray(plant.reference_output, dtype=float).copy()
    if config.reference_mode == "ar":
        reference = ARReference(order=2).fit(batch_means).predict()
    weights = np.array([20., 1., 1., 1.]) if config.scenario == "water" else np.array([1., 1., .05])
    controllers = {"scsc": SCSC(weights=weights).fit(x_train, y_train),
                   "ccsc": CCSC(centering=config.ccsc_centering).fit(x_train, y_train)}
    n = round(config.duration / config.dt)
    q = round(config.batch_duration / config.dt)
    times = (np.arange(n) + 1) * config.dt
    tolerance = np.maximum(np.abs(reference) * config.detection_fraction, 1e-8)
    runs = {}
    for method_index, name in enumerate(("normal", "fault", "scsc", "ccsc")):
        if progress:
            progress(f"simulating {name}")
        process = make_plant(config, normal=(name == "normal"))
        command = np.asarray(process.nominal_input, dtype=float).copy()
        lower, upper = process.input_bounds
        step_limit = np.maximum(np.abs(command) * config.max_step_fraction, 1e-8)
        inputs = np.empty((n, len(command)))
        outputs = np.empty((n, len(reference)))
        measured = np.empty_like(outputs)
        events = []
        # Same random measurement sequence for every controller comparison.
        rng = np.random.default_rng(config.seed + 1)
        for i in range(n):
            _check_cancelled(cancelled)
            if on_progress and i % max(1, n // 50) == 0:
                on_progress(20 + int(65 * (method_index + i / n) / 4), f"正在仿真 {name.upper()}：{i * config.dt:g} / {config.duration:g}")
            inputs[i] = command
            outputs[i] = process.step(command, i * config.dt)
            measured[i] = outputs[i] + rng.normal(size=len(reference)) * np.abs(reference) * config.measurement_noise
            if not np.isfinite(outputs[i]).all():
                raise FloatingPointError(f"nonfinite state in {name} at t={times[i]}")
            if name in controllers and (i + 1) % q == 0 and i + 1 < n:
                y_batch = measured[i + 1 - q:i + 1]
                if np.any(np.abs(y_batch.mean(axis=0) - reference) > tolerance):
                    delta = controllers[name].correction(y_batch, reference_y=reference, current_x=command)
                    gain = config.scsc_gain if name == "scsc" else config.ccsc_gain
                    requested = gain * delta
                    limited = np.clip(requested, -step_limit, step_limit)
                    updated = np.clip(command + limited, lower, upper)
                    events.append({"time": float(times[i]), "requested_delta": requested.tolist(),
                                   "applied_delta": (updated - command).tolist(),
                                   "limited": bool(np.any(np.abs(updated - command - requested) > 1e-10))})
                    command = updated
        runs[name] = {"outputs": outputs, "measured": measured, "inputs": inputs, "events": events}
    # Each metric window is explicit and identical across controllers.
    windows = {"full": np.ones(n, dtype=bool), "post_fault": times > config.fault_start,
               "tail_20_percent": times > config.duration * .8}
    scores = {name: {window: evaluate(run["outputs"][mask], reference, times[mask])
                    for window, mask in windows.items()} for name, run in runs.items()}
    _check_cancelled(cancelled)
    return {"config": asdict(config), "time": times, "reference": reference,
            "tolerance": tolerance, "input_names": list(plant.input_names),
            "output_names": list(plant.output_names), "runs": runs, "metrics": scores,
            "training": {"inputs": x_train, "outputs": y_train},
            "diagnostics": {name: controller.diagnostics() for name, controller in controllers.items()}}
