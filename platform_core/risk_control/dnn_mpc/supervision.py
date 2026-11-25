"""MPC proposals are isolated from the session clock and never advance the plant."""
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread

import numpy as np

from ..cstr import SAMPLE_MIN
from .model import Predictor


DISPATCH_RESERVE_S = 0.050
PI_KP = 11.0  # L/min per K
PI_KI = 5.0   # L/min per (K min)
TRANSFER_STEP = 5.0  # L/min per sampled interval


def backup_flow(integral, error, previous, settings):
    lower = max(settings.flow_min, previous - TRANSFER_STEP)
    upper = min(settings.flow_max, previous + TRANSFER_STEP)
    request = integral + PI_KP * error
    if not np.isfinite(request):
        raise ValueError("备用 PI 输入无效")
    flow = float(np.clip(request, lower, upper))
    # Integrate at a limit only when the error drives the controller back into range.
    if (lower < request < upper or (request >= upper and error < 0)
            or (request <= lower and error > 0)):
        integral += PI_KI * SAMPLE_MIN * error
    return flow, integral


class SolverWorker:
    """One slot per session. Each reply belongs only to the caller that submitted it."""
    def __init__(self, condition):
        self.condition = condition
        self.job = None
        self.closed = False
        self.thread = Thread(target=self._run, name="cstr-mpc", daemon=True)
        self.thread.start()

    def submit(self, func, args, kwargs):
        # The caller holds the session condition throughout admission.
        if self.job is not None or self.closed:
            return None
        reply = Future()
        self.job = (reply, func, args, kwargs)
        self.condition.notify_all()
        return reply

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.job is not None)
                if self.closed:
                    return
                reply, func, args, kwargs = self.job
            try:
                result = func(*args, **kwargs)
                error = None
            except Exception as exc:
                result, error = None, exc
            with self.condition:
                self.job = None
                if error is None:
                    reply.set_result(result)
                else:
                    reply.set_exception(error)
                self.condition.notify_all()


@dataclass
class Decision:
    command: float
    integral: float
    model: Predictor
    attempts: list[str]
    sequence: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    reason: str = ""
    outcome: str = "ok"
    rollback: str = ""
    missed: bool = False

    @property
    def source(self):
        return "pi" if self.sequence is None else "mpc"
