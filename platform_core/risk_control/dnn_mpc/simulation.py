"""Receding-horizon simulation driven by a trained state-space network."""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

REFERENCE = np.array([0., 0., np.pi, 0.])
INITIAL = REFERENCE + [0., 0., .1, 0.]
Q = np.diag([300., 1., 300., 1.])
R = .01
SOLVER_OPTIONS = {"maxiter": 100, "ftol": 1e-8}


class ArrayPredictor:
    """NumPy equivalent of a frozen network, with an analytic input Jacobian."""
    def __init__(self, model):
        if model.metadata["target_mode"] != "next_state":
            raise ValueError("This experiment requires next-state labels")
        self.w = [model.net[i].weight.detach().numpy().copy() for i in (0, 2, 4)]
        self.b = [model.net[i].bias.detach().numpy().copy() for i in (0, 2, 4)]
        self.s, self.activation = model.scaler, model.metadata["activation"]

    def step(self, state, force):
        s = self.s
        z = (np.r_[state, force] - s["x_mean"]) / s["x_std"]
        jac = np.diag(1 / s["x_std"])
        for i in range(3):
            z = self.w[i] @ z + self.b[i]
            jac = self.w[i] @ jac
            if i < 2:
                if self.activation == "tanh":
                    z = np.tanh(z)
                    jac *= (1 - z*z)[:, None]
                else:
                    jac *= (z > 0)[:, None]
                    z = np.maximum(z, 0)
        return z * s["y_std"] + s["y_mean"], jac * s["y_std"][:, None]


def terminal_weight(predictor):
    """Local LQR tail cost; this is not a nonlinear stability certificate."""
    predicted, jac = predictor.step(REFERENCE, 0.)
    if not np.isfinite(predicted).all() or not np.isfinite(jac).all():
        raise ValueError("模型在参考状态附近产生非有限值，请检查或重新训练模型。")
    a, b = jac[:, :4], jac[:, 4:5]
    try:
        p = solve_discrete_are(a, b, Q, np.array([[R]]))
        p = (p + p.T) / 2
        gain = np.linalg.solve(np.array([[R]]) + b.T @ p @ b, b.T @ p @ a)
        residual = a.T @ p @ a - p - a.T @ p @ b @ gain + Q
        radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ gain))))
        relative_residual = float(np.linalg.norm(residual) / max(1., np.linalg.norm(p)))
        if (not np.isfinite(p).all() or not np.isfinite(radius)
                or np.linalg.eigvalsh(p).min() <= 0 or radius >= 1.
                or not np.isfinite(relative_residual) or relative_residual > 1e-7):
            raise ValueError("Invalid stabilizing Riccati solution")
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError("当前模型无法构造稳定的局部终端权重，请检查数据和训练质量后重新训练。") from exc
    return p, dict(method="local_discrete_riccati", linearization_state=REFERENCE.tolist(),
                   linearization_control=0., A=a.tolist(), B=b.tolist(),
                   local_feedback_gain=gain.tolist(), local_closed_loop_radius=radius,
                   riccati_relative_residual=relative_residual,
                   reference_prediction_error=(predicted - REFERENCE).tolist())


def cost_gradient(controls, state, predictor, terminal):
    """Sum x[k]'Qx[k] + R*u[k]^2 for k=0..N-1, plus x[N]'P*x[N]."""
    state = np.asarray(state).copy()
    derivative = np.zeros((4, len(controls)))
    gradient = np.zeros(len(controls))
    error = state - REFERENCE
    value = error @ Q @ error  # Constant with respect to the control sequence.
    for i, control in enumerate(controls):
        state, jac = predictor.step(state, control)
        derivative = jac[:, :4] @ derivative
        derivative[:, i] += jac[:, 4]
        error = state - REFERENCE
        weight = terminal if i == len(controls)-1 else Q
        weighted_error = weight @ error
        value += weighted_error @ error + R * control**2
        gradient += 2 * weighted_error @ derivative
        gradient[i] += 2 * R * control
    return float(value), gradient


def simulate(model, callback=None, *, duration=5.0, horizon=10, on_step=None):
    """Run MPC; on_step receives (completed, total, state_copy, applied_control)."""
    if not np.isclose(model.metadata["dt"], .02):
        raise ValueError("Simulation requires dt=0.02 s")
    if not np.isfinite(duration) or not .2 <= duration <= 20:
        raise ValueError("仿真时长必须在 0.2～20 秒之间。")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 60:
        raise ValueError("预测步长必须是 1～60 之间的整数。")
    steps = int(np.floor(duration / .02 + 1e-9))
    predictor = ArrayPredictor(model)
    terminal, diagnostics = terminal_weight(predictor)
    state = INITIAL.copy()
    states, sequences, logs = [], [], []
    guess = np.zeros(horizon)
    for k in range(steps):
        # Optimize the sequence and apply only its first input.
        result = minimize(cost_gradient, guess, args=(state, predictor, terminal), jac=True,
                          method="SLSQP", bounds=[(-10., 10.)]*horizon,
                          options=SOLVER_OPTIONS.copy())
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError(f"Non-finite optimization at step {k}")
        sequence = np.clip(result.x, -10., 10.)
        guess = np.r_[sequence[1:], sequence[-1]]
        state, _ = predictor.step(state, sequence[0])
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite state at step {k}")
        states.append(state.copy())
        sequences.append(sequence)
        logs.append({"step": k, "success": bool(result.success), "cost": float(result.fun),
                     "iterations": int(result.nit), "message": str(result.message)})
        if on_step:
            on_step(k+1, steps, state.copy(), float(sequence[0]))
        if callback:
            callback(k+1, steps)
    sequences = np.asarray(sequences)
    return {"time": (np.arange(steps)+1)*.02, "control_time": np.arange(steps)*.02, "states": np.asarray(states),
            "controls": sequences[:, 0], "sequences": sequences, "logs": logs,
            "dt": .02, "horizon": horizon, "requested_duration": duration,
            "actual_duration": steps*.02, "source": "python_slsqp",
            "controller": dict(terminal_weight=diagnostics, Q=np.diag(Q).tolist(), P=terminal.tolist(), R=R,
                               maxiter=SOLVER_OPTIONS["maxiter"], ftol=SOLVER_OPTIONS["ftol"],
                               warm_start="shift_previous_sequence_repeat_last",
                               objective="sum(k=0..N-1, e[k]'Qe[k] + R*u[k]^2) + e[N]'Pe[N]")}
