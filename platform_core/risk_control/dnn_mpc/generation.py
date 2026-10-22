"""Python equivalent of generate_cartpole_dataset.m and cart_pole_dynamics.m.

Sampling and the explicit Euler transition follow the original source project.
NumPy's seeded generator is reproducible; its random stream differs from MATLAB.
These are simulated state transitions, not measured plant data.
"""
from io import BytesIO
import numpy as np
from scipy.io import savemat


GENERATION_SEED = 42
DT = 0.02


def plant_step(x, u):
    """Original nonlinear transition for an (..., 4) state array."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    mass, pole_mass, length, gravity = 1.0, 0.1, 0.5, 9.81
    sine, cosine = np.sin(x[..., 2]), np.cos(x[..., 2])
    angular_acceleration = (
        gravity * sine * (mass + pole_mass)
        - cosine * (u + pole_mass * length * x[..., 3] ** 2 * sine)
    ) / (length * (mass + pole_mass * (1 - cosine ** 2)))
    acceleration = (u + pole_mass * length * (
        x[..., 3] ** 2 * sine - angular_acceleration * cosine
    )) / (mass + pole_mass)
    derivative = np.stack((x[..., 1], acceleration, x[..., 3], angular_acceleration), axis=-1)
    return x + DT * derivative


def generate_dataset(sample_count, seed=GENERATION_SEED):
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or not 100 <= sample_count <= 100000:
        raise ValueError("生成样本数必须是 100～100000 之间的整数。")
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(sample_count, 4)) * [0.5, 0.5, 0.3, 0.5] + [0, 0, np.pi, 0]
    controls = rng.uniform(-10, 10, sample_count)
    x = np.column_stack((states, controls))
    y = plant_step(states, controls)
    metadata = dict(kind="generated", seed=seed, dt=DT, integrator="explicit Euler",
                    dynamics="original cart_pole_dynamics", sample_count=sample_count,
                    state_mean=[0, 0, float(np.pi), 0], state_std=[0.5, 0.5, 0.3, 0.5],
                    control_distribution="uniform", control_range=[-10, 10],
                    parameters=dict(M=1.0, m=0.1, length=0.5, g=9.81))
    return x, y, metadata


def dataset_bytes(x, y, metadata):
    """Export the original 5×N/4×N MAT convention with generation provenance."""
    stream = BytesIO()
    savemat(stream, dict(X_data=x.T, Y_data=y.T, dt=DT,
                        provenance="Simulated transitions from original cart_pole_dynamics",
                        generation=metadata), do_compression=True)
    return stream.getvalue()
