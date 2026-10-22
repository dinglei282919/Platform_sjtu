"""Correlation-based batch self-healing laws from the accompanying paper.

All observations are row vectors. ``correction`` returns the *unsigned* paper
adjustment X_e in physical input units: the caller applies the signed control
gain, limits and actuator constraints. The simulator uses
``next_x = current_x + signed_gain * correction`` (paper Eq. 12),
with negative signed gains in its default configurations.

SCSC implements Eqs. 14--17. CCSC implements Eqs. 2--5 and 24 with R1=0,
the residual assumption used in the convergence derivation. CCSC has an
explicit source ambiguity: literal Eq. 22 subtracts the *current batch* mean,
whose resulting adjustment also has zero batch mean. ``centering='batch'``
preserves that equation; default ``centering='reference'`` is the useful
engineering variant seen in the accompanying current MATLAB implementation.
It must not be described as literal Eq. 22.
"""

from __future__ import annotations

import numpy as np


def _matrix(values, name: str, *, min_rows: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1:
        raise ValueError(f"{name} must be a 2-D array with at least {min_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _vector(values, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


class Standardizer:
    """Training-only sample mean/std normalization with safe constant columns.

    A constant column uses scale 1, making its training values exactly zero.
    No statistics are recomputed when transforming operational data.
    """

    def __init__(self, min_scale: float = 1e-12):
        if not np.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be positive and finite")
        self.min_scale = float(min_scale)

    def fit(self, values) -> "Standardizer":
        values = _matrix(values, "values", min_rows=2)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0, ddof=1)
        self.active = self.std > self.min_scale * np.maximum(1.0, np.abs(self.mean))
        self.scale = np.where(self.active, self.std, 1.0)
        return self

    def transform(self, values) -> np.ndarray:
        if not hasattr(self, "mean"):
            raise RuntimeError("fit must be called before transform")
        values = _matrix(values, "values")
        if values.shape[1] != self.mean.size:
            raise ValueError("values have the wrong number of columns")
        return (values - self.mean) / self.scale

    def inverse_transform(self, values) -> np.ndarray:
        if not hasattr(self, "mean"):
            raise RuntimeError("fit must be called before inverse_transform")
        values = _matrix(values, "values")
        if values.shape[1] != self.mean.size:
            raise ValueError("values have the wrong number of columns")
        return values * self.scale + self.mean


class _CorrelationController:
    def _fit_data(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = _matrix(x, "x", min_rows=2)
        y = _matrix(y, "y", min_rows=2)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of observations")
        self.x_scaler = Standardizer().fit(x)
        self.y_scaler = Standardizer().fit(y)
        self.x_mean = self.x_scaler.mean.copy()
        self.y_mean = self.y_scaler.mean.copy()
        self.n_samples, self.n_inputs = x.shape
        self.n_outputs = y.shape[1]
        return self.x_scaler.transform(x), self.y_scaler.transform(y)

    def _operation_data(self, y_batch, reference_y, current_x):
        if not hasattr(self, "n_inputs"):
            raise RuntimeError("fit must be called before correction")
        y_batch = _matrix(y_batch, "y_batch")
        if y_batch.shape[1] != self.n_outputs:
            raise ValueError("y_batch has the wrong number of outputs")
        reference = self.y_mean if reference_y is None else _vector(
            reference_y, "reference_y", self.n_outputs
        )
        if current_x is None:
            x_mean = self.x_mean
        else:
            current_x = np.asarray(current_x, dtype=float)
            if current_x.ndim == 2:
                current_x = _matrix(current_x, "current_x")
                if current_x.shape[1] != self.n_inputs:
                    raise ValueError("current_x has the wrong number of inputs")
                x_mean = current_x.mean(axis=0)
            else:
                x_mean = _vector(current_x, "current_x", self.n_inputs)
        return y_batch, reference, x_mean

    def correction(self, y_batch, reference_y=None, current_x=None) -> np.ndarray:
        """Return the batch-mean X_e in physical input units, without gain.

        ``reference_y`` is a fixed normal vector or an AR prediction. It is
        never learned from ``y_batch``. ``current_x`` may be an input vector
        or an input batch; SCSC uses its mean as prescribed by Eq. 17.
        """
        return self.correction_batch(y_batch, reference_y, current_x).mean(axis=0)

    def diagnostics(self) -> dict:
        if not hasattr(self, "n_inputs"):
            raise RuntimeError("fit must be called before diagnostics")
        return {
            "n_samples": self.n_samples,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "x_mean": self.x_mean.tolist(),
            "y_mean": self.y_mean.tolist(),
            "x_std": self.x_scaler.std.tolist(),
            "y_std": self.y_scaler.std.tolist(),
            "constant_inputs": np.flatnonzero(~self.x_scaler.active).tolist(),
            "constant_outputs": np.flatnonzero(~self.y_scaler.active).tolist(),
        }


class SCSC(_CorrelationController):
    """Simple-correlation self-healing controller, paper Eqs. 14--17.

    ``weights`` are nonnegative output priorities (W), not actuator gains.
    The supplied CSTR example uses [1, 1, 0.05]. ``current_x=None`` retains
    the MATLAB script's normal-training input mean; callers following Eq. 17
    should pass the current input batch/mean explicitly.

    When a reference is zero or nearly zero, Eq. 15 is undefined or ill
    conditioned. Its denominator is guarded at ``reference_floor`` times
    max(1, abs(training output mean)), preserving the reference's sign.
    This numerical extension is reported in diagnostics; it does not clip
    an adjustment or assert safe closed-loop operation.
    """

    def __init__(self, weights=None, reference_floor: float = 1e-8):
        if not np.isfinite(reference_floor) or reference_floor <= 0:
            raise ValueError("reference_floor must be positive and finite")
        self._weights = None if weights is None else np.asarray(weights, dtype=float).copy()
        self.reference_floor = float(reference_floor)

    def fit(self, x, y) -> "SCSC":
        xz, yz = self._fit_data(x, y)
        self.weights = np.ones(self.n_outputs) if self._weights is None else _vector(
            self._weights, "weights", self.n_outputs
        ).copy()
        if (self.weights < 0).any():
            raise ValueError("weights must be nonnegative")
        self.correlation = np.clip(xz.T @ yz / (self.n_samples - 1), -1.0, 1.0)
        self.correlation[~self.x_scaler.active, :] = 0.0
        self.correlation[:, ~self.y_scaler.active] = 0.0
        self.last_guarded_outputs = np.empty(0, dtype=int)
        return self

    def correction_batch(self, y_batch, reference_y=None, current_x=None) -> np.ndarray:
        """Return the repeated q-by-m adjustment of Eqs. 16--17."""
        y_batch, reference, x_mean = self._operation_data(y_batch, reference_y, current_x)
        floor = self.reference_floor * np.maximum(1.0, np.abs(self.y_mean))
        guarded = np.abs(reference) < floor
        denominator = np.where(guarded, np.where(reference < 0, -floor, floor), reference)
        self.last_guarded_outputs = np.flatnonzero(guarded)
        deviation = (y_batch.mean(axis=0) - reference) / denominator
        adjustment = ((deviation * self.weights) @ self.correlation.T) * x_mean
        return np.broadcast_to(adjustment, (len(y_batch), self.n_inputs)).copy()

    def diagnostics(self) -> dict:
        return {
            **super().diagnostics(),
            "method": "SCSC",
            "weights": self.weights.tolist(),
            "correlation": self.correlation.tolist(),
            "reference_floor": self.reference_floor,
            "last_guarded_outputs": self.last_guarded_outputs.tolist(),
        }


def _symmetric_roots(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    threshold = np.finfo(float).eps * max(matrix.shape) * max(1.0, np.max(eigenvalues))
    positive = eigenvalues > threshold
    roots = np.sqrt(np.maximum(eigenvalues, 0.0))
    inverse_roots = np.zeros_like(roots)
    inverse_roots[positive] = 1.0 / roots[positive]
    return (eigenvectors * roots) @ eigenvectors.T, (eigenvectors * inverse_roots) @ eigenvectors.T


class CCSC(_CorrelationController):
    """Canonical-correlation mapping, with an explicit centering choice.

    Sample-normalized data are used to avoid mixing concentration and
    temperature scales. ``ridge * I`` regularizes both covariance matrices
    as suggested in paper Eq. 21. The final adjustment is converted back to
    physical input units. Zero-variance input directions remain inactive.

    ``centering='batch'`` uses literal Eq. 22 and therefore has essentially
    zero mean correction. ``'reference'`` subtracts the supplied/frozen
    normal reference, matching the operational idea in the MATLAB example.
    Equal row repetition (Eq. 23) does not change the mean adjustment and is
    not materialized. R1 is set to zero as in the convergence proof.

    Unlike the current MATLAB file, L uses V, not V.T: this follows Eq. 5
    and NumPy's SVD convention. This mapping is a correlation projection;
    it is not an inverse plant or an identified dynamical controller.
    """

    def __init__(self, ridge: float = 1e-8, centering: str = "reference"):
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be nonnegative and finite")
        if centering not in {"reference", "batch"}:
            raise ValueError("centering must be 'reference' or 'batch'")
        self.ridge = float(ridge)
        self.centering = centering

    def fit(self, x, y) -> "CCSC":
        xz, yz = self._fit_data(x, y)
        self.c_xx = xz.T @ xz / (self.n_samples - 1) + self.ridge * np.eye(self.n_inputs)
        self.c_yy = yz.T @ yz / (self.n_samples - 1) + self.ridge * np.eye(self.n_outputs)
        self.c_xy = xz.T @ yz / (self.n_samples - 1)
        x_root, x_inverse_root = _symmetric_roots(self.c_xx)
        _, y_inverse_root = _symmetric_roots(self.c_yy)
        phi = x_inverse_root @ self.c_xy @ y_inverse_root
        self.u, singular_values, vt = np.linalg.svd(phi, full_matrices=True)
        self.v = vt.T
        self.canonical_correlations = np.clip(singular_values, 0.0, 1.0)
        self.sigma = np.zeros((self.n_inputs, self.n_outputs))
        diagonal = np.arange(len(singular_values))
        self.sigma[diagonal, diagonal] = self.canonical_correlations
        self.j = x_inverse_root @ self.u
        self.l = y_inverse_root @ self.v
        # Eq. 24, row-vector form: Y_e @ L @ Lambda.T @ U.T @ sqrt(Cxx).
        self.mapping = self.l @ self.sigma.T @ self.u.T @ x_root
        self.mapping[:, ~self.x_scaler.active] = 0.0
        self.mapping[~self.y_scaler.active, :] = 0.0
        self.training_residual = self.j.T @ xz.T - self.sigma @ self.l.T @ yz.T
        self.x_rank = int(np.linalg.matrix_rank(xz))
        self.y_rank = int(np.linalg.matrix_rank(yz))
        return self

    def correction_batch(self, y_batch, reference_y=None, current_x=None) -> np.ndarray:
        """Return q-by-m Eq. 24 adjustments, before scalar/batch averaging."""
        y_batch, reference, _ = self._operation_data(y_batch, reference_y, current_x)
        center = y_batch.mean(axis=0) if self.centering == "batch" else reference
        deviation = (y_batch - center) / self.y_scaler.scale
        return (deviation @ self.mapping) * self.x_scaler.scale

    def diagnostics(self) -> dict:
        return {
            **super().diagnostics(),
            "method": "CCSC",
            "centering": self.centering,
            "literal_equation_22": self.centering == "batch",
            "ridge": self.ridge,
            "residual_assumption": "R1=0 in the operational adjustment",
            "canonical_correlations": self.canonical_correlations.tolist(),
            "x_rank": self.x_rank,
            "y_rank": self.y_rank,
            "standardized_mapping": self.mapping.tolist(),
            "training_residual_mean": self.training_residual.mean(axis=1).tolist(),
        }


class ARReference:
    """Frozen per-output AR reference based only on normal batch means.

    Implements the no-intercept linear lag model of Eq. 13 with a finite
    lag order. Rows supplied to ``fit`` must be *batch means*, not arbitrary
    samples. Lags run newest to oldest. Coefficients and history stay frozen
    during ``predict``; abnormal observations are never absorbed. Repeated
    ``predict(steps=1)`` calls intentionally return the same reference. To
    advance a normal forecast, request an explicit horizon in ``steps``.
    """

    def __init__(self, order: int = 2, ridge: float = 1e-8):
        if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 1:
            raise ValueError("order must be a positive integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be nonnegative and finite")
        self.order, self.ridge = int(order), float(ridge)

    def fit(self, normal_batch_means) -> "ARReference":
        values = _matrix(normal_batch_means, "normal_batch_means", min_rows=self.order + 1)
        self.normal_mean = values.mean(axis=0)
        self.history = values[-self.order:].copy()
        self.n_outputs = values.shape[1]
        self.coefficients = np.empty((self.order, self.n_outputs))
        for j in range(self.n_outputs):
            lagged = np.column_stack([
                values[self.order - lag - 1:len(values) - lag - 1, j]
                for lag in range(self.order)
            ])
            target = values[self.order:, j]
            if self.ridge:
                lagged = np.vstack((lagged, np.sqrt(self.ridge) * np.eye(self.order)))
                target = np.concatenate((target, np.zeros(self.order)))
            self.coefficients[:, j] = np.linalg.lstsq(lagged, target, rcond=None)[0]
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if not hasattr(self, "history"):
            raise RuntimeError("fit must be called before predict")
        if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        history = self.history.copy()
        for _ in range(steps):
            prediction = np.sum(history[::-1] * self.coefficients, axis=0)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("AR forecast is nonfinite; use a shorter horizon or constant reference")
            history = np.vstack((history[1:], prediction))
        return prediction.copy()

    def diagnostics(self) -> dict:
        if not hasattr(self, "history"):
            raise RuntimeError("fit must be called before diagnostics")
        return {
            "method": "AR",
            "order": self.order,
            "ridge": self.ridge,
            "normal_mean": self.normal_mean.tolist(),
            "coefficients_newest_lag_first": self.coefficients.tolist(),
            "frozen_normal_history": True,
        }
