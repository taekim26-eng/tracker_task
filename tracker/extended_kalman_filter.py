"""Extended Kalman filter used by CasTrack vessel trajectories.

The public CasTrack state layout is intentionally preserved::

    [x, y, z, u, v_lateral, v_z, a, a_z, yaw_rate,
     length, width, height, yaw, optional_features...]

Only indices 0:3 and 9:13 are consumed outside the tracker package.  The
kinematic slots 3:9 are therefore used for a vessel-oriented CTRV
(constant turn rate and velocity) model with Fossen 3-DOF kinematics while
keeping the KITTI writer and association code compatible.  The acceleration
slots a and a_z are kept only for that layout: they stay zero with zero
variance, so speed changes are absorbed by the velocity process noise.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

import numpy as np


DEFAULT_VESSEL_EKF_CONFIG = {
    # 10 Hz maritime profile for LiDAR detections of vessels roughly 15 m or
    # longer. Position/size measurements are deliberately less confident
    # because a partially observed hull can move a fitted box by metres.
    "heading_period": 2.0 * math.pi,
    "stationary_speed_threshold": 0.5,
    "bootstrap_stationary_sigma_multiplier": 3.0,
    "bootstrap_from_two_measurements": False,
    "bootstrap_yaw_rate": False,
    "max_abs_yaw_rate": 0.15,
    "measurement_position_variance": 4.0,
    "measurement_size_variance": 2.25,
    "measurement_heading_variance": 0.03,
    "measurement_feature_variance": 4.0,
    "initial_position_variance": 4.0,
    "initial_velocity_variance": 25.0,
    "initial_yaw_rate_variance": 0.04,
    "initial_size_variance": 4.0,
    "initial_heading_variance": 0.12,
    "initial_feature_variance": 16.0,
    "process_position_variance": 0.1,
    "process_velocity_variance": 0.05,
    "process_yaw_rate_variance": 0.02,
    "process_size_variance": 0.02,
    "process_heading_variance": 0.01,
    "process_feature_variance": 2.0,
}


def vessel_ekf_config(overrides: Optional[Mapping] = None) -> Mapping:
    """Return validated-model defaults with optional YAML overrides."""

    config = dict(DEFAULT_VESSEL_EKF_CONFIG)
    if overrides:
        config.update(dict(overrides))
    return config


class VesselCTRVExtendedKalmanFilter:
    """Constant-turn-rate-and-velocity EKF with Fossen 3-DOF kinematics.

    Position advances with the body-frame velocities rotated by the yaw at
    the start of the step; yaw advances by yaw_rate * dt.  The model supports
    zero speed and zero turn rate without a special division branch.
    Forward and lateral body-frame velocities are tracked separately so slow
    drift of a stationary or manoeuvring vessel can be represented.
    """

    X = 0
    Y = 1
    Z = 2
    FORWARD_SPEED = 3
    LATERAL_SPEED = 4
    VERTICAL_SPEED = 5
    FORWARD_ACCELERATION = 6       # layout placeholder, always zero (CTRV)
    VERTICAL_ACCELERATION = 7      # layout placeholder, always zero (CTRV)
    YAW_RATE = 8
    LENGTH = 9
    WIDTH = 10
    HEIGHT = 11
    YAW = 12
    FEATURE_START = 13

    MEASUREMENT_YAW = 6
    BASE_STATE_DIM = 13
    BASE_MEASUREMENT_DIM = 7

    def __init__(
        self,
        state_dim: int,
        scanning_interval: float,
        config: Optional[Mapping] = None,
    ) -> None:
        if state_dim < self.BASE_STATE_DIM:
            raise ValueError(
                f"EKF state_dim must be at least {self.BASE_STATE_DIM}, got {state_dim}"
            )
        if scanning_interval <= 0:
            raise ValueError("scanning_interval must be greater than zero")

        self.state_dim = state_dim
        self.feature_dim = state_dim - self.BASE_STATE_DIM
        self.measurement_dim = self.BASE_MEASUREMENT_DIM + self.feature_dim
        self.scanning_interval = float(scanning_interval)
        self.config = vessel_ekf_config(config)

        self.heading_period = self._read_float("heading_period", 2.0 * math.pi)
        if self.heading_period <= 0:
            raise ValueError("ekf.heading_period must be greater than zero")
        self.stationary_speed_threshold = self._read_float(
            "stationary_speed_threshold", 0.5
        )
        if self.stationary_speed_threshold < 0:
            raise ValueError("ekf.stationary_speed_threshold cannot be negative")
        self.bootstrap_stationary_sigma_multiplier = self._read_float(
            "bootstrap_stationary_sigma_multiplier", 3.0
        )
        if self.bootstrap_stationary_sigma_multiplier < 0:
            raise ValueError(
                "ekf.bootstrap_stationary_sigma_multiplier cannot be negative"
            )
        self.bootstrap_yaw_rate = bool(
            self.config.get("bootstrap_yaw_rate", False)
        )
        self.max_abs_yaw_rate = self._read_float("max_abs_yaw_rate", 0.15)
        if self.max_abs_yaw_rate <= 0:
            raise ValueError("ekf.max_abs_yaw_rate must be greater than zero")

        self.measurement_matrix = self._build_measurement_matrix()
        self.measurement_covariance = self._build_measurement_covariance()

    def _read_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"ekf.{key} must be finite")
        return value

    def _build_measurement_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.measurement_dim, self.state_dim), dtype=float)
        matrix[0:3, 0:3] = np.eye(3)
        matrix[3:6, self.LENGTH : self.HEIGHT + 1] = np.eye(3)
        matrix[self.MEASUREMENT_YAW, self.YAW] = 1.0
        if self.feature_dim:
            matrix[
                self.BASE_MEASUREMENT_DIM :, self.FEATURE_START :
            ] = np.eye(self.feature_dim)
        return matrix

    def _build_measurement_covariance(self) -> np.ndarray:
        position = self._read_float("measurement_position_variance", 4.0)
        size = self._read_float("measurement_size_variance", 2.25)
        heading = self._read_float("measurement_heading_variance", 0.03)
        feature = self._read_float("measurement_feature_variance", 4.0)
        values = [position] * 3 + [size] * 3 + [heading]
        values.extend([feature] * self.feature_dim)
        if any(value <= 0 for value in values):
            raise ValueError("all EKF measurement variances must be greater than zero")
        return np.diag(values)

    def initial_covariance(self) -> np.ndarray:
        position = self._read_float("initial_position_variance", 4.0)
        velocity = self._read_float("initial_velocity_variance", 25.0)
        acceleration = 0.0            # CTRV: acceleration slots are not estimated
        yaw_rate = self._read_float("initial_yaw_rate_variance", 0.04)
        size = self._read_float("initial_size_variance", 4.0)
        heading = self._read_float("initial_heading_variance", 0.12)
        feature = self._read_float("initial_feature_variance", 16.0)
        values = (
            [position] * 3
            + [velocity] * 3
            + [acceleration] * 2
            + [yaw_rate]
            + [size] * 3
            + [heading]
            + [feature] * self.feature_dim
        )
        if any(value < 0 for value in values):
            raise ValueError("EKF initial variances cannot be negative")
        return np.diag(values)

    def process_covariance(self, dt: Optional[float] = None) -> np.ndarray:
        dt = self._validate_dt(dt)
        position = self._read_float("process_position_variance", 0.1)
        velocity = self._read_float("process_velocity_variance", 0.05)
        acceleration = 0.0            # CTRV: acceleration slots are not estimated
        yaw_rate = self._read_float("process_yaw_rate_variance", 0.02)
        size = self._read_float("process_size_variance", 0.02)
        heading = self._read_float("process_heading_variance", 0.01)
        feature = self._read_float("process_feature_variance", 2.0)
        values = (
            [position] * 3
            + [velocity] * 3
            + [acceleration] * 2
            + [yaw_rate]
            + [size] * 3
            + [heading]
            + [feature] * self.feature_dim
        )
        if any(value < 0 for value in values):
            raise ValueError("EKF process variances cannot be negative")
        return np.diag(values) * dt

    def initialize_state(self, measurement: np.ndarray) -> np.ndarray:
        measurement = self._measurement_vector(measurement)
        state = np.zeros(self.state_dim, dtype=float)
        state[0:3] = measurement[0:3]
        state[self.LENGTH : self.HEIGHT + 1] = measurement[3:6]
        state[self.YAW] = self.wrap_angle(measurement[self.MEASUREMENT_YAW])
        if self.feature_dim:
            state[self.FEATURE_START :] = measurement[
                self.BASE_MEASUREMENT_DIM :
            ]
        return state

    def transition_function(
        self, state: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        dt = self._validate_dt(dt)
        state = self._state_vector(state)
        predicted = state.copy()

        forward_distance = state[self.FORWARD_SPEED] * dt
        lateral_distance = state[self.LATERAL_SPEED] * dt
        cosine = math.cos(state[self.YAW])
        sine = math.sin(state[self.YAW])

        predicted[self.X] += forward_distance * cosine - lateral_distance * sine
        predicted[self.Y] += forward_distance * sine + lateral_distance * cosine
        predicted[self.Z] += state[self.VERTICAL_SPEED] * dt
        predicted[self.YAW] = self.wrap_angle(
            state[self.YAW] + state[self.YAW_RATE] * dt
        )
        return predicted

    def transition_jacobian(
        self, state: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        dt = self._validate_dt(dt)
        state = self._state_vector(state)
        jacobian = np.eye(self.state_dim, dtype=float)

        forward_distance = state[self.FORWARD_SPEED] * dt
        lateral_distance = state[self.LATERAL_SPEED] * dt
        cosine = math.cos(state[self.YAW])
        sine = math.sin(state[self.YAW])

        dx_dyaw = -forward_distance * sine - lateral_distance * cosine
        dy_dyaw = forward_distance * cosine - lateral_distance * sine

        jacobian[self.X, self.FORWARD_SPEED] = dt * cosine
        jacobian[self.X, self.LATERAL_SPEED] = -dt * sine
        jacobian[self.X, self.YAW] = dx_dyaw

        jacobian[self.Y, self.FORWARD_SPEED] = dt * sine
        jacobian[self.Y, self.LATERAL_SPEED] = dt * cosine
        jacobian[self.Y, self.YAW] = dy_dyaw

        jacobian[self.Z, self.VERTICAL_SPEED] = dt
        jacobian[self.YAW, self.YAW_RATE] = dt
        return jacobian

    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._validate_dt(dt)
        state = self._state_vector(state)
        covariance = self._covariance_matrix(covariance)
        jacobian = self.transition_jacobian(state, dt)
        predicted_state = self.transition_function(state, dt)
        predicted_covariance = (
            jacobian @ covariance @ jacobian.T + self.process_covariance(dt)
        )
        return predicted_state, self._symmetrize(predicted_covariance)

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        predicted_state = self._state_vector(predicted_state)
        predicted_covariance = self._covariance_matrix(predicted_covariance)
        measurement = self._measurement_vector(measurement)

        expected_measurement = self.measurement_matrix @ predicted_state
        innovation = measurement - expected_measurement
        innovation[self.MEASUREMENT_YAW] = self.heading_residual(
            measurement[self.MEASUREMENT_YAW],
            expected_measurement[self.MEASUREMENT_YAW],
        )

        innovation_covariance = (
            self.measurement_matrix
            @ predicted_covariance
            @ self.measurement_matrix.T
            + self.measurement_covariance
        )
        cross_covariance = predicted_covariance @ self.measurement_matrix.T
        try:
            kalman_gain = np.linalg.solve(
                innovation_covariance, cross_covariance.T
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = cross_covariance @ np.linalg.pinv(
                innovation_covariance
            )

        updated_state = predicted_state + kalman_gain @ innovation
        updated_state[self.YAW] = self.wrap_angle(updated_state[self.YAW])
        updated_state[self.YAW_RATE] = np.clip(
            updated_state[self.YAW_RATE],
            -self.max_abs_yaw_rate,
            self.max_abs_yaw_rate,
        )

        identity = np.eye(self.state_dim)
        residual_matrix = identity - kalman_gain @ self.measurement_matrix
        updated_covariance = (
            residual_matrix @ predicted_covariance @ residual_matrix.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )
        return updated_state, self._symmetrize(updated_covariance)

    def bootstrap_state(
        self,
        previous_state: np.ndarray,
        measurement: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a statistically consistent state from two detections.

        Measured position, size and heading are placed at the current
        detection exactly. Kinematics are bootstrapped only when displacement
        exceeds both the physical and uncertainty-derived thresholds. Yaw
        rate stays zero by default because 3D boxes often contain pi flips.
        """

        dt = self._validate_dt(dt)
        previous_state = self._state_vector(previous_state)
        measurement = self._measurement_vector(measurement)
        state = self.initialize_state(measurement)
        covariance = self.initial_covariance()

        dx = measurement[0] - previous_state[self.X]
        dy = measurement[1] - previous_state[self.Y]
        dz = measurement[2] - previous_state[self.Z]
        heading = self.wrap_angle(
            previous_state[self.YAW]
            + self.heading_residual(
                measurement[self.MEASUREMENT_YAW], previous_state[self.YAW]
            )
        )
        cosine = math.cos(heading)
        sine = math.sin(heading)
        planar_speed = math.hypot(dx, dy) / dt
        position_variance = self.measurement_covariance[0, 0]
        uncertainty_threshold = (
            self.bootstrap_stationary_sigma_multiplier
            * math.sqrt(2.0 * position_variance)
            / dt
        )
        stationary_threshold = max(
            self.stationary_speed_threshold, uncertainty_threshold
        )

        if planar_speed <= stationary_threshold:
            state[self.FORWARD_SPEED] = 0.0
            state[self.LATERAL_SPEED] = 0.0
            state[self.VERTICAL_SPEED] = 0.0
            state[self.YAW_RATE] = 0.0
        else:
            state[self.FORWARD_SPEED] = (dx * cosine + dy * sine) / dt
            state[self.LATERAL_SPEED] = (-dx * sine + dy * cosine) / dt
            state[self.VERTICAL_SPEED] = dz / dt
            if self.bootstrap_yaw_rate:
                state[self.YAW_RATE] = np.clip(
                    self.heading_residual(
                        heading, previous_state[self.YAW]
                    ) / dt,
                    -self.max_abs_yaw_rate,
                    self.max_abs_yaw_rate,
                )
            else:
                state[self.YAW_RATE] = 0.0

        state[self.FORWARD_ACCELERATION] = 0.0
        state[self.VERTICAL_ACCELERATION] = 0.0
        state[self.YAW] = heading
        return state, covariance

    def heading_residual(
        self, measured_heading: float, reference_heading: float
    ) -> float:
        """Return the shortest heading innovation (wrapped to heading_period)."""

        return self._wrap_period(
            float(measured_heading) - float(reference_heading), self.heading_period
        )

    def wrap_angle(self, angle: float) -> float:
        return self._wrap_period(angle, self.heading_period)

    @staticmethod
    def _wrap_period(angle: float, period: float) -> float:
        half_period = 0.5 * period
        return (float(angle) + half_period) % period - half_period

    def _validate_dt(self, dt: Optional[float]) -> float:
        value = self.scanning_interval if dt is None else float(dt)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("EKF dt must be finite and greater than zero")
        return value

    def _state_vector(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=float).reshape(-1)
        if vector.shape != (self.state_dim,):
            raise ValueError(
                f"expected state shape ({self.state_dim},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("EKF state contains a non-finite value")
        return vector

    def _measurement_vector(self, measurement: np.ndarray) -> np.ndarray:
        vector = np.asarray(measurement, dtype=float).reshape(-1)
        if vector.shape != (self.measurement_dim,):
            raise ValueError(
                f"expected measurement shape ({self.measurement_dim},), "
                f"got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("EKF measurement contains a non-finite value")
        return vector

    def _covariance_matrix(self, covariance: np.ndarray) -> np.ndarray:
        matrix = np.asarray(covariance, dtype=float)
        expected = (self.state_dim, self.state_dim)
        if matrix.shape != expected:
            raise ValueError(
                f"expected covariance shape {expected}, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("EKF covariance contains a non-finite value")
        return matrix

    @staticmethod
    def _symmetrize(covariance: np.ndarray) -> np.ndarray:
        return 0.5 * (covariance + covariance.T)
