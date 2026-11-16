"""Typed CasTrack output matching ``InfoDetectedObject.msg``.

The EKF stores Cartesian position, body-frame velocity, box geometry, yaw,
and optional features.  For the message output the first four optional
features are reserved for an image box in the following order::

    [center_x_px, center_y_px, height_px, width_px]

The covariance fields are variances (unit squared), not standard deviations.
Range and bearing are horizontal, relative to the supplied sensor pose.
Speed is the horizontal speed magnitude in the tracker frame; its meaning
depends on ego-motion compensation. Heading is the EKF yaw in that frame.
The sensor pose is treated as deterministic; its uncertainty is not included
in the returned variances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping, Optional

import numpy as np

from .extended_kalman_filter import VesselCTRVExtendedKalmanFilter

INT64_FIELDS = frozenset(('id', 'target_label', 'x', 'y', 'h', 'w'))


@dataclass(frozen=True)
class InfoDetectedObject:
    """Python representation of every field in ``InfoDetectedObject.msg``."""

    id: int
    target_label: int
    score: float
    x: int
    y: int
    h: int
    w: int
    range: float
    theta: float
    speed: float
    heading: float
    x_covariances: float
    y_covariances: float
    h_covariances: float
    w_covariances: float
    range_covariances: float
    theta_covariances: float
    speed_covariances: float
    heading_covariances: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in INT64_FIELDS:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                    raise ValueError(f'{field.name} must be an int64 integer')
                value = int(value)
                if not -(2**63) <= value < 2**63:
                    raise ValueError(f'{field.name} is outside int64 range')
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'{field.name} must be finite')
                if field.name.endswith('_covariances') and value < 0:
                    raise ValueError(f'{field.name} must be a non-negative variance')
            object.__setattr__(self, field.name, value)
        if self.h < 0 or self.w < 0 or self.range < 0 or self.speed < 0:
            raise ValueError('h, w, range and speed must be non-negative')

    def as_dict(self) -> Mapping[str, Any]:
        """Return a serialization-ready dictionary with message field names."""

        return asdict(self)

    def populate_ros_message(self, message: Any) -> Any:
        """Copy this result into an instantiated ROS message and return it."""

        for field in fields(self):
            setattr(message, field.name, getattr(self, field.name))
        return message


def _as_vector(value: np.ndarray, minimum_size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size < minimum_size:
        raise ValueError(
            f"{name} must contain at least {minimum_size} values, got {vector.size}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _as_covariance(value: np.ndarray, state_size: int) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (state_size, state_size):
        raise ValueError(
            "covariance must have shape "
            f"({state_size}, {state_size}), got {covariance.shape}"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance contains a non-finite value")
    return 0.5 * (covariance + covariance.T)


def _inverse_pose(sensor_pose: Optional[np.ndarray]) -> np.ndarray:
    if sensor_pose is None:
        return np.eye(4, dtype=float)
    pose = np.asarray(sensor_pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"sensor_pose must have shape (4, 4), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("sensor_pose contains a non-finite value")
    try:
        return np.linalg.inv(pose)
    except np.linalg.LinAlgError as error:
        raise ValueError("sensor_pose must be invertible") from error


def _quadratic_variance(jacobian: np.ndarray, covariance: np.ndarray) -> float:
    value = float((jacobian @ covariance @ jacobian.T).item())
    # Tiny negative values can be introduced by floating-point roundoff.
    return max(value, 0.0)


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def info_from_ekf_state(
    *,
    track_id: int,
    target_label: int,
    score: float,
    state: np.ndarray,
    covariance: np.ndarray,
    sensor_pose: Optional[np.ndarray] = None,
    image_feature_start: int = VesselCTRVExtendedKalmanFilter.FEATURE_START,
    image_box: Optional[np.ndarray] = None,
    image_box_covariance: Optional[np.ndarray] = None,
    epsilon: float = 1e-9,
) -> InfoDetectedObject:
    """Convert one EKF posterior into all ``InfoDetectedObject`` fields.

    ``sensor_pose`` must be the same local-to-tracker transform used when the
    detection was registered by :func:`tracker.box_op.register_bbs`.
    ``theta`` is counter-clockwise from the sensor +x axis in radians.
    ``heading`` is counter-clockwise from the tracker +x axis in radians.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    required_size = image_feature_start + 4 if image_box is None else 13
    state_vector = _as_vector(state, required_size, "state")
    state_covariance = _as_covariance(covariance, state_vector.size)

    inverse_pose = _inverse_pose(sensor_pose)
    homogeneous_position = np.ones(4, dtype=float)
    homogeneous_position[:3] = state_vector[0:3]
    local_position = inverse_pose @ homogeneous_position
    local_x, local_y = float(local_position[0]), float(local_position[1])

    local_position_jacobian = np.zeros((2, state_vector.size), dtype=float)
    local_position_jacobian[:, 0:3] = inverse_pose[0:2, 0:3]
    local_position_covariance = (
        local_position_jacobian
        @ state_covariance
        @ local_position_jacobian.T
    )

    range_squared = local_x * local_x + local_y * local_y
    range_value = math.sqrt(range_squared)
    if range_value > epsilon:
        range_jacobian = np.array(
            [[local_x / range_value, local_y / range_value]], dtype=float
        )
        theta_jacobian = np.array(
            [[-local_y / range_squared, local_x / range_squared]], dtype=float
        )
        range_variance = _quadratic_variance(
            range_jacobian, local_position_covariance
        )
        theta_variance = _quadratic_variance(
            theta_jacobian, local_position_covariance
        )
        theta_value = math.atan2(local_y, local_x)
    else:
        # Bearing is undefined at the sensor origin.  Keep the numeric output
        # finite while reporting maximal angular uncertainty.
        range_variance = max(float(np.trace(local_position_covariance)), 0.0)
        theta_variance = math.pi * math.pi
        theta_value = 0.0

    forward_speed = float(
        state_vector[VesselCTRVExtendedKalmanFilter.FORWARD_SPEED]
    )
    lateral_speed = float(
        state_vector[VesselCTRVExtendedKalmanFilter.LATERAL_SPEED]
    )
    speed_value = math.hypot(forward_speed, lateral_speed)
    velocity_indices = [
        VesselCTRVExtendedKalmanFilter.FORWARD_SPEED,
        VesselCTRVExtendedKalmanFilter.LATERAL_SPEED,
    ]
    velocity_covariance = state_covariance[np.ix_(velocity_indices, velocity_indices)]
    if speed_value > epsilon:
        speed_jacobian = np.array(
            [[forward_speed / speed_value, lateral_speed / speed_value]],
            dtype=float,
        )
        speed_variance = _quadratic_variance(
            speed_jacobian, velocity_covariance
        )
    else:
        speed_variance = max(float(np.trace(velocity_covariance)), 0.0)

    if image_box is None:
        if image_box_covariance is not None:
            raise ValueError('image_box_covariance requires an explicit image_box')
        image_box = state_vector[image_feature_start : image_feature_start + 4]
        image_variances = np.maximum(
            np.diag(state_covariance)[image_feature_start : image_feature_start + 4], 0.0
        )
    else:
        image_box = np.asarray(image_box, dtype=float)
        if image_box.shape != (4,) or not np.all(np.isfinite(image_box)):
            raise ValueError('image_box must contain finite [center_x, center_y, height, width]')
        if image_box_covariance is None:
            raise ValueError('Explicit image_box requires its 4x4 covariance; missing uncertainty is not zero')
        box_covariance = _as_covariance(image_box_covariance, 4)
        if np.linalg.eigvalsh(box_covariance).min() < -1e-8:
            raise ValueError('image_box_covariance must be positive semidefinite')
        image_variances = np.maximum(np.diag(box_covariance), 0.0)
    center_x, center_y, height, width = image_box
    if height < 0 or width < 0:
        raise ValueError('image box height and width must be non-negative')

    heading_index = VesselCTRVExtendedKalmanFilter.YAW
    heading_value = _wrap_angle(state_vector[heading_index])
    heading_variance = max(float(state_covariance[heading_index, heading_index]), 0.0)

    result = InfoDetectedObject(
        id=int(track_id),
        target_label=int(target_label),
        score=float(score),
        x=int(np.rint(center_x)),
        y=int(np.rint(center_y)),
        h=max(int(np.rint(height)), 0),
        w=max(int(np.rint(width)), 0),
        range=float(range_value),
        theta=float(theta_value),
        speed=float(speed_value),
        heading=float(heading_value),
        x_covariances=float(image_variances[0]),
        y_covariances=float(image_variances[1]),
        h_covariances=float(image_variances[2]),
        w_covariances=float(image_variances[3]),
        range_covariances=float(range_variance),
        theta_covariances=float(theta_variance),
        speed_covariances=float(speed_variance),
        heading_covariances=float(heading_variance),
    )

    numeric_values = np.asarray(
        [
            result.score,
            result.range,
            result.theta,
            result.speed,
            result.heading,
            result.x_covariances,
            result.y_covariances,
            result.h_covariances,
            result.w_covariances,
            result.range_covariances,
            result.theta_covariances,
            result.speed_covariances,
            result.heading_covariances,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("tracking result contains a non-finite value")
    return result
