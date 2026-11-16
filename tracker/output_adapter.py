"""Read-only 19-field output for the unchanged upstream CasTrack manager."""
from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np

from .tracking_result import InfoDetectedObject, info_from_ekf_state

RESULT_FIELDS = tuple(field.name for field in fields(InfoDetectedObject))


@dataclass(frozen=True)
class ImageBoxEstimate:
    """[center_x, center_y, height, width] in pixels, covariance in pixels²."""
    values: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class TrackingOutput:
    timestamp: int
    objects: tuple[InfoDetectedObject, ...]
    unavailable: dict[int, str]

    def as_dict(self):
        return {'timestamp': self.timestamp, 'objects': [obj.as_dict() for obj in self.objects],
                'unavailable': [{'id': key, 'reason': value} for key, value in self.unavailable.items()]}


def tracking_output(tracker, *, target_label=0, sensor_pose=None, image_box_provider=None,
                    include_predictions=True, strict=True):
    """Read current EKF estimates without changing association or filter state.

    target_label is a detector class ID, a mapping {track_id: class_id}, or a
    callback(track_id, trajectory). It is never inferred from the track ID.
    The default image source is the original optional EKF feature block.
    A provider(track_id, state, covariance, sensor_pose) can instead return an
    ImageBoxEstimate, e.g. from calibrated projection. Missing input raises
    in strict mode; diagnostic mode lists unavailable objects explicitly.
    score is the last detector score, including on prediction-only frames.
    """
    if tracker.current_timestamp is None:
        raise ValueError('Call tracking() before reading output')
    timestamp = tracker.current_timestamp
    objects = []; unavailable = {}
    for track_id, trajectory in sorted(tracker.active_trajectories.items()):
        observation = trajectory.trajectory.get(timestamp)
        if observation is None:
            continue
        updated = observation.updated_state is not None
        if not updated and not include_predictions:
            continue
        state = observation.updated_state if updated else observation.predicted_state
        covariance = observation.updated_covariance if updated else observation.predicted_covariance
        try:
            if state is None or covariance is None:
                raise ValueError('No EKF estimate/covariance for this frame')
            if callable(target_label): label = target_label(track_id, trajectory)
            elif isinstance(target_label, Mapping): label = target_label[track_id]
            else: label = target_label
            score_observation = trajectory.trajectory[trajectory.last_updated_timestamp]
            if score_observation.score is None:
                raise ValueError('Missing detector confidence')
            kwargs = {}
            if image_box_provider is not None:
                estimate = image_box_provider(track_id, state, covariance, sensor_pose)
                kwargs = {'image_box': estimate.values, 'image_box_covariance': estimate.covariance}
            elif not tracker.tracking_features:
                raise ValueError('x/y/h/w and their variances require 2D features or a calibrated box provider')
            offset = 13 + (7 if tracker.bb_as_features else 0)
            obj = info_from_ekf_state(track_id=track_id, target_label=label,
                score=score_observation.score, state=state, covariance=covariance,
                sensor_pose=sensor_pose, image_feature_start=offset, **kwargs)
            objects.append(obj)
        except (ValueError, KeyError) as error:
            if strict:
                raise ValueError(f'Track {track_id}: {error}') from error
            unavailable[int(track_id)] = str(error)
    return TrackingOutput(int(timestamp), tuple(objects), unavailable)


class TrackingOutputAdapter:
    """Original tracking inputs -> list of all 19-field object results.

    tracker is an existing Tracker3D. For 2D detector boxes, construct it with
    tracking_features=True and pass [center_x, center_y, height, width] as its
    first four features. For LiDAR-only state, supply image_box_provider.
    """
    def __init__(self, tracker, *, target_label=0, image_box_provider=None):
        self.tracker = tracker
        self.target_label = target_label
        self.image_box_provider = image_box_provider
        self.last_output = None

    def tracking(self, bbs_3D=None, features=None, scores=None, pose=None, timestamp=None):
        self.tracker.tracking(bbs_3D=bbs_3D, features=features, scores=scores,
                              pose=pose, timestamp=timestamp)
        self.last_output = tracking_output(self.tracker, target_label=self.target_label,
            image_box_provider=self.image_box_provider, sensor_pose=pose)
        return list(self.last_output.objects)


class CalibratedBoxProjection:
    """Project a 3D EKF box; propagate its covariance into pixel box variance.

    Bounds are not clipped to the image. Outside-image coordinates may be
    negative. Boxes reaching behind the camera cannot produce valid pixel
    measurements and are rejected. Camera/ego calibration uncertainty is
    not included. The Jacobian uses finite differences of the box extrema.
    """
    def __init__(self, projection, lidar_to_camera, rectification=None):
        self.projection = np.asarray(projection, dtype=float).reshape(3, 4)
        self.lidar_to_camera = np.asarray(lidar_to_camera, dtype=float).reshape(3, 4)
        self.rectification = np.eye(3) if rectification is None else np.asarray(rectification, dtype=float).reshape(3, 3)
        if not all(np.all(np.isfinite(value)) for value in [self.projection, self.lidar_to_camera, self.rectification]):
            raise ValueError('Calibration must be finite')

    @classmethod
    def from_file(cls, path):
        matrices = {}
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            key, value = line.split(':', 1)
            matrices[key] = np.fromstring(value, sep=' ')
        return cls(matrices['P2'], matrices['Tr_velo_to_cam'], matrices.get('R0_rect'))

    def _box(self, state, inverse_pose):
        dimensions = state[9:12]
        if np.any(dimensions <= 0): raise ValueError('Non-positive 3D box dimensions')
        signs = np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                            [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
        yaw = state[12];cosine = np.cos(yaw);sine = np.sin(yaw)
        rotation = np.array([[cosine,-sine,0],[sine,cosine,0],[0,0,1]])
        world = (signs*dimensions/2)@rotation.T + state[:3]
        sensor = (inverse_pose@np.c_[world,np.ones(8)].T).T
        camera = (self.rectification@self.lidar_to_camera@sensor.T).T
        if np.any(camera[:,2] <= .01): raise ValueError('3D box is behind/intersects the front camera; pixel box unavailable')
        projected = (self.projection@np.c_[camera,np.ones(8)].T).T
        if np.any(projected[:,2] <= .01): raise ValueError('Invalid camera projection depth')
        pixels = projected[:,:2]/projected[:,2:3]
        lo = pixels.min(axis=0);hi = pixels.max(axis=0)
        return np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, hi[1]-lo[1], hi[0]-lo[0]])

    def __call__(self, track_id, state, covariance, sensor_pose=None):
        vector = np.asarray(state,dtype=float).reshape(-1)
        matrix = np.asarray(covariance,dtype=float)
        if len(vector)<13 or matrix.shape!=(len(vector),len(vector)):
            raise ValueError('Invalid EKF state/covariance dimensions')
        if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
            raise ValueError('Non-finite EKF state/covariance')
        inverse_pose = np.eye(4) if sensor_pose is None else np.linalg.inv(np.asarray(sensor_pose,dtype=float).reshape(4,4))
        values = self._box(vector, inverse_pose)
        jacobian = np.zeros((4,len(vector)))
        for column in [0,1,2,9,10,11,12]:
            step = 1e-5*max(1.0,abs(vector[column]))
            if column in [9,10,11]: step=min(step,vector[column]*.1)
            left=vector.copy();right=vector.copy();left[column]-=step;right[column]+=step
            jacobian[:,column]=(self._box(right,inverse_pose)-self._box(left,inverse_pose))/(2*step)
        box_covariance=jacobian@((matrix+matrix.T)/2)@jacobian.T
        return ImageBoxEstimate(values,(box_covariance+box_covariance.T)/2)
