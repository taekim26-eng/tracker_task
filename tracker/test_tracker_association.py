"""Regression cases for the maritime association and the vessel EKF in CasTrack."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from .extended_kalman_filter import VesselCTRVExtendedKalmanFilter
from .tracker import Tracker3D, _finite_global_assignment, _greedy_assignment


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def config(**overrides):
    result = yaml.safe_load((Path(__file__).parent/'config'/'multi_object_tracker.yaml').read_text())
    result.update(overrides)
    return AttrDict(result)


def box(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 20.0, 5.0, 6.0, 0.0]


MEASUREMENT_NOISE = np.diag([4.0] * 3 + [2.25] * 3 + [0.03])    # config ekf.measurement_*_variance


def association_tracker(detections, track_positions, scores=None, prediction_scores=None,
                        metric='euclidean', position_variances=None, matching='greedy'):
    """Tracker with stub trajectories; position_variances sets each track's predicted XY variance."""
    tracker = Tracker3D(box_type='OpenPCDet',
                        config=config(association_metric=metric, association_matching=matching))
    tracker.current_timestamp = 1
    tracker.current_bbs = np.asarray(detections, dtype=float)
    tracker.current_scores = np.ones(len(detections)) if scores is None else np.asarray(scores)
    tracker.label_seed = len(track_positions)
    for track_id, xyz in enumerate(track_positions):
        state = np.zeros((13, 1));state[:3, 0] = xyz
        prediction_score = 1.0 if prediction_scores is None else prediction_scores[track_id]
        covariance = np.eye(13) * (1.0 if position_variances is None else position_variances[track_id])
        observation = SimpleNamespace(predicted_state=state, prediction_score=prediction_score,
                                      predicted_covariance=covariance)
        tracker.active_trajectories[track_id] = SimpleNamespace(
            trajectory={1: observation}, ekf=SimpleNamespace(measurement_covariance=MEASUREMENT_NOISE))
    return tracker


class MaritimeAssociationTest(unittest.TestCase):
    def test_default_matching_is_greedy(self):
        self.assertEqual(config().association_matching, 'greedy')
        self.assertEqual(Tracker3D(box_type='OpenPCDet', config=config()).association_matching, 'greedy')

    def test_greedy_fixes_smallest_cost_first_not_detection_order(self):
        # Detection 1 m (1 to track 0, 2 to track 1) and 0.2 m (0.2 to track 0). Upstream greedy in
        # input order gives 1 m -> track 0 when it comes first; the smallest pair 0.2 m -> track 0 wins.
        positions = [(0, 0, 0), (3, 0, 0)]
        forward = association_tracker([box(1), box(.2)], positions)
        reversed_order = association_tracker([box(.2), box(1)], positions)
        self.assertEqual(forward.association(), [1, 0])
        self.assertEqual(reversed_order.association(), [0, 1])

    def test_greedy_keeps_most_certain_pair_instead_of_forcing_more_matches(self):
        # 4 m -> track 0 (cost 4) is fixed first; -5 m is 14 m from track 1 (gated) and starts a new ID.
        greedy = association_tracker([box(4), box(-5)], [(0, 0, 0), (9, 0, 0)])
        hungarian = association_tracker([box(4), box(-5)], [(0, 0, 0), (9, 0, 0)], matching='hungarian')
        self.assertEqual(greedy.association(), [0, 2])
        self.assertEqual(hungarian.association(), [1, 0])

    def test_hungarian_option_is_global_not_detection_order(self):
        positions = [(0, 0, 0), (2, 0, 0)]
        forward = association_tracker([box(.9), box(-1)], positions, matching='hungarian')
        reversed_order = association_tracker([box(-1), box(.9)], positions, matching='hungarian')
        # The global minimum is .9->1, -1->0 (greedy fixes .9->0 first).
        self.assertEqual(forward.association(), [1, 0])
        self.assertEqual(reversed_order.association(), [0, 1])

    def test_unknown_matching_is_rejected(self):
        with self.assertRaises(ValueError):
            Tracker3D(box_type='OpenPCDet', config=config(association_matching='auction'))

    def test_cost_ignores_z_distance(self):
        tracker = association_tracker([box(z=3)], [(0, 0, 0)])
        cost, ids = tracker.compute_cost_map()
        np.testing.assert_allclose(cost, [[0.0]])
        self.assertEqual(ids, [0])
        self.assertEqual(tracker.association(), [0])

    def test_prediction_score_does_not_weight_distance(self):
        tracker = association_tracker([box(.2)], [(0, 0, 0), (1, 0, 0)], prediction_scores=[1.0, .01])
        cost, _ = tracker.compute_cost_map()
        np.testing.assert_allclose(cost, [[.2, .8]])
        self.assertEqual(tracker.association(), [0])

    def test_gate_is_ten_metres_inclusive(self):
        self.assertEqual(config().association_distance_threshold, 10.0)
        self.assertEqual(association_tracker([box(10.0)], [(0, 0, 0)]).association(), [0])
        tracker = association_tracker([box(10.1)], [(0, 0, 0)])
        self.assertTrue(np.isinf(tracker.compute_cost_map()[0][0, 0]))
        self.assertEqual(tracker.association(), [1])

    def test_low_score_detection_cannot_reserve_an_id(self):
        tracker = association_tracker([box(.1), box(.2)], [(0, 0, 0)], scores=[.05, .9])
        self.assertEqual(tracker.association(), [-1, 0])
        self.assertEqual(tracker.label_seed, 1)

    def test_new_track_needs_init_score_and_consumes_no_id_otherwise(self):
        tracker = Tracker3D(box_type='OpenPCDet', config=config())
        boxes, ids = tracker.tracking(np.asarray([box()]), scores=np.asarray([.2]), timestamp=0)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(ids), 0)
        self.assertEqual(tracker.label_seed, 0)
        self.assertFalse(tracker.active_trajectories)
        _, ids = tracker.tracking(np.asarray([box()]), scores=np.asarray([.9]), timestamp=1)
        np.testing.assert_array_equal(ids, [0])

    def test_finite_assignment_handles_rectangular_gated_cost(self):
        cost = np.array([[np.inf, 3.0, np.inf], [1.0, np.inf, np.inf]])
        self.assertEqual(sorted(_finite_global_assignment(cost)), [(0, 1), (1, 0)])
        self.assertEqual(_finite_global_assignment(np.full((2, 2), np.inf)), [])

    def test_greedy_assignment_handles_rectangular_gated_cost(self):
        cost = np.array([[np.inf, 3.0, np.inf], [1.0, 2.0, np.inf]])
        self.assertEqual(_greedy_assignment(cost), [(1, 0), (0, 1)])
        self.assertEqual(_greedy_assignment(np.array([[1.0, 2.0], [1.5, 3.0]])), [(0, 0), (1, 1)])
        self.assertEqual(_greedy_assignment(np.full((2, 2), np.inf)), [])
        self.assertEqual(_greedy_assignment(np.empty((0, 3))), [])


class MahalanobisAssociationTest(unittest.TestCase):
    def test_default_metric_is_mahalanobis_with_chi_square_gate(self):
        self.assertEqual(config().association_metric, 'mahalanobis')
        self.assertEqual(config().association_gate_chi2, 9.21)
        self.assertEqual(Tracker3D(box_type='OpenPCDet', config=config()).association_metric, 'mahalanobis')

    def test_cost_is_squared_distance_over_innovation_covariance(self):
        # S = P_xy + R_xy = 5 + 4 = 9 on each axis; (3, 0) and (0, 6) away from the prediction.
        tracker = association_tracker([box(3), box(0, 6)], [(0, 0, 0)], metric='mahalanobis',
                                      position_variances=[5.0])
        cost, _ = tracker.compute_cost_map()
        np.testing.assert_allclose(cost, [[1.0], [4.0]])

    def test_cost_ignores_z_distance(self):
        tracker = association_tracker([box(z=3)], [(0, 0, 0)], metric='mahalanobis')
        np.testing.assert_allclose(tracker.compute_cost_map()[0], [[0.0]])

    def test_gate_is_chi_square_inclusive(self):
        edge = np.sqrt(9.21 * 9.0)                    # d^2 = 9.21 with S = 9
        inside = association_tracker([box(edge - 1e-6)], [(0, 0, 0)], metric='mahalanobis', position_variances=[5.0])
        outside = association_tracker([box(edge + 1e-3)], [(0, 0, 0)], metric='mahalanobis', position_variances=[5.0])
        self.assertEqual(inside.association(), [0])
        self.assertTrue(np.isinf(outside.compute_cost_map()[0][0, 0]))
        self.assertEqual(outside.association(), [1])

    def test_uncertain_track_has_a_wider_gate(self):
        # 8 m off: d^2 = 64 / 5 = 12.8 for a confident track, 64 / 29 = 2.2 for an uncertain one.
        confident = association_tracker([box(8)], [(0, 0, 0)], metric='mahalanobis', position_variances=[1.0])
        uncertain = association_tracker([box(8)], [(0, 0, 0)], metric='mahalanobis', position_variances=[25.0])
        self.assertEqual(confident.association(), [1])
        self.assertEqual(uncertain.association(), [0])

    def test_greedy_matches_smallest_squared_distance_first(self):
        # One detection inside both ellipses: 2 m from a certain track (S = 5, d^2 = 0.8) and
        # 1 m from an uncertain one (S = 29, d^2 = 0.034); the most overlapping pair wins.
        tracker = association_tracker([box(2)], [(0, 0, 0), (3, 0, 0)], metric='mahalanobis',
                                      position_variances=[1.0, 25.0])
        cost, _ = tracker.compute_cost_map()
        np.testing.assert_allclose(cost, [[4 / 5, 1 / 29]])
        self.assertEqual(tracker.association(), [1])

    def test_assignment_follows_uncertainty_not_metres(self):
        # Track 0 is 3 m away but uncertain (S = 36, d^2 = 0.25); track 1 is 2 m away and certain (S = 4, d^2 = 1).
        mahalanobis = association_tracker([box(3), box(-30)], [(0, 0, 0), (5, 0, 0)], metric='mahalanobis',
                                          position_variances=[32.0, 1e-9])
        euclidean = association_tracker([box(3), box(-30)], [(0, 0, 0), (5, 0, 0)])
        self.assertEqual(mahalanobis.association()[0], 0)
        self.assertEqual(euclidean.association()[0], 1)

    def test_low_score_detection_still_cannot_be_matched(self):
        tracker = association_tracker([box(.1), box(.2)], [(0, 0, 0)], scores=[.05, .9], metric='mahalanobis')
        self.assertEqual(tracker.association(), [-1, 0])

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            Tracker3D(box_type='OpenPCDet', config=config(association_metric='iou'))


class VesselEkfIntegrationTest(unittest.TestCase):
    def test_gap_prediction_and_reacquisition_keep_id(self):
        tracker = Tracker3D(box_type='OpenPCDet', config=config())
        for timestamp, x in [(0, 0), (1, .1)]:
            _, ids = tracker.tracking(np.asarray([box(x)]), scores=np.ones(1), timestamp=timestamp)
            np.testing.assert_array_equal(ids, [0])
        trajectory = tracker.active_trajectories[0]
        self.assertIsInstance(trajectory.ekf, VesselCTRVExtendedKalmanFilter)
        for timestamp in range(2, 5):
            tracker.tracking(np.empty((0, 7)), scores=np.empty(0), timestamp=timestamp)
            observation = trajectory.trajectory[timestamp]
            self.assertIsNone(observation.updated_state)
            self.assertTrue(np.all(np.isfinite(observation.predicted_state)))
            eigenvalues = np.linalg.eigvalsh(np.asarray(observation.predicted_covariance))
            self.assertGreaterEqual(eigenvalues.min(), -1e-9)
        _, ids = tracker.tracking(np.asarray([box(.2)]), scores=np.ones(1), timestamp=5)
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(trajectory.consecutive_missed_num, 0)

    def test_confirmed_track_lifetime_uses_maritime_forty_frames(self):
        tracker = Tracker3D(box_type='OpenPCDet', config=config())
        self.assertEqual(tracker.config.max_prediction_num, 40)
        self.assertEqual(tracker.config.max_prediction_num_for_new_object, 5)
        for timestamp in [0, 1]:
            tracker.tracking(np.asarray([box()]), scores=np.ones(1), timestamp=timestamp)
        for timestamp in range(2, 42):
            tracker.tracking(np.empty((0, 7)), scores=np.empty(0), timestamp=timestamp)
        self.assertIn(0, tracker.active_trajectories)
        tracker.tracking(np.empty((0, 7)), scores=np.empty(0), timestamp=42)
        self.assertNotIn(0, tracker.active_trajectories)
        self.assertIn(0, tracker.dead_trajectories)

    def test_ctrv_prediction_has_no_acceleration_term(self):
        ekf = VesselCTRVExtendedKalmanFilter(13, .1, config()['ekf'])
        state = np.zeros(13); state[ekf.FORWARD_SPEED] = 5.0; state[9:12] = [20, 5, 6]
        covariance = ekf.initial_covariance()
        for _ in range(10):
            state, covariance = ekf.predict(state, covariance)
        self.assertAlmostEqual(state[ekf.X], 5.0)
        self.assertAlmostEqual(state[ekf.FORWARD_SPEED], 5.0)
        for index in (ekf.FORWARD_ACCELERATION, ekf.VERTICAL_ACCELERATION):
            self.assertEqual(state[index], 0.0)
            self.assertFalse(np.any(covariance[index]))
            self.assertFalse(np.any(covariance[:, index]))

    def test_accelerating_vessel_is_absorbed_by_velocity_not_acceleration(self):
        tracker = Tracker3D(box_type='OpenPCDet', config=config())
        for timestamp in range(60):
            t = timestamp * .1
            tracker.tracking(np.asarray([box(2.0 * t + 0.25 * t * t)]), scores=np.ones(1), timestamp=timestamp)
        state = np.asarray(tracker.active_trajectories[0].trajectory[59].updated_state).reshape(-1)
        self.assertEqual(state[VesselCTRVExtendedKalmanFilter.FORWARD_ACCELERATION], 0.0)
        self.assertEqual(state[VesselCTRVExtendedKalmanFilter.VERTICAL_ACCELERATION], 0.0)
        self.assertGreater(abs(state[VesselCTRVExtendedKalmanFilter.FORWARD_SPEED]), 2.5)   # true speed 4.95 m/s

    def test_ctrv_jacobian_matches_finite_differences(self):
        ekf = VesselCTRVExtendedKalmanFilter(13, .1, config()['ekf'])
        state = np.array([3., -2, .5, 4, .7, .1, 0, 0, .08, 20, 5, 6, .9])
        numeric = np.zeros((13, 13))
        for column in range(13):
            step = np.zeros(13); step[column] = 1e-6
            numeric[:, column] = (ekf.transition_function(state + step) - ekf.transition_function(state - step)) / 2e-6
        np.testing.assert_allclose(ekf.transition_jacobian(state), numeric, atol=1e-6)

    def test_original_features_and_bb_features_remain_compatible(self):
        for bb_as_features in [False, True]:
            with self.subTest(bb_as_features=bb_as_features):
                tracker = Tracker3D(tracking_features=True, bb_as_features=bb_as_features,
                                    box_type='OpenPCDet', config=config())
                for timestamp in [0, 1]:
                    _, ids = tracker.tracking(np.asarray([box(.1*timestamp)]),
                        features=np.asarray([[100, 200, 30, 40]], dtype=float),
                        scores=np.ones(1), pose=np.eye(4), timestamp=timestamp)
                    np.testing.assert_array_equal(ids, [0])
                state = tracker.active_trajectories[0].trajectory[1].updated_state
                self.assertEqual(state.shape, (24 if bb_as_features else 17, 1))
                self.assertTrue(np.all(np.isfinite(state)))


if __name__ == '__main__': unittest.main()
