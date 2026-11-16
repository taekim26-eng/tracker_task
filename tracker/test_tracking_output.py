"""Output schema, covariance and adapter behavior tests."""
from dataclasses import fields
import json
from types import SimpleNamespace
import unittest

import numpy as np

from .output_adapter import CalibratedBoxProjection, RESULT_FIELDS, TrackingOutputAdapter, tracking_output
from .test_tracker_association import config, box
from .tracker import Tracker3D
from .tracking_result import InfoDetectedObject, info_from_ekf_state


def estimate():
    state=np.zeros(17);state[:3]=[3,4,10];state[3:5]=[3,4]
    state[9:13]=[2,3,4,.2];state[13:]=[100,200,30,40]
    return state,np.eye(17)


class TrackingResultTest(unittest.TestCase):
    def test_exact_nineteen_fields_and_json_types(self):
        state,cov=estimate()
        result=info_from_ekf_state(track_id=7,target_label=2,score=.9,state=state,covariance=cov)
        row=result.as_dict()
        self.assertEqual(len(row),19)
        self.assertEqual(tuple(row),RESULT_FIELDS)
        for name in ['id','target_label','x','y','h','w']:self.assertIs(type(row[name]),int)
        for name in set(row)-{'id','target_label','x','y','h','w'}:self.assertIs(type(row[name]),float)
        self.assertEqual(json.loads(json.dumps(row,allow_nan=False)),row)
        message=SimpleNamespace()
        self.assertEqual(vars(result.populate_ros_message(message)),row)

    def test_range_bearing_speed_and_variances(self):
        state,cov=estimate()
        result=info_from_ekf_state(track_id=0,target_label=0,score=.8,state=state,covariance=cov)
        self.assertAlmostEqual(result.range,5)
        self.assertAlmostEqual(result.theta,np.arctan2(4,3))
        self.assertAlmostEqual(result.speed,5)
        self.assertAlmostEqual(result.range_covariances,1)
        self.assertAlmostEqual(result.theta_covariances,.04)
        self.assertAlmostEqual(result.speed_covariances,1)
        self.assertAlmostEqual(result.heading_covariances,1)

    def test_pose_controls_range_and_bearing_without_mutating_state(self):
        state,cov=estimate();saved=state.copy()
        pose=np.eye(4);pose[:2,3]=[3,0]
        result=info_from_ekf_state(track_id=0,target_label=0,score=.8,state=state,covariance=cov,sensor_pose=pose)
        self.assertAlmostEqual(result.range,4)
        self.assertAlmostEqual(result.theta,np.pi/2)
        np.testing.assert_array_equal(state,saved)

    def test_explicit_pixel_geometry_requires_uncertainty(self):
        state,cov=estimate()
        with self.assertRaisesRegex(ValueError,'covariance'):
            info_from_ekf_state(track_id=0,target_label=0,score=1,state=state[:13],covariance=cov[:13,:13],image_box=np.array([10,20,30,40]))
        row=info_from_ekf_state(track_id=0,target_label=0,score=1,state=state[:13],covariance=cov[:13,:13],
            image_box=np.array([10,20,30,40]),image_box_covariance=np.diag([1,2,3,4]))
        self.assertEqual([row.x,row.y,row.h,row.w],[10,20,30,40])
        self.assertEqual([row.x_covariances,row.y_covariances,row.h_covariances,row.w_covariances],[1,2,3,4])

    def test_int64_overflow_and_nonfinite_values_are_rejected(self):
        state,cov=estimate()
        with self.assertRaisesRegex(ValueError,'int64'):
            info_from_ekf_state(track_id=2**63,target_label=0,score=1,state=state,covariance=cov)
        with self.assertRaises(ValueError):
            info_from_ekf_state(track_id=0,target_label=0,score=float('nan'),state=state,covariance=cov)

    def test_projected_box_covariance_and_invalid_depth(self):
        state,cov=estimate();state[12]=.2
        projection=CalibratedBoxProjection(np.array([[100,0,50,0],[0,100,40,0],[0,0,1,0]]),np.c_[np.eye(3),np.zeros(3)])
        output=projection(0,state,cov)
        self.assertEqual(output.values.shape,(4,))
        self.assertTrue(np.all(np.isfinite(output.values)))
        self.assertTrue(np.all(np.diag(output.covariance)>0))
        self.assertGreaterEqual(np.linalg.eigvalsh(output.covariance).min(),-1e-7)
        state[2]=-10
        with self.assertRaisesRegex(ValueError,'behind'):
            projection(0,state,cov)


class OutputAdapterIntegrationTest(unittest.TestCase):
    def test_outputs_nineteen_fields_and_prediction_uses_previous_score(self):
        model=Tracker3D(tracking_features=True,box_type='OpenPCDet',config=config())
        adapter=TrackingOutputAdapter(model,target_label=4)
        first=adapter.tracking(np.asarray([box()]),features=np.array([[100,200,30,40]]),scores=np.array([.8]),timestamp=0)
        self.assertEqual(len(first),1)
        self.assertEqual(first[0].target_label,4)
        self.assertEqual(len(first[0].as_dict()),19)
        second=adapter.tracking(np.empty((0,7)),features=np.empty((0,4)),scores=np.empty(0),timestamp=1)
        self.assertEqual(second[0].id,first[0].id)
        self.assertEqual(second[0].score,.8)
        self.assertIsNone(model.active_trajectories[0].trajectory[1].updated_state)

    def test_reader_preserves_state_and_fails_honestly_without_pixel_input(self):
        model=Tracker3D(box_type='OpenPCDet',config=config())
        model.tracking(np.asarray([box()]),scores=np.ones(1),timestamp=0)
        state=model.active_trajectories[0].trajectory[0].updated_state.copy()
        with self.assertRaisesRegex(ValueError,'2D'):
            tracking_output(model)
        output=tracking_output(model,strict=False)
        self.assertFalse(output.objects)
        self.assertIn(0,output.unavailable)
        np.testing.assert_array_equal(model.active_trajectories[0].trajectory[0].updated_state,state)


if __name__=='__main__':unittest.main()
