import numpy as np
from .object import Object
from .extended_kalman_filter import VesselCTRVExtendedKalmanFilter

class Trajectory:
    def __init__(self,init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 tracking_features=True,
                 bb_as_features=False,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
            tracking_features: bool, if track features
            bb_as_features: bool, if treat the bb as features
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label
        self.tracking_bb_size = True
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features

        self.config = config
        self.ekf_config = self.config.get('ekf', {})


        self.scanning_interval = 1./self.config.LiDAR_scanning_frequency

        if self.bb_as_features:
            if self.init_features is None:
                self.init_features = init_bb
            else:
                self.init_features = np.concatenate([init_bb,init_features],0)
        self.trajectory = {}

        self.track_dim = self.compute_track_dim() # 9+4+bb_features.shape

        self.init_ekf()
        self.init_trajectory()


        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def compute_track_dim(self):
        """
        compute tracking dimension
        :return:
        """
        track_dim=9 #x,y,z,vx,vy,vz,ax,ay,az

        if self.tracking_bb_size:
            track_dim+=4 # w,h,l,yaw
        if self.tracking_features:
            track_dim+=self.init_features.shape[0] #features.shape
        return track_dim

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = self.build_detected_state(
            self.init_bb, self.init_features
        )

        update_state = self.ekf.initialize_state(detected_state_template)
        update_covariance = self.ekf.initial_covariance()
        update_state_template = np.asmatrix(update_state).T
        update_covariance_template = np.asmatrix(update_covariance)

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features

        self.trajectory[self.init_timestamp] = object

    def build_detected_state(self, bb, features):
        """Build [x,y,z,l,w,h,yaw,features...] measurement column."""
        detected_state = np.zeros(shape=(self.track_dim-6))
        detected_state[:3] = bb[:3]

        if self.tracking_bb_size:
            detected_state[3:7] = bb[3:7]
            if self.tracking_features:
                detected_state[7:] = features[:]
        elif self.tracking_features:
            detected_state[3:] = features[:]

        return np.asmatrix(detected_state).T


    def init_ekf(self):
        """Initialize the only supported motion filter: vessel CTRV-EKF."""
        self.ekf = VesselCTRVExtendedKalmanFilter(
            state_dim=self.track_dim,
            scanning_interval=self.scanning_interval,
            config=self.ekf_config,
        )

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp-1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score

        if timestamp-1 in self.trajectory.keys():
            if self.trajectory[timestamp-1].updated_state is not None:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay*15)
            else:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)
        else:
            current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)


        predicted_state, predicted_covariance = self.ekf.predict(
            previous_state, previous_covariance
        )
        current_predicted_state = np.asmatrix(predicted_state).T
        current_predicted_covariance = np.asmatrix(predicted_covariance)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None,
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        if self.bb_as_features:
            if features is None:
                features = bb
            else:
                features = np.concatenate([bb,features],0)

        detected_state_template = self.build_detected_state(bb, features)

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        if (
            self.last_updated_timestamp == self.init_timestamp
            and self.ekf_config.get('bootstrap_from_two_measurements', False)
        ):
            previous_state = self.trajectory[
                self.last_updated_timestamp
            ].updated_state
            elapsed_time = (
                timestamp - self.last_updated_timestamp
            ) * self.scanning_interval
            updated_state_vector, updated_covariance_array = (
                self.ekf.bootstrap_state(
                    previous_state,
                    detected_state_template,
                    dt=elapsed_time,
                )
            )
        else:
            updated_state_vector, updated_covariance_array = self.ekf.update(
                predicted_state,
                predicted_covariance,
                detected_state_template,
            )
        updated_state = np.asmatrix(updated_state_vector).T
        updated_covariance = np.asmatrix(updated_covariance_array)

        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num>1:
            current_ob.prediction_score = 1
        elif self.trajectory[timestamp - 1].updated_state is not None:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*10*(self.sigmoid(score))
        else:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*(self.sigmoid(score))
        current_ob.score = score
        current_ob.features = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

    def filtering(self,config):
        """
        filtering the trajectory in a global or near online way
        """

        wind_size = int(config.LiDAR_scanning_frequency*config.latency)

        if wind_size <0:

            detected_num = 0.00001
            score_sum = 0

            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                if ob.score is not None:
                    detected_num+=1
                    score_sum+=ob.score
                if self.first_updated_timestamp<=key<=self.last_updated_timestamp and ob.updated_state is None:
                    ob.updated_state = ob.predicted_state

            score = score_sum/detected_num
            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                ob.score = score

        else:
            keys = list(self.trajectory.keys())

            for key in keys:

                min_key = int(key-wind_size)
                max_key = int(key+wind_size)
                detected_num = 0.00001
                score_sum = 0
                for key_i in range(min_key,max_key):
                    if key_i not in self.trajectory:
                        continue
                    ob = self.trajectory[key_i]
                    if ob.score is not None:
                        detected_num+=1
                        score_sum+=ob.score
                    if self.first_updated_timestamp<=key_i<=self.last_updated_timestamp and ob.updated_state is None:
                        ob.updated_state = ob.predicted_state

                score = score_sum / detected_num
                if wind_size!=0:
                    self.trajectory[key].score=score