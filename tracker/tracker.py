from .trajectory import Trajectory
from .box_op import *
import numpy as np


def _hungarian_assignment(cost):
    """Return the minimum-cost column for each row (rows <= columns).

    O(n^3) Hungarian algorithm with a vectorised inner loop so that a full
    frame of vessel detections is matched well within the 10 Hz budget.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1]:
        raise ValueError('Hungarian assignment expects a 2-D matrix with rows <= columns')
    if not np.all(np.isfinite(cost)):
        raise ValueError('Hungarian assignment cost must be finite')

    row_count, column_count = cost.shape
    row_potential = np.zeros(row_count + 1)
    column_potential = np.zeros(column_count + 1)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    previous_column = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column = 0
        minimum_slack = np.full(column_count + 1, np.inf)
        used_column = np.zeros(column_count + 1, dtype=bool)
        while True:
            used_column[column] = True
            current_row = matched_row[column]
            free = ~used_column[1:]
            reduced_cost = cost[current_row - 1] - row_potential[current_row] - column_potential[1:]
            better = free & (reduced_cost < minimum_slack[1:])
            minimum_slack[1:][better] = reduced_cost[better]
            previous_column[1:][better] = column
            candidates = np.where(free, minimum_slack[1:], np.inf)
            next_column = int(np.argmin(candidates)) + 1
            delta = candidates[next_column - 1]
            row_potential[matched_row[used_column]] += delta
            column_potential[used_column] -= delta
            minimum_slack[~used_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            previous = previous_column[column]
            matched_row[column] = matched_row[previous]
            column = previous

    assigned_column = np.full(row_count, -1, dtype=np.int64)
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assigned_column[matched_row[column] - 1] = column - 1
    return assigned_column


def _finite_global_assignment(cost):
    """Globally match detection/track pairs whose cost is finite.

    Dummy rows and columns let either side stay unmatched.  The dummy cost
    makes the solver first maximise the number of valid matches and then
    minimise their total distance.
    """
    cost = np.asarray(cost, dtype=float)
    detection_count, track_count = cost.shape
    finite = np.isfinite(cost)
    if detection_count == 0 or track_count == 0 or not np.any(finite):
        return []

    largest_valid_cost = float(np.max(cost[finite]))
    unmatched_cost = max(1.0, (largest_valid_cost + 1.0) * (min(detection_count, track_count) + 1.0))
    size = detection_count + track_count
    forbidden_cost = unmatched_cost * (4.0 * size + 1.0)
    augmented = np.full((size, size), forbidden_cost)
    augmented[:detection_count, :track_count] = np.where(finite, cost, forbidden_cost)
    augmented[:detection_count, track_count:] = unmatched_cost
    augmented[detection_count:, :track_count] = unmatched_cost
    augmented[detection_count:, track_count:] = 0.0

    assigned = _hungarian_assignment(augmented)
    return [(i, int(assigned[i])) for i in range(detection_count)
            if assigned[i] < track_count and finite[i, assigned[i]]]


def _greedy_assignment(cost):
    """Fix detection/track pairs in ascending cost order (smallest d^2 first).

    The finite pair with the smallest cost in the whole map is matched first,
    its detection and track are removed, and this repeats until no finite pair
    is left.  The result does not depend on the detection input order (exact
    cost ties are broken by detection index, then track index).
    """
    cost = np.asarray(cost, dtype=float)
    detections, tracks = np.nonzero(np.isfinite(cost))
    order = np.lexsort((tracks, detections, cost[detections, tracks]))
    used_detections, used_tracks, pairs = set(), set(), []
    for detection, track in zip(detections[order], tracks[order]):
        if detection not in used_detections and track not in used_tracks:
            used_detections.add(detection)
            used_tracks.add(track)
            pairs.append((int(detection), int(track)))
    return pairs


class Tracker3D:
    def __init__(self,tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti',
                    config = None):
        """
        initialize the the 3D tracker
        Args:
            tracking_features: bool, if tracking the features
            bb_as_features: bool, if tracking the bbs
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
        """
        self.config = config
        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type
        # Maritime association: Mahalanobis distance of the XY centre (gated by a
        # chi-square threshold), or the XY Euclidean distance in metres.
        self.association_metric = str(self.config.get('association_metric', 'mahalanobis')).lower()
        if self.association_metric not in ('mahalanobis', 'euclidean'):
            raise ValueError("association_metric must be 'mahalanobis' or 'euclidean'")
        self.association_gate_chi2 = float(self.config.get('association_gate_chi2', 9.21))
        if not np.isfinite(self.association_gate_chi2) or self.association_gate_chi2 <= 0:
            raise ValueError('association_gate_chi2 must be finite and greater than zero')
        self.association_distance_threshold = float(
            self.config.get('association_distance_threshold', 10.0))
        if not np.isfinite(self.association_distance_threshold) or self.association_distance_threshold <= 0:
            raise ValueError('association_distance_threshold must be finite and greater than zero')
        # Pairs inside the gate are fixed smallest cost first (greedy), or by a
        # global Hungarian assignment that first maximises the number of matches.
        self.association_matching = str(self.config.get('association_matching', 'greedy')).lower()
        if self.association_matching not in ('greedy', 'hungarian'):
            raise ValueError("association_matching must be 'greedy' or 'hungarian'")

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

    def tracking(self,bbs_3D = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None
                 ):
        """
        tracking the objects at the given timestamp
        Args:
            bbs: array(N,7) or array(N，7*k), 3D bounding boxes or 3D tracklets
                for tracklets, the boxes should be organized to [[box_t; box_t-1; box_t-2;...],...]
            features: array(N,k), the features of boxes or tracklets
            scores: array(N,), the detection score of boxes or tracklets
            pose: array(4,4), pose matrix to global scene
            timestamp: int, current timestamp, note that the timestamp should be consecutive

        Returns:
            bbs: array(M,7), the tracked bbs
            ids: array(M,), the assigned IDs for bbs
        """
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp

        self.trajectores_prediction()

        if self.current_bbs is None:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            if len(self.current_bbs) == 0:
                return np.zeros(shape=(0,7)),np.zeros(shape=(0))

            else:
                self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
                bbs,ids = self.trajectories_update_init(ids)

                return np.array(bbs),np.array(ids)



    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=self.config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= self.config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def _checked_scores(self):
        scores = np.asarray(self.current_scores, dtype=float).reshape(-1)
        if scores.shape != (len(self.current_bbs),):
            raise ValueError('scores must have one value per detection, '
                             f'got {scores.shape} for {len(self.current_bbs)} detections')
        if not np.all(np.isfinite(scores)):
            raise ValueError('detection scores must be finite')
        return scores

    def compute_cost_map(self):
        """
        compute the gated XY cost map between detections and predictions
        mahalanobis: cost = d^2 = v^T S^-1 v, v = detection XY - predicted XY,
        S = P_xy + R_xy (predicted state covariance + measurement noise of the
        track's EKF); pairs with d^2 > association_gate_chi2 are not matchable.
        euclidean: cost = XY distance; pairs farther than
        association_distance_threshold are not matchable.
        Detections that cannot pass update_score are not matchable either.
        Returns:
              cost, array(N,M), where N is the number of detections, M is the number of active trajectories
              all_ids, list(M,), the corresponding IDs of active trajectories
        """
        all_ids = list(self.active_trajectories.keys())
        detections = np.asarray(self.current_bbs, dtype=float)
        scores = self._checked_scores()
        predicted = [self.active_trajectories[key].trajectory[self.current_timestamp] for key in all_ids]
        predictions = np.array([np.asarray(ob.predicted_state, dtype=float).reshape(-1)[0:2]
                                for ob in predicted]).reshape(-1, 2)
        residual = detections[:, None, 0:2] - predictions[None, :, :]

        if self.association_metric == 'mahalanobis':
            inverse = np.array([
                np.linalg.inv(np.asarray(ob.predicted_covariance, dtype=float)[0:2, 0:2]
                              + np.asarray(self.active_trajectories[key].ekf.measurement_covariance,
                                           dtype=float)[0:2, 0:2])
                for key, ob in zip(all_ids, predicted)]).reshape(-1, 2, 2)
            distance = np.einsum('nmi,mij,nmj->nm', residual, inverse, residual)
            gate = self.association_gate_chi2
        else:
            distance = np.linalg.norm(residual, axis=-1)
            gate = self.association_distance_threshold
        matchable = (distance <= gate) & (scores > self.config.update_score)[:, None]
        cost = np.where(matchable, distance, np.inf)

        return cost,all_ids

    def association(self):
        """
        assign the IDs for detected state based on the gated cost map: greedy fixes
        the smallest-cost pair first (the most overlapping Mahalanobis ellipses),
        hungarian assigns globally.
        Unmatched detections get a new ID only when they pass init_score;
        -1 marks a detection that is neither matched nor initialised.
        Returns:
            ids, list(N,), assigned IDs for boxes, where N is the input boxes number
        """
        scores = self._checked_scores()
        ids = [-1] * len(self.current_bbs)
        if len(self.active_trajectories) != 0:
            cost_map, all_ids = self.compute_cost_map()
            match = _greedy_assignment if self.association_matching == 'greedy' else _finite_global_assignment
            for detection_index, trajectory_index in match(cost_map):
                ids[detection_index] = all_ids[trajectory_index]

        for i in range(len(self.current_bbs)):
            if ids[i] == -1 and scores[i] > self.config.init_score:
                ids[i] = self.label_seed
                self.label_seed+=1
        return ids


    def trajectories_update_init(self,ids):
        """
        update a exiting trajectories based on the association results, or init a new trajectory
        Args:
            ids: list or array(N), the assigned ids for boxes
        """
        assert len(ids) == len(self.current_bbs)

        valid_bbs = []
        valid_ids = []

        for i in range(len(self.current_bbs)):
            label = ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            if label in self.active_trajectories.keys() and score>self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score>self.config.init_score:
                new_tra = Trajectory(init_bb=box,
                                     init_features=features,
                                     init_score=score,
                                     init_timestamp=self.current_timestamp,
                                     label=label,
                                     tracking_features=self.tracking_features,
                                     bb_as_features=self.bb_as_features,
                                     config = self.config)
                self.active_trajectories[label] = new_tra
                valid_bbs.append(box)
                valid_ids.append(label)
            else:
                continue
        if len(valid_bbs)==0:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            return np.array(valid_bbs),np.array(valid_ids)


    def post_processing(self, config):
        """
        globally filter the trajectories
        Args:
            config: config

        Returns: dict(Trajectory)

        """
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering(config)
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering(config)
            tra[key] = track

        return tra




