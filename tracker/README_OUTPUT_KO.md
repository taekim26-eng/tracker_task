# 19개 필드 추적 결과

`TrackingOutputAdapter.tracking()`은 객체별 `InfoDetectedObject` 목록을 반환합니다. 각 객체에는 요청한 19개 필드가 모두 있습니다. `as_dict()`로 숫자 딕셔너리를 얻고, `populate_ros_message()`로 같은 필드의 ROS 메시지를 채울 수 있습니다.

출력 어댑터는 `Tracker3D`의 연결·ID·트랙 관리와 EKF 상태를 바꾸지 않고, EKF 상태와 공분산을 읽기만 합니다. 원본 `Tracker3D.tracking()` 자체는 호환성을 위해 `(boxes, ids)`를 유지합니다.

## 모델 호출

```python
from tracker.tracker import Tracker3D
from tracker.output_adapter import TrackingOutputAdapter

model = Tracker3D(tracking_features=True, box_type='OpenPCDet', config=config)
output = TrackingOutputAdapter(model, target_label=0)
objects = output.tracking(
    bbs_3D=boxes_3d, scores=detection_scores,
    features=boxes_2d_center_xyhw, pose=sensor_to_tracker, timestamp=frame_index,
)
for obj in objects:
    print(obj.as_dict())  # 모든 19개 필드
```

`features`의 첫 네 값은 `[center_x, center_y, height, width]` 픽셀값입니다. 원본의 선택적 특징 추적 기능을 사용해 이 값과 분산을 계산합니다. `target_label`은 실제 검출 클래스 ID를 지정합니다. 클래스마다 추적기를 쓰거나, 기존 모델의 ID에 대응하는 클래스 매핑을 `tracking_output(..., target_label={id: class_id})`로 전달할 수 있습니다. 트랙 번호로 클래스를 추정하지 않습니다.

이미 모델을 호출했다면 `tracking_output(model, ...)`으로 결과를 읽을 수 있습니다. 3D LiDAR 상태만 있을 경우 `CalibratedBoxProjection` 또는 명시적인 `ImageBoxEstimate` 제공 함수로 2D 좌표와 공분산을 전달합니다. 좌표나 그 분산을 알 수 없을 때 0으로 채우지 않습니다. 기본 strict 모드는 필요한 정보가 없으면 오류를 반환합니다.

## 필드 의미와 단위

이름과 타입은 제공된 InfoDetectedObject.msg와 같습니다. 단위·좌표 정의는 메시지 이름만으로 확정할 수 없어 현재 보조 함수의 기존 정의를 유지했습니다.

| 필드 | 값 |
|---|---|
| id (int64) | CasTrack 트랙 ID |
| target_label (int64) | 호출자가 지정하는 검출 클래스 ID |
| score (float64) | 마지막으로 갱신한 검출의 점수. 예측 프레임에서도 마지막 점수를 유지 |
| x / y (int64) | 2D 박스 중심 픽셀, 가장 가까운 정수로 반올림 |
| h / w (int64) | 2D 박스 높이·너비 픽셀 |
| range (float64) | 센서 기준 수평 거리 m |
| theta (float64) | 센서 +x축에서 반시계 방향 방위 rad |
| speed (float64) | EKF의 수평 속도 크기 m/s |
| heading (float64) | 추적 좌표계에서의 선박 방향 rad |
| x_covariances / y_covariances / h_covariances / w_covariances | 반올림 전 2D 박스 값의 분산 pixel² |
| range_covariances | 거리 분산 m² |
| theta_covariances / heading_covariances | 각도 분산 rad² |
| speed_covariances | 속도 분산 (m/s)² |

`*_covariances`는 스칼라 분산이며 표준편차나 전체 공분산 행렬이 아닙니다. 거리·방위·속도 분산은 EKF 공분산과 변환 Jacobian에서 계산합니다. 속도 0에서의 속도 분산은 수평 속도 공분산의 trace를 사용한 근사입니다. 센서 원점에서 방위는 정의되지 않아 보조 함수의 기존 규칙으로 theta=0, theta 분산=π²을 사용합니다.

자선 운동 보정(`pose`)을 하지 않으면 speed에 자선 운동의 영향이 포함됩니다. 이때는 실제 절대 선속으로 해석하지 마세요. 센서 포즈 및 캘리브레이션 자체의 불확실성은 포함하지 않습니다.

## ROS 메시지 빌드와 조회

ROS2 패키지는 `ros2_interfaces/castrack_interfaces`에 포함돼 있어 tracker 폴더와 함께 전달할 수 있습니다. `tracker`를 포함하는 디렉터리에서:

```bash
./tracker/build_interfaces.sh            # 기본 빌드 위치: ./.castrack_interfaces
source /opt/ros/lyrical/setup.bash
source .castrack_interfaces/install/setup.bash
ros2 interface show castrack_interfaces/msg/InfoDetectedObjectArray
```

모든 필드의 ROS 타입은 실제 생성된 메시지에서 int64 또는 float64로 지정됩니다. 배열 메시지는 프레임의 Header와 객체 목록을 담습니다.
