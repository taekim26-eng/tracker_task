# CasTrack 기반 해상 다중 선박 추적기

공식 CasTrack 코드를 기반으로, 트랙 생성·유지·삭제·후처리 구조는 그대로 두고 해상 환경에 맞게 두 부분을 바꾼 폴더입니다.

1. 운동 필터: 원본 선형 KF → 선박용 CTRV-EKF (등선회율·등속 가정 + Fossen 3자유도 운동학)
2. 데이터 연결: 원본 Greedy(검출 입력 순서, XYZ 거리 × 예측 점수 < 2.0) → XY 중심 마할라노비스 거리(카이제곱 게이트 9.21) + d²가 가장 작은 쌍부터 확정하는 Greedy(검출 순서 무관, 저신뢰 검출 사전 제외)

- 원본 저장소: https://github.com/hailanyi/3D-Multi-Object-Tracker
- 고정 커밋: `d37df41c8e512e14c198cb18a4609ed66253d7a1`
- 원본 파일 URL·SHA256: [UPSTREAM_SOURCE.json](UPSTREAM_SOURCE.json)
- 설정 비교 도구와 결과: `/home/rain/tracking_test/과제용/tuning`

## 원본과 같은 부분

`object.py`, `box_op.py`, `config.py`는 위 커밋의 원본 파일과 바이트 단위로 동일합니다. `tracker.py`에서도 `tracking()`, 예측·트랙 삭제(`trajectores_prediction`), 갱신·생성(`trajectories_update_init`), 후처리(`post_processing`)는 원본 코드 그대로입니다. 공개 호출 방식과 `(boxes, ids)` 반환도 원본과 같습니다.

```python
boxes, ids = tracker.tracking(
    bbs_3D, features=features, scores=scores,
    pose=pose, timestamp=frame_index,
)
```

VoxelNeXt(OpenPCDet) 출력 `[x, y, z, dx, dy, dz, heading]`과 점수는 `box_type='OpenPCDet'`으로 그대로 넣습니다.

## 바꾼 데이터 연결 (`tracker.py`의 `compute_cost_map`, `association`)

```text
v    = [dx, dy]                         (검출 중심 − 예측 중심, XY)
S    = P_xy + R_xy                      (트랙 EKF의 예측 위치 공분산 + 측정 위치 잡음)
cost = d² = vᵀ S⁻¹ v                    (마할라노비스 거리의 제곱)
cost > association_gate_chi2            → 연결 불가 (9.21 = 자유도 2 카이제곱 99 %)
score <= update_score                   → 기존 트랙과 연결 불가 (ID를 선점하지 않음)
매칭 (association_matching: greedy)     → 게이트 안 쌍 중 d²가 가장 작은 쌍(가장 많이 겹치는 쌍)부터 확정,
                                          확정된 검출·트랙은 제외하고 반복
연결되지 않은 검출: score > init_score일 때만 새 ID
```

모든 쌍을 d² 순으로 정렬해 고르므로 검출 입력 순서와 관계없이 같은 결과가 나옵니다(원본 CasTrack Greedy는 검출 입력 순서대로 가장 가까운 트랙을 가져갑니다). 불확실한 트랙(오래 검출이 없었거나 막 생긴 트랙)은 S가 커서 게이트가 넓어지고, 안정된 트랙은 좁아집니다. 예측 점수(`prediction_score`)는 CasTrack 방식대로 계속 계산되지만 연결 비용에는 쓰지 않습니다.

다른 설정:
- `association_matching: hungarian`: 이전 매칭(Hungarian 전역 할당, 연결 수를 최대화한 뒤 거리 합을 최소화)
- `association_metric: euclidean`: 이전 거리(XY 거리, `association_distance_threshold` 10 m 게이트)

## 설정 (`config/multi_object_tracker.yaml`)

| 설정 | 원본 (casa_mot) | 현재 |
|---|---|---|
| association_metric / association_gate_chi2 | 거리 × 예측 점수 < 2.0 | **mahalanobis / 9.21** |
| association_matching | 검출 입력 순서 Greedy | **greedy (d² 작은 쌍부터)** |
| association_distance_threshold (euclidean일 때만) | 없음 | 10 m |
| max_prediction_num | 12 frame | **40 frame (4초)** |
| max_prediction_num_for_new_object | 4 frame | **5 frame** |
| init_score / update_score | 0 / 0 | **0.30 / 0.10** |
| input_score / post_score / latency | 0 / 1.0 / 0 | 0 / 1.0 / 0 |
| prediction_score_decay / LiDAR_scanning_frequency | 0.02 / 10 Hz | 0.02 / 10 Hz |

`init_score` / `update_score`는 실제 VoxelNeXt 점수 분포에 맞춰 다시 확인해야 합니다.

## 선박용 EKF (`extended_kalman_filter.py`, `trajectory.py`)

`trajectory.py`는 EKF 구성, 상태·공분산 초기화, 비선형 예측, 측정 갱신 연결만 원본 KF 대신 바꿨습니다. 생성자·갱신 함수 인자, 상태 차원, 점수 처리, 누락 횟수, 후처리는 원본과 같습니다.

```text
[x, y, z,
 forward_speed, lateral_speed, vertical_speed,
 forward_acceleration, vertical_acceleration, yaw_rate,
 length, width, height, yaw,
 optional_features...]
```

운동 모델은 CTRV(constant turn rate and velocity)입니다. 선박의 진행·횡방향 속도와 선회율이 한 프레임 동안 일정하다고 보고, 위치는 Fossen 3자유도 운동학대로 프레임 시작의 heading으로 배 기준 속도(앞·옆)를 지도 좌표로 돌려 적분하며, heading은 선회율 × dt만큼 돌립니다. 원본 CasTrack과 상태 벡터 길이·순서를 맞추기 위해 `forward_acceleration`, `vertical_acceleration` 자리는 남겨 두지만 항상 0이며 분산도 0이라 추정에 쓰이지 않습니다. 속도 변화는 `process_velocity_variance`(0.05)로 흡수합니다. 그래서 강하게 계속 가속하는 선박에서는 추정 속도가 실제보다 늦게 따라갈 수 있습니다.

`state_func_covariance`와 `measure_func_covariance`는 원본 호환용으로만 남아 있습니다.

## 검증

`tracker`를 포함하는 디렉터리에서:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tracker/
```

Greedy 매칭(d² 작은 쌍 우선, 입력 순서 무관, 확실한 쌍을 연결 수보다 우선), Hungarian 설정(전역 할당, 연결 수 우선), Z 무시, 예측 점수 비가중, 마할라노비스 비용·카이제곱 게이트 경계·불확실도에 따른 게이트 변화, 유클리드 10 m 게이트 경계(euclidean 설정), 저신뢰 검출 ID 비선점, 초기화 점수, 트랙 수명 40 frame, EKF 누락 예측·재연결, CTRV 예측(가속 항 없음, 가속도 자리 0 유지, 야코비안 수치 검증), 특징 벡터 호환, 19개 필드 출력을 검사합니다.

## 19개 필드 결과 출력

[출력 사용법](README_OUTPUT_KO.md)의 `TrackingOutputAdapter`로 모든 19개 필드 값을 반환합니다. 출력 어댑터는 EKF 상태·공분산을 읽기만 합니다.
