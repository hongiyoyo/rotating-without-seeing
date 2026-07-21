# rotating-without-seeing

터치 전용(vision 절대 사용 안 함) z축 in-hand rotation teacher policy를 IsaacGym + PPO로 학습하기 위한, `in-hand-rotation`("Robot Synesthesia" 코드베이스)에서 필요한 부분만 추려낸 독립 프로젝트입니다.

이 문서는 이 코드를 처음 이어받는 사람/에이전트가 배경 설명 없이 바로 작업을 이어갈 수 있도록 지금까지의 작업 이력, 설계 근거, 그리고 **반드시 알아야 할 함정들**을 정리한 것입니다.

## 배경

- 목표 논문: *"Rotating without Seeing: Towards In-hand Dexterity through Touch"* (Yin et al., RSS 2023) — 16개의 이진 접촉 센서(FSR)만으로 vision 없이 물체를 z/x/y축으로 회전시키는 policy를 IsaacGym에서 PPO로 학습.
- 원본 소스: `C:\jaehong_yang\in-hand-rotation` — "Robot Synesthesia" (ICRA 2024) 코드베이스이며, 그 안에 위 논문을 재현하는 `observationType=partial_stack` 모드(= "PS" 베이스라인)가 이미 포함되어 있음.
- 이 폴더(`rotating-without-seeing`)는 그 코드베이스에서 **z축, 촉각 전용(partial_stack), objSet=C 물체 세트** 시나리오 하나에만 필요한 파일들을 골라내고, 논문 스펙에 맞게 도메인 랜덤화/보상 관련 수치들을 직접 수정한 결과물입니다. 다른 실험(baoding balls, cross wheel-wrench, visual RL, distillation)은 이 폴더에 없습니다.
- **이 개발 PC에는 GPU가 없어 실제 학습 실행 검증은 못했습니다.** 지금까지 한 것은 전부 정적 검증(문법 검사, yaml 파싱, 파일 존재 확인, 메시 volume/bbox 계산)까지이고, 실제 `python isaacgymenvs/train.py ...` 실행 및 학습 곡선 확인은 GPU 환경에서 사람이 직접 해야 합니다.

## 폴더 구조

```
rotating-without-seeing/
  pickle_utils.py
  isaacgymenvs/
    train.py, tasks/allegro_arm_morb_axis.py (핵심 태스크 파일), tasks/base/vec_task.py
    utils/, learning/ (train.py의 import chain을 만족시키기 위해 필요, AMP 관련 기능은 실제로 안 씀)
    cfg/config.yaml, cfg/task/AllegroArmMOAR.yaml, cfg/train/AllegroArmMOARPPO.yaml
  rl_games/            (PPO 구현체, 통째로 복사)
  assets/urdf/
    xarm6/xarm6_allegro_right_fsr_2023_thin.urdf + meshes/  (손 - xArm6 + Allegro Hand + 16 FSR 센서)
    objects/*.urdf + meshes/set2/*.obj                       (학습용 물체 22개, 아래 참고)
  scripts/train_z_axis.sh   (학습 실행 스크립트)
  tools/
    object_viewer.html        (물체 22개를 브라우저에서 3D로 확인하는 뷰어, mesh 데이터 내장)
    generate_new_objects.py   (새 물체 6개를 만든 스크립트 - 재현/참고용)
    regen_viewer_data.py      (object_viewer.html의 내장 mesh 데이터를 재생성하는 스크립트)
  install.md
```

## 논문 스펙 대비 수정한 항목 (파일: `isaacgymenvs/tasks/allegro_arm_morb_axis.py`, `cfg/task/AllegroArmMOAR.yaml`)

원본 `in-hand-rotation`의 `partial_stack` 구현은 논문의 대부분(16개 FSR 센서 배치, finite-difference 회전각 보상, EMA 액션 스무딩 η=0.8, 10Hz 제어, 4-step 히스토리 스택, PPO 하이퍼파라미터)을 이미 정확히 따르고 있었지만, 아래 수치/공식들은 이후 다른 실험(Robot Synesthesia)에 맞춰 바뀐 상태였고, 이 폴더에서 논문 값으로 되돌렸습니다:

| 항목 | 원본 코드 | 이 폴더 | 
|---|---|---|
| 접촉 센서 threshold | 랜덤 [1.0, 2.0]N | 랜덤 [0.005, 0.015]N (논문 0.01N 근사) |
| PD gain 랜덤화 | P×[0.30,0.60], D×[0.75,1.05] | P×[0.66,1.33], D×[0.80,1.20] |
| 손/물체 마찰 | 공유된 draw 1개, [0.2,3.0] | 독립 draw 2개, 둘 다 [0.3,3.0] |
| 물체 스케일(C 세트) | [0.95,1.1] | [0.95,1.05] |
| 초기 위치 노이즈 | ±1.0cm | ±1.5cm |
| 랜덤 외력 스케일 | 2.0 | 0.2 |
| 랜덤 외력 확률 | 죽은 코드(고정 0.25만 사용) | env별 log-uniform [0.2,0.25] 실제 사용 |
| 센서 lag 확률 | 0.2 | 0.25 |
| 관절 관측 노이즈 | ±0.06 | ±0.05 |
| 액션 노이즈 | Gaussian σ=0.04 | Uniform ±0.06 |
| Torque 보상 | `-Σ τ²` (제곱) | `-‖τ‖` (L2 norm, 논문과 일치) |
| numEnvs / minibatch | 16 / 32 (yaml 기본값) | 8192 / 16384 (논문 값 고정) |

새 yaml 키(`sensorThreshLow`, `objectFrictionLow/High`, `handFrictionLow/High`, `objectScaleLow/High`, `pGainMultLow/High`, `dGainMultLow/High`, `jointObsNoise`, `torqueRewardNorm`, `useForceProbRange`)로 위 값들을 조정 가능하게 만들어 뒀습니다.

## ⚠️ 반드시 알아야 할 함정: `ablation_mode`

**`cfg/task/AllegroArmMOAR.yaml`의 `ablation_mode`를 절대 `"multi-modality"`나 `"no-tactile"`로 바꾸지 마세요.**

원본 코드에는 `ablation_mode in ["no-tactile", "multi-modality"]`일 때 `isaacgymenvs/tasks/base/vec_task.py`의 `step()`/`reset()`/`reset_done()`이 **매 스텝 관측 버퍼에서 16차원 촉각 신호를 통째로 잘라내는** 로직이 있고, 태스크 파일도 이에 맞춰 `numObservations`를 하드코딩된 276으로 덮어씁니다(`allegro_arm_morb_axis.py:382-383`). 원본 yaml 기본값이 정확히 `ablation_mode: multi-modality`였기 때문에, 고치지 않았다면 **"vision을 안 쓰는 정책"이 아니라 "vision도 촉각도 다 못 보는 정책"이 학습될 뻔했습니다.** 크래시 없이 조용히 잘못된(목적에 안 맞는) 정책이 학습되는 종류의 버그라 발견하기 어렵습니다.

지금은 `ablation_mode: no-pc`로 고쳐뒀고, 이 값은 코드 어디에서도 특별 취급되지 않는 "안전한 기본" 값입니다(grep으로 전체 확인함). 이 필드를 건드릴 일이 있으면 반드시 `"no-tactile"`/`"multi-modality"`를 피하세요.

## 학습용 물체 (22개, `objSet: "C"`)

원본 16개 물체를 요청에 따라 리네이밍하고, 새 물체 6개를 추가했습니다. 전부 `assets/urdf/objects/*.urdf` + `assets/urdf/objects/meshes/set2/*.obj`에 있고, `isaacgymenvs/tasks/allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets["C"]`에 등록되어 있습니다. 모든 물체는 원본 mesh 기준 bounding box ~2.0 unit(= URDF `scale=".03 .03 .03"` 적용 시 실제 약 6cm)로, 기존 물체들과 동일한 크기 규격입니다.

| 라벨 | 형상 | 비고 |
|---|---|---|
| `block_1` | 정육면체 | 원래 1번 |
| `block_2` | 불규칙 블록("time") | 원래 15번 |
| `block_3` | **계단(L자) 모양** | 신규 — 정육면체에서 한 사분면(x∈[0,1],y∈[0,1], 전체 높이)을 도려냄. 충돌 메시는 볼록 박스 2개로 정확히 분해 |
| `block_4` | 모서리 깎인 블록 | 원래 6번 |
| `cylinder_1` | 원기둥 | 원래 11번 |
| `cylinder_2` | **정십각기둥** | 신규 |
| `cylinder_3` | 모서리 깎인 원기둥 | 원래 12번 |
| `cylinder_4` | 축 방향 압축 원기둥 | 원래 16번 |
| `ball_1` | **구** | 신규 (Fibonacci sphere 200점 + convex hull) |
| `ball_2` | **정12면체**(dodecahedron) | 신규 — "정12각형"을 정다면체로 해석 |
| `ball_3` | **정20면체**(icosahedron) | 신규 |
| `ball_4` | **타원체** | 신규 — 구를 두 축으로 0.8배 압축 |
| `else_1`~`else_10` | 나머지 원래 물체 (2,3,4,5,7,8,9,10,13,14번) | 순서 무관, 요청대로 일괄 리네이밍 |

**`ball` 키는 이 22개와 무관**하니 헷갈리지 마세요 — baoding balls 과제 전용의 별도 단일 구 오브젝트입니다(건드리지 않음).

새 물체 6개는 전부 볼록(convex) 도형이라 충돌 메시가 시각 메시와 동일합니다(단, `block_3`만 비볼록이라 2-파트로 분해). `tools/generate_new_objects.py`를 다시 실행하면 동일한 절차로 재생성됩니다.

**해석 관련 확인 필요**: "정12각형"/"정20각형"은 평평한 다각기둥이 아니라 3D 정다면체(12/20면체)로 해석했습니다. `ball_` 그룹(구→다면체→타원체)이 "둥근 계열"로 보여서 이렇게 판단했는데, 만약 원래 의도가 `cylinder_3`처럼 평평한 다각기둥이었다면 `generate_new_objects.py`의 `make_ball_2`/`make_ball_3`만 다시 쓰면 됩니다.

## 초기 상태 랜덤화

- **물체 초기 위치**: ±1.5cm 랜덤 (`resetPositionNoise`)
- **물체 초기 회전(yaw)**: `useInitRandomRotation: True`로 켜져 있어 z축(회전축) 기준 [-π,π] 완전 랜덤. **단, roll/pitch는 랜덤화 안 됨** — 물체는 항상 손바닥 위에 "누운" 기본 자세를 유지한 채 z축으로만 랜덤 회전한 상태로 스폰됩니다. 완전 3D 임의 자세가 필요하면 `allegro_arm_morb_axis.py`의 `randomize_rotation(...)` 호출부(else 분기, `reset_idx` 안)를 직접 고쳐야 합니다.
- **손 관절 초기 자세**: 랜덤화 **안 됨** (매 에피소드 고정된 기본 자세로 리셋). `resetDofPosRandomInterval`/`resetDofVelRandomInterval` yaml 값은 죽은 설정이라 실제로 반영 안 됩니다.

## 활성 Domain Randomization 요약

물리(질량 0.2~0.6kg, 손/물체 마찰 각각 0.3~3.0, PD gain, 위치 ±1.5cm, 스케일 0.95~1.05) + 랜덤 외력(스케일 0.2, 확률 [0.2,0.25], 0.1초마다 0.99배 감쇠) + 센서(threshold [0.005,0.015]N, 드롭율 10%, lag 25%) + 관측/액션 노이즈(관절 ±0.05, 액션 ±0.06, relScale ±5%) + 물체 초기 yaw. 자세한 항목별 설명은 이전 대화(메모리에는 없고 이 세션 히스토리에 있음)에 정리되어 있으니, 필요하면 이 README보다 대화 로그를 참고하세요.

## 실행 방법 (GPU 필요, 이 PC에서는 미실행)

```bash
cd rotating-without-seeing
bash scripts/train_z_axis.sh 0                       # GPU 0번 사용, 본 학습 (numEnvs=8192)
bash scripts/train_z_axis.sh 0 task.env.numEnvs=64   # 빠른 스모크 테스트용 소규모 실행
```

## 지금까지 확인한 것 / 다음에 확인해야 할 것

**정적으로 확인 완료:**
- `isaacgymenvs/tasks/allegro_arm_morb_axis.py` 문법 검사(`py_compile`) 통과
- `cfg/task/AllegroArmMOAR.yaml`, `cfg/train/AllegroArmMOARPPO.yaml` yaml 파싱 통과
- 22개 물체 URDF가 참조하는 모든 mesh 파일 존재 확인, `asset_files_dict`/`object_sets["C"]` 키 일치 확인
- 신규 물체 6개의 mesh volume/bounding box 계산으로 형상·크기 검증 (전부 양의 부피 = 법선 방향 정상)
- `num_training_objects`/one-hot 벡터/`numStates`가 물체 개수(16→22)에 따라 동적으로 스케일됨을 코드로 확인 (하드코딩된 16 가정 없음)
- `ablation_mode` 관련 촉각-제거 버그 발견 및 수정

**GPU 환경에서 사람이 확인해야 할 것:**
1. `bash scripts/train_z_axis.sh 0 task.env.numEnvs=64`로 소규모 스모크 테스트 — 크래시 없이 기동하는지, 관측 차원이 실제로 340인지(276이 아닌지) 로그로 확인
2. 접촉 신호(`contacts`)가 실제로 0.01N 근방에서 반응하는지, 항상 0이거나 항상 1이 아닌지 확인
3. `numEnvs=8192`로 본 학습을 돌려 Cumulative Rotation Reward가 논문 Fig.6처럼 우상향하는지 확인
4. `block_3`(계단 모양)이 물리적으로 이상 없이(끼임, 관통 등) 시뮬레이션되는지 확인 — 2-파트 볼록 분해가 실제 IsaacGym 충돌 계산에서 문제없는지는 아직 실기동 검증 전
