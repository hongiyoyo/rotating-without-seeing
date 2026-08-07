# rotating-without-seeing

터치 전용(vision 절대 사용 안 함) z축 in-hand rotation teacher policy를 IsaacGym + PPO로 학습하기 위한, `in-hand-rotation`("Robot Synesthesia" 코드베이스)에서 필요한 부분만 추려낸 독립 프로젝트입니다.

이 문서는 이 코드를 처음 이어받는 사람/에이전트가 배경 설명 없이 바로 작업을 이어갈 수 있도록 지금까지의 작업 이력, 설계 근거, 그리고 **반드시 알아야 할 함정들**을 정리한 것입니다.

## 현재 상태 요약 (가장 먼저 읽을 것)

1. **본 학습(numEnvs=8192, 20100 epoch) 완료됨.** Best 체크포인트:
   `runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/z-axis-touch-only.pth`
   (rolling reward 974.54, epoch 19927 시점 저장 — 마지막 epoch 20100 체크포인트의 reward 870.41보다 높음). 근거·확인 방법은 [본 학습 결과](#본-학습-결과) 참고.
2. **GPU 없다는 이전 버전 README 기술은 틀렸습니다.** 이 Windows PC는 WSL2를 통해 실제 NVIDIA GPU에 접근 가능하고, IsaacGym·conda 환경이 이미 다 세팅되어 있습니다. [실행 환경](#실행-환경-wsl2) 섹션 필독 — 여기부터 안 읽으면 "GPU 없어서 못 함"이라고 잘못 판단하게 됩니다.
3. 이 best 체크포인트로 **물체별 촉각+관절위치 시계열을 수집해서 9-way CNN 분류기를 학습**하는 별도 하위 프로젝트를 완료했습니다 (정책 자체는 물체 정체성을 전혀 모른 채 회전만 함 — 그 결과로 나온 감각 신호만으로 어떤 물체인지 맞히는 실험). 결과: 촉각+관절 90.3%, 관절만 87.8%, 촉각만 82.7%. 전체 내용은 [촉각 기반 물체 분류](#촉각-기반-물체-분류) 참고.
4. `scripts/train_z_axis.sh`가 **CRLF 줄바꿈 때문에 WSL bash에서 실행 자체가 안 되던 버그**를 고쳤습니다 (LF로 정규화 완료). Windows 도구로 이 저장소의 `.sh` 파일을 다시 저장/편집하면 같은 문제가 재발할 수 있으니, 이상하게 스크립트가 안 돌면 `file <script>.sh`로 line ending부터 확인하세요.
5. 다음에 자연스럽게 이어갈 수 있는 작업(요청받은 적은 없지만 참고): x/y축으로 확장, 관절+촉각 외 다른 관측(예: prev_action) 채널 추가해서 분류 정확도 개선, block_3/cylinder_1처럼 특정 물체 쌍이 계속 헷갈리는 원인을 mesh 형상 비교로 더 파보기, distillation(학생 정책으로 압축) 등.

## 실행 환경 (WSL2)

**이 Windows 개발 PC에서 GPU/IsaacGym을 쓰려면 WSL2를 거쳐야 합니다.** 네이티브 Windows Python 환경에는 torch/isaacgym이 설치되어 있지 않고, IsaacGym Preview 4는 애초에 Linux 전용입니다.

- 배포판: **WSL2 Ubuntu-24.04** (`wsl.exe -e bash -lc "..."`로 명령 실행)
- conda 환경: **`robosyn`** (`/home/openfoam/miniconda3/envs/robosyn`) — Python 3.8.20, PyTorch 2.4.1+cu121, IsaacGym Preview 4(`/home/openfoam/IsaacGym_Preview_4_Package`), wandb, hydra-core, sklearn 등 필요한 패키지가 전부 이미 설치되어 있습니다. `install.md`는 원본 세팅 안내문이지만 **이 WSL 환경에는 이미 다 되어 있으니 새로 설치할 필요 없습니다.**
- GPU: NVIDIA GeForce RTX 4070 (16GB), WSL2를 통해 정상적으로 CUDA/PhysX GPU 파이프라인 접근 가능함을 실제 학습으로 확인함.
- 이 저장소는 Windows 경로 `C:\jaehong_yang\rotating-without-seeing`에 있고, WSL에서는 `/mnt/c/jaehong_yang/rotating-without-seeing`로 동일 파일에 접근합니다. 코드 편집은 Windows 쪽에서, 실행은 WSL 쪽에서 하면 됩니다.
- 실행 명령 패턴:
  ```bash
  wsl.exe -e bash -lc "cd /mnt/c/jaehong_yang/rotating-without-seeing && source ~/miniconda3/etc/profile.d/conda.sh && conda activate robosyn && <명령>"
  ```
- **GUI 뷰어도 됩니다.** WSLg가 설정되어 있어 `headless=False`로 띄우면 Windows 데스크톱에 실제 창(타이틀 "Isaac Gym (Ubuntu-24.04)", 프로세스는 `msrdc.exe`)이 뜹니다. 단, NVIDIA Vulkan ICD가 이 WSL 배포판에는 안 잡혀서(`/usr/share/vulkan/icd.d/`에 nvidia_icd.json 없음) 어떤 렌더링 경로를 타는지는 불확실하지만, 실제로 시도했을 때 정상적으로 창이 뜨고 GPU 사용률도 올라가며 잘 동작했습니다.
- 학습/체크포인트/wandb 로그인 등은 전부 이 WSL `robosyn` 사용자(`openfoam`) 계정 하에서 이미 인증되어 있습니다(`~/.netrc`에 wandb 키 존재).

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
    objects/*.urdf + meshes/set2/*.obj                       (자산 파일 17개 존재, 실제 학습(objSet C)에는 9개만 사용 — 아래 참고)
  scripts/train_z_axis.sh   (학습 실행 스크립트 — LF로 정규화됨, CRLF 재발 주의)
  runs/                 (gitignored. 학습 체크포인트/텐서보드 로그. best 체크포인트 위치는 위 "현재 상태 요약" 참고.
                          이 디렉터리는 이 워킹 디렉터리에만 존재 — 새로 clone하면 없음, 재현하려면 재학습 필요)
  wandb/                (gitignored. wandb 로컬 캐시/로그)
  tools/
    object_viewer.html        (물체들을 브라우저에서 3D로 확인하는 뷰어, mesh 데이터 내장)
    generate_new_objects.py   (cylinder_3 등 절차적으로 생성한 물체를 만든 스크립트 - 재현/참고용)
    regen_viewer_data.py      (object_viewer.html의 내장 mesh 데이터를 재생성하는 스크립트)
    relabel_objects_2.py      (물체 목록 2차 재정리에 쓴 1회성 스크립트 - 기록용)
    collect_tactile_data.py   (신규: best 체크포인트로 정책을 굴려 촉각+관절위치 시계열 수집)
    train_tactile_classifier.py (신규: 수집한 시계열로 9-way 물체 분류 CNN 학습)
    data/
      tactile_dataset.npz         (3,230 에피소드, (N,500,32) 패딩된 배열 — tactile 16ch + joint_pos 16ch)
      sample_traces.json          (대시보드용으로 뽑은 물체별 대표 샘플 1개씩)
      tactile_classifier_dashboard.html       (촉각+관절 결과 대시보드, 메인)
      joint_pos_classifier_dashboard.html     (관절위치만 ablation 대시보드)
      tactile_only_classifier_dashboard.html  (촉각만 ablation 대시보드)
  install.md            (원본 세팅 안내 — WSL robosyn 환경엔 이미 다 설치되어 있어 참고용)
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

새 yaml 키(`sensorThreshLow`, `objectFrictionLow/High`, `handFrictionLow/High`, `objectScaleLow/High`, `pGainMultLow/High`, `dGainMultLow/High`, `jointObsNoise`, `torqueRewardNorm`, `useForceProbRange`, `deterministicObjectAssign` — 마지막 것은 이번 세션에 추가, 아래 참고)로 위 값들을 조정 가능하게 만들어 뒀습니다.

## ⚠️ 반드시 알아야 할 함정들

### 1. `ablation_mode` (기존, 여전히 유효)

**`cfg/task/AllegroArmMOAR.yaml`의 `ablation_mode`를 절대 `"multi-modality"`나 `"no-tactile"`로 바꾸지 마세요.**

원본 코드에는 `ablation_mode in ["no-tactile", "multi-modality"]`일 때 `isaacgymenvs/tasks/base/vec_task.py`의 `step()`/`reset()`/`reset_done()`이 **매 스텝 관측 버퍼에서 16차원 촉각 신호를 통째로 잘라내는** 로직이 있고, 태스크 파일도 이에 맞춰 `numObservations`를 하드코딩된 276으로 덮어씁니다(`allegro_arm_morb_axis.py:382-383` 부근). 원본 yaml 기본값이 정확히 `ablation_mode: multi-modality`였기 때문에, 고치지 않았다면 **"vision을 안 쓰는 정책"이 아니라 "vision도 촉각도 다 못 보는 정책"이 학습될 뻔했습니다.**

지금은 `ablation_mode: no-pc`로 고쳐뒀고, 이 값은 코드 어디에서도 특별 취급되지 않는 "안전한 기본" 값입니다. 실제 학습 로그로 관측 차원이 340(276 아님)임을 재확인했습니다.

### 2. 스크립트 CRLF (신규)

`scripts/train_z_axis.sh`가 Windows 도구로 편집되어 CRLF 줄바꿈으로 저장되면서 `${array[@]:1:$len}` 라인이 WSL bash에서 `syntax error: invalid arithmetic operator`로 죽는 버그가 있었습니다. `sed -i 's/\r$//'`로 LF 정규화해서 고쳤습니다. **Windows 쪽에서 `.sh` 파일을 다시 저장하면 재발할 수 있으니**, 스크립트가 원인불명으로 실행 자체가 안 되면 `file <script>.sh`로 `CRLF line terminators` 여부부터 확인하세요.

### 3. 스모크 테스트 시 `minibatch_size`도 같이 줄여야 함 (신규)

`numEnvs`만 줄이고 (예: `task.env.numEnvs=64`) `minibatch_size`(기본 16384, 8192-env 본 학습용 고정값)를 그대로 두면 `batch_size`(numEnvs×horizon_length)가 `minibatch_size`로 나누어떨어지지 않아 `AssertionError: assert(self.batch_size % self.minibatch_size == 0)`로 학습 시작 전에 즉시 죽습니다. 반드시 같이 낮추세요:
```bash
bash scripts/train_z_axis.sh 0 task.env.numEnvs=64 \
  train.params.config.minibatch_size=512 \
  train.params.config.central_value_config.minibatch_size=512
```

### 4. `deterministicObjectAssign` (신규 기능, 함정 아님이지만 알아둘 것)

물체 배정은 원래 env 생성 시(`_create_envs`) 딱 한 번 랜덤으로 뽑히고 `reset_idx`에서 절대 안 바뀝니다(즉 같은 env 슬롯은 프로세스 수명 내내 같은 물체). 시각화/데이터 수집처럼 물체가 골고루 나오길 원할 때를 위해 `task.env.deterministicObjectAssign=True`를 주면 `obj_class_indice = i % len(used_training_objects)`로 라운드로빈 배정됩니다(`allegro_arm_morb_axis.py`, `_create_envs` 안, `obj_class_indice` 대입부). 기본값 `False`라 본 학습(8192 env)에는 전혀 영향 없습니다.

### 5. `rl_games`의 `PpoPlayerContinuousCollect` (player_collect=True)는 쓰지 말 것 (신규)

이 저장소 로컬 `rl_games` 포크에 반쯤 만들어진 데이터 수집용 플레이어(`rl_games/algos_torch/players.py:338`, `train.params.config.player_collect: True`로 활성화)가 있는데, 기본 설정(`debug_viz`/`force_debug` 꺼짐)에서 `get_env_internal_info(env, 'qpos'/'target')`가 빈 리스트에 `.detach()`를 호출해 크래시합니다. 게다가 전체 rollout을 다 메모리에 쌓은 뒤 처리해서 numEnvs가 크면 비효율적입니다. **`tools/collect_tactile_data.py`는 이걸 안 쓰고 직접 커스텀 롤아웃 루프를 짰습니다** — 촉각 데이터 수집 외 다른 용도로도 이 패턴(아래)을 재사용하면 됩니다.

### 6. rl_games player로 커스텀 롤아웃 루프를 짤 때: `get_batch_size()` 빼먹지 말 것 (신규)

`player.run()`을 안 쓰고 직접 `player.get_action(obs, ...)` / `player.env_step(...)`를 부르는 코드를 짜면, `player.run()` 내부에서만 호출되는 `self.get_batch_size(obses, batch_size)`가 실행이 안 되어 `has_batch_dimension`이 `False`로 남습니다. 그러면 `get_action`이 `(num_envs, obs_dim)` 텐서를 배치 없는 단일 관측으로 오인해서 엉뚱하게 `unsqueeze`해버리고, 네트워크 forward에서 `mat1 and mat2 shapes cannot be multiplied` 같은 에러가 납니다. `env_reset` 직후 반드시:
```python
obses = player.env_reset(env)
player.get_batch_size(obses, 1)   # 이 줄이 없으면 위 에러 발생
```

## 학습용 물체 (9개, `objSet: "C"`)

세 차례 리네이밍/정리를 거쳤습니다. 전부 `assets/urdf/objects/*.urdf` + `assets/urdf/objects/meshes/set2/*.obj`에 있고, `isaacgymenvs/tasks/allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets["C"]`에 등록되어 있습니다. 모든 물체는 원본 mesh 기준 bounding box ~2.0 unit(= URDF `scale=".03 .03 .03"` 적용 시 실제 약 6cm)로 동일한 크기 규격입니다.

**1차 정리**: 원본 16개 물체를 리네이밍(`block_1~2,4`, `cylinder_1~2,4`, `else_1~10`)하고, 절차적으로 생성한 새 물체 6개(`block_3`=계단형 L자 블록, `cylinder_3`=정십각기둥, `ball_1~4`=구/정12면체/정20면체/타원체)를 추가해 총 22개.

**2차 정리**: GPU 학습 결과 `block_3`(계단형)과 `ball_1~4` 전부 reward가 확연히 낮게 나와, 이 5개 물체를 학습 세트에서 완전히 제거(mesh/urdf 파일까지 삭제)했습니다. 대신 `else_1`을 새 `block_3`으로, `else_5`를 새 `block_5`로 승격(둘 다 평범한 블록 형태, 계단형 아님)하고, 나머지 `else_2,3,4,6,7,8,9,10`을 `else_1~8`로 재정렬해 총 17개.

**3차 정리 (현재 상태)**: "학습한 물체에서만 잘 되면 충분하다"는 목표로 범용성을 포기하고, **`else_1~8`을 학습 대상(`object_sets["C"]`)에서 제외**했습니다. `else_*` 자산 파일 자체는 삭제하지 않고 `asset_files_dict`에는 여전히 남아있지만(나중에 다시 필요하면 재사용 가능), `object_sets["C"]`에는 더 이상 포함되지 않아 학습에 쓰이지 않습니다. 최종 학습 세트는 아래 9개뿐입니다.

| 라벨 | 형상 | 비고 |
|---|---|---|
| `block_1` | 정육면체 | 원래 1번 |
| `block_2` | 불규칙 블록("time") | 원래 15번 |
| `block_3` | 블록(평범한 형태) | 옛 `else_1` 승격 — **계단형이 아님** (계단형 물체는 삭제됨) |
| `block_4` | 모서리 깎인 블록 | 원래 6번 |
| `block_5` | 블록(평범한 형태) | 옛 `else_5` 승격 |
| `cylinder_1` | 원기둥 | 원래 11번 |
| `cylinder_2` | 축 방향 압축 원기둥 | 원래 16번 |
| `cylinder_3` | **정십각기둥** | 절차적 생성 |
| `cylinder_4` | 모서리 깎인 원기둥 | 원래 12번 |

**학습에서 빠졌지만 자산은 남아있는 물체**: `else_1~8` — `asset_files_dict`엔 있지만 `object_sets["C"]`엔 없음. 다시 학습에 포함시키려면 `object_sets["C"]` 리스트에 이름만 추가하면 됩니다(파일은 이미 존재).

**삭제되어 완전히 사라진 물체**: 계단형 `block_3`(구형), `ball_1`(구)/`ball_2`(정12면체)/`ball_3`(정20면체)/`ball_4`(타원체) — mesh/urdf 파일 자체를 삭제했습니다.

**`ball` 키는 이 9개와 무관**하니 헷갈리지 마세요 — baoding balls 과제 전용의 별도 단일 구 오브젝트입니다(건드리지 않음).

## 초기 상태 랜덤화

- **물체 초기 위치**: ±1.5cm 랜덤 (`resetPositionNoise`)
- **물체 초기 회전(yaw)**: `useInitRandomRotation: True`로 켜져 있어 z축(회전축) 기준 [-π,π] 완전 랜덤. **단, roll/pitch는 랜덤화 안 됨**.
- **손 관절 초기 자세**: 랜덤화 **안 됨** (매 에피소드 고정된 기본 자세로 리셋).

## 활성 Domain Randomization 요약

물리(질량 0.2~0.6kg, 손/물체 마찰 각각 0.3~3.0, PD gain, 위치 ±1.5cm, 스케일 0.95~1.05) + 랜덤 외력(스케일 0.2, 확률 [0.2,0.25], 0.1초마다 0.99배 감쇠) + 센서(threshold [0.005,0.015]N, 드롭율 10%, lag 25%) + 관측/액션 노이즈(관절 ±0.05, 액션 ±0.06, relScale ±5%) + 물체 초기 yaw.

## wandb 물체별 reward 로깅

이미 구현되어 있습니다 (이번 세션 이전부터):
- `allegro_arm_morb_axis.py`: `episode_reward_buf`/`episode_fwd_theta_buf`(env별 누적치), `_pending_obj_stats`(에피소드 종료 시 물체 이름별 기록)
- `isaacgymenvs/utils/rlgames_utils.py`의 `RLGPUAlgoObserver.after_print_stats`: 매 PPO epoch마다 `_pending_obj_stats`를 집계해 `PerObject/{obj}/rotation_count|episode_length|episode_reward`를 tensorboard writer로 기록 → `train.py`의 `wandb.init(sync_tensorboard=True)` 덕분에 별도 `wandb.log()` 없이 자동으로 wandb에도 반영됨. 콘솔에도 `[PerObject reward] epoch N | block_1: ... | ...` 형태로 출력.
- 이번 세션에 고친 것: 이 로깅 블록을 감싸던 `except Exception: pass`가 에러를 완전히 은폐하던 걸, 최초 1회만 경고를 출력하도록 수정(`self._per_object_log_warned` 플래그). 학습 자체를 죽이지 않기 위해 try/except 자체는 유지.

## 본 학습 결과

`bash scripts/train_z_axis.sh 0` (numEnvs=8192, minibatch_size=16384, 논문 스펙 고정값)로 **20,100 epoch까지 정상 완주**했습니다.

- wandb: `isaacgymenvs` 프로젝트, run `rws2_2026-08-03_12-09-19` (`https://wandb.ai/jhyang321-seoul-national-university/isaacgymenvs/runs/b5ic6zpy`)
- 체크포인트 디렉터리: `runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/`
  - `last_z-axis-touch-only_ep_<N>_rew_<R>.pth` — 100 epoch마다 저장되는 스냅샷
  - **`z-axis-touch-only.pth`** — rl_games가 rolling mean reward 갱신될 때마다 덮어쓰는 **best** 체크포인트. 마지막 갱신은 epoch 19927, reward **974.54**로, 100-epoch 단위 스냅샷 중 최고치(902.71, epoch 15500)나 마지막 체크포인트(870.41, epoch 20100)보다도 높습니다. **평균 reward 기준으로는 이 파일을 써야 합니다.**
- **"평균 reward가 가장 높은 policy"를 원하면 위 `z-axis-touch-only.pth`를 쓰세요.** "최솟값이 가장 높은" 기준(물체별 최악 케이스가 그나마 나은 policy)은 별도로 뽑지 않았습니다 — 물체마다 reward 스케일 자체가 달라(원기둥류가 원래 더 높음) 최솟값 기준은 오해를 부를 수 있다는 논의 후 평균 기준으로 확정했습니다.
- `runs/`는 gitignored라 이 워킹 디렉터리에만 존재합니다. **재현하려면 재학습(수일 소요)이 필요하니, 이 체크포인트 파일이 필요하면 별도로 백업해두는 걸 권합니다.**

### 학습된 policy 시각화

```bash
wsl.exe -e bash -lc "cd /mnt/c/jaehong_yang/rotating-without-seeing && source ~/miniconda3/etc/profile.d/conda.sh && conda activate robosyn && python isaacgymenvs/train.py test=True headless=False task.env.numEnvs=9 task.env.deterministicObjectAssign=True checkpoint='runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/z-axis-touch-only.pth'"
```
`numEnvs=9` + `deterministicObjectAssign=True`로 9개 env에 물체가 하나씩 순서대로(block_1~5, cylinder_1~4) 배정되어, 9개 물체가 동시에 회전하는 걸 한 창에서 볼 수 있습니다. Windows 데스크톱에 "Isaac Gym (Ubuntu-24.04)" 창이 뜹니다 (위 [실행 환경](#실행-환경-wsl2) 참고).

## 촉각 기반 물체 분류

**목표**: 위 best 체크포인트로 각 물체를 회전시키면서 policy가 실제로 받는 센서값(정책 자신은 물체 정체성을 절대 모름 — `object_one_hot_vector`는 critic 전용 `states_buf`에만 쓰이고 actor 관측 `last_obs_buf`/`obs_buf`에는 안 들어간다는 걸 코드로 확인함)의 시계열만으로, 9개 물체 중 무엇인지 맞히는 별도 CNN 분류기를 학습. 형상 재구성 아님, 분류만.

### 파이프라인

1. **`tools/collect_tactile_data.py`** — best 체크포인트로 결정론적(`is_deterministic=True`) 추론 롤아웃을 돌리면서, 매 control step `sensed_contacts`(촉각 16채널, policy가 실제로 보는 노이즈/threshold/lag 적용된 값)와 `last_obs_buf[:, 6:22]`(관절위치 16채널)를 읽어 env별로 버퍼링하다가, `reset_buf[e]==1`(에피소드 종료) 시점에 완성된 (T, 32) 시퀀스를 물체 라벨(`object_class_indices[e]`, env 생성 시 고정)과 함께 저장.
   - `task.env.deterministicObjectAssign=True`로 물체를 라운드로빈 배정(위 함정 #4) → 클래스 균형 보장.
   - 기본값: `numEnvs=900`(클래스당 100), `target_episodes_per_class=300`, `max_control_steps=2000`(안전 상한).
   - 결과: `tools/data/tactile_dataset.npz` — `tactile: (N, 500, 32) float32`(500=`max_episode_length`, 부족한 길이는 뒤쪽 0-padding), `lengths: (N,)`, `labels: (N,)`, `label_names`, `channels`.
   - 실행 예:
     ```bash
     python tools/collect_tactile_data.py \
       checkpoint='runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/z-axis-touch-only.pth' \
       task.env.numEnvs=900 +target_episodes_per_class=300 +max_control_steps=2000 \
       +out_path=tools/data/tactile_dataset.npz
     ```
   - 실제 수집 결과: **3,230 에피소드**, 클래스당 310~444개(불균형은 조기종료 에피소드가 섞여서; 목표 300 도달 후 정지하다 보니 다른 클래스는 그 시점까지 더 쌓인 것). 에피소드 길이는 물체마다 편차 큼(block_3/block_5가 짧은 편 — 기존 학습 reward 패턴과 일치).

2. **`tools/train_tactile_classifier.py`** — 위 npz를 로드해 라벨 층화 70/15/15 분할(seed 고정이라 재실행해도 같은 분할) 후, masked-average-pooling 1D-CNN(Conv1d 32→64→128, 커널 5/5/3, 전부 `padding="same"`이라 시간축 안 줄어듦 → 패딩 구간을 정확히 제외한 마스크 평균 풀링 가능) + FC(128→64→9)로 50 epoch 학습.
   - `--channels` 옵션으로 저장된 채널 중 일부만 골라 학습 가능 (예: `--channels joint_pos`, `--channels tactile`, 기본값은 파일에 저장된 전체). `CHANNEL_WIDTHS = {"tactile": 16, "joint_pos": 16}`가 `collect_tactile_data.py`와 동일해야 함 (import는 안 함 — `collect_tactile_data.py`가 `import isaacgym`을 torch보다 먼저 해야 하는 제약이 있어서, 순수 학습 스크립트에 그 무거운 의존성을 끌어오지 않으려고 일부러 값만 복제해뒀습니다).
   - 실행 예: `python tools/train_tactile_classifier.py --data tools/data/tactile_dataset.npz --epochs 50`

### 결과 (3가지 채널 조합으로 각각 학습 — 동일 데이터/분할/시드, 채널만 다름)

| 입력 채널 | 테스트 정확도 | 가장 약한 물체 |
|---|---|---|
| 촉각 16채널만 (`--channels tactile`) | **82.7%** | cylinder_1 61.7% (cylinder_3와 혼동) |
| 관절위치 16채널만 (`--channels joint_pos`) | **87.8%** | cylinder_1 68.1%, cylinder_3 65.2% |
| 촉각+관절 32채널 (기본값) | **90.3%** | block_3 74.6% (block_5와 혼동) |

(랜덤 베이스라인 = 1/9 = 11.1%.)

**핵심 발견**: 정책 자체는 촉각 전용으로 학습됐는데도, **촉각 단독이 세 조합 중 가장 낮은 분류 정확도**를 냄 — 관절위치(손가락이 어떤 자세로 안착하는지)가 형상 정보를 더 많이 담고 있었습니다. 게다가 세 ablation이 **서로 다른 물체 쌍에서 실패**합니다: 촉각+관절에서 헷갈리는 block_3/block_5 쌍은 관절만/촉각만 버전에서는 오히려 잘 맞히고, 반대로 촉각만·관절만 둘 다에서 헷갈리는 cylinder_1/cylinder_3 쌍은 둘을 합치면 잘 구분됩니다. 즉 촉각과 관절위치는 서로의 부분집합이 아니라 **상호보완적인, 겹치지 않는 형상 단서**를 담고 있다는 뜻입니다.

각 ablation의 confusion matrix, 학습 곡선, 물체별 원본 센서 시계열(실제 히트맵)은 `tools/data/*.html` 3개 파일에 그대로 들어있습니다 (Claude Artifact로도 게시했지만 그 URL은 이 대화 세션·사용자 계정에 종속적이라 다른 환경에서는 안 열릴 수 있음 — **재현 가능한 원본은 이 HTML 파일들과 위 두 스크립트입니다**). 브라우저로 그냥 열면 됩니다, 별도 서버 필요 없음.

## 지금까지 확인한 것 / 다음에 확인해야 할 것

**확인 완료 (이번 세션, 실제 GPU 실행으로):**
- `numEnvs=64` 스모크 테스트 → 관측 차원 340(276 아님) 확인, `ablation_mode` 수정이 실제로 반영됨을 재확인
- `numEnvs=8192` 본 학습 20,100 epoch 완주, wandb에 물체별 reward 정상 기록
- 접촉 신호가 항상 0/1로 고정되지 않고 실제로 반응함 (물체별 reward가 유의미하게 갈리는 것으로 간접 확인 — block_3/block_5가 낮고 cylinder류가 높은 일관된 패턴)
- 9개 물체 간 reward 편차: cylinder류(1000~1300대)가 block류(400~1200대)보다 대체로 높음, block_3/block_5가 특히 낮은 편 — 학습 로그와 촉각 분류 confusion 패턴 둘 다에서 일관되게 나타남
- 학습된 policy로 9개 물체 동시 시각화 (뷰어 창 정상 동작)
- 촉각+관절/관절만/촉각만 3가지 CNN 분류기 학습 및 confusion matrix 분석 완료

**아직 안 한 것 (다음 에이전트가 이어갈 수 있는 지점):**
- 논문 Fig.6과 정량적으로 비교하는 학습 곡선 분석은 안 함 (wandb 대시보드에서 직접 확인 가능)
- else_1~8, 삭제된 계단형 block_3/ball_1~4를 재학습에 포함시켰을 때 결과가 어떻게 달라지는지는 미확인
- 분류기 정확도를 precision/F1까지 확장하거나, prev_action 등 다른 관측 채널을 추가했을 때 개선되는지는 미확인
- x축/y축 회전이나 distillation(학생 정책) 등 원 논문의 다른 부분은 이 폴더 범위 밖 (README 최상단 "배경" 참고)
