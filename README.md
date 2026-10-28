# rotating-without-seeing

터치 전용(vision 절대 사용 안 함) z축 in-hand rotation teacher policy를 IsaacGym + PPO로 학습하기 위한, `in-hand-rotation`("Robot Synesthesia" 코드베이스)에서 필요한 부분만 추려낸 독립 프로젝트입니다.

이 문서는 이 코드를 처음 이어받는 사람/에이전트가 배경 설명 없이 바로 작업을 이어갈 수 있도록 작업 이력, 설계 근거, 그리고 **반드시 알아야 할 함정들**을 정리한 것입니다.

## 현재 상태 요약 (가장 먼저 읽을 것, 2026-10-28 기준)

1. **개발 PC가 2026-10-23에 포맷되었습니다.** 새 공용 PC(`C:\YangJaehong`)에 저장소를 다시 clone하고 WSL2 환경을 처음부터 다시 설치했습니다([실행 환경](#실행-환경-wsl2)). gitignored였던 `runs/`(체크포인트), `wandb/`, `tools/data/`는 모두 사라졌습니다. **학습된 체크포인트는 하나도 남아 있지 않으므로, policy가 필요하면 재학습해야 합니다.**
2. **9/29 이후 작업 중 push되지 않았던 커밋(`025d2e7`)은 사라졌고**, 기억과 wandb 기록을 바탕으로 복원해 `e1adb7f`로 push했습니다. 복원한 내용:
   - 학습 물체를 다시 **9개(`block_1~5`, `cylinder_1~4`, `objSet: "C"`)** 로 되돌림 — [학습용 물체](#학습용-물체-9개-objset-c)
   - **물체 초기 위치/yaw 랜덤화가 실제로는 시뮬레이터에 반영되지 않던 버그 수정** — [함정 #7](#7-reset-중-set_actor_rigid_body_properties가-물체-위치자세-쓰기를-지움)
   - 초기 위치 랜덤화 범위 지정(x, y ±1.5cm / z 0~1cm / yaw ±π) — [초기 상태 랜덤화](#초기-상태-랜덤화)
   - 점진적 DR(DR ramp) 기본값을 **끔**
   - 평가 스크립트에서 길이 0인 에피소드 제외
3. **README는 복원 시점(10/23)의 코드 기준입니다.** 10/20 학습 당시 코드와 완전히 같다는 보장은 없습니다. 알려진 차이는 [학습 기록](#학습-기록-wandb)에 정리했습니다.
4. **2026-10-28 스모크 테스트로 위 복원 내용이 실제 시뮬레이션에 반영되는 것을 확인했습니다** — [스모크 테스트 결과](#스모크-테스트-결과-2026-10-28). 같은 날 매 step 물체 질량을 env마다 다시 읽던 코드를 캐시(`object_env_masses`)로 바꿨습니다(결과 동일, 속도 차이는 거의 없음).
5. 바로 다음 할 일: 본 학습(`bash scripts/train_z_axis.sh 0`)으로 체크포인트를 확보하는 것(2026-10-28 시작, [학습 기록](#학습-기록-wandb) 참고). 그 뒤 [물체별 회전 성능 평가](#물체별-회전-성능-평가)와 [촉각 기반 물체 분류](#촉각-기반-물체-분류)를 새 체크포인트로 다시 돌릴 수 있습니다.

## 실행 환경 (WSL2)

**GPU/IsaacGym은 WSL2 안에서만 씁니다.** 네이티브 Windows에는 Python/torch가 없고, IsaacGym Preview 4는 Linux 전용입니다. 코드 편집은 Windows 쪽에서, 실행은 WSL 쪽에서 합니다.

- PC: Windows 11, NVIDIA GeForce RTX 4070 Ti SUPER (16GB), 공용 PC
- 저장소: Windows `C:\YangJaehong\rotating-without-seeing` = WSL `/mnt/c/YangJaehong/rotating-without-seeing`
- 배포판: **WSL2 Ubuntu-24.04**, Linux 사용자 `jaehong`
- conda 환경: **`robosyn`** (`~/miniconda3/envs/robosyn`) — Python 3.8, PyTorch 2.4.1+cu121, pytorch3d 0.7.8, IsaacGym Preview 4(`~/IsaacGym_Preview_4_Package`), numpy 1.20.3, pillow 9.5.0, hydra-core, gym, ray, open3d, tensorboard(X), wandb, scikit-learn
- wandb: `jaehong` 계정에 로그인되어 있음(`~/.netrc`)

### 실행 방법

PowerShell에서 `wsl ... bash -lc "..."`로 긴 명령을 넘기면 따옴표/괄호가 깨지기 쉽습니다. **명령을 `.sh` 파일로 써서 실행하는 쪽이 안전합니다.**

```bash
# run.sh (LF 줄바꿈으로 저장)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
cd /mnt/c/YangJaehong/rotating-without-seeing
bash scripts/train_z_axis.sh 0
```
```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/path/to/run.sh
```

**GUI 뷰어**: WSLg 덕분에 `headless=False`로 띄우면 Windows 데스크톱에 "Isaac Gym (Ubuntu-24.04)" 창이 뜹니다(옛 PC에서 확인, 새 PC에서는 아직 미확인).

### 환경을 처음부터 다시 만들 때 (PC가 또 포맷된 경우)

1. 관리자 PowerShell: `wsl --install` 후 재부팅, 이어서 `wsl --install -d Ubuntu-24.04`
2. NVIDIA에서 IsaacGym Preview 4 tar를 받아 둔다(로그인 필요).
3. WSL 안에서:
   ```bash
   sudo apt-get install -y build-essential git wget libvulkan1
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && bash /tmp/mc.sh -b -p ~/miniconda3
   source ~/miniconda3/etc/profile.d/conda.sh
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   conda create -y -n robosyn python=3.8 && conda activate robosyn
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/ld_path.sh
   conda deactivate && conda activate robosyn
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   pip install fvcore iopath
   pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"   # 미리 빌드된 wheel이 없어 소스 빌드 (수 분)
   pip install scipy imageio ninja
   mkdir -p ~/IsaacGym_Preview_4_Package && tar -xf /mnt/c/<경로>/IsaacGym_Preview_4_Package.tar -C ~/IsaacGym_Preview_4_Package
   pip install -e ~/IsaacGym_Preview_4_Package/isaacgym/python --no-deps
   pip install hydra-core gym ray open3d numpy==1.20.3 tensorboardX tensorboard wandb scikit-learn
   pip install pillow==9.5.0
   wandb login
   ```

**설치 시 함정:**
- **`LD_LIBRARY_PATH`에 `/usr/lib/wsl/lib`가 없으면** PhysX가 `libcuda.so`를 못 찾고(`PxPhysXGpuModuleLoader ... internal error : libcuda.so!`) CPU로 넘어간 뒤 `Must enable GPU pipeline to use state tensors` → `CUDA error: an illegal memory access`로 죽습니다. 위 activate.d 스크립트가 이걸 해결합니다.
- 최신 Pillow(10.x)는 `numpy==1.20.3`에 없는 `numpy.typing.NDArray`를 써서 open3d import가 실패합니다 → `pillow==9.5.0` 고정.
- `*** Warning: failed to preload CUDA lib / PhysX libs`는 무시해도 됩니다. 실제로 GPU PhysX가 잡히는지는 로그의 `+++ Using GPU PhysX`로 확인하세요.
- `install.md`는 원본 저장소의 설치 안내문(참고용)입니다. 위 순서가 실제로 검증된 방법입니다.

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
    objects/{block_1..5,cylinder_1..4}.urdf + meshes/set2/*.obj  (학습 물체 9개, 아래 참고)
  scripts/train_z_axis.sh   (학습 실행 스크립트 — LF 줄바꿈 유지, CRLF 재발 주의)
  runs/                 (gitignored. 학습 체크포인트/텐서보드 로그. 새로 clone하면 없음)
  wandb/                (gitignored. wandb 로컬 캐시/로그)
  tools/
    object_viewer.html        (9개 물체를 브라우저에서 3D로 확인하는 뷰어, mesh 데이터 내장)
    regen_viewer_data.py      (object_viewer.html의 내장 mesh 데이터를 재생성. tools/ 안에서 실행)
    generate_new_objects.py   (과거 cylinder_3(정십각기둥)/계단형 block_3/ball_1~4 등을 만든 스크립트 - 히스토리용)
    relabel_objects_2.py      (과거 물체 재정리에 쓴 1회성 스크립트 - 히스토리용, 재실행 불가)
    collect_tactile_data.py   (체크포인트로 정책을 굴려 촉각+관절위치 시계열 수집)
    train_tactile_classifier.py (수집한 시계열로 N-way 물체 분류 CNN 학습)
    evaluate_policy_per_object.py (체크포인트의 물체별 회전 성능 평가 — reward/회전속도/완주율)
    data/                     (gitignored. 위 스크립트들의 산출물. 포맷으로 전부 사라짐)
  install.md            (원본 세팅 안내 — 참고용)
```

## 논문 스펙 대비 수정한 항목 (파일: `isaacgymenvs/tasks/allegro_arm_morb_axis.py`, `cfg/task/AllegroArmMOAR.yaml`)

원본 `in-hand-rotation`의 `partial_stack` 구현은 논문의 대부분(16개 FSR 센서 배치, finite-difference 회전각 보상, EMA 액션 스무딩 η=0.8, 10Hz 제어, 4-step 히스토리 스택, PPO 하이퍼파라미터)을 이미 정확히 따르고 있었지만, 아래 수치/공식들은 이후 다른 실험(Robot Synesthesia)에 맞춰 바뀐 상태였고, 이 폴더에서 논문 값으로 되돌렸습니다:

| 항목 | 원본 코드 | 이 폴더 |
|---|---|---|
| 접촉 센서 threshold | 랜덤 [1.0, 2.0]N | 랜덤 [0.005, 0.015]N (논문 0.01N 근사) |
| PD gain 랜덤화 | P×[0.30,0.60], D×[0.75,1.05] | P×[0.66,1.33], D×[0.80,1.20] |
| 손/물체 마찰 | 공유된 draw 1개, [0.2,3.0] | 독립 draw 2개, 둘 다 [0.3,3.0] |
| 물체 스케일 | [0.95,1.1] | [0.95,1.05] |
| 초기 위치 노이즈 | ±1.0cm | x, y ±1.5cm, z 0~1cm |
| 랜덤 외력 스케일 | 2.0 | 0.2 |
| 랜덤 외력 확률 | 죽은 코드(고정 0.25만 사용) | env별 log-uniform [0.2,0.25] 실제 사용 |
| 센서 lag 확률 | 0.2 | 0.25 |
| 관절 관측 노이즈 | ±0.06 | ±0.05 |
| 액션 노이즈 | Gaussian σ=0.04 | Uniform ±0.06 |
| Torque 보상 | `-Σ τ²` (제곱) | `-‖τ‖` (L2 norm, 논문과 일치) |
| numEnvs / minibatch | 16 / 32 (yaml 기본값) | 8192 / 16384 (논문 값 고정) |

새 yaml 키(`sensorThreshLow`, `objectFrictionLow/High`, `handFrictionLow/High`, `objectScaleLow/High`, `pGainMultLow/High`, `dGainMultLow/High`, `jointObsNoise`, `torqueRewardNorm`, `useForceProbRange`, `deterministicObjectAssign`, `objInitPosNoiseXY`, `objInitPosZLow/High`, `drRamp*`)로 위 값들을 조정할 수 있습니다.

## ⚠️ 반드시 알아야 할 함정들

### 1. `ablation_mode`

**`cfg/task/AllegroArmMOAR.yaml`의 `ablation_mode`를 절대 `"multi-modality"`나 `"no-tactile"`로 바꾸지 마세요.**

원본 코드에는 `ablation_mode in ["no-tactile", "multi-modality"]`일 때 `isaacgymenvs/tasks/base/vec_task.py`의 `step()`/`reset()`/`reset_done()`이 **매 스텝 관측 버퍼에서 16차원 촉각 신호를 통째로 잘라내는** 로직이 있고, 태스크 파일도 이에 맞춰 `numObservations`를 하드코딩된 276으로 덮어씁니다. 원본 yaml 기본값이 정확히 `ablation_mode: multi-modality`였기 때문에, 고치지 않았다면 **"vision을 안 쓰는 정책"이 아니라 "vision도 촉각도 다 못 보는 정책"이 학습될 뻔했습니다.**

지금은 `ablation_mode: no-pc`로 고쳐뒀고, 이 값은 코드 어디에서도 특별 취급되지 않는 "안전한 기본" 값입니다. 관측 차원이 340(276 아님)인지로 확인할 수 있습니다.

### 2. 스크립트 CRLF

`scripts/train_z_axis.sh`가 CRLF 줄바꿈으로 저장되면 `${array[@]:1:$len}` 라인이 WSL bash에서 `syntax error: invalid arithmetic operator`로 죽습니다. **Windows 쪽에서 `.sh` 파일을 다시 저장하면 재발할 수 있으니**, 스크립트가 원인불명으로 안 돌면 `file <script>.sh`로 `CRLF line terminators` 여부부터 확인하고 `sed -i 's/\r$//' <script>.sh`로 고치세요.

### 3. 스모크 테스트 시 `minibatch_size`도 같이 줄여야 함

`numEnvs`만 줄이면 `batch_size`(numEnvs×horizon_length)가 `minibatch_size`(16384)로 나누어떨어지지 않아 `AssertionError`로 죽고, critic 쪽 `central_value_config.minibatch_size`를 그대로 두면 `ZeroDivisionError`로 죽습니다. 반드시 둘 다 낮추세요:
```bash
bash scripts/train_z_axis.sh 0 task.env.numEnvs=64 \
  train.params.config.minibatch_size=512 \
  train.params.config.central_value_config.minibatch_size=512 \
  max_iterations=3 wandb_activate=False
```

### 4. `deterministicObjectAssign` (함정은 아니지만 알아둘 것)

물체 배정은 env 생성 시(`_create_envs`) 딱 한 번 랜덤으로 뽑히고 `reset_idx`에서 안 바뀝니다(같은 env 슬롯은 프로세스 수명 내내 같은 물체). 시각화/데이터 수집처럼 물체가 골고루 나오길 원할 때는 `task.env.deterministicObjectAssign=True`로 라운드로빈(`i % len(used_training_objects)`) 배정합니다. 기본값 `False`라 본 학습에는 영향 없습니다.

### 5. `rl_games`의 `PpoPlayerContinuousCollect` (player_collect=True)는 쓰지 말 것

로컬 `rl_games` 포크의 반쯤 만들어진 데이터 수집용 플레이어(`train.params.config.player_collect: True`)는 기본 설정에서 `get_env_internal_info(env, 'qpos'/'target')`가 빈 리스트에 `.detach()`를 호출해 크래시합니다. **`tools/collect_tactile_data.py`/`evaluate_policy_per_object.py`처럼 직접 롤아웃 루프를 짜세요.**

### 6. rl_games player로 커스텀 롤아웃 루프를 짤 때: `get_batch_size()` 빼먹지 말 것

`player.run()`을 안 쓰고 직접 `player.get_action()`/`player.env_step()`을 부르면 `has_batch_dimension`이 `False`로 남아 `mat1 and mat2 shapes cannot be multiplied` 에러가 납니다. `env_reset` 직후 반드시:
```python
obses = player.env_reset(env)
player.get_batch_size(obses, 1)
```

### 7. reset 중 `set_actor_rigid_body_properties`가 물체 위치/자세 쓰기를 지움

**GPU 파이프라인에서 `set_actor_rigid_body_properties()`를 호출하면, 같은 step에 `set_actor_root_state_tensor_indexed()`로 쓴 actor의 root state(위치/자세)가 무시됩니다** (호출 순서 무관, 2026-10-23 직접 실험으로 확인. `set_actor_rigid_shape_properties()`(마찰)는 문제 없음).

예전 `reset_idx`는 매 reset마다 물체 질량을 이 함수로 다시 뽑았기 때문에, **물체 초기 위치/yaw 랜덤화가 코드상으로는 있었지만 실제 시뮬레이션에는 한 번도 반영되지 않았습니다**(물체가 매번 같은 자세에서 시작). 지금은 reset 중 질량 재설정을 제거했고, 질량은 env 생성 시 env별로 한 번만 랜덤화됩니다. **`reset_idx`에 rigid body property 설정을 다시 넣지 마세요.** 랜덤 외력 계산(`pre_physics_step`)은 env 생성 직후 읽어 둔 `self.object_env_masses`를 씁니다. 질량을 학습 중에 바꾸는 기능을 다시 넣는다면 이 캐시도 같이 갱신해야 합니다.

vec_task 쪽 DR(`randomization_params.actor_params`)이 reset 중 손 관절 속성(`dof_properties`)을 다시 뽑는 경우는 이 문제가 없습니다(2026-10-28 스모크 테스트로 확인).

검증 방법: reset 직후 `root_state_tensor`에 써 넣은 yaw와, `gym.simulate()` 한 번 뒤 `refresh_actor_root_state_tensor()`로 읽은 yaw를 비교. 정상이면 차이가 거의 0(중앙값 ~0.005rad)이고, 버그가 있으면 시뮬레이터 쪽 yaw가 전부 0 근처로 돌아가 있습니다.

### 8. 체크포인트 경로의 대괄호(`rew[7.25]`)는 Hydra가 문법으로 읽음

rl_games 체크포인트 이름에는 `last_xxxep110rew[7.25].pth`처럼 대괄호가 들어갑니다. `checkpoint=runs/.../rew[7.25].pth`로 넘기면 Hydra가 `mismatched input '[' expecting <EOF>`로 죽습니다. **bash가 벗겨내지 않도록 작은따옴표를 큰따옴표로 한 번 더 감싸서** Hydra까지 따옴표가 전달되게 하세요: `checkpoint="'runs/.../last_xxxep110rew[7.25].pth'"`. (`checkpoint='...'`만 쓰면 bash가 작은따옴표를 제거해서 똑같이 실패합니다.)

## 스모크 테스트 결과 (2026-10-28)

커밋 `8048f74` 기준(질량 캐시 적용 후 다시 확인), 새 PC(RTX 4070 Ti SUPER)에서:

| 테스트 | 결과 |
|---|---|
| 환경 검증 (512 env, 학습 없이) | 물체 9개 고르게 배정, 관측 340차원, GPU 시뮬레이션, DR ramp 꺼짐, NaN 없음 |
| 초기 위치/yaw 반영 | reset에서 써 넣은 yaw와 다음 step 시뮬레이터 yaw 차이 중앙값 ~0.005rad (첫 reset, 강제 reset, vec_task 손 DR이 같이 걸리는 reset, 낙하로 인한 일반 reset 모두). 범위도 설정대로(x, y ±1.5cm, z 0~1cm, yaw ±π) |
| 질량 | env별 0.2~0.6kg, reset 후에도 그대로. 캐시 값 = 시뮬레이터 값 |
| 64 env 학습 110 epoch (wandb 켬) | 정상 종료, 체크포인트 저장, wandb에 `PerObject/*` 포함 기록 |
| 8192 env 학습 5 epoch | 메모리 부족 없음, 초반 ~9~11k step/s (10/20 학습 초반과 같음 → 전체 ~45시간 예상) |
| `evaluate_policy_per_object.py` | 정상 동작 ([함정 #8](#8-체크포인트-경로의-대괄호rew725는-hydra가-문법으로-읽음)대로 경로를 따옴표 처리해야 함) |

참고:
- **길이 0 에피소드는 시작 시점에만 생깁니다.** 평가를 시작하면 env마다 정확히 1개씩 생기고, 실제 롤아웃 중에는 생기지 않았습니다. 8월 평가(900 env, 약 3,000 에피소드)의 "물체마다 25% 이상 길이 0"도 이 시작 기록이었을 가능성이 큽니다(900 / (3,030 + 900) ≈ 23%).
- 학습 중 `PerObject/*` 로그에도 이 시작 기록이 섞이지만 epoch 1에만 영향이 있습니다.
- 64 env처럼 작은 학습에서는 fps가 `1024000000`이나 음수로 찍히는 줄이 가끔 나옵니다. 시간 측정 출력 문제일 뿐이고, 8192 env에서는 나오지 않았습니다.

## 학습용 물체 (9개, `objSet: "C"`)

전부 `assets/urdf/objects/*.urdf` + `assets/urdf/objects/meshes/set2/*.obj`에 있고, `allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets["C"]`에 등록되어 있습니다. 모든 물체는 원본 mesh 기준 bounding box ~2.0 unit(URDF `scale=".03 .03 .03"` 적용 시 약 6cm)입니다.

| 라벨 (클래스 순서) | 형상 | 출처 (원본 in-hand-rotation 이름) |
|---|---|---|
| `block_1` | 정육면체 | `set_obj1_regular_block` |
| `block_2` | 불규칙 블록("time") | `set_obj15_irregular_block_time` |
| `block_3` | 블록 | `set_obj2_block` |
| `block_4` | 모서리 깎인 블록 | `set_obj6_block_corner` |
| `block_5` | 블록 | `set_obj7_block` |
| `cylinder_1` | 원기둥 | `set_obj11_cylinder` |
| `cylinder_2` | 축 방향 압축 원기둥 | `set_obj16_cylinder_axis` |
| `cylinder_3` | 정십각기둥 | (원본 아님, `generate_new_objects.py`로 생성) |
| `cylinder_4` | 모서리 깎인 원기둥 | `set_obj12_cylinder_corner` |

- 출처 매핑은 메시 꼭짓점/면 데이터와 URDF를 직접 비교해서 확인했습니다(8개 모두 형상 동일).
- 원본 16개 중 나머지 8개(`set_obj3/4/5/8/9/10/13/14`)는 삭제했습니다. 필요하면 커밋 `71e2df7`에서 꺼낼 수 있습니다.
- **클래스 순서는 과거(8월) 9-object 학습과 같습니다.** 그래서 아래 과거 평가/분류 결과의 물체 이름이 지금 물체와 그대로 대응합니다.
- 물체 세트 변천: 원본 16개 → 22개(절차적 생성 6개 추가) → 17개 → 9개(`C`, 8월 학습) → 16개 원본 복원(9/29, `71e2df7`) → **9개(`C`)로 재복귀(10월~, 현재)**.
- `ball`, `cross4_0~4` 키는 파일이 없는 죽은 키입니다(다른 과제용, 건드리지 않음).
- critic의 `states_buf`에는 물체 one-hot(`num_training_objects`차원)이 들어가므로, 물체 수가 다른 설정으로 학습한 체크포인트는 로드되지 않습니다.

## 초기 상태 랜덤화

매 reset마다 (`reset_idx`):
- **물체 초기 위치**: 기본 시작 위치 기준 x, y ~ U(-1.5cm, +1.5cm), z ~ U(0, +1cm) (`objInitPosNoiseXY: 0.015`, `objInitPosZLow: 0.0`, `objInitPosZHigh: 0.01`). DR ramp 영향을 받지 않고 항상 전체 범위.
- **물체 초기 회전(yaw)**: `useInitRandomRotation: True`일 때 월드 z축 기준 U(-π, π). **roll/pitch는 랜덤화 안 됨.**
- **손 관절 초기 자세**: 랜덤화 안 됨(고정된 기본 자세).

옛 키 `resetPositionNoise`는 yaml에 남아 있지만 물체 초기화에는 더 이상 쓰이지 않습니다.

## 활성 Domain Randomization 요약

- 물리: 질량 0.2~0.6kg(**env 생성 시 env별 1회**), 손/물체 마찰 각각 0.3~3.0(매 reset), PD gain(매 reset), 스케일 0.95~1.05(env 생성 시 1회)
- 랜덤 외력: 스케일 0.2, 확률 [0.2,0.25], 0.1초마다 0.99배 감쇠
- 센서: threshold [0.005,0.015]N, 드롭율 10%, lag 25%
- 관측/액션 노이즈: 관절 ±0.05, 액션 ±0.06, relScale ±5%
- 물체 초기 위치/yaw (위 절 참고)

### 점진적 도메인 랜덤화 (DR ramp, 기본값 꺼짐)

위 DR들을 학습 초반엔 좁게(또는 0에서) 시작해서 정해진 step 수에 걸쳐 선형으로 설정값까지 넓히는 기능입니다(`allegro_arm_morb_axis.py`의 `_dr_ramp_frac()`/`_apply_dr_ramp()`). 9/29에 16개 물체 학습용으로 추가됐고, **9개 물체 학습(8월, 10/20)은 이 기능 없이 처음부터 전체 DR로 학습했기 때문에 기본값은 꺼져 있습니다.**

- 켜는 법: `bash scripts/train_z_axis.sh 0 task.env.drRampEnable=True` (`drRampStartFrac`, `drRampSteps`로 스케줄 조정)
- 대상: 손/물체 마찰, PD gain 배수, 센서 threshold/노이즈/lag, 관절 관측 노이즈, 랜덤 외력. 범위형은 중앙값 쪽으로 좁히고, 크기형은 0부터 선형으로 키움.
- 대상 아님: 물체 스케일·질량(env 생성 시 고정), 물체 초기 위치/yaw, 액션 노이즈·중력(vec_task 쪽 DR)
- 효과는 검증된 적 없습니다.

## wandb 로깅

- `allegro_arm_morb_axis.py`의 `reset_idx`가 에피소드 종료 시 `{obj, rotations, ep_len, ep_reward}`를 `_pending_obj_stats`에 쌓고, `isaacgymenvs/utils/rlgames_utils.py`의 `RLGPUAlgoObserver.after_print_stats`가 매 epoch `PerObject/{obj}/rotation_count|episode_length|episode_reward`를 tensorboard에 기록합니다. `train.py`의 `wandb.init(sync_tensorboard=True)` 덕분에 wandb에도 자동 반영됩니다. 콘솔에는 `[PerObject reward] epoch N | block_1: ... | ...`로 출력됩니다.
- **wandb에는 체크포인트(`.pth`)가 올라가지 않습니다.** 올라가는 것은 설정(`config.yaml`), 콘솔 로그(`output.log`), tensorboard 이벤트 파일, git 커밋 해시(커밋 안 된 변경이 있으면 `diff.patch`)입니다. PC가 날아가면 체크포인트는 복구할 수 없으니 **중요한 체크포인트는 따로 백업하세요.**
- 프로젝트: `jhyang321-seoul-national-university/isaacgymenvs`

## 학습 기록 (wandb)

| 날짜 | wandb run (id) | 커밋 | 물체 | 결과 | 체크포인트 |
|---|---|---|---|---|---|
| 2026-08-03 | `rws3 260803` (`b5ic6zpy`) | `901f8aa` | 9개 (`C`) | 20,100 epoch 완주, best rolling reward **974.54** (epoch 19927) | 사라짐 |
| 2026-09-29 | `rws-set16` (`adrn0fwz`) | `71e2df7` | 16개 | 30분 만에 crash | 없음 |
| 2026-10-20 | `rws2-randinit` (`5ng9uo5d`) | `3e0c2fe` | 9개 | 12분 후 중단 | 없음 |
| 2026-10-20 | `rws2-randinit-zup` (`im67809a`) | `025d2e7` (push 안 됨, 사라짐) | 9개 | 20,100 epoch 완주(45.4h), best **826.83** (epoch 19605), 마지막 740.65 | 사라짐 |
| 2026-10-28 | `rws2-retrain` (`lxgr93nn`) | `8048f74` + 질량 캐시/README 수정(시작 시점 미커밋, wandb `diff.patch`에 기록) | 9개 | 진행 중 (새 PC, 8192 env, 예상 ~45h) | `runs/z-axis-touch-only/` — **끝나면 저장소 밖에 백업** |

10/20 `rws2-randinit-zup` 마지막 epoch 물체별 reward: block_1 834, block_2 615, block_3 417, block_4 683, block_5 674, cylinder_1 905, cylinder_2 547, cylinder_3 957, cylinder_4 1121.

**10/20 학습 설정과 현재 코드의 차이** (wandb `config.yaml` 비교):
- 10/20 학습에는 DR ramp 키가 없었음 → 현재 기본값도 꺼짐으로 맞춤.
- z 초기 위치를 `resetPositionNoiseZ: [0, 0.01]`라는 키로 지정 → 현재는 `objInitPosZLow/High`(범위 동일).
- `forceObject: None`이라는 키가 있었음 — 용도 불명, 현재 코드에는 없음.
- 그 밖의 보상 계수, DR 범위, PPO 하이퍼파라미터, 물체 세트는 동일.
- 질량 처리/초기 자세 버그를 당시 어떻게 고쳤는지는 알 수 없음(현재는 [함정 #7](#7-reset-중-set_actor_rigid_body_properties가-물체-위치자세-쓰기를-지움) 방식).

10/20 run의 `config.yaml`/`output.log`/tensorboard 파일은 저장소 밖 `C:\YangJaehong\wandb_recovery\`에 내려받아 두었습니다(wandb에서 다시 받을 수도 있음).

## 물체별 회전 성능 평가

**아래 결과는 8월 9-object 체크포인트(best 974.54) 기준입니다.** 물체 이름은 지금과 같지만, 그 체크포인트는 사라졌고 당시에는 초기 위치/yaw 랜덤화가 실제로 반영되지 않았습니다([함정 #7](#7-reset-중-set_actor_rigid_body_properties가-물체-위치자세-쓰기를-지움)). 또 이 결과에는 **길이 0인 에피소드가 포함**되어 있습니다(물체마다 25% 이상이 생존 시간 0). 현재 스크립트는 이를 제외하므로, 다시 돌리면 수치가 달라집니다.

**[촉각 기반 물체 분류](#촉각-기반-물체-분류)와는 별개의 실험입니다** — 데이터셋이 다릅니다(평가 3,030 에피소드 vs 분류 3,230 에피소드).

### 파이프라인

- **`tools/evaluate_policy_per_object.py`** — 체크포인트로 결정론적(`is_deterministic=True`) 롤아웃을 돌리면서, 학습 중 로깅과 같은 에피소드별 기록(`env._pending_obj_stats`)을 수집합니다.
  - `task.env.deterministicObjectAssign=True`로 라운드로빈 배정(클래스 균형).
  - 기본값: `numEnvs=900`(클래스당 100), `target_episodes_per_class=300`, `max_control_steps=3000`.
  - `env_reset()` 직후의 더미 레코드는 버리고 시작하며, **길이 0인 에피소드(`ep_len == 0`)는 결과와 목표 개수 집계에서 제외**합니다. 제외 개수는 콘솔과 결과 파일(`zero_len_dropped`)에 남습니다.
  - 결과: `tools/data/policy_eval.npz` — `labels/rotations/ep_len/ep_reward/full_duration: (N,)`, `label_names`, `max_episode_length`, `zero_len_dropped`. `full_duration`은 `ep_len >= max_episode_length - 1`(499), 즉 낙하 없이 끝까지(50초) 버틴 에피소드 여부.
  - 실행 예:
    ```bash
    python tools/evaluate_policy_per_object.py \
      checkpoint="'runs/<experiment>/<run>/nn/<name>.pth'" \
      task.env.numEnvs=900 +target_episodes_per_class=300 +max_control_steps=3000 \
      +out_path=tools/data/policy_eval.npz
    ```

### 결과 (8월 체크포인트, 길이 0 에피소드 포함)

| 물체 | n | reward (mean) | 회전 속도 (rev/s) | 완주율 (499 step) |
|---|---|---|---|---|
| block_1 | 309 | 621.63 | 0.169 | 55.7% |
| block_2 | 335 | 492.86 | 0.156 | 37.0% |
| block_3 | 371 | 364.99 | 0.137 | 26.2% |
| block_4 | 341 | 631.67 | 0.200 | 41.4% |
| block_5 | 388 | 327.25 | 0.134 | 19.3% |
| cylinder_1 | 320 | 747.86 | 0.234 | 51.6% |
| cylinder_2 | 347 | 572.16 | 0.177 | 36.6% |
| cylinder_3 | 311 | 753.72 | 0.231 | 52.4% |
| cylinder_4 | 308 | 765.41 | 0.250 | 49.7% |
| **전체** | **3,030** | **575.24** | **0.184** | **40.2%** |

- cylinder류가 block류보다 전반적으로 높고, **block_3/block_5가 reward·완주율 모두 최하위**. 학습 로그의 물체별 reward 서열, 촉각 분류 confusion 패턴과 일관됩니다.
- 이 평가의 평균 reward(575.24)가 학습 중 rolling reward(974.54)보다 크게 낮았던 것은, 길이 0 에피소드가 섞여 평균을 끌어내린 영향이 클 것으로 보입니다(재평가로 확인 필요).
- 결과 대시보드(Claude Artifact): https://claude.ai/artifact/PwSqHSJMarGBrUDk83JhYr (계정 종속, 다른 환경에서는 안 열릴 수 있음)

## 촉각 기반 물체 분류

**아래 결과도 8월 9-object 체크포인트 기준입니다**(데이터·체크포인트는 사라짐, 물체 이름은 지금과 같음). 파이프라인 코드는 클래스 수 무관하게 동작하므로 새 체크포인트로 그대로 재현할 수 있습니다.

**목표**: policy가 실제로 받는 센서값의 시계열만으로 9개 물체 중 무엇인지 맞히는 별도 CNN 분류기 학습(형상 재구성 아님, 분류만). 정책 자신은 물체 정체성을 모릅니다 — `object_one_hot_vector`는 critic 전용 `states_buf`에만 쓰이고 actor 관측(`last_obs_buf`/`obs_buf`)에는 들어가지 않습니다.

### 파이프라인

1. **`tools/collect_tactile_data.py`** — 결정론적 롤아웃을 돌리면서 매 control step `sensed_contacts`(촉각 16채널, 노이즈/threshold/lag 적용된 값)와 `last_obs_buf[:, 6:22]`(관절위치 16채널)를 env별로 버퍼링하다가, 에피소드 종료 시 (T, 32) 시퀀스를 물체 라벨과 함께 저장.
   - `task.env.deterministicObjectAssign=True`로 클래스 균형.
   - 기본값: `numEnvs=900`, `target_episodes_per_class=300`, `max_control_steps=2000`.
   - 결과: `tools/data/tactile_dataset.npz` — `tactile: (N, 500, 32)`(뒤쪽 0-padding), `lengths`, `labels`, `label_names`, `channels`.
   - 실행 예:
     ```bash
     python tools/collect_tactile_data.py \
       checkpoint="'runs/<experiment>/<run>/nn/<name>.pth'" \
       task.env.numEnvs=900 +target_episodes_per_class=300 +max_control_steps=2000 \
       +out_path=tools/data/tactile_dataset.npz
     ```
2. **`tools/train_tactile_classifier.py`** — 라벨 층화 70/15/15 분할(seed 고정) 후, masked-average-pooling 1D-CNN(Conv1d 32→64→128) + FC(128→64→N)로 50 epoch 학습.
   - `--channels tactile` / `--channels joint_pos`로 채널 일부만 학습 가능. `CHANNEL_WIDTHS`는 `collect_tactile_data.py`와 같아야 함(isaacgym 의존성을 피하려고 import 대신 값만 복제).
   - 실행 예: `python tools/train_tactile_classifier.py --data tools/data/tactile_dataset.npz --epochs 50`

### 결과 (8월 체크포인트, 3,230 에피소드)

| 입력 채널 | 테스트 정확도 | 가장 약한 물체 |
|---|---|---|
| 촉각 16채널만 | **82.7%** | cylinder_1 61.7% (cylinder_3와 혼동) |
| 관절위치 16채널만 | **87.8%** | cylinder_1 68.1%, cylinder_3 65.2% |
| 촉각+관절 32채널 | **90.3%** | block_3 74.6% (block_5와 혼동) |

(랜덤 베이스라인 = 1/9 = 11.1%.)

**핵심 발견**: 정책은 촉각 전용으로 학습됐는데도 **촉각 단독이 가장 낮은 분류 정확도**를 냈고, 세 조합이 **서로 다른 물체 쌍에서 실패**했습니다(촉각+관절은 block_3/block_5, 촉각만·관절만은 cylinder_1/cylinder_3). 촉각과 관절위치가 **상호보완적인 형상 단서**를 담고 있다는 뜻입니다.

## 다음 할 일

- **최우선**: 현재 기본 설정(9개 물체, 초기 자세 랜덤화 수정, DR ramp 꺼짐)으로 본 학습을 다시 돌려 체크포인트 확보 → **완료되면 `runs/`의 best 체크포인트를 저장소 밖에도 백업**
- 새 체크포인트로 `evaluate_policy_per_object.py`(길이 0 제외) 재평가 → 8월 결과와 비교
- 초기 자세 랜덤화가 실제로 들어간 학습이 8월(974.54)/10월(826.83) 학습과 비교해 어떤지 확인
- `forceObject` 키의 용도 확인(10/20 학습 설정에만 있음)
- 촉각 분류 파이프라인을 새 체크포인트로 재실행
- (기존 TODO) 분류기 precision/F1, prev_action 등 다른 관측 채널 추가, x/y축 회전, distillation(학생 정책)
