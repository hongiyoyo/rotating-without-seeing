# rotating-without-seeing

터치 전용(vision 절대 사용 안 함) z축 in-hand rotation teacher policy를 IsaacGym + PPO로 학습하기 위한, `in-hand-rotation`("Robot Synesthesia" 코드베이스)에서 필요한 부분만 추려낸 독립 프로젝트입니다.

이 문서는 이 코드를 처음 이어받는 사람/에이전트가 배경 설명 없이 바로 작업을 이어갈 수 있도록 지금까지의 작업 이력, 설계 근거, 그리고 **반드시 알아야 할 함정들**을 정리한 것입니다.

## 현재 상태 요약 (가장 먼저 읽을 것)

0. **⚠️ 물체 세트가 이번 세션에 통째로 되돌려졌습니다 (가장 최신, 가장 중요).** 그동안 쓰던 9개 물체(`block_1~5`, `cylinder_1~4`, `objSet: "C"`)는 **완전히 삭제**되었고, 원본 `in-hand-rotation` 프로젝트의 물체 16개(`set_obj1_regular_block` ~ `set_obj16_cylinder_axis`, `objSet: "set16"`, 이제 기본값)를 원래 이름 그대로 복원했습니다. 아래 1~6번(본 학습 완료, 촉각 분류, 물체별 회전 성능 평가)은 **전부 이제는 존재하지 않는 9개 물체 이름 체계로 학습/평가된 과거 결과**입니다 — 역사적 기록으로는 유효하지만, **그대로 재현 실행은 안 됩니다** (아래 참고). 자세한 내용·이유·전체 이름 매핑은 [학습용 물체](#학습용-물체-16개-원본-in-hand-rotation-세트) 참고.
1. **(과거 기록) 9개 물체 기준 본 학습(numEnvs=8192, 20100 epoch) 완료됨.** Best 체크포인트:
   `runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/z-axis-touch-only.pth`
   (rolling reward 974.54, epoch 19927 시점 저장 — 마지막 epoch 20100 체크포인트의 reward 870.41보다 높음). 근거·확인 방법은 [본 학습 결과](#본-학습-결과) 참고. **이 체크포인트는 9-object critic(`num_training_objects=9`)에 맞게 학습되어, 16-object 기본 설정(`objSet: "set16"`)으로는 네트워크 차원이 달라 그대로 로드/시각화할 수 없습니다** — `objSet=C`를 명시해도 물체 자산 파일 자체가 삭제되어 로드가 안 됩니다. 16개 세트로 다시 학습해야 새 체크포인트가 생깁니다.
2. **GPU 없다는 이전 버전 README 기술은 틀렸습니다.** 이 Windows PC는 WSL2를 통해 실제 NVIDIA GPU에 접근 가능하고, IsaacGym·conda 환경이 이미 다 세팅되어 있습니다. [실행 환경](#실행-환경-wsl2) 섹션 필독 — 여기부터 안 읽으면 "GPU 없어서 못 함"이라고 잘못 판단하게 됩니다.
3. **(과거 기록, 9개 물체 기준)** 위 체크포인트로 **물체별 촉각+관절위치 시계열을 수집해서 9-way CNN 분류기를 학습**하는 별도 하위 프로젝트를 완료했습니다 (정책 자체는 물체 정체성을 전혀 모른 채 회전만 함 — 그 결과로 나온 감각 신호만으로 어떤 물체인지 맞히는 실험). 결과: 촉각+관절 90.3%, 관절만 87.8%, 촉각만 82.7%. 전체 내용은 [촉각 기반 물체 분류](#촉각-기반-물체-분류) 참고. 파이프라인 코드 자체는 클래스 수에 무관하게 동작하므로(동적으로 `label_names` 길이를 읽음) 16개 세트로 재학습한 체크포인트만 있으면 그대로 재사용 가능합니다.
4. `scripts/train_z_axis.sh`가 **CRLF 줄바꿈 때문에 WSL bash에서 실행 자체가 안 되던 버그**를 고쳤습니다 (LF로 정규화 완료). Windows 도구로 이 저장소의 `.sh` 파일을 다시 저장/편집하면 같은 문제가 재발할 수 있으니, 이상하게 스크립트가 안 돌면 `file <script>.sh`로 line ending부터 확인하세요.
5. 다음에 자연스럽게 이어갈 수 있는 작업(요청받은 적은 없지만 참고): **16개 물체 세트(`objSet=set16`)로 본 학습을 처음부터 다시 돌리기** (가장 우선순위 높음 — 위 0번 때문에 기존 체크포인트를 못 씀), 그 다음 x/y축으로 확장, 관절+촉각 외 다른 관측(예: prev_action) 채널 추가해서 분류 정확도 개선, distillation(학생 정책으로 압축) 등.
6. **(과거 기록, 9개 물체 기준)** 위 체크포인트를 촉각 분류와는 **별개로**, `tools/evaluate_policy_per_object.py`로 물체별 회전 성능(reward, 회전 속도, 완주율)만 따로 평가한 결과도 있습니다. **주의: 이 평가의 에피소드 수는 3,030개로, 촉각 분류용 데이터셋(3,230개, `tactile_dataset.npz`)과는 다른 별도 수집입니다 — 헷갈리지 마세요.** 자세한 내용은 [물체별 회전 성능 평가](#물체별-회전-성능-평가) 참고.

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
    objects/*.urdf + meshes/set2/*.obj                       (원본 in-hand-rotation 물체 16개, set_obj1~16 원래 이름 그대로 — 전부 학습(objSet set16)에 사용, 아래 참고)
  scripts/train_z_axis.sh   (학습 실행 스크립트 — LF로 정규화됨, CRLF 재발 주의)
  runs/                 (gitignored. 학습 체크포인트/텐서보드 로그. best 체크포인트 위치는 위 "현재 상태 요약" 참고.
                          이 디렉터리는 이 워킹 디렉터리에만 존재 — 새로 clone하면 없음, 재현하려면 재학습 필요)
  wandb/                (gitignored. wandb 로컬 캐시/로그)
  tools/
    object_viewer.html        (물체들을 브라우저에서 3D로 확인하는 뷰어, mesh 데이터 내장 — 이번 세션에 16개 원본 물체로 재생성됨)
    generate_new_objects.py   (과거 cylinder_3/계단형 block_3/ball_1~4 등 절차적 생성 물체를 만든 스크립트 - 그 물체들은 이제 전부 삭제되어 순수 히스토리 기록용)
    regen_viewer_data.py      (object_viewer.html의 내장 mesh 데이터를 재생성하는 스크립트 — CATALOG_INFO가 이번 세션에 16개 원본 이름으로 갱신됨)
    relabel_objects_2.py      (과거 물체 목록 2차 재정리에 쓴 1회성 스크립트 - 그 리네이밍은 이번 세션에 전부 되돌려져 순수 히스토리 기록용, 재실행 불가)
    collect_tactile_data.py   (best 체크포인트로 정책을 굴려 촉각+관절위치 시계열 수집 — 클래스 수에 무관하게 동작, 16개 세트에도 그대로 재사용 가능)
    train_tactile_classifier.py (수집한 시계열로 N-way 물체 분류 CNN 학습 — num_classes를 `label_names` 길이로 동적으로 읽으므로 16개 세트에도 그대로 재사용 가능)
    evaluate_policy_per_object.py (촉각 분류와 무관하게, best 체크포인트의 물체별 회전 성능만 따로 평가 — reward/회전속도/완주율. 역시 클래스 수 무관)
    data/  (아래 파일들은 전부 옛 9-object("C") 체크포인트 기준 결과입니다 — 히스토리로만 유효, 16-object 재학습 후 다시 수집해야 함)
      tactile_dataset.npz         (3,230 에피소드, (N,500,32) 패딩된 배열 — tactile 16ch + joint_pos 16ch. collect_tactile_data.py 산출물, 촉각 분류기 학습용)
      sample_traces.json          (대시보드용으로 뽑은 물체별 대표 샘플 1개씩)
      tactile_classifier_dashboard.html       (촉각+관절 결과 대시보드, 메인)
      joint_pos_classifier_dashboard.html     (관절위치만 ablation 대시보드)
      tactile_only_classifier_dashboard.html  (촉각만 ablation 대시보드)
      policy_eval.npz             (3,030 에피소드, 물체별 {rotations, ep_len, ep_reward} 기록. evaluate_policy_per_object.py 산출물 — tactile_dataset.npz와 별개의 수집, 분류기 학습에는 안 쓰임)
      policy_eval_summary.json    (policy_eval.npz의 물체별 통계 요약 — 대시보드용)
      training_curve_per_object.json (학습 전체 20,100 epoch의 물체별 reward 추이, 25-epoch 이동평균 — 콘솔 학습 로그에서 파싱)
      (물체별 회전 성능 대시보드 HTML도 별도로 존재 — 정확한 파일명은 이 워킹 디렉터리에 파일이 없어 미확인, 위 세 데이터 파일로부터 재생성 가능)
  install.md            (원본 세팅 안내 — WSL robosyn 환경엔 이미 다 설치되어 있어 참고용)
```

## 논문 스펙 대비 수정한 항목 (파일: `isaacgymenvs/tasks/allegro_arm_morb_axis.py`, `cfg/task/AllegroArmMOAR.yaml`)

원본 `in-hand-rotation`의 `partial_stack` 구현은 논문의 대부분(16개 FSR 센서 배치, finite-difference 회전각 보상, EMA 액션 스무딩 η=0.8, 10Hz 제어, 4-step 히스토리 스택, PPO 하이퍼파라미터)을 이미 정확히 따르고 있었지만, 아래 수치/공식들은 이후 다른 실험(Robot Synesthesia)에 맞춰 바뀐 상태였고, 이 폴더에서 논문 값으로 되돌렸습니다:

| 항목 | 원본 코드 | 이 폴더 |
|---|---|---|
| 접촉 센서 threshold | 랜덤 [1.0, 2.0]N | 랜덤 [0.005, 0.015]N (논문 0.01N 근사) |
| PD gain 랜덤화 | P×[0.30,0.60], D×[0.75,1.05] | P×[0.66,1.33], D×[0.80,1.20] |
| 손/물체 마찰 | 공유된 draw 1개, [0.2,3.0] | 독립 draw 2개, 둘 다 [0.3,3.0] |
| 물체 스케일(기본 objSet 세트) | [0.95,1.1] | [0.95,1.05] |
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

## 학습용 물체 (16개, 원본 in-hand-rotation 세트)

### 배경: 9개로 줄었다가, 이번 세션에 16개 원본으로 전부 되돌아감

과거 세 차례 리네이밍/정리를 거쳐 원본 16개 → 22개(절차적 생성 6개 추가) → 17개(reward 낮은 5개 삭제, 일부 승격/재넘버링) → **9개**(`block_1~5`, `cylinder_1~4`, `objSet: "C"`, `else_1~8`은 자산만 남기고 학습에서 제외)까지 축소된 상태였습니다. **이번 세션에 이 전체 히스토리를 되돌려서, 원본 `in-hand-rotation`(`assets/urdf/objects/`, `set_obj1`~`set_obj16`)의 16개 물체를 원래 파일명 그대로 복원**했습니다.

구체적으로 한 일:
- `assets/urdf/objects/`의 기존 물체 파일(`block_1~5.urdf`, `cylinder_1~4.urdf`, `else_1~8.urdf`, 관련 `meshes/set2/*.obj` 전부)을 **모두 삭제**.
- `in-hand-rotation/assets/urdf/objects/`의 `set_obj1_regular_block.urdf` ~ `set_obj16_cylinder_axis.urdf` 16개와 그 mesh 파일(`meshes/set2/`, 시각 mesh + 모든 convex decompose 파트)을 **원래 이름 그대로** 복사해옴 (`cross4_0~4`는 애초에 이 저장소에 파일 자체가 없어 제외 대상도 아니었음).
- `isaacgymenvs/tasks/allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets`를 새 이름에 맞게 전면 재작성 — `block_N`/`cylinder_N`/`else_N` 키를 전부 제거하고 `set_obj1_regular_block` ~ `set_obj16_cylinder_axis` 16개 키로 교체, `object_sets["C"]`(9개)는 완전히 삭제하고 `object_sets["set16"]`(16개 전부)로 대체.
- `cfg/task/AllegroArmMOAR.yaml`의 `objSet` 기본값을 `"C"` → `"set16"`으로 변경.
- `tools/regen_viewer_data.py`의 `CATALOG_INFO`를 16개 새 이름으로 갱신하고 **실제로 재실행**해서 `tools/object_viewer.html`의 내장 mesh 데이터를 16개 물체로 재생성함 (GPU 불필요, 확인 완료).
- `tools/evaluate_policy_per_object.py`/`tools/collect_tactile_data.py`/`tools/train_tactile_classifier.py`는 클래스 수를 하드코딩하지 않고 `label_names`/`used_training_objects` 길이로 동적으로 읽기 때문에 **코드 수정 없이 16개 세트에도 그대로 재사용 가능** — 주석·docstring의 "9개"/"numEnvs=900" 예시 문구만 16개 기준으로 고쳐뒀습니다.

**결과적으로 `object_sets["C"]`, `block_N`/`cylinder_N`/`else_N`이라는 이름은 이 코드베이스 어디에도 더 이상 존재하지 않습니다.** 과거 `else_1~8`이 "C 세트에서 제외됐던 원본 물체 8개"였고, `cylinder_3`(정십각기둥)·계단형 `block_3`·`ball_1~4`는 원본 16개에 없던 절차적 생성 물체였다는 것은 git 히스토리(`relabel_objects_2.py`, `generate_new_objects.py`)로만 남아있습니다.

### ⚠️ 기존 9-object 체크포인트/데이터와의 호환성

`num_training_objects`(=9였다가 16이 됨)는 critic 전용 `states_buf`의 object one-hot 구간 크기를 직접 결정합니다(`allegro_arm_morb_axis.py`의 `num_states` 계산, `object_one_hot_vector` 대입부). 따라서 **기존 best 체크포인트(`z-axis-touch-only.pth`, 9-object 기준으로 학습됨)는 새 기본 설정(`objSet: "set16"`, 16개)으로 그대로 로드/시각화/평가할 수 없습니다** — 네트워크 입력 차원이 다릅니다. `objSet=C`를 강제로 지정해도 `block_1` 등 물체 자산 파일 자체가 삭제되어 env 생성 단계에서 실패합니다. [물체별 회전 성능 평가](#물체별-회전-성능-평가)/[촉각 기반 물체 분류](#촉각-기반-물체-분류) 절의 결과와 `tools/data/*`의 npz/HTML은 모두 이 옛 9-object 체크포인트 기준이라 **히스토리로만 유효**하며, 16개 세트로 그대로 재현되지 않습니다. 다음에 할 일은 `objSet=set16`으로 본 학습을 처음부터 다시 돌리는 것입니다.

### 물체 목록

전부 `assets/urdf/objects/set_obj*.urdf` + `assets/urdf/objects/meshes/set2/*.obj`에 있고, `isaacgymenvs/tasks/allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets["set16"]`에 등록되어 있습니다. 모든 물체는 원본 mesh 기준 bounding box ~2.0 unit(= URDF `scale=".03 .03 .03"` 적용 시 실제 약 6cm)로 동일한 크기 규격입니다.

| 라벨 (`asset_files_dict` 키) | 형상 | 원래 번호 |
|---|---|---|
| `set_obj1_regular_block` | 정육면체 | 1 |
| `set_obj2_block` | 블록 | 2 |
| `set_obj3_block` | 블록 | 3 |
| `set_obj4_block` | 블록 | 4 |
| `set_obj5_block` | 블록 | 5 |
| `set_obj6_block_corner` | 모서리 깎인 블록 | 6 |
| `set_obj7_block` | 블록 | 7 |
| `set_obj8_short_block` | 짧은 블록 | 8 |
| `set_obj9_thin_block` | 얇은 블록 | 9 |
| `set_obj10_thin_block_corner` | 모서리 깎인 얇은 블록 | 10 |
| `set_obj11_cylinder` | 원기둥 | 11 |
| `set_obj12_cylinder_corner` | 모서리 깎인 원기둥 | 12 |
| `set_obj13_irregular_block` | 불규칙 블록 | 13 |
| `set_obj14_irregular_block_cross` | 불규칙 블록("cross") | 14 |
| `set_obj15_irregular_block_time` | 불규칙 블록("time") | 15 |
| `set_obj16_cylinder_axis` | 축 방향 압축 원기둥 | 16 |

`object_sets["set16"]` = 위 16개 전부. **`ball` 키는 이 16개와 무관**하니 헷갈리지 마세요 — baoding balls 과제 전용의 별도 단일 구 오브젝트입니다(건드리지 않음, 다만 이 저장소에는 애초에 `ball.urdf` 파일 자체가 없어 실제로는 죽은 키입니다). `cross4_0~4` 키도 마찬가지로 파일이 없는 죽은 키입니다.

## 초기 상태 랜덤화

- **물체 초기 위치**: ±1.5cm 랜덤 (`resetPositionNoise`)
- **물체 초기 회전(yaw)**: `useInitRandomRotation: True`로 켜져 있어 z축(회전축) 기준 [-π,π] 완전 랜덤. **단, roll/pitch는 랜덤화 안 됨**.
- **손 관절 초기 자세**: 랜덤화 **안 됨** (매 에피소드 고정된 기본 자세로 리셋).

## 활성 Domain Randomization 요약

물리(질량 0.2~0.6kg, 손/물체 마찰 각각 0.3~3.0, PD gain, 위치 ±1.5cm, 스케일 0.95~1.05) + 랜덤 외력(스케일 0.2, 확률 [0.2,0.25], 0.1초마다 0.99배 감쇠) + 센서(threshold [0.005,0.015]N, 드롭율 10%, lag 25%) + 관측/액션 노이즈(관절 ±0.05, 액션 ±0.06, relScale ±5%) + 물체 초기 yaw.

### 점진적 도메인 랜덤화 (Progressive DR, 16-object 학습 기본값으로 활성화됨)

물체 세트를 9개→16개로 늘리면서 탐색 난이도가 올라간 것을 완화하기 위해, 위 DR들을 학습 초반엔 좁게(또는 아예 끄고) 시작해서 서서히 원래 설정값까지 넓혀가는 기능을 추가했습니다(`allegro_arm_morb_axis.py`의 `_dr_ramp_frac()`/`_apply_dr_ramp()`, `pre_physics_step`에서 매 control step마다 호출). **`objSet: "set16"`과 함께 기본으로 켜져 있어서, 추가 옵션 없이 `bash scripts/train_z_axis.sh 0`만 실행하면 바로 16개 물체 + progressive DR로 학습이 시작됩니다.**

- **대상**: 물체/손 마찰, 질량, PD gain 배수, 센서 threshold, 센서 노이즈/lag 확률, 관절 관측 노이즈, 리셋 위치 노이즈, 랜덤 외력 스케일/확률 — 위에서 나열한 DR 중 **물체 스케일만 제외**(env 생성 시 한 번 고정되는 값이라 매 리셋마다 다시 뽑는 다른 DR들과 달리 학습 중 못 바꿈).
- **방식**: 각 `[low, high]` 범위는 중점(= "랜덤화 없음" 지점)으로 수축시키고, 확률/노이즈 크기류(센서 노이즈, lag, 관절 노이즈, 위치 노이즈, 외력 스케일 등)는 0에서부터 설정값까지 선형 스케일. **성능 기반으로 자동 조절되는 진짜 ADR이 아니라, 정해진 step 수에 걸쳐 선형으로 넓히는 스케줄 방식**입니다.
- **설정 (`cfg/task/AllegroArmMOAR.yaml`)**: `drRampEnable: True`(기본값), `drRampStartFrac: 0.0`(시작 시점 DR 강도 비율), `drRampSteps: 50000`(control step 기준 — 논문 스펙 학습 1회가 `horizon_length(16) x max_epochs(20100) = 321,600` step이므로 기본값은 전체 학습의 약 15% 구간에 걸쳐 DR을 최대치까지 끌어올림).
- **끄는 법**(기존처럼 처음부터 풀 DR로 학습하고 싶으면): `bash scripts/train_z_axis.sh 0 task.env.drRampEnable=False` (필요시 `task.env.drRampSteps=...`, `task.env.drRampStartFrac=...`로 스케줄 자체도 조정 가능).
- **아직 안 한 것**: 실제로 이 기능을 켜고 학습을 돌려서 껐을 때 대비 효과가 있는지는 검증하지 않았습니다 — 코드만 작성/기본 활성화됨. 16-object 본 학습 자체도 아직 안 돌아갔습니다(다음 섹션 참고).

## wandb 물체별 reward 로깅

이미 구현되어 있습니다 (이번 세션 이전부터):
- `allegro_arm_morb_axis.py`: `episode_reward_buf`/`episode_fwd_theta_buf`(env별 누적치), `_pending_obj_stats`(에피소드 종료 시 물체 이름별 기록)
- `isaacgymenvs/utils/rlgames_utils.py`의 `RLGPUAlgoObserver.after_print_stats`: 매 PPO epoch마다 `_pending_obj_stats`를 집계해 `PerObject/{obj}/rotation_count|episode_length|episode_reward`를 tensorboard writer로 기록 → `train.py`의 `wandb.init(sync_tensorboard=True)` 덕분에 별도 `wandb.log()` 없이 자동으로 wandb에도 반영됨. 콘솔에도 `[PerObject reward] epoch N | block_1: ... | ...` 형태로 출력.
- 이번 세션에 고친 것: 이 로깅 블록을 감싸던 `except Exception: pass`가 에러를 완전히 은폐하던 걸, 최초 1회만 경고를 출력하도록 수정(`self._per_object_log_warned` 플래그). 학습 자체를 죽이지 않기 위해 try/except 자체는 유지.

## 본 학습 결과

**⚠️ 이하 이 섹션 전체는 옛 9-object(`objSet: "C"`) 세트 기준 결과입니다.** [학습용 물체](#학습용-물체-16개-원본-in-hand-rotation-세트)에서 설명한 대로 그 9개 물체 이름(`block_1~5`, `cylinder_1~4`)과 자산 파일은 이번 세션에 삭제되었고 `objSet` 기본값도 `"set16"`(16개)으로 바뀌어, 아래 체크포인트는 새 기본 설정으로 로드되지 않습니다. 히스토리로는 유효하지만 그대로 재현 실행은 안 됩니다 — 16개 세트로 재학습해야 합니다.

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
`numEnvs=9` + `deterministicObjectAssign=True`로 9개 env에 물체가 하나씩 순서대로(옛 block_1~5, cylinder_1~4) 배정되어, 9개 물체가 동시에 회전하는 걸 한 창에서 볼 수 있었습니다 (Windows 데스크톱에 "Isaac Gym (Ubuntu-24.04)" 창, 위 [실행 환경](#실행-환경-wsl2) 참고). **이 명령은 위 체크포인트 호환성 문제 때문에 지금은 그대로 실행되지 않습니다** — 16개 세트로 재학습한 새 체크포인트가 생기면 `task.env.numEnvs=16`으로 바꿔서 같은 패턴을 재사용하면 됩니다(기본값이 이미 `objSet=set16`이라 별도 지정 불필요).

## 물체별 회전 성능 평가

**⚠️ 이 섹션도 옛 9-object(`objSet: "C"`) 체크포인트 기준 결과입니다** (위 [본 학습 결과](#본-학습-결과) 경고 참고) — 아래 표의 `block_*`/`cylinder_*` 이름은 이제 자산 파일이 없습니다. 파이프라인 코드(`tools/evaluate_policy_per_object.py`) 자체는 클래스 수 무관하게 동작하므로, 16개 세트로 재학습한 체크포인트만 있으면 그대로 재사용해 같은 형식의 새 결과를 만들 수 있습니다.

**아래 [촉각 기반 물체 분류](#촉각-기반-물체-분류)와는 완전히 별개의 실험입니다.** 촉각 분류용 데이터셋(`tactile_dataset.npz`, 3,230 에피소드)과 헷갈리지 마세요 — 여기서 다루는 데이터셋은 에피소드 수부터 다릅니다(3,030개).

**목표**: best 체크포인트(`z-axis-touch-only.pth`, rolling reward 974.54)가 실제로 각 물체를 얼마나 잘 회전시키는지, 탐험 노이즈나 학습 중 배치 노이즈 없이 "배포했을 때의 성능"에 가깝게 물체별로 깔끔하게 재평가.

### 파이프라인

- **`tools/evaluate_policy_per_object.py`** — best 체크포인트로 결정론적(`is_deterministic=True`) 롤아웃을 돌리면서, 학습 중 wandb 물체별 reward 로깅에 이미 쓰이던 것과 동일한 에피소드별 기록(`env._pending_obj_stats`, `allegro_arm_morb_axis.py`의 `reset_idx`에서 채워짐 — `{obj, rotations, ep_len, ep_reward}`)을 그대로 수집. 촉각/관절 시계열 자체는 기록하지 않고 물체별 요약 통계만 모으므로 `collect_tactile_data.py`보다 훨씬 가볍습니다.
  - `task.env.deterministicObjectAssign=True`로 라운드로빈 배정(클래스 균형 보장) — 촉각 분류 데이터 수집과 동일한 방식.
  - 기본값: `numEnvs=900`(클래스당 100), `target_episodes_per_class=300`, `max_control_steps=3000`(안전 상한).
  - `env_reset()` 직후 한 번 밀려 들어오는 all-zero 더미 레코드(`rotations=0, ep_len=0, ep_reward=0`)는 버리고 시작합니다.
  - 결과: `tools/data/policy_eval.npz` — `labels/rotations/ep_len/ep_reward/full_duration: (N,)`, `label_names`, `max_episode_length`. `full_duration`은 `ep_len >= max_episode_length - 1`(499) 여부로, 낙하 등으로 조기 종료되지 않고 끝까지(50초) 버틴 에피소드인지를 나타냅니다.
  - 실행 예:
    ```bash
    python tools/evaluate_policy_per_object.py \
      checkpoint='runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-08-03_12-09-19-83810/nn/z-axis-touch-only.pth' \
      task.env.numEnvs=900 +target_episodes_per_class=300 +max_control_steps=3000 \
      +out_path=tools/data/policy_eval.npz
    ```
  - 실제 수집 결과: **3,030 에피소드**, 클래스당 308~388개(불균형 원인은 촉각 데이터 수집 때와 동일 — 목표 300 도달 후 정지, 에피소드가 짧게 끝나는 물체일수록 그 사이 더 많이 쌓임).
  - 추가로 `tools/data/policy_eval_summary.json`(물체별 통계 요약)과, 전체 학습 콘솔 로그(20,100 epoch)에서 파싱한 `tools/data/training_curve_per_object.json`(물체별 reward의 25-epoch 이동평균 추이)도 함께 만들어 대시보드에 사용했습니다.

### 결과

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

- **패턴 재확인**: cylinder류(reward 572~765, 완주율 37~52%)가 block류(reward 327~632, 완주율 19~56%)보다 전반적으로 높음. **block_3/block_5가 reward·완주율 둘 다 최하위**(각각 365/327, 26%/19%) — 학습 로그 상의 물체별 reward 서열, 그리고 촉각 분류 confusion matrix에서 이 둘이 두드러지는 패턴과 모두 일관됩니다.
- **참고(미해명)**: 이 평가의 전체 평균 reward(575.24)는 체크포인트 저장 시점의 rolling mean reward(974.54)보다 상당히 낮습니다. 둘은 계산 방식이 다릅니다(rolling mean은 학습 중 슬라이딩 윈도 평균, 이쪽은 이 평가 롤아웃 3,030개 에피소드의 단순 산술 평균) — 정확한 괴리 원인은 아직 분석하지 않았습니다.
- 물체별 reward histogram(이분법적으로 "낙하 초반" 봉우리와 "완주" 봉우리로 갈리는 정도)과 물체별 학습 곡선(25-epoch 이동평균)까지 포함한 시각화는 별도 대시보드 HTML로 만들어져 있습니다 (정확한 파일명은 위 [폴더 구조](#폴더-구조) 참고 — 이 워킹 디렉터리에 파일 자체는 없어 미확인).

## 촉각 기반 물체 분류

**⚠️ 이 섹션도 옛 9-object(`objSet: "C"`) 체크포인트 기준 결과입니다** (위 [본 학습 결과](#본-학습-결과) 경고 참고). 파이프라인 코드(`tools/collect_tactile_data.py`, `tools/train_tactile_classifier.py`)는 클래스 수 무관하게 동작하므로, 16개 세트로 재학습한 체크포인트만 있으면 그대로 재사용해 16-way 분류로 재현할 수 있습니다.

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

**이번 세션에 한 일 (GPU 불필요, 파일/코드 작업만 — 아직 학습은 안 돌림):**
- 9-object(`objSet: "C"`) 세트를 완전히 삭제하고 원본 in-hand-rotation 16-object 세트를 원래 이름(`set_obj1_regular_block`~`set_obj16_cylinder_axis`)으로 복원 — 메시 지오메트리 diff로 16개 전부 바이트 단위 동일함을 먼저 확인한 뒤 진행
- `allegro_arm_morb_axis.py`의 `asset_files_dict`/`object_sets`를 새 이름 기준으로 전면 재작성, `cfg/task/AllegroArmMOAR.yaml`의 `objSet` 기본값을 `"set16"`으로 변경
- `tools/regen_viewer_data.py` 갱신 후 실제 재실행 → `object_viewer.html`이 16개 물체를 정상적으로 보여줌을 확인 (GPU 불필요라 직접 실행/검증 가능했음)
- `tools/evaluate_policy_per_object.py`/`collect_tactile_data.py`/`train_tactile_classifier.py`는 클래스 수 하드코딩이 없어 수정 없이 재사용 가능함을 코드 확인 (실행 자체는 GPU 필요라 안 함)
- **아직 안 한 것(다음으로 가장 시급함): `objSet=set16`으로 본 학습 자체를 돌려보는 것.** 코드는 "실행 준비 완료" 상태까지만 만들어뒀고, 실제 실행(수일 소요되는 학습 작업)은 이번 세션에서 하지 않았습니다 — 위 [실행 환경](#실행-환경-wsl2)의 WSL2 명령 패턴으로 `bash scripts/train_z_axis.sh 0`를 돌리면 됩니다. 아래는 그 전 세션들의 9-object 기준 과거 기록입니다.

**확인 완료 (더 이전 세션, 실제 GPU 실행으로 — 지금은 삭제된 9-object 세트 기준):**
- `numEnvs=64` 스모크 테스트 → 관측 차원 340(276 아님) 확인, `ablation_mode` 수정이 실제로 반영됨을 재확인
- `numEnvs=8192` 본 학습 20,100 epoch 완주, wandb에 물체별 reward 정상 기록
- 접촉 신호가 항상 0/1로 고정되지 않고 실제로 반응함 (물체별 reward가 유의미하게 갈리는 것으로 간접 확인 — block_3/block_5가 낮고 cylinder류가 높은 일관된 패턴)
- 9개 물체 간 reward 편차: cylinder류(1000~1300대)가 block류(400~1200대)보다 대체로 높음, block_3/block_5가 특히 낮은 편 — 학습 로그와 촉각 분류 confusion 패턴 둘 다에서 일관되게 나타남
- 학습된 policy로 9개 물체 동시 시각화 (뷰어 창 정상 동작)
- 촉각+관절/관절만/촉각만 3가지 CNN 분류기 학습 및 confusion matrix 분석 완료
- `evaluate_policy_per_object.py`로 별도의 결정론적 롤아웃(3,030 에피소드) 돌려 물체별 reward/회전속도/완주율 정량 평가 완료 — [물체별 회전 성능 평가](#물체별-회전-성능-평가) 참고

**아직 안 한 것 (다음 에이전트가 이어갈 수 있는 지점):**
- **최우선**: `objSet=set16`(16개 원본 물체)으로 본 학습을 처음부터 다시 돌리기 — 지금 기본 설정이지만 아직 학습된 체크포인트가 없음
- 16개 세트로 학습 완료 후: `evaluate_policy_per_object.py`/촉각 분류 파이프라인을 16-way로 재실행해서 아래 9-object 시절 결과와 비교 (특히 원래 `else_*`였던 8개 물체가 9-object 세트에서 빠졌던 이유 — reward가 낮아서가 아니라 "범용성 포기" 방침 때문이었으므로, 16개로 학습하면 이 물체들 성능이 어떻게 나올지는 미지수)
- 논문 Fig.6과 정량적으로 비교하는 학습 곡선 분석은 안 함 (wandb 대시보드에서 직접 확인 가능)
- 분류기 정확도를 precision/F1까지 확장하거나, prev_action 등 다른 관측 채널을 추가했을 때 개선되는지는 미확인 (9-object 시절 기준 TODO, 16-object 재현 시에도 유효)
- x축/y축 회전이나 distillation(학생 정책) 등 원 논문의 다른 부분은 이 폴더 범위 밖 (README 최상단 "배경" 참고)
- (9-object 시절 기록) 결정론적 평가(575.24)와 체크포인트 저장 시점 rolling reward(974.54) 사이의 괴리 원인은 미분석
