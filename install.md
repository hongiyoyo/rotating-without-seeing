# 설치 안내

이 폴더는 `in-hand-rotation`(Robot Synesthesia 코드베이스)에서 "Rotating without Seeing" 논문의
촉각 전용(z축) teacher policy 학습 시나리오 하나에만 필요한 파일들을 추려낸 것입니다.
설치 방법은 원본 저장소와 동일합니다.

## 1. Conda 환경 + pytorch3d

```
conda create -n rotating-without-seeing python=3.8
conda activate rotating-without-seeing
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

(참고: `pytorch3d`/`open3d`는 이 폴더의 `vec_task.py`/`allegro_arm_morb_axis.py`가 모듈
최상단에서 무조건 import하기 때문에, 실제로는 point cloud/카메라 기능을 쓰지 않는
`partial_stack`(촉각 전용) 시나리오에서도 설치가 필요합니다.)

## 2. IsaacGym

NVIDIA 웹사이트에서 Isaac Gym Preview 4 릴리즈를 받아 설치합니다.

```
pip install scipy imageio ninja
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e . --no-deps
```

## 3. 나머지 의존성

```
pip install hydra-core gym ray open3d numpy==1.20.3 tensorboardX tensorboard wandb
```

## 4. 실행 (GPU 필요)

이 폴더 루트에서:

```
bash scripts/train_z_axis.sh 0
```

`0`은 사용할 GPU 번호(`CUDA_VISIBLE_DEVICES`)입니다. 추가 Hydra 오버라이드는 뒤에 이어서
전달할 수 있습니다 (예: `bash scripts/train_z_axis.sh 0 task.env.numEnvs=64`로 소규모
스모크 테스트).
