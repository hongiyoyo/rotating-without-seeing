# Phase 1: run the trained policy under evaluation conditions (900 envs, round-robin objects,
# deterministic actions) and record every block_4 episode that ends before the time limit.
# Frames are recorded once per physics step (1/60 s): object pose (7) + hand joint positions (22).
# Output: C:\YangJaehong\train_logs\viewer\block4_failed_episodes.pkl
#
# Run in WSL (robosyn env):  python record_failed_episodes.py [max_control_steps]
import isaacgym  # noqa: F401  (must precede torch)
import os, sys, math, pickle
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
TARGET = os.environ.get("TARGET", "block_4")
OUT = os.environ.get("OUT", f"/mnt/c/YangJaehong/train_logs/viewer/{TARGET.replace('_', '')}_failed_episodes.pkl")
PICK = int(os.environ.get("PICK", "0"))  # >0: keep only this many, spread over the failure-length distribution
CK = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import isaacgymenvs
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, quat_mul, quat_conjugate

N = 900
MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500  # the 10-30 evaluation stopped at step 1498


with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[f"task.env.numEnvs={N}", "headless=True", "test=True",
                                                   "task.env.deterministicObjectAssign=True", f"checkpoint={CK}"])

from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def create_env_thunk(**kwargs):
    return isaacgymenvs.make(cfg.seed, cfg.task_name, N, cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                             True, False, False, False, cfg, **kwargs)


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {'vecenv_type': 'RLGPU', 'env_creator': create_env_thunk})
runner = Runner(); runner.load(omegaconf_to_dict(cfg.train))
player = runner.create_player(); player.restore(CK)
env = player.env

names = list(env.used_training_objects)
cls = torch.as_tensor(env.object_class_indices).cpu().numpy()
tgt_envs = np.where(cls == names.index(TARGET))[0]
tgt_t = torch.as_tensor(tgt_envs, device=env.device)
obj_idx = env.object_indices[tgt_t]
print(f"{TARGET}: {len(tgt_envs)} of {N} envs")

frames = {int(i): [] for i in tgt_envs}  # env -> list of (29,) frames
last_frame = None


def snapshot():
    o = env.root_state_tensor[obj_idx, 0:7]
    q = env.arm_hand_dof_pos[tgt_t]
    return torch.cat([o, q], dim=1).cpu().numpy()


orig_uc = env.update_controller


def uc_wrapped():
    orig_uc()  # refreshes all state tensors first; we only copy them afterwards
    s = snapshot()
    for k, i in enumerate(tgt_envs):
        frames[int(i)].append(s[k])


env.update_controller = uc_wrapped
orig_pp = env.post_physics_step


def pp_wrapped():
    global last_frame
    orig_pp()
    last_frame = snapshot()


env.post_physics_step = pp_wrapped

obses = player.env_reset(env)
player.get_batch_size(obses, 1)
for i in frames:
    frames[i] = []
ep_count = {int(i): 0 for i in tgt_envs}
episodes, n_total, n_timeout = [], 0, 0
max_len = env.max_episode_length

for step in range(MAX_STEPS):
    action = player.get_action(obses, is_deterministic=True)
    obses, rew, done, info = player.env_step(env, action)
    d = torch.as_tensor(done).cpu().numpy().astype(bool)[tgt_envs]
    if not d.any():
        continue
    L = env.progress_buf[tgt_t].cpu().numpy()
    x = env.object_pos[tgt_t, 0].cpu().numpy()
    gd = torch.norm(env.object_pos[tgt_t] - env.goal_pos[tgt_t], dim=-1).cpu().numpy()
    main = env.spin_axis[tgt_t]
    q0 = env.object_init_quat[tgt_t]; q = env.object_rot[tgt_t]
    cur = quat_rotate(q, quat_rotate_inverse(q0, main))
    tilt = torch.rad2deg(torch.arccos(torch.clamp((main * cur).sum(-1), -1, 1))).cpu().numpy()
    q00 = env.object_init_state[tgt_t, 3:7]
    r = quat_mul(q0, quat_conjugate(q00)); yaw = torch.rad2deg(2 * torch.atan2(r[:, 2], r[:, 3])).cpu().numpy()
    off = (env.object_init_pos[tgt_t] - env.object_init_state[tgt_t, 0:3]).cpu().numpy()
    for k in np.where(d)[0]:
        i = int(tgt_envs[k])
        n_total += 1
        if L[k] >= max_len - 1:
            n_timeout += 1
        elif L[k] > 0:
            fr = np.stack(frames[i] + [last_frame[k]]).astype(np.float32)
            episodes.append(dict(
                env=i, ep_in_env=ep_count[i], control_steps=int(L[k]), frames=fr,
                tilt_deg=float(tilt[k]), x=float(x[k]), goal_dist=float(gd[k]),
                hit_tilt=bool(tilt[k] > 72.0), hit_x=bool(x[k] < 0.59), hit_fall=bool(gd[k] >= env.fall_dist),
                start_offset_cm=(off[k] * 100).tolist(), start_yaw_deg=float(yaw[k]), global_step=step))
        ep_count[i] += 1
        frames[i] = []
    if (step + 1) % 100 == 0:
        print(f"step {step + 1}/{MAX_STEPS}: {TARGET} episodes {n_total}, failed {len(episodes)}, completed {n_timeout}", flush=True)

if PICK > 0 and len(episodes) > PICK:
    order = sorted(range(len(episodes)), key=lambda j: episodes[j]["control_steps"])
    keep = sorted({order[round(q * (len(order) - 1))] for q in np.linspace(0, 1, PICK)})
    episodes = [episodes[j] for j in keep]
    print("picked", len(episodes), "episodes with lengths", [e["control_steps"] for e in episodes])
meta = dict(object=TARGET, physics_dt=float(env.dt), control_dt=float(env.dt * env.control_freq_inv),
            fall_dist=float(env.fall_dist), tilt_limit_deg=72.0, x_limit=0.59, max_episode_length=int(max_len),
            n_total=n_total, n_completed=n_timeout, checkpoint=CK)
with open(OUT, "wb") as f:
    pickle.dump(dict(meta=meta, episodes=episodes), f)
lens = np.array([e["control_steps"] for e in episodes])
print(f"saved {len(episodes)} failed {TARGET} episodes ({n_timeout} completed, {n_total} total) -> {OUT}")
print("failed lengths: <=5:", (lens <= 5).sum(), "6-50:", ((lens > 5) & (lens <= 50)).sum(), ">50:", (lens > 50).sum())
print("RECORD_DONE")
