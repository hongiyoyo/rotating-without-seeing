# Same diagnosis as early_term.py, but with the trained policy acting (as in the eval script).
# For every episode records: object, episode index within its env (0 = first after startup),
# length, end reason, and the start pose offset written in reset_idx.
import isaacgym  # noqa: F401
import os, sys, math
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
import numpy as np
import torch
from collections import Counter, defaultdict
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import isaacgymenvs

CK = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
N = 900
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

start = {}  # env -> (x_off, y_off, z_off, yaw)
orig = env.reset_idx


def wrapped(env_ids, goal_env_ids, is_test=False):
    orig(env_ids, goal_env_ids, is_test)
    ids = env_ids.cpu().numpy()
    off = (env.object_init_pos[env_ids] - env.object_init_state[env_ids, 0:3]).cpu().numpy()
    q = env.object_init_quat[env_ids]; q0 = env.object_init_state[env_ids, 3:7]
    from isaacgym.torch_utils import quat_mul, quat_conjugate
    r = quat_mul(q, quat_conjugate(q0)); yaw = (2 * torch.atan2(r[:, 2], r[:, 3])).cpu().numpy()
    for k, i in enumerate(ids):
        start[int(i)] = (off[k, 0], off[k, 1], off[k, 2], yaw[k])


env.reset_idx = wrapped
obses = player.env_reset(env)
player.get_batch_size(obses, 1)
ep_idx = np.zeros(N, int)
eps = []
for t in range(700):
    action = player.get_action(obses, is_deterministic=True)
    obses, rew, done, info = player.env_step(env, action)
    d = done.cpu().numpy().astype(bool)
    if d.any():
        x = env.object_pos[:, 0].cpu().numpy()
        gd = torch.norm(env.object_pos - env.goal_pos, dim=-1).cpu().numpy()
        L = env.progress_buf.cpu().numpy()
        for i in np.where(d)[0]:
            if L[i] >= env.max_episode_length - 1:
                reason = "timeout"
            elif x[i] < 0.59:
                reason = "x<0.59"
            elif gd[i] >= env.fall_dist:
                reason = "fall_dist"
            else:
                reason = "tilt"
            eps.append(dict(env=int(i), obj=names[cls[i]], idx=int(ep_idx[i]), len=int(L[i]), reason=reason, start=start.get(int(i))))
            ep_idx[i] += 1

print(f"\n{len(eps)} episodes")
short = [e for e in eps if e["len"] <= 10]
print("len<=10:", len(short), "| reasons:", Counter(e["reason"] for e in short))
print("len<=10 by episode index (0 = first after startup):", sorted(Counter(min(e["idx"], 3) for e in short).items()))
print("all episodes by index:", sorted(Counter(min(e["idx"], 3) for e in eps).items()))
print("all reasons:", Counter(e["reason"] for e in eps))
for n in names:
    es = [e for e in eps if e["obj"] == n]; s = [e for e in es if e["len"] <= 10]
    print(f"  {n:11s} {len(s):3d}/{len(es):3d} short | {dict(Counter(e['reason'] for e in s))}")
st = np.array([e["start"] for e in short if e["start"] is not None]); al = np.array([e["start"] for e in eps if e["start"] is not None])
if len(st):
    for j, nm in enumerate(["x_off", "y_off", "z_off", "yaw"]):
        print(f"start {nm}: short mean {st[:, j].mean():+.4f} (|.| {np.abs(st[:, j]).mean():.4f}) vs all mean {al[:, j].mean():+.4f} (|.| {np.abs(al[:, j]).mean():.4f})")
    # yaw dependence: fraction short by |yaw| bucket, per object (symmetry of shapes)
    for n in names:
        es = [e for e in eps if e["obj"] == n and e["start"] is not None]
        if not es: continue
        yaw = np.array([e["start"][3] for e in es]); sh = np.array([e["len"] <= 10 for e in es])
        b = np.digitize(np.mod(yaw, 2 * np.pi), np.linspace(0, 2 * np.pi, 9)[1:-1])
        print(f"  {n:11s} short-rate by yaw octant:", [f"{sh[b == k].mean():.2f}" if (b == k).any() else "-" for k in range(8)])
print("POLICY_DIAG_DONE")
