# Re-run the per-object evaluation (same settings and seed as tools/evaluate_policy_per_object.py:
# 900 envs, round-robin objects, deterministic actions, seed 42, stop when every object has 300
# episodes) and record, for every episode, which termination condition ended it.
# Conditions (compute_hand_reward_finger): tilt > 0.4*pi (72 deg), object x < 0.59, distance to goal >= fallDistance.
# Output: C:\YangJaehong\train_logs\viewer\reset_reasons.json
import isaacgym  # noqa: F401
import os, sys, json
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
OUT = os.environ.get("OUT", "/mnt/c/YangJaehong/train_logs/viewer/reset_reasons.json")
CK = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import isaacgymenvs
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, quat_mul, quat_conjugate

N, TARGET_PER_CLASS, MAX_STEPS = 900, 300, 3000
with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[f"task.env.numEnvs={N}", "headless=True", "test=True",
                                                   "task.env.deterministicObjectAssign=True", f"checkpoint={CK}"] + sys.argv[1:])
set_np_formatting()
cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

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
max_len = env.max_episode_length

obses = player.env_reset(env)
player.get_batch_size(obses, 1)
if hasattr(env, "_pending_obj_stats"):
    env._pending_obj_stats.clear()

eps = []
counts = np.zeros(len(names), int)
for step in range(MAX_STEPS):
    action = player.get_action(obses, is_deterministic=True)
    obses, rew, done, info = player.env_step(env, action)
    env._pending_obj_stats.clear()
    d = torch.as_tensor(done).cpu().numpy().astype(bool)
    if d.any():
        ids = torch.as_tensor(np.where(d)[0], device=env.device)
        L = env.progress_buf[ids].cpu().numpy()
        x = env.object_pos[ids, 0].cpu().numpy()
        gd = torch.norm(env.object_pos[ids] - env.goal_pos[ids], dim=-1).cpu().numpy()
        main = env.spin_axis[ids]; q0 = env.object_init_quat[ids]
        cur = quat_rotate(env.object_rot[ids], quat_rotate_inverse(q0, main))
        tilt = torch.rad2deg(torch.arccos(torch.clamp((main * cur).sum(-1), -1, 1))).cpu().numpy()
        off = (env.object_init_pos[ids] - env.object_init_state[ids, 0:3]).cpu().numpy()
        r = quat_mul(q0, quat_conjugate(env.object_init_state[ids, 3:7]))
        yaw = torch.rad2deg(2 * torch.atan2(r[:, 2], r[:, 3])).cpu().numpy()
        for k, i in enumerate(np.where(d)[0]):
            if L[k] <= 0:
                continue  # startup record, dropped like the eval script does
            c = int(cls[i]); counts[c] += 1
            eps.append(dict(obj=names[c], len=int(L[k]), timeout=bool(L[k] >= max_len - 1),
                            tilt=bool(tilt[k] > 72.0), x=bool(x[k] < 0.59), fall=bool(gd[k] >= env.fall_dist),
                            tilt_deg=round(float(tilt[k]), 1), z_off_mm=round(float(off[k, 2] * 1000), 2),
                            yaw_deg=round(float(yaw[k]), 1)))
    if counts.min() >= TARGET_PER_CLASS:
        print(f"reached {TARGET_PER_CLASS} episodes/object at step {step + 1}")
        break

json.dump(dict(objects=names, episodes=eps, control_dt=0.1), open(OUT, "w"))
print("episodes per object:", dict(zip(names, counts.tolist())))
fails = [e for e in eps if not e["timeout"]]
from collections import Counter
print("failed:", len(fails), "| combos:", Counter((e["tilt"], e["x"], e["fall"]) for e in fails))
print("REASONS_DONE")
