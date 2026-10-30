# Which termination condition ends episodes within the first few control steps?
# Conditions in compute_hand_reward_finger: goal_dist >= fallDistance, tilt (angle_difference > 0.4*pi),
# deviation = object x - 0.59 < 0. Uses zero actions (hand holds its default pose).
import isaacgym  # noqa: F401
import os, sys, math
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
import isaacgymenvs

N = 900
with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[f"task.env.numEnvs={N}", "headless=True",
                                                   "task.env.deterministicObjectAssign=True"])
env = isaacgymenvs.make(cfg.seed, cfg.task_name, N, cfg.sim_device, cfg.rl_device,
                        cfg.graphics_device_id, True, False, False, False, cfg)
names = list(env.used_training_objects)
cls = torch.as_tensor(env.object_class_indices).cpu().numpy()
print("nominal object start x (unique):", np.unique(env.object_init_state[:, 0].cpu().numpy().round(4)),
      "| deviation threshold x = 0.59 | fallDistance", env.fall_dist)

zeros = torch.zeros((N, env.num_actions), device=env.device)
env.step(zeros)  # initial reset of all envs + first step
start_x = env.object_init_pos[:, 0].cpu().numpy().copy()  # written start pose of this episode
records = {}  # env -> (step, x, goal_dist, reason)
alive = np.ones(N, bool)
for t in range(1, 12):
    obs, rew, done, info = env.step(zeros)
    # state at end of this step, before the next step's reset
    x = env.object_pos[:, 0].cpu().numpy()
    gd = torch.norm(env.object_pos - env.goal_pos, dim=-1).cpu().numpy()
    d = done.cpu().numpy().astype(bool)
    for i in np.where(d & alive)[0]:
        reason = "x<0.59" if x[i] < 0.59 else ("fall_dist" if gd[i] >= env.fall_dist else "tilt/other")
        records[i] = (t, x[i], gd[i], reason)
    alive &= ~d

print(f"\nended within 11 steps: {len(records)}/{N}")
from collections import Counter
print("by reason:", Counter(r[3] for r in records.values()))
print("by step:", sorted(Counter(r[0] for r in records.values()).items()))
for k, n in enumerate(names):
    ids = np.where(cls == k)[0]
    ended = [i for i in ids if i in records]
    print(f"  {n:11s} {len(ended):3d}/{len(ids)} ended early | reasons {dict(Counter(records[i][3] for i in ended))}")
ended = np.array(sorted(records))
ok = np.array([i for i in range(N) if i not in records])
print("\nstart x of early-ended envs: min %.4f max %.4f mean %.4f" % (start_x[ended].min(), start_x[ended].max(), start_x[ended].mean()))
print("start x of surviving envs:   min %.4f max %.4f mean %.4f" % (start_x[ok].min(), start_x[ok].max(), start_x[ok].mean()))
print("fraction early-ended when start x < 0.595: %.2f, when start x >= 0.605: %.2f" % (
    np.mean([i in records for i in np.where(start_x < 0.595)[0]]), np.mean([i in records for i in np.where(start_x >= 0.605)[0]])))
print("EARLY_DONE")
