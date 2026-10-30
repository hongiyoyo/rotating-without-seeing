# Env-level smoke check for rotating-without-seeing (no training).
# Verifies: config values, object set, obs dim, GPU sim, per-env mass randomization,
# and that the object init pose written in reset_idx actually reaches the sim
# (README pitfall #7), incl. resets where vec_task DR re-applies hand dof props.
import isaacgym  # noqa: F401  (must precede torch)
import os
import sys
import math

REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)

import numpy as np
import torch
from hydra import initialize_config_dir, compose
import isaacgymenvs
from isaacgym.torch_utils import quat_mul, quat_conjugate

N = int(os.environ.get("N", "512"))

with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[f"task.env.numEnvs={N}", "headless=True"])

env = isaacgymenvs.make(cfg.seed, cfg.task_name, N, cfg.sim_device, cfg.rl_device,
                        cfg.graphics_device_id, True, False, False, False, cfg)
gym, sim = env.gym, env.sim

print("\n==== CONFIG ====")
print("objSet:", env.object_set_id, "| objects:", list(env.used_training_objects))
print("num_obs:", env.num_obs, "| num_states:", env.num_states, "| num_actions:", env.num_actions)
print("device:", env.device, "| use_gpu_pipeline:", env.sim_params.use_gpu_pipeline,
      "| physx.use_gpu:", env.sim_params.physx.use_gpu)
print("obs_type:", env.obs_type, "| ablation_mode:", env.cfg["env"].get("ablation_mode"))
print("drRampEnable:", env.dr_ramp_enable, "| useInitRandomRotation:", env.use_initial_rotation)
print("objInitPos xy/zlo/zhi:", env.obj_init_pos_noise_xy, env.obj_init_pos_z_low, env.obj_init_pos_z_high)
print("mass range:", env.randomize_mass_lower, env.randomize_mass_upper,
      "| episodeLength:", env.max_episode_length, "| controlFrequencyInv:", env.control_freq_inv)
print("object class counts:", np.bincount(torch.as_tensor(env.object_class_indices).cpu().numpy(), minlength=len(env.used_training_objects)).tolist())


def obj_masses():
    return np.array([gym.get_actor_rigid_body_properties(env.envs[i], env.objects[i])[0].mass
                     for i in range(N)])


m0 = obj_masses()
print("cached object_env_masses == sim masses:", bool(np.allclose(env.object_env_masses.cpu().numpy(), m0)))
print("mass at creation: min %.3f max %.3f mean %.3f unique %d" % (m0.min(), m0.max(), m0.mean(), len(np.unique(m0.round(5)))))

captured = []
orig_reset_idx = env.reset_idx


def wrapped(env_ids, goal_env_ids, is_test=False):
    orig_reset_idx(env_ids, goal_env_ids, is_test)
    oi = env.object_indices[env_ids]
    captured.append((env_ids.clone(), env.root_state_tensor[oi, :7].clone(), env.object_init_state[env_ids, :7].clone()))


env.reset_idx = wrapped


def rel_yaw(q, q0):
    r = quat_mul(q, quat_conjugate(q0))  # xyzw
    return 2.0 * torch.atan2(r[:, 2], r[:, 3])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


results = {}


def evaluate(label):
    gym.refresh_actor_root_state_tensor(sim)
    for env_ids, written, nominal in captured:
        obs = env.root_state_tensor[env.object_indices[env_ids], :7]
        w_off = (written[:, :3] - nominal[:, :3]).cpu().numpy()
        o_off = (obs[:, :3] - nominal[:, :3]).cpu().numpy()
        w_yaw = rel_yaw(written[:, 3:7], nominal[:, 3:7])
        o_yaw = rel_yaw(obs[:, 3:7], nominal[:, 3:7])
        dyaw = wrap(o_yaw - w_yaw).abs().cpu().numpy()
        dxy = np.linalg.norm(o_off[:, :2] - w_off[:, :2], axis=1)
        r = results.setdefault(label, dict(n=0, dyaw=[], dxy=[], wyaw=[], oyaw_abs=[], woff=[], ooffxy=[]))
        r["n"] += len(env_ids)
        r["dyaw"].append(dyaw); r["dxy"].append(dxy)
        r["wyaw"].append(w_yaw.cpu().numpy()); r["oyaw_abs"].append(o_yaw.abs().cpu().numpy())
        r["woff"].append(w_off); r["ooffxy"].append(o_off[:, :2])
    captured.clear()


zeros = torch.zeros((N, env.num_actions), device=env.device)
nan_obs = 0
pending_total = 0

# step 0: initial reset of all envs (first vec_task randomization -> hand dof props for all envs)
env.step(zeros); evaluate("initial reset (all envs, DR first call)")
for t in range(1, 10):
    env.step(zeros)

# forced reset, even envs, no vec_task actor-param re-randomization
env.reset_buf[0::2] = 1
env.step(zeros); evaluate("forced reset (no actor DR)")
for t in range(5):
    env.step(zeros)

# forced reset, odd envs, with vec_task actor-param re-randomization (randomize_buf >= frequency)
env.randomize_buf[:] = 10_000
env.reset_buf[1::2] = 1
env.step(zeros); evaluate("forced reset (with actor DR)")

# random actions: natural resets (drops) over 600 control steps (> episodeLength)
for t in range(600):
    a = torch.rand((N, env.num_actions), device=env.device) * 2 - 1
    obs_dict, rew, done, info = env.step(a)
    o = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
    nan_obs += int(torch.isnan(o).any().item()) + int(torch.isnan(rew).any().item())
    if captured:
        evaluate("natural resets (random actions)")
    pending_total += len(env._pending_obj_stats)
    if t == 0:
        sample = env._pending_obj_stats[:2]
    env._pending_obj_stats.clear()

print("\n==== INIT POSE: written in reset_idx vs. sim after next step ====")
for label, r in results.items():
    dyaw = np.concatenate(r["dyaw"]); dxy = np.concatenate(r["dxy"])
    wyaw = np.concatenate(r["wyaw"]); oyaw = np.concatenate(r["oyaw_abs"])
    woff = np.concatenate(r["woff"]); ooff = np.concatenate(r["ooffxy"])
    corr_x = np.corrcoef(woff[:, 0], ooff[:, 0])[0, 1] if len(woff) > 2 else float("nan")
    print(f"- {label}: n={r['n']}")
    print(f"    |yaw sim - yaw written| median {np.median(dyaw):.4f} rad, p95 {np.percentile(dyaw, 95):.4f}, frac<0.1rad {np.mean(dyaw < 0.1):.3f}")
    print(f"    |xy sim - xy written|  median {np.median(dxy)*1000:.2f} mm, p95 {np.percentile(dxy, 95)*1000:.2f} mm")
    print(f"    written yaw range [{wyaw.min():.3f}, {wyaw.max():.3f}] std {wyaw.std():.3f} (U(-pi,pi) std=1.814); sim |yaw| mean {oyaw.mean():.3f} (bug => ~0)")
    print(f"    written x off [{woff[:,0].min()*100:.2f},{woff[:,0].max()*100:.2f}]cm y off [{woff[:,1].min()*100:.2f},{woff[:,1].max()*100:.2f}]cm z off [{woff[:,2].min()*100:.2f},{woff[:,2].max()*100:.2f}]cm; corr(written x, sim x)={corr_x:.3f}")

m1 = obj_masses()
print("\n==== OTHER ====")
print("mass unchanged after resets:", bool(np.allclose(m0, m1)), "| max diff", float(np.abs(m0 - m1).max()))
print("NaN in obs/reward (count of steps):", nan_obs)
print("per-object episode records during random phase:", pending_total, "| sample:", sample)
print("GPU mem allocated (torch): %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("ENV_CHECK_DONE")
