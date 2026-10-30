# Policy-switch check: block_4 in every env; the generalist acts for the first SWITCH_STEPS control
# steps of each episode, then the block_4 specialist takes over. Records episodes that run past the
# switch (object pose + hand joints every physics step, plus which policy acted at each frame) for
# replay_viewer.py, and prints simple switch diagnostics.
#
# Run in WSL (robosyn env):  python record_switch.py
#   env: OUT, PICK (episodes to keep, default 10), SWITCH_SEC (default 5), STEPS (control steps, default 1000)
import isaacgym  # noqa: F401  (must precede torch)
import os, sys, copy, pickle
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
import isaacgymenvs
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, quat_mul, quat_conjugate

GEN = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
SPEC = REPO + "/runs/finetune_init/block4_ft_best_snapshot.pth"
OUT = os.environ.get("OUT", "/mnt/c/YangJaehong/train_logs/viewer/block4_switch_episodes.pkl")
PICK = int(os.environ.get("PICK", "10"))
SWITCH_SEC = float(os.environ.get("SWITCH_SEC", "5"))
STEPS = int(os.environ.get("STEPS", "1000"))
N = 100
TARGET = "block_4"

with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[f"task.env.numEnvs={N}", "headless=True", "test=True",
                                                   f"task.env.forceObject={TARGET}", "task.env.objInitPosZHigh=0.001",
                                                   f"checkpoint={GEN}"])
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
player = runner.create_player()
player.restore(GEN); model_gen = copy.deepcopy(player.model)
player.restore(SPEC); model_spec = player.model
spec_ck = torch.load(SPEC, map_location="cpu")
print(f"specialist snapshot: epoch {spec_ck.get('epoch')}, best rolling reward {spec_ck.get('last_mean_rewards')}")
env = player.env
switch_steps = int(round(SWITCH_SEC / (env.dt * env.control_freq_inv)))
max_len = env.max_episode_length

frames = {i: [] for i in range(N)}
cur_policy = np.zeros(N, np.int8)  # 0 = generalist, 1 = specialist, for frames recorded this control step
last_frame = None


def snapshot():
    o = env.root_state_tensor[env.object_indices, 0:7]
    q = env.arm_hand_dof_pos
    return torch.cat([o, q], dim=1).cpu().numpy()


orig_uc = env.update_controller


def uc_wrapped():
    orig_uc()
    s = snapshot()
    for i in range(N):
        frames[i].append(np.concatenate([s[i], [cur_policy[i]]]))


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
prev_action = torch.zeros((N, env.num_actions), device=env.device)
step_deltas, switch_deltas = [], []
episodes, n_total = [], 0
ep_count = np.zeros(N, int)

for step in range(STEPS):
    # steps already taken in the current episode; an env flagged for reset starts a new episode in this step
    prog = torch.where(env.reset_buf.bool(), torch.zeros_like(env.progress_buf), env.progress_buf)
    use_spec = (prog >= switch_steps)
    player.model = model_gen; a_gen = player.get_action(obses, is_deterministic=True)
    player.model = model_spec; a_spec = player.get_action(obses, is_deterministic=True)
    action = torch.where(use_spec.unsqueeze(1), a_spec, a_gen)
    cur_policy[:] = use_spec.cpu().numpy().astype(np.int8)
    d = (action - prev_action).norm(dim=1)
    at_switch = (prog == switch_steps)
    if at_switch.any():
        switch_deltas += d[at_switch].cpu().tolist()
    mid = (prog > 1) & ~at_switch
    step_deltas += d[mid].cpu().tolist()
    prev_action = action.clone()
    obses, rew, done, info = player.env_step(env, action)
    dn = torch.as_tensor(done).cpu().numpy().astype(bool)
    if not dn.any():
        continue
    ids = torch.as_tensor(np.where(dn)[0], device=env.device)
    L = env.progress_buf[ids].cpu().numpy()
    x = env.object_pos[ids, 0].cpu().numpy()
    gd = torch.norm(env.object_pos[ids] - env.goal_pos[ids], dim=-1).cpu().numpy()
    main = env.spin_axis[ids]; q0 = env.object_init_quat[ids]
    tilt = torch.rad2deg(torch.arccos(torch.clamp((main * quat_rotate(env.object_rot[ids], quat_rotate_inverse(q0, main))).sum(-1), -1, 1))).cpu().numpy()
    r = quat_mul(q0, quat_conjugate(env.object_init_state[ids, 3:7]))
    yaw = torch.rad2deg(2 * torch.atan2(r[:, 2], r[:, 3])).cpu().numpy()
    off = (env.object_init_pos[ids] - env.object_init_state[ids, 0:3]).cpu().numpy()
    rot = env.episode_fwd_theta_buf[ids].cpu().numpy() / (2 * np.pi) if hasattr(env, "episode_fwd_theta_buf") else np.zeros(len(ids))
    for k, i in enumerate(np.where(dn)[0]):
        if L[k] > 0:
            n_total += 1
            completed = bool(L[k] >= max_len - 1)
            fr = np.stack(frames[i] + [np.concatenate([last_frame[i], [cur_policy[i]]])]).astype(np.float32)
            episodes.append(dict(env=int(i), ep_in_env=int(ep_count[i]), control_steps=int(L[k]), frames=fr[:, :29],
                                 policy=fr[:, 29].astype(np.int8), completed=completed, switch_step=switch_steps,
                                 tilt_deg=float(tilt[k]), x=float(x[k]), goal_dist=float(gd[k]),
                                 hit_tilt=bool(not completed and tilt[k] > 72.0), hit_x=bool(not completed and x[k] < 0.59),
                                 hit_fall=bool(not completed and gd[k] >= env.fall_dist),
                                 start_offset_cm=(off[k] * 100).tolist(), start_yaw_deg=float(yaw[k]), global_step=step))
        ep_count[i] += 1
        frames[i] = []
    if (step + 1) % 100 == 0:
        print(f"step {step + 1}/{STEPS}: episodes {n_total}", flush=True)

past = [e for e in episodes if e["control_steps"] > switch_steps]
fail_after = [e for e in past if not e["completed"]]
within1s = [e for e in fail_after if e["control_steps"] <= switch_steps + 10]
print(f"\nepisodes {len(episodes)}; reached the switch at {SWITCH_SEC:g} s: {len(past)}; "
      f"completed {sum(e['completed'] for e in past)}; failed after switch {len(fail_after)} "
      f"({100 * len(fail_after) / max(len(past), 1):.1f}%); failed within 1 s after switch {len(within1s)}")
print(f"action change per step: median {np.median(step_deltas):.3f}, p95 {np.percentile(step_deltas, 95):.3f} | "
      f"at the switch step: median {np.median(switch_deltas):.3f}, p95 {np.percentile(switch_deltas, 95):.3f}, n {len(switch_deltas)}")

# keep PICK episodes that ran past the switch: failures after the switch first (up to half), then completed ones
fails_sorted = sorted(fail_after, key=lambda e: e["control_steps"])
keep = fails_sorted[:PICK // 2]
keep += [e for e in past if e["completed"]][:PICK - len(keep)]
keep += [e for e in fails_sorted[PICK // 2:]][:PICK - len(keep)]
keep = sorted(keep, key=lambda e: e["global_step"])
meta = dict(object=TARGET, physics_dt=float(env.dt), control_dt=float(env.dt * env.control_freq_inv),
            fall_dist=float(env.fall_dist), tilt_limit_deg=72.0, x_limit=0.59, max_episode_length=int(max_len),
            n_total=len(past), n_completed=int(sum(e["completed"] for e in past)), switch_sec=SWITCH_SEC,
            policies=["GENERALIST", f"SPECIALIZED ({TARGET.upper()})"], generalist=GEN, specialist=SPEC,
            specialist_epoch=spec_ck.get("epoch"))
with open(OUT, "wb") as f:
    pickle.dump(dict(meta=meta, episodes=keep), f)
print("saved", len(keep), "episodes:", [(e["control_steps"], "completed" if e["completed"] else "failed") for e in keep], "->", OUT)
print("SWITCH_DONE")
