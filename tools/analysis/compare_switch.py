# Compare policy-switch configurations on block_4 in ONE simulation: envs are split into equal
# groups, each group has its own (policy before the switch, policy after the switch). Only the FIRST
# episode of every env is counted, so completion is not biased toward short episodes.
#
# Run in WSL (robosyn env):  python compare_switch.py
#   env: PER_GROUP (default 200), SWITCH_SEC (default 5), OUT (json)
import isaacgym  # noqa: F401
import os, sys, copy, json, glob, re
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
import isaacgymenvs
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

GEN = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
SPEC_DIR = REPO + "/runs/finetune_init/compare"  # copies of the specialist checkpoints (training keeps writing the originals)
PER_GROUP = int(os.environ.get("PER_GROUP", "200"))
SWITCH_SEC = float(os.environ.get("SWITCH_SEC", "5"))
OUT = os.environ.get("OUT", "/mnt/c/YangJaehong/train_logs/compare_switch.json")
TARGET = "block_4"

specs = {}
for f in sorted(glob.glob(SPEC_DIR + "/*.pth")):
    specs[os.path.basename(f)[:-4]] = f
spec_names = sorted(specs, key=lambda n: int(re.search(r"(\d+)", n).group(1)))
last = spec_names[-1]
CONFIGS = [("generalist only", "G", "G")] + [(f"G -> {n}", "G", n) for n in spec_names] + [(f"{last} only", last, last)]
G = len(CONFIGS)
N = PER_GROUP * G

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
models = {}
player.restore(GEN); models["G"] = copy.deepcopy(player.model)
for n in spec_names:
    player.restore(specs[n]); models[n] = copy.deepcopy(player.model)
env = player.env
switch_steps = int(round(SWITCH_SEC / (env.dt * env.control_freq_inv)))
max_len = env.max_episode_length
group = torch.arange(N, device=env.device) // PER_GROUP
print("configs:", [c[0] for c in CONFIGS], "| envs per group", PER_GROUP, "| switch at", switch_steps, "steps")

obses = player.env_reset(env)
player.get_batch_size(obses, 1)
finished = np.zeros(N, bool)
length = np.zeros(N, int); completed = np.zeros(N, bool)
end_tilt = np.zeros(N, bool); end_x = np.zeros(N, bool); end_fall = np.zeros(N, bool)
first_started = np.zeros(N, bool)

for step in range(max_len + 20):
    prog = torch.where(env.reset_buf.bool(), torch.zeros_like(env.progress_buf), env.progress_buf)
    acts = {}
    for name, m in models.items():
        player.model = m
        acts[name] = player.get_action(obses, is_deterministic=True)
    action = torch.zeros_like(acts["G"])
    for gi, (_, pre, post) in enumerate(CONFIGS):
        sel = (group == gi)
        use_post = sel & (prog >= switch_steps)
        use_pre = sel & (prog < switch_steps)
        action[use_pre] = acts[pre][use_pre]
        action[use_post] = acts[post][use_post]
    obses, rew, done, info = player.env_step(env, action)
    dn = torch.as_tensor(done).cpu().numpy().astype(bool)
    newly = dn & ~finished
    if newly.any():
        ids = torch.as_tensor(np.where(newly)[0], device=env.device)
        L = env.progress_buf[ids].cpu().numpy()
        main = env.spin_axis[ids]
        tilt = torch.rad2deg(torch.arccos(torch.clamp((main * quat_rotate(env.object_rot[ids], quat_rotate_inverse(env.object_init_quat[ids], main))).sum(-1), -1, 1))).cpu().numpy()
        x = env.object_pos[ids, 0].cpu().numpy(); gd = torch.norm(env.object_pos[ids] - env.goal_pos[ids], dim=-1).cpu().numpy()
        for k, i in enumerate(np.where(newly)[0]):
            if L[k] <= 0:
                continue  # startup record
            finished[i] = True; length[i] = L[k]; completed[i] = L[k] >= max_len - 1
            end_tilt[i] = (not completed[i]) and tilt[k] > 72; end_x[i] = (not completed[i]) and x[k] < 0.59
            end_fall[i] = (not completed[i]) and gd[k] >= env.fall_dist
    if finished.all():
        break

res = []
for gi, (name, pre, post) in enumerate(CONFIGS):
    s = slice(gi * PER_GROUP, (gi + 1) * PER_GROUP)
    f, L, c = finished[s], length[s], completed[s]
    reached = f & (L > switch_steps)
    r = dict(config=name, n=int(f.sum()), completion=float(c[f].mean()),
             early_1s=float(((L <= 10) & ~c)[f].mean()),
             reached_switch=int(reached.sum()),
             fail_after_switch=float((~c[reached]).mean()) if reached.any() else None,
             fail_within_1s_after=float(((L > switch_steps) & (L <= switch_steps + 10) & ~c)[f].sum() / max(reached.sum(), 1)),
             tilt_share=float(end_tilt[s][f & ~c].mean()) if (f & ~c).any() else None,
             mean_len_s=float(L[f].mean() * 0.1))
    res.append(r)
    print(f"{name:28s} n={r['n']:3d} completion {100 * r['completion']:5.1f}% | reached 5 s {r['reached_switch']:3d}, "
          f"failed after {100 * (r['fail_after_switch'] or 0):5.1f}% (within 1 s: {100 * r['fail_within_1s_after']:4.1f}%) | "
          f"ends within 1 s {100 * r['early_1s']:4.1f}% | tilt share {100 * (r['tilt_share'] or 0):5.1f}%")
json.dump(dict(per_group=PER_GROUP, switch_sec=SWITCH_SEC, results=res), open(OUT, "w"), indent=1)
print("COMPARE_DONE")
