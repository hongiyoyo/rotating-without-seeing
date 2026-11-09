# Closed-loop policy switch on block_4: the generalist starts every episode; at K seconds a K-second
# tactile+joint classifier looks at the sensor history so far and, only if it says "block_4", the env
# switches to the block_4 specialist. Envs are split into equal groups in ONE simulation:
#   generalist only | oracle switch at K (always switch) | classifier switch at K   (for each K)
# Only the FIRST episode of every env is counted. Every object is block_4 (forceObject), so the
# classifier can only miss block_4 (no false switches on other objects by design).
#
# The classifier input is built exactly like tools/collect_tactile_data.py: after every env step,
# read [sensed_contacts, last_obs_buf[:, 6:22]]; the frame read on a done step belongs to the ending
# episode, so an episode's frames are the ones read after its own steps. Frame count == progress.
#
# Run in WSL (robosyn env):  python closed_loop_switch.py
#   env: PER_GROUP (default 1000), KS (default "2,5"), SPEC (specialist ckpt), CLS_SEED (default 0),
#        OUT (json), NPZ (per-env arrays)
import isaacgym  # noqa: F401
import os, sys, copy, json
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/tools")
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
import isaacgymenvs
from train_tactile_classifier import TactileCNN

GEN = REPO + "/runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth"
SPEC = os.environ.get("SPEC", REPO + "/runs/finetune_init/compare_best/spec_ep3311_best.pth")
PER_GROUP = int(os.environ.get("PER_GROUP", "1000"))
KS = [float(k) for k in os.environ.get("KS", "2,5").split(",")]
CLS_SEED = int(os.environ.get("CLS_SEED", "0"))
OUT = os.environ.get("OUT", "/mnt/c/YangJaehong/train_logs/closed_loop_switch.json")
NPZ = os.environ.get("NPZ", OUT[:-5] + ".npz")
TARGET = "block_4"

CONFIGS = [("generalist only", "none", None)]
for K in KS:
    CONFIGS += [(f"oracle switch @{K:g}s", "oracle", K), (f"classifier switch @{K:g}s", "classifier", K)]
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
player.restore(GEN); gen_model = copy.deepcopy(player.model)
player.restore(SPEC); spec_model = copy.deepcopy(player.model)
env = player.env
dev = env.device
ctrl_dt = env.dt * env.control_freq_inv
max_len = env.max_episode_length

# classifiers, one per K
cls = {}
for K in KS:
    path = f"{REPO}/tools/data/classifier_z1mm/k{K:g}_tactile+joint_pos_s{CLS_SEED}.pt"
    ck = torch.load(path, map_location=dev)
    assert list(ck["channels"]) == ["tactile", "joint_pos"], ck["channels"]
    m = TactileCNN(ck["in_channels"], ck["num_classes"]).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    steps = int(ck["window_steps"])
    assert steps == int(round(K / ctrl_dt)), (steps, K, ctrl_dt)
    cls[K] = dict(model=m, steps=steps, b4=list(ck["label_names"]).index(TARGET), names=list(ck["label_names"]))
max_steps = max(c["steps"] for c in cls.values())

group = torch.arange(N, device=dev) // PER_GROUP
grp_mode = [c[1] for c in CONFIGS]
grp_K = [c[2] for c in CONFIGS]
print("configs:", [c[0] for c in CONFIGS], "| envs per group", PER_GROUP, "| K steps", {K: c["steps"] for K, c in cls.items()})

buf = torch.zeros(N, max_steps, 32, device=dev)     # sensor history of the current episode
count = torch.zeros(N, dtype=torch.long, device=dev)  # frames read in the current episode
in_first = torch.ones(N, dtype=torch.bool, device=dev)  # still in the first real episode
switched = torch.zeros(N, dtype=torch.bool, device=dev)
pred = {K: torch.full((N,), -1, dtype=torch.long, device=dev) for K in KS}   # classifier output at K (all groups)
p_b4 = {K: torch.full((N,), float("nan"), device=dev) for K in KS}
act_jump = []  # |a_spec - a_gen| at the first post-switch step (switch smoothness)

obses = player.env_reset(env)
player.get_batch_size(obses, 1)
finished = np.zeros(N, bool)
length = np.zeros(N, int); completed = np.zeros(N, bool)

for step in range(max_len + 20):
    prog = torch.where(env.reset_buf.bool(), torch.zeros_like(env.progress_buf), env.progress_buf)
    player.model = gen_model; a_gen = player.get_action(obses, is_deterministic=True)
    player.model = spec_model; a_spec = player.get_action(obses, is_deterministic=True)
    use_spec = switched & in_first
    first_post = use_spec & torch.zeros_like(use_spec)
    for gi in range(G):
        if grp_K[gi] is not None:
            first_post |= (group == gi) & use_spec & (prog == cls[grp_K[gi]]["steps"])
    if first_post.any():
        act_jump.append((a_spec[first_post] - a_gen[first_post]).norm(dim=-1).cpu().numpy())
    action = torch.where(use_spec.unsqueeze(-1), a_spec, a_gen)
    obses, rew, done, info = player.env_step(env, action)

    dn = torch.as_tensor(done, device=dev).bool().view(-1)
    frame = torch.cat([env.sensed_contacts, env.last_obs_buf[:, 6:22]], dim=1)
    idx = count.clamp(max=max_steps - 1)
    write = count < max_steps
    buf[write, idx[write]] = frame[write]
    count += 1

    # classify envs whose current (first) episode just reached K steps
    for K, c in cls.items():
        hit = in_first & (count == c["steps"]) & ~dn
        if hit.any():
            ids = hit.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                x = buf[ids, :c["steps"]]
                logits = c["model"](x, torch.full((len(ids),), c["steps"], dtype=torch.long, device=dev))
                pr = torch.softmax(logits, -1)
            pred[K][ids] = pr.argmax(-1); p_b4[K][ids] = pr[:, c["b4"]]
            for gi in range(G):
                if grp_K[gi] != K:
                    continue
                sel = group[ids] == gi
                if grp_mode[gi] == "oracle":
                    switched[ids[sel]] = True
                elif grp_mode[gi] == "classifier":
                    switched[ids[sel]] = pr[sel].argmax(-1) == c["b4"]

    dn_np = dn.cpu().numpy()
    if dn_np.any():
        L = env.progress_buf.cpu().numpy()
        for i in np.where(dn_np)[0]:
            if finished[i] or L[i] <= 0:   # startup record (same rule as compare_switch.py)
                continue
            finished[i] = True; length[i] = L[i]; completed[i] = L[i] >= max_len - 1
        in_first = torch.as_tensor(~finished, device=dev)
        count[dn] = 0; buf[dn] = 0
    if finished.all():
        break

g_np = group.cpu().numpy(); sw = switched.cpu().numpy()
res = []
for gi, (name, mode, K) in enumerate(CONFIGS):
    s = g_np == gi
    r = dict(config=name, mode=mode, K=K, n=int(finished[s].sum()), completion=float(completed[s].mean()),
             mean_len_s=float(length[s].mean() * ctrl_dt))
    for KK in ([K] if K is not None else KS):
        st = cls[KK]["steps"]
        reached = s & (length > st)
        r[f"reached_{KK:g}s"] = int(reached.sum())
        r[f"fail_after_{KK:g}s"] = float((~completed[reached]).mean())
        r[f"fail_within_1s_after_{KK:g}s"] = float(((length <= st + 10) & ~completed)[reached].mean())
    if K is not None:
        reached = s & (length > cls[K]["steps"])
        r["switched_share"] = float(sw[reached].mean())
        # outcome split by switch decision (classifier groups)
        for flag, tag in [(True, "switched"), (False, "kept")]:
            m = reached & (sw == flag)
            r[f"n_{tag}"] = int(m.sum())
            r[f"fail_after_{tag}"] = float((~completed[m]).mean()) if m.any() else None
    res.append(r)
    print(r)

# online recognition, pooled over all groups (identical generalist rollout up to K)
recog = {}
for K, c in cls.items():
    pk = pred[K].cpu().numpy(); pb = p_b4[K].cpu().numpy()
    valid = pk >= 0
    hist = np.bincount(pk[valid], minlength=len(c["names"]))
    recog[f"{K:g}"] = dict(n=int(valid.sum()), recall=float((pk[valid] == c["b4"]).mean()),
                           pred_counts={c["names"][j]: int(hist[j]) for j in range(len(hist)) if hist[j]},
                           p_b4_quantiles={q: float(np.quantile(pb[valid], q)) for q in [0.01, 0.05, 0.1, 0.5]})
    print("recognition", K, recog[f"{K:g}"])
aj = np.concatenate(act_jump) if act_jump else np.zeros(0)
json.dump(dict(per_group=PER_GROUP, Ks=KS, spec=os.path.basename(SPEC), cls_seed=CLS_SEED, seed=int(cfg.seed),
               results=res, recognition=recog,
               switch_action_jump_median=float(np.median(aj)) if len(aj) else None), open(OUT, "w"), indent=1)
np.savez(NPZ, group=g_np, length=length, completed=completed, switched=sw,
         **{f"pred_{K:g}": pred[K].cpu().numpy() for K in KS}, **{f"p_b4_{K:g}": p_b4[K].cpu().numpy() for K in KS})
print("CLOSED_LOOP_DONE")
