import json, glob, numpy as np
OUT = "/mnt/c/Users/기계공학부/AppData/Local/Temp/claude/c--YangJaehong/c8046ee3-2a6a-4dba-93ce-5b4b346830c7/scratchpad/"
d = np.load("/mnt/c/YangJaehong/rotating-without-seeing/tools/data/policy_eval.npz", allow_pickle=True)
names = [str(n) for n in d["label_names"]]; lab = d["labels"]; rot = d["rotations"].astype(float)
L = d["ep_len"].astype(float); R = d["ep_reward"].astype(float); full = d["full_duration"].astype(float)
dt = 0.1; T = L * dt; rate = np.where(L > 0, rot / np.maximum(T, 1e-9), 0.0)
def st(x):
    p = np.percentile(x, [25, 50, 75])
    return {"mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4), "min": round(float(x.min()), 4),
            "p25": round(float(p[0]), 4), "median": round(float(p[1]), 4), "p75": round(float(p[2]), 4), "max": round(float(x.max()), 4)}
r_edges = np.linspace(0, R.max(), 21); q_edges = np.linspace(0, rate.max(), 21)
def hist(x, e):
    c, _ = np.histogram(np.clip(x, e[0], e[-1]), bins=e); return {"counts": c.tolist(), "edges": [round(float(v), 3) for v in e]}
objs = {}
for i, n in enumerate(names):
    m = lab == i
    objs[n] = {"n": int(m.sum()), "reward": st(R[m]), "rotations": st(rot[m]), "rotation_rate": st(rate[m]), "time_alive_s": st(T[m]),
               "full_duration_rate": round(float(full[m].mean()), 4), "reward_hist": hist(R[m], r_edges), "rotrate_hist": hist(rate[m], q_edges)}
EVAL = {"label_names": names, "n_total": int(len(R)), "control_dt": dt, "max_episode_length": int(d["max_episode_length"]),
        "objects": objs, "overall": {"reward": st(R), "rotation_rate": st(rate), "time_alive_s": st(T), "full_duration_rate": round(float(full.mean()), 4)}}
json.dump(EVAL, open(OUT + "eval.json", "w"), separators=(",", ":"))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = glob.glob("/mnt/c/YangJaehong/rotating-without-seeing/runs/z-axis-touch-only/*/summaries/events*")[0]
ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
E = 20100; B = 25
TRAIN = {"epochs": [int(s + (B + 1) // 2) for s in range(0, E - B + 1, B)]}
for n in names:
    v = np.full(E + 1, np.nan)
    for ev in ea.Scalars(f"PerObject/{n}/episode_reward"):
        if 1 <= ev.step <= E: v[ev.step] = ev.value
    TRAIN[n] = [None if np.all(np.isnan(v[s + 1:s + B + 1])) else round(float(np.nanmean(v[s + 1:s + B + 1])), 2) for s in range(0, E - B + 1, B)]
json.dump(TRAIN, open(OUT + "train.json", "w"), separators=(",", ":"))
print("overall rate per-episode mean:", EVAL["overall"]["rotation_rate"]["mean"], "reward", EVAL["overall"]["reward"]["mean"], "full", EVAL["overall"]["full_duration_rate"], "time", EVAL["overall"]["time_alive_s"]["mean"])
for n in names: print(n, objs[n]["rotation_rate"]["mean"], objs[n]["reward"]["median"], objs[n]["time_alive_s"]["mean"], "| train final", TRAIN[n][-1])
print("train points:", len(TRAIN["epochs"]), TRAIN["epochs"][:2], TRAIN["epochs"][-1])
