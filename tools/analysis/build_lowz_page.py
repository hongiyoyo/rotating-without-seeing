# Builds the standalone 0-1 mm evaluation page from the 0-10 mm page: same layout, new EVAL / REASONS data.
import json, sys, re, numpy as np
S = sys.argv[1]
d = np.load("/mnt/c/YangJaehong/rotating-without-seeing/tools/data/policy_eval_z1mm.npz", allow_pickle=True)
names = [str(n) for n in d["label_names"]]; lab = d["labels"]; rot = d["rotations"].astype(float)
L = d["ep_len"].astype(float); R = d["ep_reward"].astype(float); full = d["full_duration"].astype(float)
T = L * 0.1; rate = np.where(L > 0, rot / np.maximum(T, 1e-9), 0.0)
def st(x):
    p = np.percentile(x, [25, 50, 75])
    return {"mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4), "min": round(float(x.min()), 4),
            "p25": round(float(p[0]), 4), "median": round(float(p[1]), 4), "p75": round(float(p[2]), 4), "max": round(float(x.max()), 4)}
re_ = np.linspace(0, R.max(), 21); qe = np.linspace(0, rate.max(), 21)
def hist(x, e):
    c, _ = np.histogram(np.clip(x, e[0], e[-1]), bins=e); return {"counts": c.tolist(), "edges": [round(float(v), 3) for v in e]}
objs = {}
for i, n in enumerate(names):
    m = lab == i
    o = {"n": int(m.sum()), "reward": st(R[m]), "rotations": st(rot[m]), "rotation_rate": st(rate[m]), "time_alive_s": st(T[m]),
         "full_duration_rate": round(float(full[m].mean()), 4), "reward_hist": hist(R[m], re_), "rotrate_hist": hist(rate[m], qe)}
    o["rotation_rate"]["mean"] = round(float(rot[m].sum() / T[m].sum()), 4)  # time-weighted
    objs[n] = o
ov = {"reward": st(R), "rotation_rate": st(rate), "time_alive_s": st(T), "full_duration_rate": round(float(full.mean()), 4)}
ov["rotation_rate"]["mean"] = round(float(rot.sum() / T.sum()), 4)
EVAL = {"label_names": names, "n_total": int(len(R)), "control_dt": 0.1, "max_episode_length": int(d["max_episode_length"]), "objects": objs, "overall": ov}
rr = json.load(open("/mnt/c/YangJaehong/train_logs/viewer/reset_reasons_z1mm.json")); E = rr["episodes"]
REAS = {"obj": [names.index(e["obj"]) for e in E], "len": [e["len"] for e in E],
        "code": [0 if e["timeout"] else (1 if e["tilt"] else 0) + (2 if e["x"] else 0) + (4 if e["fall"] else 0) for e in E],
        "z": [round(e["z_off_mm"], 2) for e in E]}
src = open(S + "/rws2_retrain_eval.html", encoding="utf-8").read().split("\n")
out, ne, nr = [], 0, 0
for line in src:
    if line.startswith("const EVAL = "):
        line = "const EVAL = " + json.dumps(EVAL, separators=(",", ":")) + ";"; ne += 1
    elif line.startswith("const REASONS = "):
        line = "const REASONS = " + json.dumps(REAS, separators=(",", ":")) + ";"; nr += 1
    out.append(line)
assert ne == 1 and nr == 1, (ne, nr)
open(S + "/rws2_lowz_paper.html", "w", encoding="utf-8").write("\n".join(out))
print("overall", ov["rotation_rate"]["mean"], ov["full_duration_rate"], round(ov["reward"]["mean"], 2), len(R), "| reasons eps", len(E))
fast = sorted(names, key=lambda n: -objs[n]["rotation_rate"]["mean"])
print("fastest", fast[0], objs[fast[0]]["rotation_rate"]["mean"], "slowest", fast[-1], objs[fast[-1]]["rotation_rate"]["mean"])
