import json, numpy as np, sys
D = "/mnt/c/YangJaehong/rotating-without-seeing/tools/data/"; V = "/mnt/c/YangJaehong/train_logs/viewer/"
OUT = sys.argv[1]
def evals(path):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["label_names"]]; lab = d["labels"]; rot = d["rotations"].astype(float)
    L = d["ep_len"].astype(float); R = d["ep_reward"].astype(float); full = d["full_duration"].astype(bool)
    def st(m):
        return {"n": int(m.sum()), "reward": round(float(R[m].mean()), 2), "reward_median": round(float(np.median(R[m])), 2),
                "full": round(float(full[m].mean()), 4), "rate": round(float(rot[m].sum() / (L[m].sum() * 0.1)), 4),
                "time": round(float(L[m].mean() * 0.1), 2), "early": round(float(((L[m] <= 10) & ~full[m]).mean()), 4),
                "rot": round(float(rot[m].mean()), 2)}
    return names, {"objects": {n: st(lab == i) for i, n in enumerate(names)}, "overall": st(np.ones(len(L), bool))}
def reasons(path):
    d = json.load(open(path)); names = d["objects"]; E = d["episodes"]
    return {"obj": [names.index(e["obj"]) for e in E], "len": [e["len"] for e in E],
            "code": [0 if e["timeout"] else (1 if e["tilt"] else 0) + (2 if e["x"] else 0) + (4 if e["fall"] else 0) for e in E]}
names, base = evals(D + "policy_eval.npz"); _, low = evals(D + "policy_eval_z1mm.npz")
out = {"labels": names, "base": {"eval": base, "reasons": reasons(V + "reset_reasons.json")},
       "low": {"eval": low, "reasons": reasons(V + "reset_reasons_z1mm.json")}}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
for k in ["base", "low"]:
    o = out[k]["eval"]["overall"]; print(k, o)
for n in names:
    b, l = base["objects"][n], low["objects"][n]
    print(f"{n:11s} full {b['full']:.3f}->{l['full']:.3f} early {b['early']:.3f}->{l['early']:.3f} rate {b['rate']:.3f}->{l['rate']:.3f} rew {b['reward']:.0f}->{l['reward']:.0f}")
