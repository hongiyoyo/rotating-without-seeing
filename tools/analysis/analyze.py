import glob, numpy as np, re, os
d = np.load("/mnt/c/YangJaehong/rotating-without-seeing/tools/data/policy_eval.npz", allow_pickle=True)
names = list(d["label_names"]); lab = d["labels"]; rot = d["rotations"]; L = d["ep_len"]; R = d["ep_reward"]; full = d["full_duration"]
print("EVAL per object: n, reward, rev/s(sum rot / sum time, dt=0.1s), full%")
for i, n in enumerate(names):
    m = lab == i
    print(f"  {n:11s} {m.sum():4d} {R[m].mean():8.2f} {rot[m].sum()/(L[m].sum()*0.1):.3f} {100*full[m].mean():5.1f}%")
print(f"  ALL        {len(R):4d} {R.mean():8.2f} {rot.sum()/(L.sum()*0.1):.3f} {100*full.mean():5.1f}%")
print("  per-object mean of means:", np.mean([R[lab==i].mean() for i in range(len(names))]).round(2), "| worst object:", min((R[lab==i].mean(), names[i]) for i in range(len(names))))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = glob.glob("/mnt/c/YangJaehong/rotating-without-seeing/runs/z-axis-touch-only/*/summaries/events*")[0]
ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
tags = [t for t in ea.Tags()["scalars"] if t.startswith("PerObject/") and t.endswith("/episode_reward")]
E = 20100
M = np.full((len(tags), E + 1), np.nan)
for j, t in enumerate(tags):
    for ev in ea.Scalars(t):
        if ev.step <= E: M[j, ev.step] = ev.value
print("step range check:", [ (t, len(ea.Scalars(t))) for t in tags[:2] ])
W = 200
def smooth(x):
    out = np.full_like(x, np.nan)
    for e in range(W, len(x)):
        w = x[e-W+1:e+1]; out[e] = np.nanmean(w)
    return out
S = np.vstack([smooth(M[j]) for j in range(len(tags))])
mean_c = np.nanmean(S, 0); min_c = np.nanmin(S, 0)
obj = [t.split("/")[1] for t in tags]
for label, c in [("max of object-mean", mean_c), ("max of worst-object", min_c)]:
    e = int(np.nanargmax(c))
    print(f"{label}: epoch {e} | mean {mean_c[e]:.1f} worst {min_c[e]:.1f} ({obj[int(np.nanargmin(S[:, e]))]})")
for e in [17500, 20100]:
    print(f"epoch {e}: mean {mean_c[e]:.1f} worst {min_c[e]:.1f} ({obj[int(np.nanargmin(S[:, e]))]})")
# top-5 candidates by each criterion among saved-checkpoint epochs (multiples of 100)
ck = sorted(int(re.search(r"_ep_(\d+)_", p).group(1)) for p in glob.glob("/mnt/c/YangJaehong/rotating-without-seeing/runs/z-axis-touch-only/*/nn/last_*_ep_*"))
print("saved last_ checkpoints:", len(ck), "range", ck[0], ck[-1])
for label, c in [("mean", mean_c), ("worst", min_c)]:
    top = sorted(ck, key=lambda e: -c[e])[:5]
    print(f"top5 saved by {label}:", [(e, round(mean_c[e], 1), round(min_c[e], 1)) for e in top])
