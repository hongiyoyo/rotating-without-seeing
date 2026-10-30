import json, numpy as np
d = json.load(open("/mnt/c/YangJaehong/train_logs/viewer/reset_reasons.json"))
E = d["episodes"]; names = d["objects"]
z = np.array([e["z_off_mm"] for e in E]); L = np.array([e["len"] for e in E]); obj = np.array([names.index(e["obj"]) for e in E])
early = (L <= 10) & np.array([not e["timeout"] for e in E])
fail = np.array([not e["timeout"] for e in E])
bins = [0, 2, 4, 6, 8, 10.01]
print("z bin (mm): early(<=10 step) fail rate | any fail rate | n")
for a, b in zip(bins[:-1], bins[1:]):
    m = (z >= a) & (z < b); print(f"  {a:.0f}-{b:.0f}: {early[m].mean():.3f} | {fail[m].mean():.3f} | {m.sum()}")
for gname, idxs in [("block_4,cyl", [3, 5, 6, 7, 8]), ("block_1,2,3,5", [0, 1, 2, 4])]:
    g = np.isin(obj, idxs)
    print(gname, [f"{early[g & (z >= a) & (z < b)].mean():.2f}" for a, b in zip(bins[:-1], bins[1:])])
print("z of early fails mean %.2f vs all %.2f" % (z[early].mean(), z.mean()))
