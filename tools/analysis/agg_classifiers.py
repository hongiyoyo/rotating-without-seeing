import json, glob, os, sys, numpy as np
D = "/mnt/c/YangJaehong/rotating-without-seeing/tools/data/classifier_z1mm/"
OUT = sys.argv[1]
d = np.load("/mnt/c/YangJaehong/rotating-without-seeing/tools/data/tactile_dataset_z1mm.npz")
lengths, labels, names = d["lengths"], d["labels"], [str(n) for n in d["label_names"]]
Ks = [1, 2, 5, 10, 20, 50]; CH = ["tactile", "joint_pos", "tactile+joint_pos"]
res = {"labels": names, "Ks": Ks, "channels": CH, "runs": {}, "dataset": {
    "n": int(len(labels)), "per_class": [int((labels == c).sum()) for c in range(len(names))],
    "eligible": {str(K): [int(((labels == c) & (lengths >= min(int(round(K / 0.1)), 499))).sum()) for c in range(len(names))] for K in Ks}}}
for K in Ks:
    for ch in CH:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}k{K}_{ch}_s*.json"))]
        res["runs"][f"{K}|{ch}"] = {
            "acc": [r["test_acc"] for r in runs], "n_train": runs[0]["n_train"], "n_val": runs[0]["n_val"], "n_test": runs[0]["n_test"],
            "per_class": [[r["per_class"][n]["acc"] for n in names] for r in runs],
            "per_class_n": [runs[0]["per_class"][n]["n"] for n in names],
            "confusion": np.sum([r["confusion"] for r in runs], axis=0).tolist()}
json.dump(res, open(OUT, "w"), separators=(",", ":"))
for K in Ks:
    print(K, "n_test", res["runs"][f"{K}|tactile"]["n_test"], " | ".join(f"{ch}: {np.mean(res['runs'][f'{K}|{ch}']['acc']):.3f}±{np.std(res['runs'][f'{K}|{ch}']['acc']):.3f}" for ch in CH))
for K in [5, 50]:
    pc = np.mean(res["runs"][f"{K}|tactile+joint_pos"]["per_class"], axis=0)
    print(K, "per-class t+j:", dict(zip(names, pc.round(2).tolist())))
