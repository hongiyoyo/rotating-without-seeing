import numpy as np
d = np.load("/mnt/c/YangJaehong/rotating-without-seeing/tools/data/policy_eval.npz", allow_pickle=True)
names = [str(n) for n in d["label_names"]]; lab = d["labels"]; L = d["ep_len"]; R = d["ep_reward"]; rot = d["rotations"]
print("zero_len_dropped in file:", d["zero_len_dropped"] if "zero_len_dropped" in d.files else "n/a")
print("total", len(L), "| ep_len==0:", (L == 0).sum(), "| min ep_len:", L.min(), "| ep_len<=1:", (L <= 1).sum(), "<=5:", (L <= 5).sum(), "<=10:", (L <= 10).sum(), "<=30:", (L <= 30).sum(), "| >=498:", (L >= 498).sum())
edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 498, 500]
h, _ = np.histogram(L, bins=edges)
print("ep_len histogram:"); [print(f"  [{a},{b}): {c}") for a, b, c in zip(edges[:-1], edges[1:], h)]
print("most common short lengths:", sorted(((int(v), int((L == v).sum())) for v in np.unique(L[L <= 30])), key=lambda x: -x[1])[:10])
for i, n in enumerate(names):
    m = lab == i; Lm = L[m]
    print(f"{n:11s} n={m.sum():3d} <=10:{(Lm<=10).sum():3d} ({100*(Lm<=10).mean():4.1f}%) 11-497:{((Lm>10)&(Lm<498)).sum():3d} full:{(Lm>=498).sum():3d} | median len {np.median(Lm):.0f}")
print("reward of len<=10 episodes: mean", R[L <= 10].mean().round(2), "max", R[L <= 10].max().round(2), "| rotations max", rot[L <= 10].max())
