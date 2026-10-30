import json, sys
S = sys.argv[1]
new = json.load(open(S + "/eval.json")); old = json.load(open(S + "/aug_eval.json"))
ZERO = 100  # assumed zero-length records per object in the Aug eval (900 envs, 1 per env at start)
cmp = {"assumed_zero_len_per_object": ZERO, "objects": {}}
tot = {"old_rot": 0, "old_time": 0, "old_full": 0, "old_rew": 0, "old_n": 0, "new_rot": 0, "new_time": 0}
for name in new["label_names"]:
    o, n = old["objects"][name], new["objects"][name]
    on = o["n"]; oreal = on - ZERO
    o_rot, o_time = o["rotations"]["mean"] * on, o["time_alive_s"]["mean"] * on
    n_rot, n_time = n["rotations"]["mean"] * n["n"], n["time_alive_s"]["mean"] * n["n"]
    c = {"aug_n": oreal, "aug_reward": o["reward"]["mean"] * on / oreal, "aug_full": o["full_duration_rate"] * on / oreal, "aug_rate": o_rot / o_time,
         "new_n": n["n"], "new_reward": n["reward"]["mean"], "new_full": n["full_duration_rate"], "new_rate": n_rot / n_time}
    cmp["objects"][name] = {k: round(v, 4) for k, v in c.items()}
    n["rotation_rate"]["mean"] = round(n_rot / n_time, 4)  # time-weighted
    tot["old_rot"] += o_rot; tot["old_time"] += o_time; tot["old_full"] += o["full_duration_rate"] * on; tot["old_rew"] += o["reward"]["mean"] * on; tot["old_n"] += oreal
    tot["new_rot"] += n_rot; tot["new_time"] += n_time
cmp["overall"] = {"aug_n": tot["old_n"], "aug_reward": round(tot["old_rew"] / tot["old_n"], 2), "aug_full": round(tot["old_full"] / tot["old_n"], 4), "aug_rate": round(tot["old_rot"] / tot["old_time"], 4),
                  "new_n": new["n_total"], "new_reward": round(new["overall"]["reward"]["mean"], 2), "new_full": new["overall"]["full_duration_rate"], "new_rate": round(tot["new_rot"] / tot["new_time"], 4)}
new["overall"]["rotation_rate"]["mean"] = cmp["overall"]["new_rate"]
json.dump(new, open(S + "/eval.json", "w"), separators=(",", ":")); json.dump(cmp, open(S + "/compare.json", "w"), separators=(",", ":"))
print(json.dumps(cmp["overall"]))
for k, v in cmp["objects"].items(): print(k, v)
