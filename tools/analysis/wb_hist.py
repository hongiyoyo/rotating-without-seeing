import wandb, datetime, collections
api = wandb.Api()
r = api.run("jhyang321-seoul-national-university/isaacgymenvs/lxgr93nn")
rows = list(r.scan_history(page_size=5000))
print("total rows:", len(rows))
last_seen = {}
for row in rows:
    for k, v in row.items():
        if v is not None and not k.startswith("_"):
            last_seen[k] = (row.get("_step"), row.get("_timestamp"), row.get("global_step"))
by_time = collections.Counter()
for k, (s, t, g) in sorted(last_seen.items(), key=lambda x: x[1][0] or 0):
    by_time[(s, g)] += 1
for (s, g), n in sorted(by_time.items(), key=lambda x: x[0][0] or 0)[-8:]:
    ex = [k for k, v in last_seen.items() if v[0] == s][:4]
    t = [v[1] for k, v in last_seen.items() if v[0] == s][0]
    print(f"_step {s} global_step {g} time {datetime.datetime.fromtimestamp(t) if t else None}: {n} keys, e.g. {ex}")
print("last row:", {k: v for k, v in rows[-1].items() if v is not None and not k.startswith('PerObject')})
