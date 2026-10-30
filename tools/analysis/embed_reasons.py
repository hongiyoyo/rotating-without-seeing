import json, sys
src, html = sys.argv[1], sys.argv[2]
d = json.load(open(src)); names = d["objects"]; E = d["episodes"]
comp = {"obj": [names.index(e["obj"]) for e in E], "len": [e["len"] for e in E],
        "code": [0 if e["timeout"] else (1 if e["tilt"] else 0) + (2 if e["x"] else 0) + (4 if e["fall"] else 0) for e in E],
        "z": [round(e["z_off_mm"], 1) for e in E]}
bad = sum(1 for c in comp["code"] if c == 0 and False)
s = open(html, encoding="utf-8").read()
assert s.count("__REASONS_DATA__") == 1
s = s.replace("__REASONS_DATA__", json.dumps(comp, separators=(",", ":")))
open(html, "w", encoding="utf-8").write(s)
unknown = sum(1 for e in E if not e["timeout"] and not (e["tilt"] or e["x"] or e["fall"]))
print("embedded", len(E), "episodes; failed with no flagged condition:", unknown)
