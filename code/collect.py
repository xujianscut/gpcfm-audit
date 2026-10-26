import os, json, glob, sys
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
import numpy as np
from collections import defaultdict
d = defaultdict(dict)
for f in glob.glob(f"{RES}/res/*_[0-9].json"):
    try: rows = json.load(open(f))
    except Exception: continue
    for r in rows: d[r["arm"]][r["seed"]] = r["w2"]
arms = ["icfm", "gp", "gpfix", "mmstoch", "mmdet", "bump"]
print(f"{'arm':>9} {'n':>4} {'mean':>7} {'std':>7} {'se':>7}")
for a in arms:
    if a not in d: continue
    v = np.array([d[a][s] for s in sorted(d[a])])
    print(f"{a:>9} {len(v):>4} {v.mean():>7.3f} {v.std(ddof=1):>7.3f} {v.std(ddof=1)/len(v)**.5:>7.4f}")
print("\npaired vs gp (same seed; negative = arm is better than the released GP-CFM):")
if "gp" in d:
    for a in arms:
        if a == "gp" or a not in d: continue
        common = sorted(set(d[a]) & set(d["gp"]))
        if len(common) < 5: continue
        diff = np.array([d[a][s] - d["gp"][s] for s in common])
        se = diff.std(ddof=1) / len(diff)**.5
        print(f"  {a:>9} - gp : n={len(common):>3}  mean diff {diff.mean():+.4f}  95% CI [{diff.mean()-1.96*se:+.4f}, {diff.mean()+1.96*se:+.4f}]")
    common = sorted(set(d.get("icfm", {})) & set(d["gp"]))
    if len(common) >= 5:
        diff = np.array([d["gp"][s] - d["icfm"][s] for s in common])
        se = diff.std(ddof=1)/len(diff)**.5
        print(f"\n  reproduction gate  gp - icfm : n={len(common)}  {diff.mean():+.4f}  95% CI [{diff.mean()-1.96*se:+.4f}, {diff.mean()+1.96*se:+.4f}]  (paper: 1.51-1.54 = -0.03)")
