import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from collections import defaultdict
D = defaultdict(dict)
for f in glob.glob(f"{RES}/res2/*_[0-9].json"):
    try: rows = json.load(open(f))
    except Exception: continue
    for r in rows: D[r["arm"]][r["seed"]] = r["w2"]
levels = [("2.0", 0.0442), ("1.0", 0.1745), ("0.5", 0.5933), ("0.35", 0.8628), ("0.25", 0.9815)]
print(f"{'cov sd':>7} {'l':>5} | {'GP (coupled)':>22} | {'decoupled (no GP)':>22} | {'paired diff GP-dec':>26}")
for L, sd in levels:
    g, b = D.get(f"gpfix_l{L}", {}), D.get(f"bump_l{L}", {})
    if not g and not b: continue
    def fmt(d):
        if not d: return f"{'-':>22}"
        v = np.array(list(d.values()))
        return f"{v.mean():>8.3f} +-{v.std(ddof=1)/len(v)**.5:>6.3f} (n={len(v):>2})"
    common = sorted(set(g) & set(b))
    if len(common) >= 5:
        df = np.array([g[s] - b[s] for s in common]); se = df.std(ddof=1)/len(df)**.5
        dtxt = f"{df.mean():>+8.3f}  95%CI[{df.mean()-1.96*se:+.2f},{df.mean()+1.96*se:+.2f}]"
    else: dtxt = f"{'-':>26}"
    print(f"{sd:>7.4f} {L:>5} | {fmt(g)} | {fmt(b)} | {dtxt}")
print("\n(positive diff = the GP parametrisation is WORSE than decoupling mean from variance)")
