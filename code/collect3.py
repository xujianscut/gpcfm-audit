import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from collections import defaultdict
from scipy.stats import wilcoxon, binomtest
D = defaultdict(dict)
for f in glob.glob(f"{RES}/res2/*_[0-9].json"):
    try: rows = json.load(open(f))
    except Exception: continue
    for r in rows: D[r["arm"]][r["seed"]] = r["w2"]
levels = [("2.0", .0442), ("1.0", .1745), ("0.5", .5933), ("0.35", .8628), ("0.25", .9815)]
BLOW = 5.0   # W2 well beyond anything a working model produces here (all baselines ~1.5)
def row(d):
    if not d: return "        -            -        -  "
    v = np.array(list(d.values()))
    return f"{np.median(v):>8.3f} [{np.percentile(v,25):>6.3f},{np.percentile(v,75):>7.3f}] {100*(v>BLOW).mean():>5.0f}%"
print(f"{'cov sd':>7} {'l':>5} | {'GP median [IQR]':>24} {'>5':>5} | {'decoupled median [IQR]':>24} {'>5':>5} | {'GP loses':>9} {'sign p':>9}")
for L, sd in levels:
    g, b = D.get(f"gpfix_l{L}", {}), D.get(f"bump_l{L}", {})
    common = sorted(set(g) & set(b))
    if len(common) >= 5:
        w = sum(1 for s in common if g[s] > b[s])
        p = binomtest(w, len(common), 0.5).pvalue
        tail = f"{w:>3}/{len(common):<3}  {p:>9.2e}"
    else: tail = " " * 19
    print(f"{sd:>7.4f} {L:>5} | {row(g)} | {row(b)} | {tail}")
print(f"\n'GP loses' = # of the 40 paired seeds where the GP parametrisation is worse; sign test vs 50/50.")
print(f"'>5' = fraction of seeds with W2 > {BLOW} (divergent run; every working baseline here sits at ~1.5).")
mm = {k: v for k, v in D.items() if k.startswith("mmdet")}
if mm:
    print("\nattribution control - mmdet (GP's own (m_t,v_t), but NO joint GP sampler at all):")
    for L, sd in levels:
        d = D.get(f"mmdet_l{L}", {})
        if d: print(f"  cov sd {sd:>6.4f} (l={L:>4}) : {row(d)}  n={len(d)}")
