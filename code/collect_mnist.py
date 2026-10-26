import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from collections import defaultdict
from scipy.stats import wilcoxon, binomtest
D = defaultdict(dict)
for f in glob.glob(f"{RES}/res_mnist/*_[0-9]*.json"):
    try: r = json.load(open(f))
    except Exception: continue
    D[r["arm"]][r["seed"]] = r
arms = ["icfm", "gp", "gpfix", "indep", "mmdet"]
paper = {"icfm": (44.50, 0.18, 0.0040), "gp": (43.55, 0.13, 0.0032)}
print(f"{'arm':>6} {'n':>3} | {'FID mean':>8} {'se':>6} {'sd':>6} {'median':>7} {'max':>7} | {'KID mean':>8} {'se':>7} | {'fail':>5} {'train s':>7} | paper FID (se)")
for a in arms:
    if a not in D: continue
    rs = [D[a][s] for s in sorted(D[a])]
    f = np.array([r["fid"] for r in rs]); k = np.array([r["kid"] for r in rs])
    fl = np.mean([r["sampler_fail"] for r in rs]); ts = np.mean([r["train_sec"] for r in rs])
    pp = f"{paper[a][0]:.2f} ({paper[a][1]:.2f})" if a in paper else ""
    print(f"{a:>6} {len(f):>3} | {f.mean():>8.2f} {f.std(ddof=1)/len(f)**.5:>6.2f} {f.std(ddof=1):>6.2f} {np.median(f):>7.2f} {f.max():>7.2f} | {k.mean():>8.4f} {k.std(ddof=1)/len(k)**.5:>7.4f} | {fl:>5.1f} {ts:>7.0f} | {pp}")
print("\npaired vs icfm (same seed = same init, data order, x0, t):  negative = better than I-CFM")
for a in arms[1:]:
    common = sorted(set(D.get(a, {})) & set(D.get("icfm", {})))
    if len(common) < 5: continue
    d = np.array([D[a][s]["fid"] - D["icfm"][s]["fid"] for s in common]); se = d.std(ddof=1)/len(d)**.5
    w = sum(1 for x in d if x < 0)
    try: pw = wilcoxon(d).pvalue
    except Exception: pw = float("nan")
    print(f"  {a:>6} - icfm : n={len(common):>2}  dFID {d.mean():+6.2f}  95%CI [{d.mean()-1.96*se:+.2f},{d.mean()+1.96*se:+.2f}]  wins {w}/{len(d)}  wilcoxon p={pw:.3f}")
print("\npaired vs gp (released code):")
for a in ["gpfix", "indep", "mmdet"]:
    common = sorted(set(D.get(a, {})) & set(D.get("gp", {})))
    if len(common) < 5: continue
    d = np.array([D[a][s]["fid"] - D["gp"][s]["fid"] for s in common]); se = d.std(ddof=1)/len(d)**.5
    try: pw = wilcoxon(d).pvalue
    except Exception: pw = float("nan")
    print(f"  {a:>6} - gp   : n={len(common):>2}  dFID {d.mean():+6.2f}  95%CI [{d.mean()-1.96*se:+.2f},{d.mean()+1.96*se:+.2f}]  wilcoxon p={pw:.3f}")
