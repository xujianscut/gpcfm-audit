import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from scipy import stats
rows = {}
for f in sorted(glob.glob(f"{RES}/res_mnsweep/*.json")):
    r = json.load(open(f))
    rows.setdefault((r["arm"].split("_l")[0], r["kern_l"]), {})[r["seed"]] = (r["fid"], r["cov_sd"], r["final_loss"])
levels = [(5.0,"paper"), (1.0,""), (0.5,""), (0.35,""), (0.25,"")]
print("%8s %7s | %-24s | %-24s | %-22s" % ("cov sd","l","gpfix (GP chart)","bump (decoupled)","paired gpfix-bump"))
print("-"*100)
for L,tagn in levels:
    g, b = rows[("gpfix",L)], rows[("bump",L)]
    cov = list(g.values())[0][1]
    gv = np.array([g[s][0] for s in sorted(g)]); bv = np.array([b[s][0] for s in sorted(b)])
    common = sorted(set(g) & set(b)); d = np.array([g[s][0]-b[s][0] for s in common])
    se = d.std(ddof=1)/np.sqrt(len(d))
    t = stats.wilcoxon(d).pvalue if len(d) > 5 else float("nan")
    print("%8.4f %7s | %8.2f +-%5.2f (n=%2d) | %8.2f +-%5.2f (n=%2d) | %+6.2f [%+.2f,%+.2f] p=%.3f"
          % (cov, ("%g"%L)+("*" if tagn else ""), gv.mean(), gv.std(ddof=1)/np.sqrt(len(gv)), len(gv),
             bv.mean(), bv.std(ddof=1)/np.sqrt(len(bv)), len(bv),
             d.mean(), d.mean()-1.96*se, d.mean()+1.96*se, t))
print("\n* = the setting the paper uses.   FID on 1280 generated vs 1280 fixed test images.")
print("\ndivergent / degenerate runs (FID > 100):")
for L,_ in levels:
    for a in ["gpfix","bump"]:
        v = np.array([x[0] for x in rows[(a,L)].values()])
        k = int((v > 100).sum())
        if k: print("   l=%g %s: %d/%d   (max FID %.0f)" % (L, a, k, len(v), v.max()))
print("\nfinal training loss (mean):")
for L,_ in levels:
    print("   l=%-5g gpfix %.4f   bump %.4f" % (L, np.mean([x[2] for x in rows[("gpfix",L)].values()]),
                                                np.mean([x[2] for x in rows[("bump",L)].values()])))
