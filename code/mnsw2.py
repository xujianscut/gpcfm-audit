import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from scipy import stats
R = {}
for f in sorted(glob.glob(f"{RES}/res_mnsweep/*.json")):
    r = json.load(open(f)); R[(r["arm"].split("_l")[0], r["kern_l"], r["seed"])] = r["fid"]
base = {}
for f in sorted(glob.glob(f"{RES}/res_mnist/*_[0-9]*.json")):
    r = json.load(open(f)); base[(r["arm"], r["seed"])] = r["fid"]
S = range(10)
def pair(x, y, nx, ny):
    d = np.array([x[s]-y[s] for s in S]); se = d.std(ddof=1)/np.sqrt(len(d))
    return "%-34s %+6.2f  95%%CI [%+.2f,%+.2f]  wilcoxon p=%.3f" % (
        nx+" - "+ny, d.mean(), d.mean()-2.262*se, d.mean()+2.262*se, stats.wilcoxon(d).pvalue)
g = lambda L: {s: R[("gpfix",L,s)] for s in S}
b = lambda L: {s: R[("bump",L,s)] for s in S}
icfm = {s: base[("icfm",s)] for s in S}
gp   = {s: base[("gp",s)] for s in S}
print("Does widening coverage help on MNIST?  (paired over seeds 0-9)")
for L in [1.0, 0.5, 0.35, 0.25]:
    print("  ", pair(g(L), g(5.0), "gpfix l=%g"%L, "gpfix l=5 (paper)"))
print()
for L in [1.0, 0.5]:
    print("  ", pair(b(L), b(5.0), "bump  l=%g"%L, "bump  l=5"))
print("\nBest GP-chart setting vs the two published reference points:")
print("  ", pair(g(1.0), icfm, "gpfix l=1", "I-CFM"))
print("  ", pair(g(1.0), gp,   "gpfix l=1", "gp (released code)"))
print("  ", pair(g(5.0), icfm, "gpfix l=5", "I-CFM"))
print("\nsanity: gpfix l=5 here vs gpfix in the main MNIST run (same seeds, same config)")
d = np.array([R[("gpfix",5.0,s)] - base[("gpfix",s)] for s in S])
print("   mean %+.3f  max|.| %.3f   (should be 0: identical arm and seeds)" % (d.mean(), np.abs(d).max()))
