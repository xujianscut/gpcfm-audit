import os, json, glob, numpy as np
import os
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from scipy.stats import fisher_exact
def get(a, L):
    fs = glob.glob(f"{RES}/res2/{a}_l{L}_[0-9].json"); assert fs, (a, L)
    return np.array([r["w2"] for f in fs for r in json.load(open(f))])
rows = [("gpfix","GP","GP"), ("mmdet","GP","GP"), ("gpmean_decvar","GP","decoupled"),
        ("straight_gpvar","straight","GP"), ("bump","straight","decoupled")]
hdr = "%15s %9s %10s | %-26s | %-26s" % ("arm","mean m_t","var v_t","l=0.35 (cov 0.86)","l=0.25 (cov 0.98)")
print(hdr); print("-"*len(hdr))
tot = {}
for a, mm, vv in rows:
    cells = []; k_t = n_t = 0
    for L in ["0.35", "0.25"]:
        v = get(a, L); k = int((v > 5).sum()); k_t += k; n_t += len(v)
        cells.append("med %.2f   diverged %d/%d" % (np.median(v), k, len(v)))
    tot[a] = (k_t, n_t)
    print("%15s %9s %10s | %-26s | %-26s" % (a, mm, vv, cells[0], cells[1]))
print("\npooled over the two high-coverage levels:")
for a, _, _ in rows: print("   %15s: %2d/%d" % (a, tot[a][0], tot[a][1]))
def fe(x, y):
    (k1,n1),(k2,n2) = tot[x], tot[y]
    return fisher_exact([[k1,n1-k1],[k2,n2-k2]])[1]
print("\n  does the GP MEAN cause it?  gpmean_decvar vs bump  (same v_t, mean differs)  p=%.2e" % fe("gpmean_decvar","bump"))
print("  does the GP VARIANCE?       straight_gpvar vs bump (same mean, v_t differs)  p=%.3f" % fe("straight_gpvar","bump"))
print("  GP-mean-only vs full GP:    gpmean_decvar vs gpfix                           p=%.3f" % fe("gpmean_decvar","gpfix"))
print("  GP-var-only  vs full GP:    straight_gpvar vs gpfix                          p=%.2e" % fe("straight_gpvar","gpfix"))
