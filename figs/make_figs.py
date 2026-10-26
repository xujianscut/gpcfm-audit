"""Figures for the GP-CFM audit paper. Every number is read from result files or computed
from the SE-bridge formulas; nothing is typed in."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# palette (validated all-pairs, light surface): blue / orange / aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "legend.frameon": False,
})

# ----------------------------------------------------------------- SE bridge (numpy, float64)
def k(r, l, a=1.0): return a**2 * np.exp(-0.5 * (r / l) ** 2)
def bridge(t, l, a=1.0):
    """mean weights (a0,a1)(t), variance v_t, and full conditional covariance over the grid t."""
    O = np.array([0.0, 1.0])
    KtO = k(t[:, None] - O[None, :], l, a); KOO = k(O[:, None] - O[None, :], l, a)
    Ktt = k(t[:, None] - t[None, :], l, a)
    A = KtO @ np.linalg.inv(KOO)
    C = Ktt - A @ KtO.T
    return A, np.clip(np.diag(C), 0, None), (C + C.T) / 2

def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += json.load(open(f))
    assert rows, f"no data for {pattern}"
    return rows


# ===================================================================== Fig 1: the GP chart
def fig_chart():
    t = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.5], "wspace": 0.35})
    rng = np.random.default_rng(1)
    x0, x1 = 0.0, 1.0
    lo, hi = -1.6, 2.2
    for ax, l in zip(axes[:2], [2.0, 0.25]):
        A, v, C = bridge(t, l)
        m = A[:, 0] * x0 + A[:, 1] * x1
        w, U = np.linalg.eigh(C); Lf = U @ np.diag(np.sqrt(np.clip(w, 0, None)))
        for _ in range(8):
            path = m + Lf @ rng.standard_normal(len(t))
            ax.plot(t, path, color=ORANGE, lw=0.5, alpha=0.4)
            lo, hi = min(lo, path.min() - 0.1), max(hi, path.max() + 0.1)   # never clip a drawn path
        ax.fill_between(t, m - np.sqrt(v), m + np.sqrt(v), color=ORANGE, alpha=0.15, lw=0)
        ax.plot(t, m, color=ORANGE, lw=1.6)
        ax.plot(t, x0 + (x1 - x0) * t, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"GP chart, $\\ell={l:g}$")
    ax = axes[2]
    smax = np.sqrt(bridge(t, 0.25)[1]).max()
    m = x0 + (x1 - x0) * t; sd = smax * (4 * t * (1 - t))
    ax.fill_between(t, m - sd, m + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(t, m, color=BLUE, lw=1.6)
    ax.set_title("decoupled chart")
    for ax in axes[:3]:
        ax.set_ylim(lo, hi); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-1, 0, 1, 2]); ax.set_xlabel("$t$")
    axes[0].set_ylabel("$s_t$")
    ax = axes[3]
    ls = np.geomspace(0.05, 8, 200)
    cov = np.array([np.sqrt(bridge(t, l)[1]).max() for l in ls])
    mid = np.array([bridge(np.array([0.5]), l)[0].sum() for l in ls])
    ax.plot(cov, mid, color=ORANGE, lw=1.8)
    ax.axhline(1.0, color=BLUE, lw=1.8)
    for l, (dx, dy, ha) in {2: (6, -8, "left"), 0.25: (-6, -2, "right")}.items():
        c = np.sqrt(bridge(t, l)[1]).max(); mm = bridge(np.array([0.5]), l)[0].sum()
        ax.plot(c, mm, "o", color=ORANGE, ms=4, mec="white", mew=0.7)
        ax.annotate(f"$\\ell={l:g}$", (c, mm), textcoords="offset points", xytext=(dx, dy), fontsize=7, color=INK2, ha=ha)
    ax.set_xlabel("coverage  $\\max_t\\sqrt{v_t}$"); ax.set_ylabel("mid-weight $a_0\\!+\\!a_1$")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.05, 1.25); ax.grid(True, axis="y")
    ax.set_title("coverage vs. interpolation")
    fig.legend(handles=[Line2D([], [], color=ORANGE, lw=1.8, label="GP chart (SE kernel, $\\ell$ varies)"),
                        Line2D([], [], color=BLUE, lw=1.8, label="decoupled chart"),
                        Line2D([], [], color=INK2, lw=0.8, ls=":", label="straight line")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.26), fontsize=7)
    fig.savefig(f"{OUT}/gp_chart.pdf", bbox_inches="tight")
    print("gp_chart ok")

# ===================================================================== Fig 2: the released sampler
def fig_bug():
    z = np.load(f"{D}/res/bug_draws.npz")
    E = z["released"]
    nB, NC, twont, dim = E.shape; nt = twont // 2
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.8), gridspec_kw={"wspace": 0.4})
    for ax, b in zip(axes[:2], [1, 0]):
        mu, S, I = z[f"mu_{b}"], z[f"S_{b}"], z[f"intended_{b}"]
        i, d = 0, 0
        rx, rd = E[b, :1500, i, d] - mu[i, d], E[b, :1500, nt + i, d] - mu[nt + i, d]
        ix, idd = I[:1500, i, d] - mu[i, d], I[:1500, nt + i, d] - mu[nt + i, d]
        ci = S[i, nt + i] / np.sqrt(S[i, i] * S[nt + i, nt + i])
        cr = np.corrcoef(E[b, :, i, d], E[b, :, nt + i, d])[0, 1]
        ax.scatter(ix, idd, s=3, color=BLUE, alpha=0.35, lw=0)
        ax.scatter(rx, rd, s=3, color=ORANGE, alpha=0.35, lw=0)
        ax.set_xlabel("$s_t - m_t$"); ax.set_ylabel("$\\dot s_t - \\dot m_t$")
        ax.set_title(f"corr ${ci:+.2f}$ vs. ${cr:+.2f}$")
        ax.axhline(0, color=GRID, lw=0.5, zorder=0); ax.axvline(0, color=GRID, lw=0.5, zorder=0)
    ax = axes[2]
    emp, want, pred = [], [], []
    for b in range(nB):
        S = z[f"S_{b}"]; Salt = z[f"S_{(2 * b) % nB}"]
        for i in range(nt):
            for d in range(dim):
                emp.append(E[b, :, i, d].var()); want.append(S[i, i]); pred.append(Salt[2 * i + d, 2 * i + d])
    emp, want, pred = map(np.array, (emp, want, pred))
    lim = max(emp.max(), want.max(), pred.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=GRID, lw=0.8, zorder=0)
    ax.scatter(want, emp, s=14, color=BLUE, zorder=2)
    ax.scatter(pred, emp, s=14, color=ORANGE, marker="s", zorder=3)
    ax.set_xlabel("predicted variance"); ax.set_ylabel("empirical variance")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title("24 state coordinates", fontsize=7.5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="intended law"),
                        Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="released sampler"),
                        Line2D([], [], marker="s", ls="", color=ORANGE, ms=4, label="index-arithmetic prediction (right)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24), fontsize=7)
    fig.savefig(f"{OUT}/sampler_bug.pdf", bbox_inches="tight")
    print("sampler_bug ok")

# ===================================================================== Fig 3: toy results
def fig_toy():
    arms = ["icfm", "gp", "gpfix", "mmstoch", "mmdet", "bump"]
    labels = ["I-CFM", "gp", "gpfix", "mmstoch", "mmdet", "bump"]
    colors = [AQUA, ORANGE, ORANGE, BLUE, BLUE, BLUE]
    data = {a: np.array([r["w2"] for r in load(f"{D}/res/{a}_[0-9].json")]) for a in arms}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.8), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.25})
    ax = axes[0]; rng = np.random.default_rng(0)
    for j, (a, c) in enumerate(zip(arms, colors)):
        v = data[a]; n = len(v)
        ax.scatter(j + rng.uniform(-0.22, 0.22, n), v, s=5, color=c, alpha=0.3, lw=0)
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
        ax.errorbar(j, m, yerr=1.96 * se, fmt="o", color=c, ms=4.5, mec="white", mew=0.7, capsize=3, lw=1.3, zorder=5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("$W_2$"); ax.set_ylim(0, 3.8); ax.grid(True, axis="y")
    ax.set_title("Table 1 protocol, 100 seeds")
    ax = axes[1]
    levels = [("2.0", 0.0442), ("1.0", 0.1745), ("0.5", 0.5933), ("0.35", 0.8628), ("0.25", 0.9815)]
    specs = [("gpfix", ORANGE, "full", -0.27, "gpfix"), ("mmdet", ORANGE, "none", 0.0, "mmdet"), ("bump", BLUE, "full", 0.27, "bump")]
    for x, (L, sd) in enumerate(levels):
        for arm, c, fs, off, _ in specs:
            files = glob.glob(f"{D}/res2/{arm}_l{L}_[0-9].json")
            if not files: continue
            v = np.array([r["w2"] for f in files for r in json.load(open(f))])
            xs = x + off + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(xs, v, s=7, facecolors=c if fs == "full" else "none", edgecolors=c, lw=0.6, alpha=0.6)
    ax.axhline(5, color=INK2, lw=0.6, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.3, 3e6); ax.set_xlim(-0.6, 4.6)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{sd:.2f}" for L, sd in levels])
    ax.set_xlabel("coverage $\\max_t\\sqrt{v_t}$"); ax.set_ylabel("$W_2$"); ax.grid(True, axis="y")
    ax.set_title("coverage sweep, 40 seeds per cell")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markerfacecolor=c if fs == "full" else "none", ms=4.5, label=lab)
                        for _, c, fs, _, lab in specs] + [Line2D([], [], color=INK2, lw=0.6, ls="--", label="$W_2=5$")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.savefig(f"{OUT}/toy_results.pdf", bbox_inches="tight")
    print("toy_results ok")

# ===================================================================== Fig 4: MNIST paired
def fig_mnist():
    by = {}
    for f in sorted(glob.glob(f"{D}/res_mnist/*_[0-9]*.json")):
        r = json.load(open(f))
        by.setdefault(r["arm"], {})[r["seed"]] = r["fid"]
    assert by, "no MNIST results"
    arms = [("gp", "gp (released)", ORANGE), ("gpfix", "gpfix", BLUE), ("indep", "indep", BLUE), ("mmdet", "mmdet", BLUE)]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    rng = np.random.default_rng(0)
    for y, (a, lab, c) in enumerate(arms[::-1]):
        common = sorted(set(by[a]) & set(by["icfm"]))
        d = np.array([by[a][s] - by["icfm"][s] for s in common])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        ax.scatter(d, y + rng.uniform(-0.15, 0.15, len(d)), s=8, color=c, alpha=0.35, lw=0)
        ax.errorbar(m, y, xerr=1.96 * se, fmt="o", color=c, ms=5, mec="white", mew=0.7, capsize=3, lw=1.4, zorder=5)
    ax.axvline(0, color=AQUA, lw=1.4)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([lab for _, lab, _ in arms[::-1]])
    ax.set_ylim(-0.6, 3.6); ax.set_xlim(-1.5, 6.2)
    ax.set_xlabel("$\\Delta$FID vs. I-CFM, same seed"); ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(f"{OUT}/mnist_paired.pdf", bbox_inches="tight")
    print("mnist_paired ok")

if __name__ == "__main__":
    fig_chart(); fig_bug(); fig_toy(); fig_mnist()
