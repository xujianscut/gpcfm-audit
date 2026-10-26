"""
De-risk: does GP-CFM's stream machinery do anything beyond a per-time Gaussian
conditional path (m_t, v_t)?

Arms (all identical compute: 5000 steps, nt=10 targets/step, same MLP, Adam):
  icfm     : I-CFM, sigma=0                       (paper baseline)
  gp       : GP-I-CFM, joint GP over nt=10 times  (paper method, verbatim code)
  mmstoch  : matched per-t marginal (m_t,v_t,c_t,w_t) but INDEPENDENT times
             -> isolates the effect of cross-time correlation
  mmdet    : matched (m_t,v_t), target = analytic E[sdot_t | s_t] (no eta noise)
             -> isolates the effect of the stochastic velocity target
  bump     : NO GP at all: linear mean, v_t = vmax*4t(1-t) matched to GP's max
             variance, analytic conditional velocity
             -> is a 1-parameter schedule enough to reproduce the gain?
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torchdyn.core import NeuralODE
import ot

# ---------------------------------------------------------------- data (verbatim)
def make_data():
    N, yend, p = 100, 10, .5
    np.random.seed(0)
    x0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N)
    z = np.random.binomial(1, p, N)[:, None]
    x1 = z * np.random.multivariate_normal([-3, yend], [[.1, 0], [0, .1]], N) + \
         (1 - z) * np.random.multivariate_normal([3, yend], [[.1, 0], [0, .1]], N)
    x1 = torch.from_numpy(x1).to(torch.float32)
    N_test = 1000
    np.random.seed(1)
    zt = np.random.binomial(1, p, N_test)[:, None]
    x1_test = zt * np.random.multivariate_normal([-3, yend], [[.1, 0], [0, .1]], N_test) + \
              (1 - zt) * np.random.multivariate_normal([3, yend], [[.1, 0], [0, .1]], N_test)
    return x1, torch.from_numpy(x1_test).to(torch.float32)

# ---------------------------------------------------------------- model (verbatim)
class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        out_dim = out_dim or dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, out_dim))
    def forward(self, x): return self.net(x)

class torch_wrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, t, x, *a, **k):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

# ---------------------------------------------------------------- GP kernels (verbatim)
def calc_r(ti, tj):
    r = ti[..., None] - tj[..., None, :]
    r[r == 0] = 1e-15
    return r
def k11(r, alpha, l): return (alpha**2) * torch.exp(-0.5 * ((r / l)**2))
def k12(r, alpha, l): return (alpha**2 / l**2) * r * torch.exp(-0.5 * ((r / l)**2))
def k22(r, alpha, l): return (alpha**2 / l**4) * (l**2 - r**2) * torch.exp(-0.5 * ((r / l)**2))

def cov_mat2(ti, tj, alpha, l, sig2_diag=1e-8):
    r = calc_r(ti, tj); nB, nt = r.shape[0], r.shape[1]
    Sig11 = k11(r, alpha, l) + (torch.eye(nt) * sig2_diag).repeat(nB, 1, 1)
    Sig12 = k12(r, alpha, l); Sig21 = Sig12.permute(0, 2, 1); Sig22 = k22(r, alpha, l)
    Sig = torch.cat([torch.cat([Sig11, Sig12], 2), torch.cat([Sig21, Sig22], 2)], 1)
    return (Sig + Sig.permute(0, 2, 1)) / 2

def samp_x_dx2(t_mat, alpha, l, x_obs, t_obs, sig2_diag=1e-8):
    """verbatim from GP-CFM/1_varRed.ipynb"""
    nB, nt, dim = x_obs.shape[0], t_mat.shape[1], x_obs.shape[2]
    nt_obs = t_obs.shape[0]
    r_obs_x = calc_r(t_obs, t_mat); r_obs_obs = calc_r(t_obs, t_obs)
    Sig_11 = cov_mat2(t_mat, t_mat, alpha, l, sig2_diag)
    k_obs_x, k_obs_dx = k11(r_obs_x, alpha, l), k12(r_obs_x, alpha, l)
    Sig_21 = torch.cat([k_obs_x, k_obs_dx], dim=2); Sig_12 = Sig_21.permute(0, 2, 1)
    Sig_22_sing = k11(r_obs_obs, alpha, l) + torch.eye(nt_obs) * sig2_diag
    Sig_22_inv = torch.linalg.inv(Sig_22_sing).repeat(nB, 1, 1)
    Sig_cond = Sig_11 - torch.bmm(torch.bmm(Sig_12, Sig_22_inv), Sig_21)
    Sig_cond = (Sig_cond + Sig_cond.permute(0, 2, 1)) / 2
    bad = torch.sum((torch.linalg.eigvals(Sig_cond).real >= 0).T, axis=0) != Sig_cond.shape[1]
    if bad.any():
        U, S, Vh = torch.linalg.svd(Sig_cond[bad, :, :])
        add = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), torch.diag_embed(S + 1e-8)), Vh)
        Sig_cond[bad, :, :] = (add + add.permute(0, 2, 1)) / 2
    mu_A = torch.bmm(Sig_12, Sig_22_inv)
    mu_new = torch.bmm(mu_A, x_obs.reshape(nB, nt_obs, dim)).reshape(nB, 2 * nt, dim)
    mu_flat = mu_new.view(nB * dim, 2 * nt)          # NOTE: matches original (see assert below)
    Sig_flat = Sig_cond.repeat(dim, 1, 1)
    dist = MultivariateNormal(loc=mu_flat, covariance_matrix=Sig_flat)
    s = dist.rsample().view(nB, 2 * nt, dim)
    return s[:, :nt, :], s[:, nt:, :]


def samp_x_dx2_fixed(t_mat, alpha, l, x_obs, t_obs, sig2_diag=1e-8):
    """Correct version: draw (s_{t_1..t_nt}, sdot_{t_1..t_nt}) from the conditional GP,
    with per-(stream,dimension) mean paired to that stream's own covariance."""
    nB, nt, dim = x_obs.shape[0], t_mat.shape[1], x_obs.shape[2]
    nt_obs = t_obs.shape[0]
    r_obs_x = calc_r(t_obs, t_mat); r_obs_obs = calc_r(t_obs, t_obs)
    Sig_11 = cov_mat2(t_mat, t_mat, alpha, l, sig2_diag)
    Sig_21 = torch.cat([k11(r_obs_x, alpha, l), k12(r_obs_x, alpha, l)], dim=2)
    Sig_12 = Sig_21.permute(0, 2, 1)
    Sig_22_inv = torch.linalg.inv(k11(r_obs_obs, alpha, l) + torch.eye(nt_obs) * sig2_diag).repeat(nB, 1, 1)
    Sig_cond = Sig_11 - torch.bmm(torch.bmm(Sig_12, Sig_22_inv), Sig_21)
    Sig_cond = (Sig_cond + Sig_cond.permute(0, 2, 1)) / 2
    lam, U = torch.linalg.eigh(Sig_cond.double())
    L = U @ torch.diag_embed(lam.clamp_min(0.0).sqrt())           # (nB,2nt,2nt) sqrt factor
    mu_A = torch.bmm(Sig_12, Sig_22_inv)
    mu_new = torch.bmm(mu_A, x_obs.reshape(nB, nt_obs, dim)).double()   # (nB,2nt,dim)
    z = torch.randn(nB, 2 * nt, dim, dtype=torch.float64)
    s = (mu_new + torch.bmm(L, z)).to(x_obs.dtype)
    return s[:, :nt, :], s[:, nt:, :]

# ------------------------------------------------- per-time stats of the SAME GP
def gp_stats(t, alpha, l, t_obs):
    """t: (M,) times.  returns A:(M,2,2) mapping [x0;x1]->[m_t; mdot_t],
       and Sig:(M,2,2) = [[v,c],[c,w]] conditional covariance of (s_t, sdot_t).
    Computed in float64: v_t is O(1e-3) here and float32 loses c_t = vdot_t/2 entirely."""
    dt_in = t.dtype
    M = t.shape[0]
    t = t.double(); t_obs = t_obs.double()
    tm = t.reshape(M, 1)
    r_obs_x = calc_r(t_obs.expand(M, -1), tm)                 # (M,2,1)
    Sig_11 = cov_mat2(tm, tm, alpha, l, 0.0)                  # (M,2,2)
    Sig_21 = torch.cat([k11(r_obs_x, alpha, l), k12(r_obs_x, alpha, l)], dim=2)  # (M,2,2)
    Sig_12 = Sig_21.permute(0, 2, 1)
    r_oo = calc_r(t_obs, t_obs)
    Sig_22_inv = torch.linalg.inv(k11(r_oo, alpha, l)).expand(M, -1, -1)
    A = torch.bmm(Sig_12, Sig_22_inv)
    Sig = Sig_11 - torch.bmm(A, Sig_21)
    Sig = (Sig + Sig.permute(0, 2, 1)) / 2
    return A.to(dt_in), Sig.to(dt_in)

# ---------------------------------------------------------------- training arms
def train(arm, x1, model, opt, n_epochs, nt=10, batch_size=100, alpha=1., l=2., vmax=None):
    N, dim = x1.shape[0], x1.shape[1]
    t_obs = torch.tensor([0., 1.])
    nbatch = N // batch_size
    idx = np.reshape(np.arange(N), [nbatch, batch_size])
    train.n_sampler_fail = 0
    xt_b = ut_b = None
    model.train()
    for k in range(n_epochs):
        for bb in range(nbatch):
            x0 = torch.randn((batch_size, dim))
            x1b = x1[idx[bb, :], :]
            if arm == "icfm":
                t = torch.rand(nt * batch_size)
                x0e, x1e = x0.repeat(nt, 1), x1b.repeat(nt, 1)
                xt = t[:, None] * x1e + (1 - t[:, None]) * x0e     # sigma = 0
                ut = x1e - x0e
            elif arm in ("gp", "gpfix"):
                x_obs = torch.stack([x0, x1b], dim=1)              # (B,2,dim)
                t_mat = torch.rand((batch_size, nt))
                sampler = samp_x_dx2 if arm == "gp" else samp_x_dx2_fixed
                # GP_FM2 in the released notebook wraps this in a bare `try/except: pass`,
                # i.e. on failure the PREVIOUS minibatch's (xt, ut) is silently reused.
                try:
                    xt_b, ut_b = sampler(t_mat, alpha, l, x_obs, t_obs, 0.0)
                except Exception:
                    train.n_sampler_fail += 1
                    if xt_b is None:      # nothing to reuse yet: skip this step
                        continue
                t = t_mat.reshape(-1)
                xt = xt_b.reshape(-1, dim); ut = ut_b.reshape(-1, dim)
            else:
                t = torch.rand(nt * batch_size)
                x0e, x1e = x0.repeat(nt, 1), x1b.repeat(nt, 1)
                if arm in ("bump", "gpmean_decvar"):
                    # decoupled variance profile; mean is straight (bump) or the GP's (gpmean_decvar)
                    if arm == "bump":
                        m = t[:, None] * x1e + (1 - t[:, None]) * x0e
                        md = x1e - x0e
                    else:
                        A, _ = gp_stats(t.double(), alpha, l, t_obs)
                        mm = torch.bmm(A, torch.stack([x0e, x1e], dim=1).double())
                        m, md = mm[:, 0, :].float(), mm[:, 1, :].float()
                    g = 4 * t * (1 - t)
                    v = (vmax * g**2).clamp_min(1e-30)             # v ~ t^2 at the ends
                    c = vmax * g * 4 * (1 - 2 * t)                 # = v_dot/2
                    slope = c / v.sqrt(); resid = torch.zeros_like(v)
                elif arm == "straight_gpvar":
                    # the GP's variance curve under a straight mean
                    m = t[:, None] * x1e + (1 - t[:, None]) * x0e
                    md = x1e - x0e
                    _, S = gp_stats(t.double(), alpha, l, t_obs)
                    v64, c64, w64 = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
                    ok = v64 > 1e-9
                    slope = torch.where(ok, c64 / v64.clamp_min(1e-30).sqrt(), torch.sign(c64) * w64.sqrt())
                    assert slope.abs().max() < 10 * w64.sqrt().max() + 1e-6, "slope blew up"
                    v = v64.clamp_min(0).float(); slope = slope.float(); resid = torch.zeros_like(v)
                else:
                    A, S = gp_stats(t.double(), alpha, l, t_obs)   # (M,2,2) float64
                    ends = torch.stack([x0e, x1e], dim=1).double()  # (M,2,dim)
                    mm = torch.bmm(A, ends)                         # (M,2,dim)
                    m, md = mm[:, 0, :].float(), mm[:, 1, :].float()
                    v64, c64, w64 = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
                    # near the pinned endpoints v_t -> 0 and float64 cancellation can make it
                    # ~1e-15 or negative, so c/sqrt(v) explodes; analytically it -> sign(c) sqrt(w)
                    # and the residual sd -> 0 there. Use those limits where v is unreliable.
                    ok = v64 > 1e-9
                    slope = torch.where(ok, c64 / v64.clamp_min(1e-30).sqrt(), torch.sign(c64) * w64.sqrt())
                    resid = torch.where(ok, (w64 - c64**2 / v64.clamp_min(1e-30)).clamp_min(0), torch.zeros_like(w64))
                    assert slope.abs().max() < 10 * w64.sqrt().max() + 1e-6, "slope blew up"
                    v = v64.clamp_min(0).float(); slope = slope.float(); resid = resid.float()
                eps = torch.randn_like(m)
                xt = m + v.sqrt()[:, None] * eps
                # E[sdot_t | s_t] = mdot_t + (c/v)(s_t - m_t) = mdot_t + (c/sqrt(v)) * eps
                ut = md + slope[:, None] * eps
                if arm == "mmstoch":
                    ut = ut + resid.sqrt()[:, None] * torch.randn_like(ut)
            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)
            loss.backward(); opt.step(); opt.zero_grad()
    return model

# ---------------------------------------------------------------- eval (verbatim)
def w2(model, x1_test, seed, dim=2):
    node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint",
                     atol=1e-4, rtol=1e-4)
    torch.manual_seed(seed)
    x_start = torch.randn(x1_test.shape[0], dim)
    with torch.no_grad():
        traj = node.trajectory(x_start, t_span=torch.linspace(0, 1, 2))
    g = traj[-1, :, :].numpy(); r = x1_test.numpy()
    a = np.ones(r.shape[0]) / r.shape[0]; b = np.ones(g.shape[0]) / g.shape[0]
    return float(ot.emd2(a, b, ot.dist(r, g)))

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--nseeds", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", default="")
    ap.add_argument("--kern_l", type=float, default=2.0)
    ap.add_argument("--sigma_max", type=float, default=-1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    x1, x1_test = make_data()
    dim = 2

    # vmax for the "bump" arm: max over t of the GP's conditional variance v_t
    tg = torch.linspace(1e-4, 1 - 1e-4, 20001).double()
    _, Sg = gp_stats(tg, 1., a.kern_l, torch.tensor([0., 1.]))
    vg, cg = Sg[:, 0, 0], Sg[:, 0, 1]
    vmax = float(vg.max())
    # ---- sanity assert: c_t == 0.5 * d v_t / dt (continuity-equation consistency),
    #      checked on the interior where the finite difference is meaningful
    dv = torch.gradient(vg, spacing=(tg,))[0]
    # normalise GLOBALLY: v_dot_t is exactly 0 at the symmetric midpoint, so a
    # pointwise relative error there is 0/0 roundoff, not a violation.
    lo, hi = 200, len(tg) - 200
    rel = (cg - 0.5 * dv).abs()[lo:hi].max() / (0.5 * dv).abs()[lo:hi].max()
    assert rel < 1e-6, f"c_t != v_dot/2 (rel err {rel:.2e}) -- theory broken"
    assert 1e-6 < vmax < 1e6, f"degenerate GP conditional variance {vmax}"
    gp_sd = vmax ** 0.5
    if a.sigma_max > 0:                    # decoupled arm: match coverage, free the mean path
        vmax = a.sigma_max ** 2
    print(f"[check] l={a.kern_l} -> GP max sd {gp_sd:.4f}; arm uses max sd {vmax**0.5:.4f}; "
          f"c_t = v_dot_t/2 to rel {rel:.1e}", flush=True)
    assert abs(gp_sd - (vmax ** 0.5)) < 1e-6 or a.sigma_max > 0, "coverage mismatch not requested"

    if a.warm and not os.path.exists(a.warm):     # arm-specific warm start (paper protocol)
        torch.manual_seed(12345); np.random.seed(12345)
        m0 = MLP(dim=dim, out_dim=dim, time_varying=True)
        o0 = torch.optim.Adam(m0.parameters(), lr=1e-3)
        train(a.arm, x1, m0, o0, a.epochs, l=a.kern_l, vmax=vmax)
        torch.save(m0.state_dict(), a.warm)

    res = []
    for s in range(a.seed0, a.seed0 + a.nseeds):
        t0 = time.time()
        torch.manual_seed(1000 + s); np.random.seed(1000 + s)
        m = MLP(dim=dim, out_dim=dim, time_varying=True)
        if a.warm: m.load_state_dict(torch.load(a.warm))
        o = torch.optim.Adam(m.parameters(), lr=2e-3)
        train(a.arm, x1, m, o, a.epochs, l=a.kern_l, vmax=vmax)
        d = w2(m, x1_test, s)
        res.append({"arm": a.arm + a.tag, "seed": s, "w2": d, "sec": time.time() - t0,
                    "kern_l": a.kern_l, "cov_sd": vmax ** 0.5,
                    "sampler_fail": getattr(train, "n_sampler_fail", 0)})
        with open(a.out, "w") as f: json.dump(res, f)
        print(f"{a.arm} seed {s}: W2={d:.4f} ({time.time()-t0:.1f}s)", flush=True)
