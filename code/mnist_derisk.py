"""
MNIST de-risk for the GP-CFM audit (paper Table 3 protocol, 5_mnist.ipynb).

Arms (identical UNet init, data order, x0 draws and t draws per seed -> paired):
  icfm   : I-CFM, sigma=0                                     (paper baseline)
  gp     : released GP sampler verbatim (nt=1, alpha=1, l=5)  (paper method, incl. its bug)
  gpfix  : the law the paper INTENDS: per pixel (s_t, sdot_t) ~ N(mu_t, Sigma_t) jointly
  indep  : state ~ N(m_t, v_t), velocity ~ N(mdot_t, w_t) INDEPENDENT of the state
           (= the structural content of the bug, without the pixel scramble)
  mmdet  : state ~ N(m_t, v_t), velocity = E[sdot_t | s_t]  (deterministic target)

Metrics: FID and KID exactly as in the notebook (1280 generated vs 1280 test images,
their metric/Fid_score.py, torchmetrics KID subset_size=40).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets, transforms
from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derisk import samp_x_dx2, gp_stats          # verbatim sampler + float64 per-t stats
from metric.Fid_score import calculate_fid
from torchmetrics.image.kid import KernelInceptionDistance

ALPHA, L = 1.0, 5.0          # paper's MNIST setting; --kern_l overrides L
T_OBS = torch.tensor([0., 1.])


def train(arm, seed, n_epochs, device, data_root, kern_l=L, sigma_max=-1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    trainset = datasets.MNIST(data_root, train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize((0.5,), (0.5,))]))
    g_data = torch.Generator().manual_seed(seed)                     # shared across arms
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                         drop_last=True, generator=g_data)
    g_shared = torch.Generator(device=device).manual_seed(seed)      # x0 and t: shared
    g_arm = torch.Generator(device=device).manual_seed(10**6 + seed) # arm-specific noise
    fm = ConditionalFlowMatcher(sigma=0.0)
    n_fail = 0; xt_b = ut_b = None; last_loss = None
    model.train()
    for ep in range(n_epochs):
        for x1, _ in loader:
            x1 = x1.to(device)
            B = x1.shape[0]
            x0 = torch.randn(x1.shape, generator=g_shared, device=device)
            t = torch.rand(B, generator=g_shared, device=device)
            if arm == "icfm":
                xt = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
                ut = x1 - x0
                if ep == 0 and last_loss is None:   # one-off check: identical to torchcfm sigma=0
                    _, xt_ref, ut_ref = fm.sample_location_and_conditional_flow(x0, x1, t=t)
                    assert torch.allclose(xt, xt_ref) and torch.allclose(ut, ut_ref)
            elif arm == "gp":
                x01 = torch.zeros(B, 2, 784, device=device)
                x01[:, 0, :] = x0.reshape(B, -1); x01[:, 1, :] = x1.reshape(B, -1)
                t_mat = t.reshape(B, 1)
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g_arm, device=device)))
                try:   # notebook: bare `except: print('fail'); pass` -> stale batch reused
                    with torch.device(device):          # notebook sets cuda as default device
                        xt_b, ut_b = samp_x_dx2(t_mat, ALPHA, L, x01, T_OBS.to(device), 0.0)
                except Exception:
                    n_fail += 1
                    if xt_b is None: continue
                xt = xt_b.reshape(B, 1, 28, 28); ut = ut_b.reshape(B, 1, 28, 28)
            elif arm == "bump":
                # decoupled chart: straight mean, pinned variance profile v_t = smax^2 (4t(1-t))^2
                tt = t[:, None, None, None]
                m = tt * x1 + (1 - tt) * x0
                md = x1 - x0
                g = 4 * t * (1 - t)
                sv = (sigma_max * g)[:, None, None, None]                  # sqrt(v_t)
                slope = (sigma_max * 4 * (1 - 2 * t))[:, None, None, None] # = (v_dot/2)/sqrt(v)
                e1 = torch.randn(x1.shape, generator=g_arm, device=device)
                xt = m + sv * e1
                ut = md + slope * e1
            else:
                A, S = gp_stats(t.double().cpu(), ALPHA, kern_l, T_OBS)     # (B,2,2) float64
                A = A.to(device).float(); S = S.to(device)
                m  = A[:, 0, 0, None, None, None] * x0 + A[:, 0, 1, None, None, None] * x1
                md = A[:, 1, 0, None, None, None] * x0 + A[:, 1, 1, None, None, None] * x1
                v = S[:, 0, 0]; c = S[:, 0, 1]; w = S[:, 1, 1]
                # Near the pinned endpoints v_t ~ t^2 -> 0 and float64 cancellation makes it
                # ~1e-15 or negative, so c/sqrt(v) explodes (seen: 1e6). Analytically
                # c/sqrt(v) -> sign(c) sqrt(w) there and the residual sd -> 0; use the limits.
                ok = v > 1e-9
                vs = v.clamp_min(0).sqrt()
                slope = torch.where(ok, c / vs.clamp_min(1e-30), torch.sign(c) * w.sqrt())
                resid = torch.where(ok, (w - c**2 / v.clamp_min(1e-30)).clamp_min(0).sqrt(), torch.zeros_like(w))
                assert slope.abs().max() < 10 * w.sqrt().max() + 1e-6, "slope blew up"
                sv = vs.float()[:, None, None, None]
                slope = slope.float()[:, None, None, None]; resid = resid.float()[:, None, None, None]
                e1 = torch.randn(x1.shape, generator=g_arm, device=device)
                e2 = torch.randn(x1.shape, generator=g_arm, device=device)
                xt = m + sv * e1
                if arm == "gpfix":
                    ut = md + slope * e1 + resid * e2
                elif arm == "indep":
                    ut = md + w.sqrt().float()[:, None, None, None] * e2
                elif arm == "mmdet":
                    ut = md + slope * e1
                else:
                    raise ValueError(arm)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = loss.item()
    return model, n_fail, last_loss


@torch.no_grad()
def generate(model, device, rep_eval=40, batch_eval=32):
    node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    out = []
    for i in range(rep_eval):
        torch.manual_seed(i)
        x0 = torch.randn(batch_eval, 1, 28, 28, device=device)
        traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=device))
        out.append(traj[-1].clip(-1, 1).cpu().numpy())
    return np.concatenate(out, 0)                                     # (1280,1,28,28) in [-1,1]


def real_subset(data_root, n=1280, seed=0):
    testset = datasets.MNIST(data_root, train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(testset), generator=g)[:n]
    return torch.stack([testset[int(i)][0] for i in idx]).numpy()     # (n,1,28,28) in [-1,1]


def fid_like_notebook(gen, real, device):
    up = torch.nn.Upsample(size=(299, 299), mode="bilinear")
    def prep(x):
        x = up(torch.tensor((x + 1) / 2)).cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return np.repeat(x, 3, axis=3)
    return float(calculate_fid(prep(gen), prep(real), use_multiprocessing=False, batch_size=32))  # notebook: 4; batch only affects speed (eval-mode inception)


def kid_like_notebook(gen, real, device, subset_size=40):
    kid = KernelInceptionDistance(subset_size=subset_size).to(device)
    A = torch.from_numpy(255 * (np.repeat(real, 3, axis=1) + 1) / 2).type(torch.uint8).to(device)
    B = torch.from_numpy(255 * (np.repeat(gen, 3, axis=1) + 1) / 2).type(torch.uint8).to(device)
    kid.update(A, real=True); kid.update(B, real=False)
    m, s = kid.compute()
    return float(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--kern_l", type=float, default=L)
    ap.add_argument("--sigma_max", type=float, default=-1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    import torch as _t
    tg = _t.linspace(1e-4, 1 - 1e-4, 20001).double()
    _, Sg = gp_stats(tg, ALPHA, a.kern_l, T_OBS)
    gp_sd = float(Sg[:, 0, 0].max().sqrt())
    smax = a.sigma_max if a.sigma_max > 0 else gp_sd
    assert a.arm != "bump" or a.sigma_max > 0, "bump needs --sigma_max"
    print(f"[check] l={a.kern_l} -> GP max sd {gp_sd:.4f}; arm uses max sd {smax:.4f}", flush=True)
    model, n_fail, last_loss = train(a.arm, a.seed, a.epochs, device, a.data,
                                     kern_l=a.kern_l, sigma_max=smax)
    t_train = time.time() - t0
    model.eval()
    gen = generate(model, device)
    del model; torch.cuda.empty_cache()
    assert gen.shape == (1280, 1, 28, 28) and np.isfinite(gen).all(), "generation produced NaN/inf"
    real = real_subset(a.data)
    fid = fid_like_notebook(gen, real, device)
    kid = kid_like_notebook(gen, real, device)
    res = {"arm": a.arm + a.tag, "kern_l": a.kern_l, "cov_sd": smax, "seed": a.seed, "fid": fid, "kid": kid, "train_sec": t_train,
           "eval_sec": time.time() - t0 - t_train, "sampler_fail": n_fail,
           "final_loss": last_loss, "gen_absmax": float(np.abs(gen).max())}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res), flush=True)
