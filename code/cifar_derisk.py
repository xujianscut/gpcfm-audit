"""
CIFAR-10 de-risk for the GP-CFM audit (paper Sec. 5.1 / Table 4 protocol).

Setup copied from torchcfm examples/images/cifar10/train_cifar10.py (the paper says it
follows Tong et al.): UNet 128ch, 2 res blocks, Adam 2e-4 + 5k warmup, grad-clip 1.0,
EMA 0.9999, batch 128, random h-flip, 400k steps.  Paper specifics: sigma = 1e-3 for the
linear interpolant, diagonal white noise 1e-6 in the GP covariance.  Kernel params are not
reported for CIFAR; we use the MNIST notebook's (alpha=1, l=5).

Arms (same seed -> same init, data order, x0 and t draws across arms => paired):
  icfm  : I-CFM, sigma = 1e-3
  gp    : released GP sampler verbatim (nt=1 flavour, as in 5_mnist.ipynb), sig2_diag=1e-6
  gpfix : the law the paper intends: per pixel (s_t, sdot_t) ~ N(mu_t, Sigma_t) jointly
  mmdet : state ~ N(m_t, v_t), velocity = E[sdot_t | s_t]

FID: cleanfid, legacy_tensorflow mode, CIFAR-10 train stats, dopri5 (tol 1e-5) from the
EMA model, exactly as compute_fid.py; num_gen configurable (10k at checkpoints, 50k final).
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch
from torchvision import datasets, transforms
from torchdiffeq import odeint
from torchcfm.models.unet.unet import UNetModelWrapper
from cleanfid import fid as cleanfid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derisk import samp_x_dx2, gp_stats

ALPHA, L = 1.0, 5.0
T_OBS = torch.tensor([0., 1.])


def make_model(device):
    return UNetModelWrapper(dim=(3, 32, 32), num_res_blocks=2, num_channels=128,
                            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
                            attention_resolutions="16", dropout=0.1).to(device)


def ema_update(src, tgt, decay):
    sd, td = src.state_dict(), tgt.state_dict()
    for k in sd:
        td[k].data.copy_(td[k].data * decay + sd[k].data * (1 - decay))


def infinite(loader):
    while True:
        for x, _ in loader:
            yield x


def gp_targets(arm, t, x0, x1, g_arm, device):
    """matched-marginal arms; per-pixel iid, shared scalar (m,v,c,w) per sample."""
    A, S = gp_stats(t.double().cpu(), ALPHA, L, T_OBS)
    A = A.to(device).float(); S = S.to(device)
    m = A[:, 0, 0, None, None, None] * x0 + A[:, 0, 1, None, None, None] * x1
    md = A[:, 1, 0, None, None, None] * x0 + A[:, 1, 1, None, None, None] * x1
    v, c, w = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    ok = v > 1e-9
    vs = v.clamp_min(0).sqrt()
    slope = torch.where(ok, c / vs.clamp_min(1e-30), torch.sign(c) * w.sqrt())
    resid = torch.where(ok, (w - c**2 / v.clamp_min(1e-30)).clamp_min(0).sqrt(), torch.zeros_like(w))
    assert slope.abs().max() < 10 * w.sqrt().max() + 1e-6, "slope blew up"
    sv, slope, resid = [z.float()[:, None, None, None] for z in (vs, slope, resid)]
    e1 = torch.randn(x1.shape, generator=g_arm, device=device)
    e2 = torch.randn(x1.shape, generator=g_arm, device=device)
    xt = m + sv * e1
    if arm == "gpfix":
        ut = md + slope * e1 + resid * e2
    elif arm == "mmdet":
        ut = md + slope * e1
    else:
        raise ValueError(arm)
    return xt, ut


@torch.no_grad()
def compute_fid(ema_model, device, num_gen, seed, bs=1024, tol=1e-5):
    ema_model.eval()
    g = torch.Generator(device=device).manual_seed(seed)

    def gen_1(_):
        x = torch.randn(bs, 3, 32, 32, generator=g, device=device)
        traj = odeint(ema_model, x, torch.linspace(0, 1, 2, device=device),
                      rtol=tol, atol=tol, method="dopri5")
        return (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)

    s = cleanfid.compute_fid(gen=gen_1, dataset_name="cifar10", batch_size=bs, dataset_res=32,
                             num_gen=num_gen, dataset_split="train", mode="legacy_tensorflow",
                             verbose=False)
    ema_model.train()
    return float(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400001)
    ap.add_argument("--ckpt_every", type=int, default=50000)
    ap.add_argument("--fid_every", type=int, default=100000)
    ap.add_argument("--fid_gen", type=int, default=10000)
    ap.add_argument("--fid_gen_final", type=int, default=50000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.arm}_s{a.seed}"
    log = os.path.join(a.outdir, tag + ".json")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    net = make_model(device); ema = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(s, 5000) / 5000)
    ds = datasets.CIFAR10(a.data, train=True, download=True, transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.5,) * 3, (0.5,) * 3)]))
    g_data = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=4,
                                         drop_last=True, generator=g_data)
    it = infinite(loader)
    g_shared = torch.Generator(device=device).manual_seed(a.seed)
    g_arm = torch.Generator(device=device).manual_seed(10**6 + a.seed)

    # resume
    start = 0; hist = []
    ck = os.path.join(a.outdir, tag + "_last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=device)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start = st["step"] + 1
        g_shared.set_state(st["g_shared"].cpu()); g_arm.set_state(st["g_arm"].cpu())
        hist = json.load(open(log)) if os.path.exists(log) else []
        for _ in range(start % len(loader)):
            next(it)   # approximate data-order resume
        print(f"resumed at step {start}", flush=True)

    n_fail = 0; xt_b = ut_b = None; t0 = time.time(); ema_loss = None
    net.train()
    for step in range(start, a.steps):
        x1 = next(it).to(device); B = x1.shape[0]
        x0 = torch.randn(x1.shape, generator=g_shared, device=device)
        t = torch.rand(B, generator=g_shared, device=device)
        if a.arm == "icfm":
            tt = t[:, None, None, None]
            xt = tt * x1 + (1 - tt) * x0 + 1e-3 * torch.randn(x1.shape, generator=g_arm, device=device)
            ut = x1 - x0
        elif a.arm == "gp":
            x01 = torch.stack([x0.reshape(B, -1), x1.reshape(B, -1)], dim=1)
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g_arm, device=device)))
            try:
                with torch.device(device):
                    xt_b, ut_b = samp_x_dx2(t.reshape(B, 1), ALPHA, L, x01, T_OBS.to(device), 1e-6)
            except Exception:
                n_fail += 1
                if xt_b is None:
                    continue
            xt = xt_b.reshape(B, 3, 32, 32); ut = ut_b.reshape(B, 3, 32, 32)
        else:
            xt, ut = gp_targets(a.arm, t, x0, x1, g_arm, device)
        vt = net(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); ema_update(net, ema, 0.9999)
        ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
        if step % 1000 == 0:
            rate = (time.time() - t0) / max(1, step - start + 1) * 1000
            print(f"{tag} step {step} loss {ema_loss:.4f} fail {n_fail} {rate:.0f}s/1k", flush=True)
        last = step == a.steps - 1
        if (step % a.ckpt_every == 0 and step > 0) or last:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step,
                        "g_shared": g_shared.get_state(), "g_arm": g_arm.get_state()}, ck)
            if step % 100000 == 0 or last:
                torch.save({"ema": ema.state_dict(), "step": step},
                           os.path.join(a.outdir, f"{tag}_ema_{step}.pt"))
        if (step % a.fid_every == 0 and step > 0) or last:
            n = a.fid_gen_final if last else a.fid_gen
            tf = time.time(); f = compute_fid(ema, device, n, seed=777 + step)
            hist.append({"arm": a.arm, "seed": a.seed, "step": step, "fid": f, "num_gen": n,
                         "loss": ema_loss, "n_fail": n_fail, "fid_sec": time.time() - tf})
            json.dump(hist, open(log, "w"), indent=1)
            print(f"{tag} FID@{step} ({n} gen) = {f:.3f}  [{time.time()-tf:.0f}s]", flush=True)
    print("done", flush=True)
