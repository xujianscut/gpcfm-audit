"""Dump draws from the released sampler and from the intended conditional law, same streams."""
import torch, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derisk import samp_x_dx2, cov_mat2, calc_r, k11, k12
torch.manual_seed(0)
nB, nt, dim, alpha, l = 4, 3, 2, 1.0, 2.0
t_obs = torch.tensor([0., 1.])
x_obs = torch.randn(nB, 2, dim); t_mat = torch.rand(nB, nt)
def intended(b):
    tm = t_mat[b:b+1]; r = calc_r(t_obs.expand(1, -1), tm)
    S11 = cov_mat2(tm, tm, alpha, l, 0.0)
    S21 = torch.cat([k11(r, alpha, l), k12(r, alpha, l)], 2); S12 = S21.permute(0, 2, 1)
    S22i = torch.linalg.inv(k11(calc_r(t_obs, t_obs), alpha, l))[None]
    A = torch.bmm(S12, S22i); mu = torch.bmm(A, x_obs[b:b+1])[0]; S = (S11 - torch.bmm(A, S21))[0]
    return mu.numpy(), ((S + S.T) / 2).numpy()
NC = 3000
acc = []
for _ in range(NC):
    x, d = samp_x_dx2(t_mat, alpha, l, x_obs, t_obs, 0.0)
    acc.append(torch.cat([x, d], 1))
E = torch.stack(acc).permute(1, 0, 2, 3).numpy()      # (nB, NC, 2nt, dim)
out = {"released": E, "t_mat": t_mat.numpy(), "x_obs": x_obs.numpy()}
rng = np.random.default_rng(0)
for b in range(nB):
    mu, S = intended(b)
    out[f"mu_{b}"] = mu; out[f"S_{b}"] = S
    # intended draws per dim (same S for both dims)
    w_, U_ = np.linalg.eigh(S); L = U_ @ np.diag(np.sqrt(np.clip(w_, 0, None)))
    z = rng.standard_normal((NC, dim, 2 * nt))
    out[f"intended_{b}"] = (mu[None] + np.einsum("ij,ndj->nid", L, z))   # (NC, 2nt, dim)
np.savez("res/bug_draws.npz", **out)
print("saved", E.shape)
