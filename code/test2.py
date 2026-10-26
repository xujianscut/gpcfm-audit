import torch, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derisk import samp_x_dx2, cov_mat2, calc_r, k11, k12
torch.manual_seed(0)
nB, nt, dim, alpha, l = 4, 3, 2, 1.0, 2.0
t_obs = torch.tensor([0.,1.])
x_obs = torch.randn(nB,2,dim); t_mat = torch.rand(nB,nt)

def intended(b):
    tm=t_mat[b:b+1]; r=calc_r(t_obs.expand(1,-1),tm)
    S11=cov_mat2(tm,tm,alpha,l,0.0)
    S21=torch.cat([k11(r,alpha,l),k12(r,alpha,l)],2); S12=S21.permute(0,2,1)
    S22i=torch.linalg.inv(k11(calc_r(t_obs,t_obs),alpha,l))[None]
    A=torch.bmm(S12,S22i); mu=torch.bmm(A,x_obs[b:b+1])[0]; S=(S11-torch.bmm(A,S21))[0]
    return mu,(S+S.T)/2

NC=6000
acc=[]
for _ in range(NC):
    x,d=samp_x_dx2(t_mat,alpha,l,x_obs,t_obs,0.0)
    acc.append(torch.cat([x,d],1))
E=torch.stack(acc).permute(1,0,2,3)      # (nB, NC, 2nt, dim)
print(f"n = {NC} draws per stream\n")
print(f"{'b':>2} {'mean relerr':>11} | {'corr(s_t1,sdot_t1) empirical':>28} {'intended':>9} | {'Var(s_t1) emp':>13} {'intended':>9} {'Sigcond[1,1] of stream 2b%nB':>29}")
for b in range(nB):
    mu_i,S_i=intended(b); Eb=E[b]
    mu_e=Eb.mean(0); mrel=((mu_e-mu_i).abs().max()/mu_i.abs().max()).item()
    ce=np.corrcoef(Eb[:,0,0].numpy(),Eb[:,nt,0].numpy())[0,1]
    ci=(S_i[0,nt]/(S_i[0,0]*S_i[nt,nt]).sqrt()).item()
    ve=Eb[:,0,0].var().item(); vi=S_i[0,0].item()
    _,S_alt=intended((2*b)%nB); alt=S_alt[0,0].item()
    print(f"{b:>2} {mrel:>11.4f} | {ce:>28.4f} {ci:>9.4f} | {ve:>13.4f} {vi:>9.4f} {alt:>29.4f}")

# structural prediction: sample(b,i,d) == X[row 2b + (i>=nt), col 2*(i%nt)+d]
# => Var of (b, x at time i, dim d) should equal Sigcond_of_stream(2b%nB)[2i+d, 2i+d]
print("\nprediction: Var[ s_{t_i}^{(d)} ] == Sig_cond(stream (2b)%nB)[2i+d, 2i+d]")
print(f"{'b':>2} {'i':>2} {'d':>2} {'empirical Var':>14} {'predicted-by-bug':>17} {'intended':>10}")
for b in range(nB):
    _,S_alt=intended((2*b)%nB); _,S_i=intended(b)
    for i in range(nt):
        for d in range(dim):
            print(f"{b:>2} {i:>2} {d:>2} {E[b][:,i,d].var().item():>14.4f} {S_alt[2*i+d,2*i+d].item():>17.4f} {S_i[i,i].item():>10.4f}")
