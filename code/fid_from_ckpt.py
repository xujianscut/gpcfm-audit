"""Compute the final 50k-sample FID from a saved EMA checkpoint and append it to the run log."""
import sys, json, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_derisk import make_model, compute_fid
arm, seed = sys.argv[1], int(sys.argv[2]); bs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
dev = torch.device("cuda")
ema = make_model(dev); st = torch.load(f"res_cifar/{arm}_s{seed}_ema_400000.pt", map_location=dev)
assert st["step"] == 400000; ema.load_state_dict(st["ema"])
t0 = time.time(); f = compute_fid(ema, dev, 50000, seed=777 + 400000, bs=bs)
log = f"res_cifar/{arm}_s{seed}.json"; hist = json.load(open(log)) if os.path.exists(log) else []
assert all(h["step"] != 400000 for h in hist), "final FID already present"
hist.append({"arm": arm, "seed": seed, "step": 400000, "fid": f, "num_gen": 50000, "loss": None, "n_fail": None, "fid_sec": time.time() - t0, "note": "from ema ckpt"})
json.dump(hist, open(log, "w"), indent=1)
print(f"{arm}_s{seed} FID@400000 (50000 gen) = {f:.3f}  [{time.time()-t0:.0f}s]", flush=True)
