# What Does a Stream Model Buy You in Flow Matching?

Code and per-seed results for *What Does a Stream Model Buy You in Flow Matching?*
(Jian Xu, RIKEN Center for Advanced Intelligence Project; submitted to *Neural Networks*).
Table and figure numbers below refer to that manuscript.

## Layout

    code/      experiment and analysis scripts
    results/   every per-seed result file behind the tables and figures
    figs/      figure script; running it regenerates the four PDFs from results/
    code/warm, code/warm2
               the warm-start checkpoints used for the 2-Gaussian runs (Table 1 protocol and
               coverage sweep); derisk.py creates them on first use if absent

## Reproducing the paper's numbers without retraining

Every table can be recomputed from `results/` in seconds:

    cd code
    python collect.py         # Table 2  2-Gaussian, six arms, 100 seeds
    python collect3.py        # Table 3  coverage sweep, medians and divergence counts
    python iso.py             # Table 4  mean/variance 2x2 attribution
    python collect_mnist.py   # Table 5  MNIST, 20 seeds x 5 arms
    python mnsw.py            # Table 6  MNIST coverage sweep
    python mnsw2.py           #          paired comparisons across coverage levels
    cd ../figs && python make_figs.py     # Figures 1-4

Table 7 (CIFAR-10) is read directly from the per-seed files in `results/res_cifar/`.

Set `RES=/path/to/results` to point the analysis scripts elsewhere.

## Re-running the experiments

`derisk.py` (2-Gaussian), `mnist_derisk.py`, `cifar_derisk.py`. Arms are selected with
`--arm`; the coverage sweeps additionally take `--kern_l` and `--sigma_max`.

    # 2-Gaussian, one arm, 100 seeds (paper protocol, warm start as in the released notebook)
    python derisk.py --arm gpfix --nseeds 100 --epochs 5000 --warm warm/gpfix.pt --out res/gpfix_0.json

    # 2-Gaussian coverage sweep / 2x2 crossing
    python derisk.py --arm gpmean_decvar --kern_l 0.25 --sigma_max 0.9815 --nseeds 40 ...

    # MNIST (5 epochs, paper protocol) and its coverage sweep
    python mnist_derisk.py --arm gpfix --seed 0 --out res_mnist/gpfix_0.json
    python mnist_derisk.py --arm bump --kern_l 0.5 --sigma_max 0.5933 --seed 0 --out ...

    # CIFAR-10 (400k steps, Tong et al. configuration; ~30 h on one A6000)
    python cifar_derisk.py --arm gpfix --seed 0 --outdir res_cifar

`cifar_run.sh`, `mnsweep.sh`, `mnist_worker.sh`, `run_sweep.sh` are the batch wrappers we
used; they resume from the last checkpoint if a run is killed.

### Arms

| arm | state | velocity target |
|---|---|---|
| `icfm` | linear interpolant | `x1 - x0` |
| `gp` | the released `samp_x_dx2`, verbatim (including its indexing bug) | |
| `gpfix` | joint draw from the conditional GP, correctly paired | |
| `mmstoch` | GP per-time law, times drawn independently | |
| `mmdet` | GP `(m_t, v_t)` | analytic `E[sdot_t | s_t]` |
| `indep` | GP `(m_t, v_t)` | drawn independently of the state (images) |
| `bump` | straight mean, `v_t = smax^2 (4t(1-t))^2` | analytic |
| `gpmean_decvar` | GP mean, decoupled variance | analytic |
| `straight_gpvar` | straight mean, GP variance | analytic |

## The sampler audit

    python test2.py       # empirical moments of the released sampler vs the law it derives
    python dump_bug.py    # writes results/res/bug_draws.npz, used by Figure 2

Both call the released `samp_x_dx2` unmodified. Running them needs the original
repository (the flow-matching-with-GP-streams code the paper audits) on the path;
`derisk.py` carries a verbatim copy of the kernel and sampler functions so the
analysis scripts and figures run without it.

## Requirements

python 3.10, torch 2.5 (cu124), torchdyn, torchdiffeq, torchcfm, POT, torchmetrics,
clean-fid, scipy, numpy, matplotlib. CIFAR-10 FID uses `clean-fid` in
`legacy_tensorflow` mode; MNIST FID/KID use the audited repository's own metric code.

## Note on determinism

Per seed, one generator drives initialisation, one the data order, one the shared `x0`
and `t` draws, and one the arm-specific noise, so arms are paired exactly. cuDNN
convolution backward is nondeterministic by default: re-running the same arm and seed
on a different GPU moved MNIST FID by at most 0.33 (mean +0.02) over ten seeds.

## Third-party code

`derisk.py` contains a verbatim copy of the kernel and sampler functions (`k11`, `k12`, `k22`,
`cov_mat2`, `samp_x_dx2`, ...) from the audited repository,
[weigcdsb/GP-CFM](https://github.com/weigcdsb/GP-CFM) (`1_varRed.ipynb`, commit `87fc8ae`),
reproduced unmodified so that the audit tests the released code itself. Those functions remain
the work of their authors; the upstream repository carries no license.

## License

MIT for the code written for this study (see `LICENSE`); this excludes the third-party
functions listed above.
