"""Emergent-intensity library I(lambda, mu) in T_eff' bins for the dump-3200 per-point run.

FASTWIND's pformalsol normally writes flux profiles only. The modified build FW_10.6.4.1/v10.6_HHe_imu (formalsol.f90,
changes marked [M424]) also writes OUT_IMU.<line>_VTV010: the emergent continuum and line intensity of every ray of the
formal solution (impact parameter p in units of the inner-boundary radius), at the same wavelengths as OUT.*; the OUT
files themselves are unchanged. This script:
  1. picks one representative model per 10 K bin of T_eff' (the saved model closest to the bin's mean T_eff'), from
     extracted model directories (--raw, several allowed);
  2. reruns the modified pformalsol on it (0.5 s; no new atmosphere) in RUNS/P<idx>/;
  3. keeps, per line, the rays with p <= R_max (the radius beyond which the continuum intensity is < 1e-3 of the
     disc-centre value; ~1.013), with node coordinate s = p / R_max = sqrt(1 - mu^2), and interpolates each ray's
     intensities onto the common velocity grid fw_disc.Y;
  4. checks that the flux 2 int I mu dmu, with I interpolated linearly in p between the rays (how FASTWIND treats it),
     reproduces the model's own flux profile.
Output: OUT/imu_library_dT10.npz with edges, tmean, count, idx_rep, teff_rep, rmax (nb, 3), nnode (nb, 3),
s (nb, 3, K) (NaN-padded), Ic, Il (nb, 3, K, ny) float32; empty bins point to the nearest filled bin (src).

The FASTWIND binaries need the host's libraries, so step 2 runs on the host (fw_imu_run.sh), not in the container:
    ./run.sh fw_imu_library.py --raw DIR [DIR ...] --stage select     -> OUT/representatives.txt (bin, idx, T_eff', dir)
    ./fw_imu_run.sh OUT/representatives.txt OUT/runs 20                 (modified pformalsol, host)
    ./run.sh fw_imu_library.py --raw DIR [DIR ...] --stage build      -> OUT/imu_library_dT10.npz
"""
import argparse
import glob
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", nargs="+", required=True, help="directories holding extracted model directories P<idx>/")
ap.add_argument("--out", default="/scratch/ppathak/fastwind_imu")
ap.add_argument("--stage", choices=["select", "build"], required=True)
a = ap.parse_args()
BUILD = "/scratch/ppathak/FW_10.6.4.1/v10.6_HHe_imu"
FW = "/scratch/ppathak/FW_10.6.4.1"
FORMAL_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastwind", "FORMAL_INPUT_He3")
MODEL_FILES = ["MODEL", "NLTE_POP", "LTE_POP", "ENION", "TAU_ROS", "FLUXCONT", "CLUMPING_OUTPUT", "CONT_FORMAL", "CONT_FORMAL_ALL"]
RUNS = os.path.join(a.out, "runs")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


# ---- 1. representatives ----
flib = fd.library()                                     # flux library: edges, tmean, count
edges, tmean, count = flib["edges"], flib["tmean"], flib["count"]
nb = edges.size - 1
cands = []
for r in a.raw:
    for d in sorted(glob.glob(os.path.join(r, "P*"))):
        if os.path.exists(os.path.join(d, "meta.txt")) and os.path.exists(os.path.join(d, "CONT_FORMAL")):
            f = open(os.path.join(d, "meta.txt")).read().split()
            cands.append((int(f[0]), float(f[1]), d))
idx_c = np.array([c[0] for c in cands]); t_c = np.array([c[1] for c in cands])
b_c = np.clip(np.digitize(t_c, edges) - 1, 0, nb - 1)
rep = {}
for b in np.where(count > 0)[0]:
    k = np.where(b_c == b)[0]
    if k.size:
        rep[int(b)] = int(k[np.argmin(np.abs(t_c[k] - tmean[b]))])
missing = [int(b) for b in np.where(count > 0)[0] if int(b) not in rep]
log(f"{len(cands)} candidate models; representatives for {len(rep)} of {int((count > 0).sum())} filled bins"
    + (f"; MISSING bins {missing[:10]}" if missing else ""))
assert not missing, "extract models for the missing bins first"

sel = os.path.join(a.out, "representatives.txt")
if a.stage == "select":
    os.makedirs(a.out, exist_ok=True)
    with open(sel, "w") as f:
        for b in sorted(rep):
            i, t, d = cands[rep[b]]
            f.write(f"{b} {i} {t:.3f} {d}\n")
    log(f"wrote {sel}; now run fw_imu_run.sh {sel} {RUNS} 20 on the host, then --stage build")
    raise SystemExit(0)

# ---- 2. (host) modified pformalsol: fw_imu_run.sh wrote RUNS/P<idx>/P<idx>/OUT_IMU.* ----
mdir_of = {b: os.path.join(RUNS, f"P{cands[rep[b]][0]:06d}", f"P{cands[rep[b]][0]:06d}") for b in rep}
bad = [b for b, m in mdir_of.items() if not all(os.path.exists(os.path.join(m, f"OUT_IMU.{ln}_VTV010")) for ln in fd.LINES)]
log(f"{len(mdir_of)} representatives; missing OUT_IMU: {bad}")
assert not bad

# ---- 3. library ----
K = 0
per = {}
for b, m in mdir_of.items():
    for j, ln in enumerate(fd.LINES):
        lam, p, Ic, Il = fd.read_imu(os.path.join(m, f"OUT_IMU.{ln}_VTV010"))
        rmax = fd.r_outer(p, Ic)
        keep = p <= rmax
        per[b, j] = (lam, p[keep], rmax, Ic[:, keep], Il[:, keep])
        K = max(K, int(keep.sum()))
ny = fd.Y.size
S = np.full((nb, 3, K), np.nan)
IC = np.zeros((nb, 3, K, ny), np.float32)
IL = np.zeros((nb, 3, K, ny), np.float32)
RM = np.zeros((nb, 3)); NN = np.zeros((nb, 3), int)
chk = []
mf = np.linspace(0.0, 1.0, 4001)
for (b, j), (lam, p, rmax, Ic, Il) in per.items():
    yl = fd.C_KMS * np.log(lam / fd.LREF[j])
    n = p.size
    S[b, j, :n] = p / rmax
    RM[b, j], NN[b, j] = rmax, n
    IC[b, j, :n] = fd.interp_rows(np.repeat(yl[None, :], n, 0), Ic.T, fd.Y)
    IL[b, j, :n] = fd.interp_rows(np.repeat(yl[None, :], n, 0), Il.T, fd.Y)
    # check: flux with I linear in p (s) vs the model's own flux profile, on the model's wavelength grid
    sf = np.sqrt(1.0 - mf ** 2)
    Lf = np.array([np.interp(sf, p / rmax, row) for row in Il]); Cf = np.array([np.interp(sf, p / rmax, row) for row in Ic])
    fn = np.trapz(Lf * 2 * mf, mf, axis=1) / np.trapz(Cf * 2 * mf, mf, axis=1)
    w0, f0 = np.genfromtxt(os.path.join(mdir_of[b], f"OUT.{fd.LINES[j]}_VTV010"), usecols=[2, 4], max_rows=161).T
    chk.append((j, np.abs(fn - f0).max(), np.trapz(1 - fn, w0) - np.trapz(1 - f0, w0)))
chk = np.array(chk)
for j in range(3):
    c = chk[chk[:, 0] == j]
    log(f"{fd.LINES[j]}: flux from intensities vs FASTWIND flux, over {len(c)} models: max|dF| {c[:, 1].max():.1e}, "
        f"max|dEW| {np.abs(c[:, 2]).max():.1e} A; rays kept {NN[:, j][NN[:, j] > 0].min()}-{NN[:, j].max()}, R_max "
        f"{RM[:, j][RM[:, j] > 0].min():.4f}-{RM[:, j].max():.4f}")
filled = np.array(sorted(rep))
src = np.array([filled[np.argmin(np.abs(filled - b))] for b in range(nb)])
for b in range(nb):
    if src[b] != b:
        S[b], IC[b], IL[b], RM[b], NN[b] = S[src[b]], IC[src[b]], IL[src[b]], RM[src[b]], NN[src[b]]
idx_rep = np.array([cands[rep[src[b]]][0] for b in range(nb)]); teff_rep = np.array([cands[rep[src[b]]][1] for b in range(nb)])
out = os.path.join(a.out, "imu_library_dT10.npz")
np.savez(out[:-4] + ".tmp.npz", edges=edges, tmean=tmean, count=count, src=src, idx_rep=idx_rep, teff_rep=teff_rep,
         rmax=RM, nnode=NN, s=S, Ic=IC, Il=IL)
os.replace(out[:-4] + ".tmp.npz", out)
log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
