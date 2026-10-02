"""Validation of the all-dump disc integration (fw_disc_dumps.py; T_eff' library interpolation + each dump's velocities).

Tests (all for the 8 lines of sight of fw_disc.los8, max |dF| in units of the continuum):
  V1  dump 3200 through the pipeline vs the exact per-point sums (every point's own model, fw_disc_los.py ->
      RUN/disc_los8.npz), with and without velocities; intensity method (T_eff'-interpolated) vs fw_disc_imu.py
      (nearest-bin intensity library, RUN/disc_los8_imu.npz; --no-imu skips it, it needs ~8 GB and 2 min to set up).
  V3  points beyond the T_eff' node range (clamped to the end node): for the dumps with the most such points and with the
      coolest point, clamping vs (a) leaving these points out and (b) extrapolating linearly with the slope between the
      end node and the node >= 300 K inside.
  V4  linear interpolation in T_eff' vs the nearest 10 K bin (fw_disc.integrate_lib), for later dumps.
  V5  node merging: --nmin 1, 5, 100 vs the default 20.
  V6  1 km/s rounding of the Doppler shifts vs continuous shifts, for a random subset of visible points of later dumps
      (direct per-point sum of the interpolated library profiles, shifted by interpolation; an independent code path).
The hold-out test (library from half of the dump-3200 models, predicting the other half) is fw_disc_holdout.py; the
systematics of the FASTWIND outputs and of the method (flux vs intensity, library variants), which are time-variable and
invisible to these tests, are measured on the production time series by fw_disc_systematics.py.
Output: DISC_DUMPS/validate.npz (all numbers) and the printed table.

Usage:  ./run.sh fw_disc_dumps_validate.py [--dumps 4000,4800] [--nsub 20000]
"""
import argparse
import os
import time

import numpy as np

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", default="3600,4000,4400,4800", help="later dumps for V4 and V6")
ap.add_argument("--nsub", type=int, default=20000)
ap.add_argument("--nmin", type=int, default=20)
ap.add_argument("--no-imu", action="store_true")
ap.add_argument("--ranges", default=None, help="npy with per-dump (dump, tmin, tmax, n_lo, n_hi) to pick the V3 dumps")
a = ap.parse_args()
T0 = time.time()
R = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


lib = np.load(os.path.join(fd.RUN, "library_dT10.npz"))
NODES = fd.lib_nodes(lib, nmin=a.nmin)
FX = fd.DiscFlux(NODES)
pts = np.load(os.path.join(fd.RUN, "points.npz"))
rhat, that, phat = fd.unit_vectors(pts["theta"], pts["phi"])
LOS = fd.los8()
MU, TN, PN = (np.ascontiguousarray((u @ LOS.T).T) for u in (rhat, that, phat))
del rhat, that, phat


def sample(d):
    s = np.load(os.path.join(fd.SAMPLES, f"d{d:04d}.npz"))
    return s["teff"].astype(np.float64), s["ur"].astype(np.float64), s["uth"].astype(np.float64), s["uph"].astype(np.float64)


def run(integ, d, pairs=None, mask=None):
    """F, F0 (8, 3, ny) of dump d; pairs = (k0, k1, a) override; mask: points to include."""
    teff, ur, uth, uph = sample(d)
    k0, k1, w = integ.pairs(teff) if pairs is None else pairs
    F, F0 = np.zeros((8, 3, fd.Y.size)), np.zeros((8, 3, fd.Y.size))
    for k in range(8):
        mu = MU[k] if mask is None else np.where(mask, MU[k], 0.0)
        F[k], F0[k] = integ(mu, ur * MU[k] + uth * TN[k] + uph * PN[k], k0, k1, w)[:2]
    return F, F0


def dmax(A, B):
    return np.abs(A - B).max(axis=(0, 2))            # per line


def fmt(x):
    return " ".join(f"{v:.1e}" for v in x)


# ---- V1: dump 3200 vs the exact per-point result ----
ex = np.load(os.path.join(fd.RUN, "disc_los8.npz"))
F, F0 = run(FX, 3200)
R["V1_flux_F"], R["V1_flux_F0"] = dmax(F, ex["F"]), dmax(F0, ex["F0"])
R["V1_flux_dEW"] = np.array([np.abs([fd.diagnostics(F[k, j], j)["ew"] - fd.diagnostics(ex["F"][k, j], j)["ew"]
                                      for k in range(8)]).max() for j in range(3)])
log(f"V1 flux, dump 3200 vs exact per-point sums: max|dF| {fmt(R['V1_flux_F'])} (no velocities {fmt(R['V1_flux_F0'])}); "
    f"max|dEW| {fmt(R['V1_flux_dEW'])} A")
F3200 = F
if not a.no_imu:
    exi = np.load(os.path.join(fd.RUN, "disc_los8_imu.npz"))
    Fi, Fi0 = run(fd.DiscImu(), 3200)
    R["V1_imu_F"], R["V1_imu_F0"] = dmax(Fi, exi["F"]), dmax(Fi0, exi["F0"])
    log(f"V1 imu, dump 3200 (T_eff' interpolation) vs fw_disc_imu.py (nearest bin): max|dF| {fmt(R['V1_imu_F'])} "
        f"(no velocities {fmt(R['V1_imu_F0'])})")

# ---- V4: interpolation vs nearest bin; V5: node merging ----
dl = [int(x) for x in a.dumps.split(",")]
R["V4_dumps"] = np.array(dl)
R["V4"] = np.zeros((len(dl), 3))
for i, d in enumerate(dl):
    teff, ur, uth, uph = sample(d)
    F, _ = run(FX, d)
    b = np.clip(np.digitize(teff, lib["edges"]) - 1, 0, lib["edges"].size - 2)
    Fn = np.array([fd.integrate_lib(lib, teff, ur * MU[k] + uth * TN[k] + uph * PN[k], fd.weights(MU[k], 1.0)[:, None] * lib["fc"][b])
                   for k in range(8)])
    R["V4"][i] = dmax(F, Fn)
    log(f"V4 dump {d}: T_eff' interpolation vs nearest 10 K bin: max|dF| {fmt(R['V4'][i])}")
    if i == 0:
        Fref_d, dref = F, d
for nm in (1, 5, 100):
    F, _ = run(fd.DiscFlux(fd.lib_nodes(lib, nmin=nm)), dref)
    R[f"V5_nmin{nm}"] = dmax(F, Fref_d)
    log(f"V5 dump {dref}: nmin {nm} vs {a.nmin}: max|dF| {fmt(R[f'V5_nmin{nm}'])}")

# ---- V3: points beyond the node range ----
if a.ranges:
    rg = np.load(a.ranges)
    v3 = sorted({int(rg[np.argmax(rg[:, 3] + rg[:, 4]), 0]), int(rg[np.argmin(rg[:, 1]), 0]), int(rg[np.argmax(rg[:, 2]), 0])})
else:
    v3 = [4391, 4169]
R["V3_dumps"] = np.array(v3)
R["V3_leaveout"], R["V3_extrap"], R["V3_n"] = np.zeros((len(v3), 3)), np.zeros((len(v3), 3)), np.zeros((len(v3), 2))
tn = FX.t
jlo = int(np.searchsorted(tn, tn[0] + 300.0))
jhi = int(np.searchsorted(tn, tn[-1] - 300.0)) - 1
for i, d in enumerate(v3):
    teff = sample(d)[0]
    lo, hi = teff < tn[0], teff > tn[-1]
    R["V3_n"][i] = lo.sum(), hi.sum()
    F, _ = run(FX, d)
    Fo, _ = run(FX, d, mask=~(lo | hi))
    k0, k1, w = FX.pairs(teff)
    k0, k1, w = k0.copy(), k1.copy(), w.copy()
    k0[lo], k1[lo], w[lo] = 0, jlo, (teff[lo] - tn[0]) / (tn[jlo] - tn[0])            # w < 0: extrapolation
    k0[hi], k1[hi], w[hi] = jhi, tn.size - 1, (teff[hi] - tn[jhi]) / (tn[-1] - tn[jhi])  # w > 1
    Fe, _ = run(FX, d, pairs=(k0, k1, w))
    R["V3_leaveout"][i], R["V3_extrap"][i] = dmax(F, Fo), dmax(F, Fe)
    log(f"V3 dump {d}: {int(lo.sum())} points below {tn[0]:.0f} K (min {teff.min():.0f}), {int(hi.sum())} above {tn[-1]:.0f} K "
        f"(max {teff.max():.0f}); clamped vs left out: max|dF| {fmt(R['V3_leaveout'][i])}; vs extrapolated: {fmt(R['V3_extrap'][i])}")

# ---- V6: 1 km/s rounding vs continuous shifts (direct per-point sum, subset of visible points) ----
rng = np.random.default_rng(7)
R["V6"] = np.zeros((len(dl), 3))
P = NODES["prof"]
for i, d in enumerate(dl):
    teff, ur, uth, uph = sample(d)
    k = i % 8
    v = ur * MU[k] + uth * TN[k] + uph * PN[k]
    sub = np.sort(rng.choice(np.where(MU[k] > 0)[0], a.nsub, replace=False))
    k0, k1, w = FX.pairs(teff[sub])
    mask = np.zeros(teff.size, bool)
    mask[sub] = True
    Fh = FX(np.where(mask, MU[k], 0.0), v, *FX.pairs(teff))[0]
    s = -fd.C_KMS * np.log(1.0 - v[sub] / fd.C_KMS)
    for j in range(3):
        fc = (1 - w) * NODES["fc"][k0, j] + w * NODES["fc"][k1, j]
        num = np.zeros(fd.Y.size)
        for c0 in range(0, sub.size, 2000):
            c = slice(c0, c0 + 2000)
            line = (1 - w[c, None]) * NODES["fc"][k0[c], j, None] * P[k0[c], j] + w[c, None] * NODES["fc"][k1[c], j, None] * P[k1[c], j]
            num += MU[k][sub[c]] @ fd.interp_rows(fd.Y[None, :] - s[c, None], line, fd.Y)
        Fd = num / np.sum(MU[k][sub] * fc)
        R["V6"][i, j] = np.abs(Fh[j] - Fd).max()
    log(f"V6 dump {d}, los{k + 1}, {a.nsub} points: 1 km/s rounded vs continuous shifts: max|dF| {fmt(R['V6'][i])}")

out = os.path.join(fd.DISC_DUMPS, "validate.npz")
os.makedirs(fd.DISC_DUMPS, exist_ok=True)
np.savez(out, lines=np.array(fd.LINES), nmin=a.nmin, **R)
log(f"wrote {out}")
