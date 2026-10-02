"""Disc integration of dump 3200 with emergent intensities I(lambda, mu) (the SPAMMS approach, Abdul-Masih et al. 2020).

For a line of sight n every visible point i (mu_i = r_i . n > 0) contributes the specific intensity its model emits in
direction mu_i, from the intensity library (fw_imu_library.py; bin of the point's T_eff'), Doppler-shifted by
v_i = u_i . n, weighted by its projected area mu_i dA (equal dA):
    F(lambda) / F_c(lambda) = sum_i mu_i I_l,i(lambda / (1 - v_i/c), mu_i) / sum_i mu_i I_c,i(lambda / (1 - v_i/c), mu_i).
Limb darkening and the centre-to-limb change of the line are thus included (the flux method, fw_disc_los.py, used each
model's flux profile with weight mu F_c instead). I(mu) is interpolated linearly in s = sqrt(1 - mu^2) = p / R_max between
the rays of the formal solution, which reproduces FASTWIND's flux for a uniform star. Numerically, the weights
mu_i (1 - t_i) and mu_i t_i go to the two bracketing ray nodes; for every (T_eff' bin, node) the node's line and continuum
intensities are convolved with its histogram of line-of-sight velocities (1 km/s, as in the flux method).
Checks: (1) a uniform star (all points in the bin of 38 230 K) without velocities must return that model's FASTWIND flux
profile; (2) comparison with the flux method (RUN/disc_los8.npz).
Output: RUN/disc_los8_imu.npz (same fields as disc_los8.npz: F, F0, diag_*, vmean_w, sigma_w; weights mu I_c(mu)).

Usage:  ./run.sh fw_disc_imu.py [--lib /scratch/ppathak/fastwind_imu/imu_library_dT10.npz]
"""
import argparse
import os
import time

import numpy as np
from scipy.signal import fftconvolve

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default="/scratch/ppathak/fastwind_imu/imu_library_dT10.npz")
ap.add_argument("--chunk", type=int, default=2000, help="(bin, node) rows per FFT batch")
a = ap.parse_args()
T0 = time.time()
V = fd.VSHIFT
ny = fd.Y.size


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


L = np.load(a.lib)
edges, S, NN, IC, IL = L["edges"], L["s"], L["nnode"], L["Ic"], L["Il"]
nb, _, K = S.shape
log(f"intensity library: {nb} bins, up to {K} ray nodes, representatives T_eff' {L['teff_rep'].min():.0f}-{L['teff_rep'].max():.0f} K")
p = np.load(os.path.join(fd.RUN, "profiles.npz"))
teff, theta, phi = p["teff"], p["theta"], p["phi"]
smp = np.load(os.path.join(fd.SAMPLES, "d3200.npz"))
rhat, that, phat = fd.unit_vectors(theta, phi)
bins_all = np.clip(np.digitize(teff, edges) - 1, 0, nb - 1)
los = fd.los8()


def node_weights(b, s, j):
    """For points in bins b at s = sqrt(1 - mu^2): lower node index k and fraction t (linear in s = p / R_max)."""
    k = np.empty(b.size, int); t = np.empty(b.size)
    for bb in np.unique(b):
        m = b == bb
        n = NN[bb, j]
        sn = S[bb, j, :n]
        kk = np.clip(np.searchsorted(sn, s[m], side="right") - 1, 0, n - 2)
        k[m] = kk
        t[m] = np.clip((s[m] - sn[kk]) / (sn[kk + 1] - sn[kk]), 0.0, 1.0)
    return k, t


def integrate(b, s, mu, v, j, shift=True):
    """Normalised disc-integrated profile of line j and the continuum-weighted <v>, sigma of v."""
    k, t = node_weights(b, s, j)
    m = np.clip(np.rint(-fd.C_KMS * np.log(1.0 - v / fd.C_KMS) / fd.DV).astype(int), -V, V) if shift else np.zeros(b.size, int)
    nv = 2 * V + 1
    row = np.concatenate([b * K + k, b * K + k + 1])              # (bin, node) rows
    col = np.concatenate([m, m]) + V
    wgt = np.concatenate([mu * (1 - t), mu * t])
    H = np.bincount(row * nv + col, weights=wgt, minlength=nb * K * nv).reshape(nb * K, nv)
    rows = np.where(H.sum(axis=1) > 0)[0]
    num, den = np.zeros(ny), np.zeros(ny)
    Ilf, Icf = IL[:, j].reshape(nb * K, ny), IC[:, j].reshape(nb * K, ny)
    for r0 in range(0, rows.size, a.chunk):
        rr = rows[r0:r0 + a.chunk]
        h = H[rr][:, ::-1]
        for A, acc in ((Ilf, num), (Icf, den)):
            Ip = np.pad(A[rr].astype(np.float64), ((0, 0), (V, V)), mode="edge")
            acc += fftconvolve(Ip, h, mode="valid", axes=1).sum(axis=0)
    # continuum-weighted velocity moments (weight mu I_c at the line centre)
    ic0 = (1 - t) * IC[b, j, k, ny // 2] + t * IC[b, j, k + 1, ny // 2]
    w = mu * ic0
    vm = np.sum(w * v) / w.sum()
    return num / den, vm, np.sqrt(np.sum(w * (v - vm) ** 2) / w.sum())


F = np.zeros((8, 3, ny)); F0 = np.zeros_like(F)
vmean, vsig = np.zeros((8, 3)), np.zeros((8, 3))
for kk, n in enumerate(los):
    mu, v = fd.mu_vlos(n, rhat, that, phat, smp["ur"], smp["uth"], smp["uph"])
    vis = mu > 0
    s = np.sqrt(1.0 - mu[vis] ** 2)
    for j in range(3):
        F[kk, j], vmean[kk, j], vsig[kk, j] = integrate(bins_all[vis], s, mu[vis], v[vis], j)
        F0[kk, j] = integrate(bins_all[vis], s, mu[vis], v[vis], j, shift=False)[0]
    log(f"los{kk + 1}: {int(vis.sum())} visible points, <v> {vmean[kk, 0]:+.2f}, sigma_v {vsig[kk, 0]:.2f} km/s")

# check (1): uniform star at the reference T_eff, no velocities -> the representative model's own flux profile
bref = int(np.clip(np.digitize(38230.0, edges) - 1, 0, nb - 1))
mu, v = fd.mu_vlos(los[0], rhat, that, phat, smp["ur"], smp["uth"], smp["uph"])
vis = mu > 0
checks = {}
rep_dir = os.path.join("/scratch/ppathak/fastwind_imu/runs", f"P{int(L['idx_rep'][bref]):06d}", f"P{int(L['idx_rep'][bref]):06d}")
for j in range(3):
    Fu = integrate(np.full(int(vis.sum()), bref), np.sqrt(1 - mu[vis] ** 2), mu[vis], v[vis], j, shift=False)[0]
    f0 = np.genfromtxt(os.path.join(rep_dir, f"OUT.{fd.LINES[j]}_VTV010"), usecols=[4], max_rows=161)
    w0 = fd.read_imu(os.path.join(rep_dir, f"OUT_IMU.{fd.LINES[j]}_VTV010"))[0]   # OUT prints lambda to 0.01 A only
    Fr = fd.interp_rows(fd.C_KMS * np.log(w0[None, :] / fd.LREF[j]), f0[None, :], fd.Y)[0]
    checks[f"uniform_{fd.LINES[j]}"] = float(np.abs(Fu - Fr).max())
    log(f"{fd.LINES[j]} check (1): uniform star at {L['teff_rep'][bref]:.0f} K, no velocities vs its FASTWIND flux: "
        f"max|dF| {checks[f'uniform_{fd.LINES[j]}']:.1e}, EW {fd.diagnostics(Fu, j)['ew']:.4f} vs {fd.diagnostics(Fr, j)['ew']:.4f} A")

keys = ["ew", "v1", "sigma", "fwhm", "depth"]
dF = np.array([[[fd.diagnostics(F[k, j], j)[q] for q in keys] for j in range(3)] for k in range(8)])
dF0 = np.array([[[fd.diagnostics(F0[k, j], j)[q] for q in keys] for j in range(3)] for k in range(8)])
ref = np.load(os.path.join(fd.RUN, "disc_los8.npz"))
for j in range(3):
    log(f"{fd.LINES[j]}: intensity method EW {dF[:, j, 0].mean():.4f} (flux method {ref['diag_F'][:, j, 0].mean():.4f}) A; "
        f"FWHM {dF[:, j, 3].mean():.0f} ({ref['diag_F'][:, j, 3].mean():.0f}), no vel. {dF0[:, j, 3].mean():.0f} "
        f"({ref['diag_F0'][:, j, 3].mean():.0f}) km/s; depth {dF[:, j, 4].mean():.3f} ({ref['diag_F'][:, j, 4].mean():.3f}); "
        f"max|F_imu - F_flux| {np.abs(F[:, j] - ref['F'][:, j]).max():.4f}; sigma_v {vsig[:, j].mean():.1f} ({ref['sigma_w'][:, j].mean():.1f}) km/s")
out = os.path.join(fd.RUN, "disc_los8_imu.npz")
np.savez(out[:-4] + ".tmp.npz", Y=fd.Y, LREF=fd.LREF, los=los, F=F, F0=F0, diag_keys=np.array(keys), diag_F=dF, diag_F0=dF0,
         vmean_w=vmean, sigma_w=vsig, check_keys=np.array(list(checks)), check_vals=np.array(list(checks.values())))
os.replace(out[:-4] + ".tmp.npz", out)
log(f"wrote {out}")
