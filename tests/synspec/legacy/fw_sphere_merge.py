"""Stage 3 of the per-point FASTWIND run: merge the packed per-node results into one file.

Reads RUN_DIR/results/<tag>/part_*.tar.gz (one directory P<idx>/ per point with meta.txt and
OUT.<LINE>_VTV010) and writes RUN_DIR/<out> (default profiles.npz, written atomically):
    idx, teff, status (str), niter, t_pnlte, t_formal, T_tau23     per point
    teff_nudge             Teff used minus Teff of points.npz (0, or +1 K per retry of a failed model)
    lam, fcont, fnorm      float32 [npoint, 3, 161]  (air wavelength A, continuum flux, F/F_cont)
    lines                  the three line names
    r, theta, phi, x, y, z, ur_kms, relT   copied from RUN_DIR/points.npz for each idx (coordinates
                           of the point in the simulation frame, for the hemispherical integration)
Duplicates (a point packed twice after a restart) keep the successful entry. Points with
status != ok have NaN profiles. Profiles are at rest; u_r for the Doppler shift is in points.npz.

Failed models: a FASTWIND failure is deterministic (e.g. point 571348 of the dump-3200 run,
"error in ne -- nlteopt" at iteration 19, fails again with the same input but converges with
Teff + 1 K), so missing.txt lists a failed point with its Teff raised by TEFF_NUDGE (1 K) more than
its last failed attempt; points that were never run keep their Teff. 1 K is 3e-5 of Teff, far below
the fluctuations (~1e-2) and the convergence noise of the profiles (~0.5 % in EW).

Large runs: merge each task separately (in parallel, --tags task_0007 --out merged/task_0007.npz),
then concatenate with --combine (reads merged/task_*.npz, writes profiles.npz and missing.txt =
points of points.txt without a successful model, for resubmission with fw_sphere_multinode.sbatch
RUN_DIR RUN_DIR/missing.txt; copy it to a new name, e.g. missing2.txt, for a second round, because
a list's result directories skip every point they already hold). See fw_sphere_merge.sbatch.

Usage:  ./run.sh fw_sphere_merge.py RUN_DIR [--tags task_* | pilot] [--out profiles.npz]
        ./run.sh fw_sphere_merge.py RUN_DIR --combine
"""
import argparse
import glob
import io
import os
import tarfile

import numpy as np

LINES = ["HEI4026", "HEII4200", "HEI4922"]
NROW = 161
TEFF_NUDGE = 1.0      # K added to Teff for each retry of a failed model
TEFF_NUDGE_MAX = 10.0

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--tags", default="task_*")
ap.add_argument("--out", default="profiles.npz")
ap.add_argument("--combine", action="store_true")
a = ap.parse_args()


def save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


if a.combine:
    fs_ = sorted(glob.glob(os.path.join(a.run_dir, "merged", "task_*.npz")))
    parts_ = [np.load(f) for f in fs_]
    keys = [k for k in parts_[0].files if k != "lines"]
    out = {k: np.concatenate([p[k] for p in parts_]) for k in keys}
    out["lines"] = parts_[0]["lines"]
    # largest Teff nudge already tried for each point that has failed (before duplicates are dropped)
    tried = {}
    for i, dn in zip(out["idx"][out["status"] != "ok"].tolist(), out["teff_nudge"][out["status"] != "ok"].tolist()):
        tried[i] = max(tried.get(i, -np.inf), dn)
    o = np.argsort(out["idx"], kind="stable")
    out = {k: (v[o] if k != "lines" else v) for k, v in out.items()}
    # a point merged twice (e.g. from two tasks after a re-split) keeps its successful entry
    _, first = np.unique(out["idx"], return_index=True)
    ok = out["status"] == "ok"
    keep = np.zeros(out["idx"].size, bool)
    for i0, i1 in zip(first, np.append(first[1:], out["idx"].size)):
        j = i0 + (np.argmax(ok[i0:i1]) if ok[i0:i1].any() else 0)
        keep[j] = True
    out = {k: (v[keep] if k != "lines" else v) for k, v in out.items()}
    save(out, os.path.join(a.run_dir, "profiles.npz"))
    allidx = np.loadtxt(os.path.join(a.run_dir, "points.txt"))
    good = set(out["idx"][out["status"] == "ok"].tolist())
    miss = allidx[[int(i) not in good for i in allidx[:, 0]]]
    nfail = 0
    for row in miss:
        if int(row[0]) in tried:
            row[1] += tried[int(row[0])] + TEFF_NUDGE
            nfail += 1
    np.savetxt(os.path.join(a.run_dir, "missing.txt"), miss, fmt=["%d", "%.3f"])
    st, cnt = np.unique(out["status"], return_counts=True)
    print(f"combined {len(fs_)} task files: {out['idx'].size} points (" + ", ".join(f"{s} {c}" for s, c in zip(st, cnt))
          + f"); {miss.shape[0]} of {allidx.shape[0]} points without a successful model -> missing.txt"
          + f" ({nfail} failed models listed with Teff + {TEFF_NUDGE:g} K per failed attempt, "
          + f"{miss.shape[0] - nfail} never run)")
    nud = out["teff_nudge"] != 0
    if nud.any():
        print(f"{nud.sum()} points in profiles.npz were computed with a Teff nudge (max {out['teff_nudge'].max():g} K)")
    raise SystemExit(0)

rec = {}


def finish(pdir, files):
    """Parse one point's files (meta.txt, OUT.*) collected while streaming a part."""
    f = files["meta.txt"].decode().split()
    idx, teff, status, niter, tr23, tp, tf_ = int(f[0]), float(f[1]), f[2], int(f[3]), f[4], float(f[5]), float(f[6])
    prof = np.full((3, 3, NROW), np.nan, np.float32)
    if status == "ok":
        for j, ln in enumerate(LINES):
            arr = np.genfromtxt(io.BytesIO(files[f"OUT.{ln}_VTV010"]), usecols=[2, 3, 4], max_rows=NROW)
            prof[:, j, :] = arr.T
    old = rec.get(idx)
    if old is None or (old[2] != "ok" and status == "ok"):
        rec[idx] = (idx, teff, status, niter, tr23, tp, tf_, prof)


parts = sorted(glob.glob(os.path.join(a.run_dir, "results", a.tags, "part_*.tar.gz")))
WANT = ("meta.txt", "OUT.")
for part in parts:
    # stream the archive once, in order (random access in a .tar.gz re-decompresses from the start);
    # a point's files are contiguous, so collect them per directory and parse when it changes
    cur, files = None, {}
    with tarfile.open(part, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            pdir, fname = m.name.lstrip("./").rsplit("/", 1)
            if pdir != cur:
                if cur is not None and "meta.txt" in files:
                    finish(cur, files)
                cur, files = pdir, {}
            if fname.startswith(WANT):
                files[fname] = tf.extractfile(m).read()
    if cur is not None and "meta.txt" in files:
        finish(cur, files)

idx = np.array(sorted(rec), dtype=np.int32)
r = [rec[i] for i in idx]
prof = np.array([x[7] for x in r], np.float32) if r else np.zeros((0, 3, 3, NROW), np.float32)
out = dict(idx=idx, teff=np.array([x[1] for x in r]), status=np.array([x[2] for x in r]),
           niter=np.array([x[3] for x in r], np.int16), T_tau23=np.array([float(x[4]) for x in r]),
           t_pnlte=np.array([x[5] for x in r], np.float32), t_formal=np.array([x[6] for x in r], np.float32),
           lam=prof[:, 0], fcont=prof[:, 1], fnorm=prof[:, 2], lines=np.array(LINES))
pts = np.load(os.path.join(a.run_dir, "points.npz"))
assert np.all(pts["idx"][idx] == idx), "points.npz index mismatch"
for k in ("r", "theta", "phi", "x", "y", "z", "ur_kms", "relT"):
    out[k] = pts[k][idx]
dteff = out["teff"] - pts["teff"][idx]                   # points.txt has Teff to 1e-3 K
out["teff_nudge"] = np.where(np.abs(dteff) <= 1e-3, 0.0, np.round(dteff, 3)).astype(np.float32)
assert np.all((out["teff_nudge"] >= 0) & (out["teff_nudge"] <= TEFF_NUDGE_MAX)), \
    "Teff in results differs from points.npz by more than a retry nudge"
path = os.path.join(a.run_dir, a.out)
save(out, path)
st, cnt = np.unique(out["status"], return_counts=True)
print(f"{len(parts)} parts, {len(idx)} points: " + ", ".join(f"{s} {c}" for s, c in zip(st, cnt))
      + (f"; {int((out['teff_nudge'] != 0).sum())} with a Teff nudge" if (out["teff_nudge"] != 0).any() else ""))
if len(idx) and (out["status"] == "ok").any():
    ok = out["status"] == "ok"
    print(f"pnlte time: median {np.median(out['t_pnlte'][ok]):.0f} s, max {out['t_pnlte'][ok].max():.0f} s; "
          f"iterations median {int(np.median(out['niter'][ok]))}, max {out['niter'][ok].max()}")
print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
