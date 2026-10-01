Frozen legacy sources and tables of the M424 project (stellar-atmosphere-KU-Leuven, commit 67e042f, 2026-10-01),
used by the synspec regression tests as a fixed reference: the project scripts are being rewritten on top of
ppmpy.synspec, so the tests must not read the live project files.

fw_disc.py                          shared functions of the original pipeline (diagnostics, kernels, DiscFlux, ...);
                                    test_library.py runs its library() (on synthetic runs in tmp, chunk 400),
                                    lam_corrections() (synthetic files; DISC_DUMPS patched to tmp), lib_nodes() and
                                    node_pairs(); the M424 coverage test uses its unit_vectors() and los8()
fig_disc_zerocross_spectrum.py      original spectrum()/series() of the zero-crossing power spectra
figures/fw_disc_{vmac,profiles}{,_imu}_d3200_r4050_N1236544.csv   tables printed by the original fig_disc_vmac.py
                                    and fig_disc_profiles.py

PPMPY_SYNSPEC_M424_PROJECT can point the tests at another copy.

Added for test_sphere.py / test_parallel.py (M2, 2026-10-01; scripts with module-level argparse, so the tests
execute selected source lines, located by their text, instead of importing them):
fw_disc_dumps.py                    all-dump driver: MU, TN, PN (rhat @ LOS.T), v = ur*MU[k] + ..., rank split
fw_disc_los.py                      exact per-point sums: weighted mean/rms line-of-sight velocity (vmean, vsig)
                                    test_library.py also runs its library by-product (edges/bins, stream(),
                                    the per-line division, the assembly; synthetic runs): the producer of the
                                    M424 library_dT10.npz (blocks of 5000, 20 workers)
fw_sphere_extract.py                equal-area grid (theta, phi) and x, y, z of points.npz

Added for test_fwresults.py (M2, 2026-10-01):
fw_sphere_merge.py                  per-tag merge and --combine; run as a script by the tests (synthetic runs in
                                    tmp_path, and a copy of the M424 merged/task_missing_0000.npz) and compared
                                    byte for byte with fwresults.merge_task / combine
fig_fw_sphere_ew.py                 producer of the M424 ew.npz; the tests execute its EW block (located by text)

Added for test_library.py (M2, 2026-10-01):
fw_disc_holdout.py                  hold-out validation of the T_eff' library (M3 port pending); test_library.py runs
                                    its library part (bins/halves, stream(), the weight matrix with velocity groups
                                    and per-half library rows, the per-half bin means; synthetic runs, pool.map ->
                                    map) and compares FluxLibrary.build(select=half, prof_dtype=float64,
                                    fill_empty=False) and lib_nodes of it bit for bit

Used by test_disc.py (M2, 2026-10-01; no new copies):
fw_disc.py                          DiscFlux (incl. pairs), integrate_exact and integrate_lib, called directly on
                                    synthetic nodes/stars and compared bit for bit with disc.DiscFlux,
                                    disc.integrate_exact and disc.integrate_library_nearest
fw_disc_los.py                      the tests execute lines 56-145 (from "rhat, that, phat = fd.unit_vectors" to
                                    "del A, M, MC": MU/V/SH, bins, check subset, the per-line weight matrices,
                                    stream(), shift_add, check (a)), 147-155 (the library assembly, from "# library
                                    (same format" to "count=cnt, prof=lib_prof") and 159-177 (check (b),
                                    diagnostics, vmean/vsig, from "wl = fd.weights(MU[0], 1.0)" to
                                    "vsig[k, j] = np.sqrt") on a synthetic star (Pool -> builtin map, nothing
                                    written) and compare disc.integrate_exact_stream bit for bit

Added for test_dumps.py (M2, 2026-10-01):
fw_disc_collect.py                  time-series assembly of the per-dump files; run as a script (subprocess, --out a tmp
                                    directory) on synthetic per-dump files written by dumps.run_disc_dumps and on
                                    symlinks to the M424 production files flux/d3200-d3209.npz, and compared byte for
                                    byte with dumps.collect_timeseries
Used by test_dumps.py (no new copies):
fw_disc_dumps.py                    the tests execute line 55 (NAME, from "NAME = a.method"), lines 71-75 (projections,
                                    from "rhat, that, phat = fd.unit_vectors" to "del rhat, that, phat, pts"), line 76
                                    (KEYS) and process() (lines 79-110, from "def process(d):" to its last message line)
                                    on a toy library / sphere / samples (fd.SAMPLES and fd.DISC_DUMPS patched to tmp,
                                    the lamfix correction as the lamfix_dT10.npz cache there) and compare the per-dump
                                    files byte for byte with dumps.run_disc_dumps (flux, smooth, lamfix variants)
fw_disc.py                          lib_nodes, DiscFlux, lam_corrections (cache), diagnostics, los8, Y, VSHIFT via the
                                    process() lines above
