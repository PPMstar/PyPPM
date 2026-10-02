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

Added for test_validate.py (M3, 2026-10-01):
fw_disc_dumps_validate.py           validation V1, V3-V6 of the all-dump method; run as a whole script (fw_disc.RUN,
                                    SAMPLES, DISC_DUMPS patched to tmp; --no-imu --dumps 3201,3203 --nsub 400 --ranges)
                                    on a synthetic run on the M424 grid; its validate.npz is compared bit for bit with
                                    validate.v1_exact / v3_tails / v4_nearest / v5_node_merging / v6_rounding
fw_disc_validate.py                 the first library-vs-exact check (nearest-bin library vs per-point sums with
                                    continuous shifts on fw_disc.directions(ndir)); run as a whole script (--ndir 2;
                                    fd.library(dT=...) / fd.load_run() given fd.RUN, whose defaults are bound at import)
                                    on the same synthetic run; its last direction's Fx, Fl are compared bit for bit with
                                    validate.exact_continuous and validate.NearestBin through v1_exact
Used by test_validate.py (no new copies):
fw_disc_holdout.py                  run as a whole script (Pool -> builtin map; --nproc 2 --block 500) on the same
                                    synthetic run; its holdout.npz is compared bit for bit with validate.v2_holdout
                                    (serial, fork, spawn; library file or built in the pass)
fw_disc.py                          DiscImu (V1 of the intensity method on M424, slow), and the module both scripts import

Added for test_imulib.py (M4, 2026-10-02):
fw_imu_library.py                   intensity library (select representatives, build imu_library_dT10.npz, flux check);
                                    the tests execute its lines (located by text) on synthetic raw/runs directories in
                                    tmp: candidates + selection (from "edges, tmean, count = flib" to "assert not
                                    missing"), representatives.txt (from "sel = os.path.join(a.out" to its log line),
                                    the build (from "mdir_of = {b: os.path.join(RUNS" to "os.replace(out[:-4]") and the
                                    flux check (line 105, "mf = np.linspace", and from "sf = np.sqrt(1.0 - mf ** 2)" to
                                    "fn = np.trapz("), and compare library.find_candidates, select_representatives,
                                    Representatives.write, build_imu_library (+ ImuLibrary.save(meta=False): same file
                                    bytes) and flux_from_rays bit for bit
Used by test_imulib.py (no new copies):
fw_disc.py                          read_imu, r_outer, flux_from_p, imu_from_rays (bitwise against library.r_outer,
                                    flux_from_p, imu_from_rays), and Y, LREF, LINES, interp_rows for the build lines

Added for test_discimu.py (M4, 2026-10-02):
fw_disc_imu.py                      the first intensity method (dump 3200, nearest 10 K bin of the intensity library,
                                    disc_los8_imu.npz, the uniform-star check); run as a whole script (fw_disc.RUN,
                                    SAMPLES patched to tmp; --lib a toy intensity library on the M424 grid; the
                                    representatives' directory "/scratch/ppathak/fastwind_imu/runs" replaced by a tmp
                                    directory holding toy OUT / OUT_IMU files) on a toy star; its disc_los8_imu.npz is
                                    compared bit for bit with disc.integrate_imu_nearest (+ save_disc_los)
Used by test_discimu.py (no new copies):
fw_disc.py                          DiscImu (constructed from a toy library file on the M424 grid; set-up arrays and
                                    calls compared bit for bit with disc.DiscImu), read_imu / interp_rows / mu_vlos /
                                    diagnostics through fw_disc_imu.py

Added for test_fwresults.py (M4, 2026-10-02):
fw_imu_extract.sh                   extraction of selected model directories from the per-point archives (GNU tar -x
                                    of './P<idx>' per line of a parts file, xargs -P); run as a script (bash, in the
                                    container) on synthetic parts with ledgers, given the parts file that
                                    fwresults.locate_points yields, and its output tree (names, bytes, modes, mtimes)
                                    compared with fwresults.extract_points
Used by test_dumps.py (M4, 2026-10-02; no new copies):
fw_disc_dumps.py, fw_disc.py        process() with a.method = 'imu' and INT = the frozen fw_disc.DiscImu of a toy
                                    intensity library on the M424 grid; its per-dump files are compared byte for byte
                                    with dumps.run_disc_dumps(dumps.imu_integrator, ...) (serial, fork, spawn, batched)
Used by test_validate.py (M4, 2026-10-02; no new copies):
fw_disc.py                          DiscImu on a toy intensity library: validate.v1_exact gives the same arrays with it
                                    as with disc.DiscImu

Added for test_moms.py (M5, 2026-10-02):
sphere_sample.py                    all-dump moms sampling (relT, teff, ur, uth, uph float32 + T9_mean, t_s per dump;
                                    producer of samples_r4050_N1236544/dNNNN.npz); from commit 67e042f
Used by test_moms.py (no new copies):
fw_sphere_extract.py                one-dump moms sampling (points.npz, points.txt, meta.json); the M424 tests compare
                                    moms.sample_moms_sphere + write_points_table with its stored products, and the toy
                                    tests execute its sampling lines (from "T9 = m.get_spherical_interpolation" to
                                    "ur_kms = m.get_spherical_interpolation") on a MomsDataSet of a synthetic dump

Added for the FASTWIND runner tests (M6, 2026-10-02), from the project's committed versions before their migration
(project commit c14658a): fw_sphere_point.sh, fw_sphere_task.sh, fastwind_run.sh, fw_imu_run.sh (the shell runner the
ppmpy.synspec.fastwind runner reproduces) and fastwind/INDAT_*.DAT (the project's INDAT templates, for the awk test).
