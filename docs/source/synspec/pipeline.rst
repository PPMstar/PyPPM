.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

The pipeline
============

Overview
--------

::

    moms dumps (3D cubes)
      |  1. sampling on the sphere                       moms.sample_moms_sphere / sample_moms_dumps
      v
    points.npz, points.txt (library dump)  +  samples/dNNNN.npz (every dump)
      |  2. FASTWIND, one model per sphere point          python3 -m ppmpy.synspec.fastwind run
      |     or one model per T_eff' node (library mode)   libmode.plan_teff_nodes + the same runner
      v
    results/<tag>/part_*.tar.gz + part_*.idx
      |  3. merge                                        fwresults.merge_tasks, fwresults.combine
      v
    merged/<tag>.npz  ->  profiles.npz, missing.txt (failed models, rerun with T_eff + 1 K)
      |  4. T_eff' libraries                             library.FluxLibrary, library.build_imu_library
      v
    library_dT10.npz (flux),  imu_library_dT10.npz (intensities I(y, mu))
      |  5. disc integration per dump                    dumps.run_disc_dumps with DiscFlux or DiscImu
      v
    <name>/dNNNN.npz
      |  6. time series                                  dumps.collect_timeseries
      v
    <name>_timeseries.npz
      |  7. LPV analysis                                 lpv, spectrum, diagnostics, plotting
      v
    residual spectra, zero-crossing tracks, coherence times, temporal power spectra

The key premise: the local models of a run differ only in T_eff' (every other INDAT input is fixed;
:func:`ppmpy.synspec.fwresults.check_indat_premise` checks it on archived models). So the models computed for
one dump form a T_eff' library for every dump. Each dump contributes only its own T_eff' field (interpolated
between library nodes) and its own velocities (Doppler shifts); later dumps need no new FASTWIND runs.

Two disc-integration methods use the same library models:

* the **flux method** (:class:`ppmpy.synspec.disc.DiscFlux`): each point contributes its continuum-normalised
  flux profile with weight mu F_c (I(mu) = const, Lambert's cosine law);
* the **intensity method** (:class:`ppmpy.synspec.disc.DiscImu`, the SPAMMS approach): each point contributes its
  model's emergent intensities in its own direction mu, so limb darkening and the centre-to-limb change of the line
  enter. It needs FASTWIND's pformalsol patched to write OUT_IMU files (:doc:`fastwind`). M424 uses it for figures
  and the comparison with observations; the flux method is the validated reference.

Stage 1: sampling the moms dumps
--------------------------------

:class:`ppmpy.synspec.moms.MomsSource` describes a directory of decompressed moms dumps: file pattern, slot map,
grid spacing (``deex`` of an rprof header) and dump times. The slot map has to be known for the run (M424:
``['xc', 'ux', 'uy', 'uz', 'slot4_unknown', 'dUr', '|w|', 'T9', 'rho', 'dT9']``).

* :func:`ppmpy.synspec.moms.sample_moms_sphere`: one dump on the equal-area sphere of radius ``radius`` with
  ``npoints`` points (:func:`ppmpy.synspec.sphere.fibonacci_sphere`): relT = (T - <T>) / <T>,
  T_eff' = teff0 (1 + relT), and u_r, u_theta, u_phi in km/s (velocity slots times ``velocity_scale`` = 1e3).
* :func:`ppmpy.synspec.moms.write_points_table`: ``points.npz``, ``points.txt`` (``idx teff`` lines, the input
  of the FASTWIND runner) and ``meta.json`` of the library dump.
* :func:`ppmpy.synspec.moms.sample_moms_dumps`: ``samples/dNNNN.npz`` for every dump (relT, teff, ur, uth, uph as
  float32, T9_mean, t_s); restartable, split over ranks, worker processes.

The default ``backend='slab'`` reads only the boxes of the block files around the sphere (0.3-0.4 GB and a few
seconds per M424 dump); ``backend='momsdataset'`` is the legacy path through a ppmpy ``MomsDataSet`` (18 GB per
dump). Both give the same bits. M424: radius 4050 Mm, 1 236 544 points, teff0 = 38 230 K, dump 3200 is the
library dump.

Stage 2: FASTWIND
-----------------

FASTWIND is external (:doc:`fastwind`). The runner, ``python3 -m ppmpy.synspec.fastwind``, uses only the Python
standard library and runs under the host python, where the FASTWIND binaries run. It reads an ``idx teff`` table,
writes INDAT.DAT from a template with MODNAM and TEFF set per model, runs pnlte and pformalsol, and packs the results
into ``results/<tag>/part_*.tar.gz`` with a ledger ``part_*.idx`` per part.

* **Per point** (M424 production): the table is ``points.txt`` of the library dump, one model per sphere point.
* **Per T_eff' node** (library mode, :mod:`ppmpy.synspec.libmode`, M7, in development; see its module docstring
  for the current interface): :func:`ppmpy.synspec.libmode.plan_teff_nodes` places nodes every dT (e.g. 10 K) over
  the T_eff' range of all dumps (:func:`ppmpy.synspec.validate.teff_ranges`) and writes the table; a few hundred
  models replace a million. Its V8 test on M424 (dT = 10 K, one model per node, 551 models) puts the time-variable
  profile deviation at <= 1.4 % of the LPV rms; the library must then be built with ``nmin = 1``.

Python entry points: :func:`ppmpy.synspec.fastwind.run_model` (one model),
``ppmpy.synspec.fastwind.batch.run_models`` (one task's share of a table),
:class:`ppmpy.synspec.fastwind.FastwindInstall`.

Stage 3: merging the results
----------------------------

* :func:`ppmpy.synspec.fwresults.merge_tasks` (per tag, in parallel; one tag:
  :func:`~ppmpy.synspec.fwresults.merge_task`): ``results/<tag>/`` -> ``merged/<tag>.npz`` (profiles as float32
  (npoint, nline, nrow): lam, fcont, fnorm; idx, teff, status, niter, T_tau23; the coordinates copied from
  ``points.npz``). ``lines``, ``suffix`` and ``nrow``
  default to M424 (``HEI4026 HEII4200 HEI4922``, ``VTV010``, 161 rows).
* :func:`ppmpy.synspec.fwresults.combine`: all merged files -> ``profiles.npz`` and ``missing.txt``, streamed
  (0.2 GB for M424). ``missing.txt`` lists points without a successful model; failed ones get T_eff + 1 K per failed
  attempt, because FASTWIND failures are deterministic. Run the runner again on it (list mode, a new list name per
  round) and merge again (``incremental=True`` re-merges only tags with new parts).
* :class:`ppmpy.synspec.fwresults.ProfileStore`: zero-copy access to ``profiles.npz``;
  :func:`~ppmpy.synspec.fwresults.ew_per_point`, :func:`~ppmpy.synspec.fwresults.status_summary` (with the iteration
  cap, :func:`~ppmpy.synspec.fwresults.niter_cap_from_indat`).
* :func:`ppmpy.synspec.fwresults.locate_points` and :func:`~ppmpy.synspec.fwresults.extract_points`: find and unpack
  the archived model directories of chosen points (e.g. the representatives of the intensity library).

Stage 4: T_eff' libraries
-------------------------

* Flux: :meth:`ppmpy.synspec.library.FluxLibrary.from_profiles_npz` (or
  :meth:`~ppmpy.synspec.library.FluxLibrary.build`) interpolates every model onto the velocity grid and averages
  them in T_eff' bins of width dT (M424: 10 K, 351 bins, 303 filled).
  :meth:`~ppmpy.synspec.library.FluxLibrary.save` writes ``library_dT10.npz``.
  :func:`ppmpy.synspec.library.lib_nodes` merges consecutive bins until a node holds ``nmin`` models (M424: nmin 20,
  245 nodes, 35 829-38 891 K; options ``smooth`` and ``corr`` give the library variants ``flux_sm335`` and
  ``flux_lamfix``, the latter with :func:`~ppmpy.synspec.library.wavelength_rounding_correction`).
* Intensities: one representative model per bin.
  :func:`ppmpy.synspec.library.find_candidates` (extracted model directories with ``meta.txt`` and ``CONT_FORMAL``,
  i.e. runs with ``--keep model``) -> :func:`~ppmpy.synspec.library.select_representatives` ->
  ``representatives.txt``; pformalsol of the patched build is rerun on them (``rerun-formal``, 0.5 s per model, into
  ``runs/P<idx>/P<idx>/``); :func:`~ppmpy.synspec.library.build_imu_library` keeps the rays inside R_max and
  interpolates them onto the grid -> :class:`~ppmpy.synspec.library.ImuLibrary`, ``imu_library_dT10.npz``
  (M424: 303 representatives, 42 rays, 1.9 GB). In library mode,
  :func:`ppmpy.synspec.libmode.library_from_models` builds both libraries from the node models.

Stage 5: disc integration of every dump
---------------------------------------

:func:`ppmpy.synspec.dumps.run_disc_dumps` integrates each dump with an integrator built by a factory:
:func:`ppmpy.synspec.dumps.flux_integrator` (library file -> nodes -> :class:`~ppmpy.synspec.disc.DiscFlux`) or
:func:`ppmpy.synspec.dumps.imu_integrator` (intensity library -> :class:`~ppmpy.synspec.disc.DiscImu`). For each
dump it loads the samples, projects the points onto the lines of sight (:func:`ppmpy.synspec.sphere.project_los`),
forms v = u . n (:func:`ppmpy.synspec.sphere.los_velocity`), interpolates T_eff' between nodes (clamped at the ends),
shifts by whole grid steps and sums, by FFT convolution of each node's profile with its histogram of shifts. It
writes ``<outdir>/<name>/dNNNN.npz``: F and F0 (with and without Doppler shifts; float32 (nlos, nline, ny)),
diagnostics (ew, v1, sigma, fwhm, depth), weighted velocity moments, points beyond the node range, clipped shifts,
T_eff' statistics.

* Restartable: existing files are skipped; ``_run.json`` records the configuration and a restart with another
  configuration into the same directory is refused.
* ``nproc`` worker processes ('fork' or 'spawn'), ``rank`` / ``nranks`` split the dumps over nodes, a watchdog
  (``timeout``) raises :class:`ppmpy.synspec.parallel.PoolStalled` when a worker is killed.
* DiscImu modes: ``fft='precomputed'`` (default, the production bits, 7.7 GB of library FFTs), ``fft='lazy'`` and
  ``dtype='float32'`` for small memory; each mode writes to its own directory (``imu_lazy``, ``imu_f32``, ...).

References for validation: :func:`ppmpy.synspec.disc.integrate_exact_stream` (every point's own model, the dump-3200
reference ``disc_los8.npz``, which also yields the flux library), :func:`~ppmpy.synspec.disc.integrate_imu_nearest`,
:func:`~ppmpy.synspec.disc.integrate_exact` (continuous shifts). One dump in memory:
:func:`ppmpy.synspec.dumps.disc_dump`.

Stage 6: time series
--------------------

:func:`ppmpy.synspec.dumps.collect_timeseries` streams the per-dump files into ``<name>_timeseries.npz``: dumps,
t_s, the grid Y, LREF, los, F, F0 (ndump, nlos, nline, ny), diag_F, diag_F0 and the per-dump statistics
(M424: 1601 dumps, 1.66 GB per run, 9 s).

Stage 7: line-profile variability
---------------------------------

* :func:`ppmpy.synspec.lpv.open_timeseries` (memory maps of F, F0), :func:`~ppmpy.synspec.lpv.residual_spectra`
  (R = F - <F>_t), :func:`~ppmpy.synspec.lpv.residual_summary`, :func:`~ppmpy.synspec.lpv.zero_crossing_track` (the
  central zero crossing of the residual), :func:`~ppmpy.synspec.lpv.fill_gaps`,
  :func:`~ppmpy.synspec.lpv.lag_correlation` and :func:`~ppmpy.synspec.lpv.coherence_time`,
  :func:`~ppmpy.synspec.lpv.systematics` (one run against another).
* :func:`ppmpy.synspec.spectrum.temporal_power_spectrum` / :func:`~ppmpy.synspec.spectrum.temporal_power_spectra`
  (padded FFT as the PPMstar notebooks, or a direct DFT on any frequency grid; :doc:`conventions`).
* :mod:`ppmpy.synspec.diagnostics`: line moments, broadening kernels, macroturbulence and v sin i fits.
* :mod:`ppmpy.synspec.plotting`: dynamic spectra, profile bundles, power spectra with a d\ :sup:`-1` axis (draws on
  axes given by the caller).

Example: a new run on a small machine
-------------------------------------

Library mode keeps FASTWIND affordable on a workstation or laptop with ~8 cores and 16 GB: the dumps are sampled
on a sphere of 20 000 points and one FASTWIND model is computed per 10 K node. The paths, the slot map and teff0
below are placeholders for your run; the moms dumps must be decompressed (``moms/myavsbq``) and the FASTWIND
binaries must run on this machine (:doc:`fastwind`). The library-mode calls follow the current
:mod:`ppmpy.synspec.libmode`, which is still being written.

Step 1, sample all dumps and plan the nodes (numpy python)::

    import os
    from ppmpy.synspec import libmode, moms, validate

    RUN, WORK = "/data/M999", "/data/M999_synspec"
    NPTS, RADIUS, TEFF0 = 20000, 4050.0, 38230.0          # points, sphere radius [Mm], T_eff of the 1D model [K]
    SLOTS = ["xc", "ux", "uy", "uz", "s4", "dUr", "|w|", "T9", "rho", "dT9"]

    if __name__ == "__main__":                            # needed for 'spawn' worker processes
        src = moms.MomsSource(RUN + "/moms/myavsbq", SLOTS, rprof=RUN + "/prfs")
        dumps = src.dumps()
        moms.sample_moms_dumps(src, dumps, RADIUS, NPTS, TEFF0, WORK + "/samples", nproc=4)
        ranges = validate.teff_ranges(WORK + "/samples", dumps, nproc=4)
        plan = libmode.plan_teff_nodes(ranges, dT=10.0)
        os.makedirs(WORK + "/nodes", exist_ok=True)
        plan.write(WORK + "/nodes/points.txt")            # 'idx teff' table for the runner
        plan.write_points_npz(WORK + "/nodes/points.npz") # idx, teff for the merge

Step 2, FASTWIND (host python; ``--formal-build`` with the OUT_IMU build also gives the intensity library)::

    export PYTHONPATH=/path/to/PyPPM
    export PPMPY_FASTWIND_ROOT=/opt/FW_10.6.4.1 PPMPY_FASTWIND_BUILD=v10.6_HHe_gfortran
    python3 -m ppmpy.synspec.fastwind check
    python3 -m ppmpy.synspec.fastwind run /data/M999_synspec/nodes \
        --template INDAT_M999.DAT --formal FORMAL_INPUT --nworkers 8
    python3 -m ppmpy.synspec.fastwind status /data/M999_synspec/nodes/results \
        --table /data/M999_synspec/nodes/points.txt

Step 3, merge, libraries, disc integration, time series, a first spectrum (numpy python; ``WORK``, ``NPTS`` and
``dumps`` as in step 1)::

    from ppmpy.synspec import dumps as dm, fwresults, libmode, lpv, sphere, spectrum
    from ppmpy.synspec.spectral import LineSet, VelocityGrid

    LINES = LineSet(["HEI4026", "HEII4200", "HEI4922"], [4026.22, 4199.90, 4921.93])
    GRID = VelocityGrid()                                 # dv 1, vmax 2700, vshift 400 km/s

    if __name__ == "__main__":
        N = WORK + "/nodes"
        fwresults.merge_tasks(N + "/results", N + "/merged", N + "/points.npz", nproc=2, copy_keys=())
        fwresults.combine(N + "/merged/task_*.npz", N + "/points.txt", N)    # profiles.npz, missing.txt
        lib = libmode.library_from_models(N + "/profiles.npz", GRID, LINES, plan=N + "/points.npz")
        # with OUT_IMU files: add results_dir=N + "/results", extract_dir=N + "/imu_models" for the I(mu) library
        files = lib.save(WORK + "/lib")                   # 'flux' (+ 'imu' when built), report

        theta, phi = sphere.fibonacci_sphere(NPTS)        # the points the samples were taken at
        dm.run_disc_dumps(dumps, WORK + "/samples", WORK + "/disc", "flux", dm.flux_integrator,
                          (files["flux"],), theta, phi, "thompson2024", nproc=4, lref=LINES,
                          factory_kwargs=dict(nmin=1, grid=GRID))
        dm.collect_timeseries(WORK + "/disc", "flux")

        ts = lpv.open_timeseries(WORK + "/disc/flux_timeseries.npz")
        R, Fref = lpv.residual_spectra(ts["F"][:, 0, 0])  # line of sight 1, first line
        ew = ts["diag_F"][:, 0, 0, 0]                     # diag_keys: ew, v1, sigma, fwhm, depth
        dt, spread = spectrum.sample_spacing(ts["t_s"])
        f_muhz, power = spectrum.temporal_power_spectrum(ew, dt, detrend="ratio")

With an intensity library (``files["imu"]``), use ``dm.imu_integrator`` with ``factory_kwargs=dict(grid=GRID,
fft="lazy")`` (2-3 GB per process) and the run name ``"imu_lazy"``. Validate the run before using it
(:doc:`validation`): at least :func:`ppmpy.synspec.validate.run_validation` with V1 and the brute force on a few
dumps, and ``python -m ppmpy.synspec.testing`` once per installation.

Batch commands on a cluster (Trillium)
--------------------------------------

The M424 production ran on Trillium (192 cores and ~750 GB per node). One multi-node job with one ``srun`` task per
node is preferred over job arrays. Compute nodes cannot write ``/home``: write job outputs to ``/scratch``.

Moms sampling: one node, a script like step 1 of the example run in the numpy container with ``nproc=64``
(``apptainer exec --bind /home,/scratch,/project $SIF python sample.py``); 64 workers sampled all 1601 M424 dumps
at 1 236 544 points in 125 s. The library dump also needs ``points.npz`` / ``points.txt``::

    s = moms.sample_moms_sphere(src, 3200, 4050.0, 1236544, 38230.0)
    moms.write_points_table(RUN_DIR, s, 38230.0)

FASTWIND per point (``python3 -m ppmpy.synspec.fastwind run``, host python, 192 models per node; M424: 40 nodes for
9.4-9.7 h)::

    #!/bin/bash
    #SBATCH --nodes=40 --ntasks-per-node=1 --cpus-per-task=192 --mem=0 --time=12:00:00
    #SBATCH --signal=USR1@900
    export PYTHONPATH=/home/$USER/PyPPM
    K=$SLURM_NNODES
    srun --nodes=$K --ntasks=$K --ntasks-per-node=1 --cpus-per-task=192 --cpu-bind=none --kill-on-bad-exit=0 \
         /usr/bin/python3 -m ppmpy.synspec.fastwind run $RUN_DIR -K $K --nworkers 192 \
         --root /scratch/$USER/FW_10.6.4.1 --build v10.6_HHe --keep model \
         --template INDAT_M424test.DAT --formal FORMAL_INPUT_He3

Task k (``$SLURM_PROCID``) takes lines k, k + K, ... of ``points.txt``. SIGUSR1 15 min before the time limit packs
what is finished and exits with status 3: submit the same script again to resume. Then merge (one node, numpy
python), and run the failed points from ``missing.txt`` with a new list name (``run $RUN_DIR missing.txt -K ...``)::

    python -c "from ppmpy.synspec import fwresults as fr; R = '$RUN_DIR'
    fr.merge_tasks(R + '/results', R + '/merged', R + '/points.npz', nproc=40)
    fr.combine(R + '/merged/task_*.npz', R + '/points.txt', R)"

Disc integration of all dumps: a script calling :func:`ppmpy.synspec.dumps.run_disc_dumps` as in the example,
run inside the container on one node (``rank`` / ``nranks`` split the dumps over nodes), then
:func:`ppmpy.synspec.dumps.collect_timeseries` once all ranks are done. M424: flux 1601 dumps in 21 s with 96
workers; imu 526 s with 48 workers in the production (legacy integrator) and 873 s with ppmpy's DiscImu in the
reproduction (a performance regression still to be fixed).
In the project, ``fw_disc_dumps.py --method flux|imu`` and ``fw_disc_dumps.sbatch`` are these thin callers.
