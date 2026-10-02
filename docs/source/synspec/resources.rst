.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

Memory and time
===============

Measured numbers for the M424 run (1 236 544 points, 3 lines, 8 lines of sight, velocity grid of 5401 points, 1601
dumps), from the module docstrings and the project log (2026-09-25 to 2026-10-02). "Login node" is a Trillium login
node, "node" a Trillium compute node (192 cores, ~750 GB). Memory is peak RSS per process unless stated. There are
no laptop measurements: the laptop column gives what the per-process numbers imply for a machine with ~8 cores and
16 GB, and which option to choose there.

Sizes that set the cost: per-point arrays scale with the number of points N (the 20 000-point example of
:doc:`pipeline` is 60 times smaller than M424); library arrays scale with the number of T_eff' nodes (M424: 245
flux nodes, 303 intensity bins x 42 rays) and the grid; FASTWIND costs scale with the number of models.

Per stage
---------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - stage
     - M424, measured
     - laptop
   * - 1\. moms sampling (:func:`~ppmpy.synspec.moms.sample_moms_dumps`)
     - slab backend: 0.33-0.44 GB, 2-6 s per dump and process (15 s on a cold Lustre read); all 1601 dumps on one
       node with 64 workers in 125 s. momsdataset backend: 18 GB, ~50 s per dump. A dump is ~3.7 GB on disk
       (8 block files); the slab backend reads only the boxes around the sphere (4 of 10 slots)
     - slab backend, a few workers. The moms files (5.9 TB for 1601 M424 dumps) usually stay on the cluster:
       sample there and copy the samples (M424: 25 MB per dump; 20 000 points: 0.4 MB)
   * - 2\. FASTWIND, one model
     - ~100-150 s alone (reference model: pnlte ~150 s, 59 iterations); 0.17 GB; capped models (100 iterations)
       median 352 s against 179 s
     - same per model and core (a gfortran build runs as fast as ifort)
   * - 2\. FASTWIND, many models
     - one node, 192 at once: 127 s per model, ~5250 models per hour (scaling test); production with packing:
       ~3100-3400 models per node-hour, pnlte mean 212 s. All 1 236 544 models: 40 nodes x 9.4-9.7 h (73 066
       core-h). Storage per point: 7.7 kB packed with profiles only, 3.4 MB with the model files (``--keep model``;
       M424 results 3.9 TB)
     - library mode: ~550 models (10 K nodes) at 8 at once, roughly 3-4 h (estimate from the per-model time);
       per-point runs are cluster work
   * - 3\. merge (:func:`~ppmpy.synspec.fwresults.merge_tasks`, :func:`~ppmpy.synspec.fwresults.combine`)
     - one tag (~30 900 points) ~12 min on a login node, ~520 s CPU, 0.43 GB; 40 tags in parallel on one node
       12 min. Combine of all tags: 11 s (warm cache), 0.18 GB; ``profiles.npz`` 7.35 GB
     - seconds for a library-mode run
   * - 4\. flux library (:meth:`~ppmpy.synspec.library.FluxLibrary.build`)
     - 1.24 M models: 10 min serially, 2.7 min with 4 spawn workers, 130 s with 8 fork workers (1.2-1.7 GB per
       worker); 412 s with a cold page cache. ``rows`` <= 1000 keeps a process near 0.7 GB
     - fine (library mode: seconds)
   * - 4\. exact per-point sums (:func:`~ppmpy.synspec.disc.integrate_exact_stream`, the dump-3200 reference)
     - 69 s with 8 fork workers; parent 2.8 GB anonymous (7.1 GB with memory-mapped file pages), 1.5 GB per worker
     - only for a per-point run
   * - 4\. intensity library (:func:`~ppmpy.synspec.library.build_imu_library`)
     - selection 5 s, 0.12 GB; pformalsol reruns 0.5 s per representative; build 19.5-21.5 s, 2.0-2.1 GB; file
       1.91 GB
     - fine
   * - 5\. DiscFlux per dump (:func:`~ppmpy.synspec.dumps.run_disc_dumps`)
     - 0.7 s per dump alone; 1601 dumps in 21 s with 96 workers on a node (scaling test: 67 dumps per s at 96
       workers, node memory <= 91 GB); ~2.5-3 min with 8 workers on a login node
     - fine; < 1 GB per worker
   * - 5\. DiscImu per dump, default (``fft='precomputed'``, float64)
     - 7.7 GB of library FFTs, 9.3 GB peak; set-up 21-200 s on a login node (first touch, mostly system time);
       9-10 s per dump alone; 'fork' workers share the FFTs: 8 workers did 1601 dumps in 39 min (9.4 GB for all
       9 processes); on a node 48 workers took 873 s (the legacy class 526 s: open performance regression)
     - needs ~10 GB in one process; use 'fork' workers, not 'spawn' (each spawn worker builds its own copy)
   * - 5\. DiscImu, low-memory modes
     - ``fft='lazy'``: 2.2-2.8 GB per process (1.6 GB of it clean, shared pages of the memory-mapped library), set-up
       < 0.1 s, 11-13 s per dump; the same F as the default. ``dtype='float32'``: 3.85 GB held, max abs(dF) 6e-8
     - ``fft='lazy'``, or ``'precomputed'`` float32 for one line (1.4 GB) on <= 3 GB
   * - 6\. time series (:func:`~ppmpy.synspec.dumps.collect_timeseries`)
     - 8.8 s and 90 MB for 1601 dumps; 1.66 GB per run file
     - fine; open it with memory maps (:func:`~ppmpy.synspec.lpv.open_timeseries`)
   * - 7\. LPV and spectra
     - EW conservation plus LPV rms of one time series 60-70 s
     - fine
   * - full reproduction
     - samples and the four disc runs (flux, flux_lamfix, flux_sm335, imu) for all 1601 dumps: one node, 20.7 min
       (job 2483431)
     - \-

Validation
----------

M424 on a login node (2026-10-01/02): V1 and V3-V6 serially 65-71 s and 1.7 GB; V2 (hold-out) with 8 fork workers
72 s (parent 5.9 GB, 3.7 GB per worker including inherited pages); brute force 34-38 s per dump with 8 workers
(< 1 GB per process); :func:`~ppmpy.synspec.validate.teff_ranges` of all 1601 dumps 2 s with 8 workers; V1 of the
intensity method 15 s after a 67 s set-up (9.6 GB); intensity brute force of all points of a dump 143 s with 8
workers (~1 GB each). The self-test (no data): 10 s serially, 0.57 GB; 39 s and 1.35 GB on the M424 grid.

Machine notes
-------------

* Trillium login nodes: 3600 s of CPU time per process (``ulimit -t``; soft limit SIGXCPU, hard limit 5400 s),
  memory throttled above ~405 GB, no CPU quota. About 12 heavy processes at once thrash (99 % system time from page
  faults): run one heavy job at a time, with <= ~20 workers. In a single process the CPU time of all dumps adds
  up: use ``nproc >= 2`` with ``maxtasksperchild`` for long runs there (a lazy DiscImu, ~11 s of CPU per dump, would
  be killed after ~300 dumps).
* Large temporaries are slow where page faults are expensive (transparent huge pages 'always' on fragmented
  memory): keep ``rows`` <= ~1000 in library builds, or disable THP for the process.
* Compute nodes cannot write ``/home``; read and write on ``/scratch``. FASTWIND throughput is highest at 192 models
  per node (one per core, the most tested); DiscImu saturates at ~48 workers per node (memory bandwidth); DiscFlux
  scales to 96-192 workers.
* 'spawn' workers (macOS, Windows, or ``PPMPY_SYNSPEC_START_METHOD=spawn``) rebuild the integrator in every worker
  and replacement: prefer 'fork' on Linux, the lazy DiscImu with spawn, and a large ``maxtasksperchild``.
