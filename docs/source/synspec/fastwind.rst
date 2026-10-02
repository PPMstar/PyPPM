.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

FASTWIND
========

FASTWIND (J. Puls and collaborators; 1D non-LTE model atmospheres and line formation) is **not part of
ppmpy**. The user installs and builds it and tells ppmpy where it is. ``ppmpy.synspec.fastwind`` drives it: it
writes the inputs, runs the two codes per model, decides whether a model succeeded and packs the results in the
layout that :mod:`ppmpy.synspec.fwresults` reads. M424 used FASTWIND v10.6.4.1 (from F. Backs, KU Leuven; upstream
https://github.com/uh101aw/Fastwind_base) with the H+He model atom ``A10HHe``.

What ppmpy needs
----------------

An install root and a build (:class:`ppmpy.synspec.fastwind.FastwindInstall`)::

    <root>/inicalc/{DATA,OP_DATA_NEW,ATOMDAT_NEW,RaymondSmith}     data read by the codes
    <root>/inicalc/HOPFPARA_ALL_{HHe,met}                          Hopf-parameter tables
    <root>/<build>/pnlte_<tag>.eo, pformalsol_<tag>.eo             the executables
    <root>/<build>/ATOM_FILE, <tag>.dat                            model atom (tag = ATOM_FILE's first line
                                                                   without '.dat', e.g. A10HHe)

* Build: ``make`` in the build directory (ifort or gfortran; ~25 s for A10HHe). The ifort and gfortran builds of
  M424 (``v10.6_HHe``, ``v10.6_HHe_gfortran``) gave the same speed and results.
* Optional, for the intensity method: a second build whose pformalsol is patched to also write
  ``OUT_IMU.<line>_<suffix>``, the emergent continuum and line intensities of every ray of the formal solution
  (M424: ``v10.6_HHe_imu``). The patch is not in ppmpy; it is kept in the M424 project
  (``stellar-atmosphere-KU-Leuven``, ``project/analysis/fastwind/formalsol_imu.patch``). Give it as
  ``formal_build``; :meth:`~ppmpy.synspec.fastwind.FastwindInstall.has_imu_patch` detects it. The ``OUT`` files are
  unchanged by the patch.
* Inputs per run: an INDAT.DAT template (all physics except T_eff; MODNAM and TEFF are set per model by
  :class:`ppmpy.synspec.fastwind.Indat`, byte-identical to the legacy awk edit) and a FORMAL_INPUT line list
  (:class:`~ppmpy.synspec.fastwind.FormalInput`). The M424 templates are ``INDAT_M424test.DAT`` and
  ``FORMAL_INPUT_He3`` of the project.
* INDAT pitfalls found for M424: OPTTLUCY (second switch of line 8) must be T for arbitrary stellar parameters (with F
  the code needs an exact (T_eff, log g, Y_He) entry in the Hopf table); pnlte reads the Hopf tables from the
  *parent* of its run directory (the staged root provides them). FASTWIND needs Mdot > 0 (M424: 1e-10 Msun/yr).

The codes open their data relative to the working directory and write scratch files there, so every model runs in
its own directory one level below a *staged root* (links or copies of inicalc, the Hopf tables and the executables;
:meth:`~ppmpy.synspec.fastwind.FastwindInstall.stage`). The batch runner stages into ``/dev/shm`` when it has room.

Host python, not the container
------------------------------

The M424 binaries need the ELF interpreter of the cluster software stack (``/cvmfs/...``), which the Python 3.9
container does not have: FASTWIND runs on the host. The whole subpackage therefore uses only the Python standard
library and runs under the host ``/usr/bin/python3`` (3.9, no numpy) with ``PYTHONPATH`` pointing at the PyPPM
checkout. :meth:`~ppmpy.synspec.fastwind.FastwindInstall.check` reports a missing interpreter. The numpy side
(merging, libraries, disc integration) runs in the container as usual. The tests of real FASTWIND runs (marker
``fastwind``) work in the container only with ``/cvmfs`` bound.

The install is given with ``--root`` / ``--build`` / ``--formal-build`` / ``--launcher`` or the environment variables
``PPMPY_FASTWIND_ROOT``, ``PPMPY_FASTWIND_BUILD``, ``PPMPY_FASTWIND_FORMAL_BUILD``, ``PPMPY_FASTWIND_LAUNCHER``.

The command line
----------------

::

    python3 -m ppmpy.synspec.fastwind check  [--fingerprint]
    python3 -m ppmpy.synspec.fastwind stage  DEST [--mode link|copy]
    python3 -m ppmpy.synspec.fastwind one    IDX TEFF --template T --formal F --out RES [--name NAME]
    python3 -m ppmpy.synspec.fastwind run    RUN_DIR [LIST] --template T --formal F [-K N] [-k i] [--nworkers N]
    python3 -m ppmpy.synspec.fastwind status RESULTS_DIR [--table points.txt] [--tags PATTERN] [--json]
    python3 -m ppmpy.synspec.fastwind recover DIR [--all]
    python3 -m ppmpy.synspec.fastwind rerun-formal [MODEL_DIR ...] [--list FILE] --formal F --run-root R

* ``check``: data, executables, ELF interpreter; ``--fingerprint`` adds the sha256 of the executables.
* ``one``: one model, e.g. the reference model of a star
  (``one 0 38230 --name M424_T38230 --template T --formal F --out RES``).
* ``run``: one task's share of ``RUN_DIR/points.txt`` (or of LIST, e.g. ``missing.txt``) into
  ``RUN_DIR/results/task_%04d`` (``task_<list>_%04d`` for a list). Line i of the table goes to task i % K. The task
  index comes from ``-k``, ``$FW_TASK``, ``$SLURM_ARRAY_TASK_ID`` or ``$SLURM_PROCID``; K from ``-K`` or
  ``$FW_NTASKS`` and is never guessed from Slurm. ``--nworkers`` models run at once (default ``$NW``; in a Slurm job
  ``$SLURM_CPUS_PER_TASK`` with a warning). Model options: ``--keep profiles|model`` (``model`` keeps the files a
  pformalsol rerun needs: 3.4 MB per point packed instead of 7.7 kB), ``--extras full|digest|none``,
  ``--pnlte-timeout`` (default 3600 s), ``--formal-timeout`` (600 s), ``--vturb`` ('10 0.1'), ``--iescat`` (0),
  ``--retry N --retry-step 1`` (immediate retries of failed models at T_eff + 1 K per attempt; default 0).
* ``status``: progress of a run from its ledgers (done, failed, orphans, temporaries).
* ``recover``: rebuild a missing ledger from its archive, remove stale temporaries (also done by every ``run`` start).
* ``rerun-formal``: pformalsol only, from saved model files (0.5 s per model), e.g. with the OUT_IMU build for the
  representatives of the intensity library; outputs in ``<run-root>/<name>/<name>/``.

Python: :func:`ppmpy.synspec.fastwind.run_model`, :func:`~ppmpy.synspec.fastwind.rerun_formal`,
``ppmpy.synspec.fastwind.batch.run_models`` and ``batch.status``.

Outputs
-------

Per model a directory ``P<idx>/`` (byte-compatible with the legacy shell runner): ``INDAT.DAT``, ``OUT.*`` (one file
per line; 161 rows for the M424 builds), ``meta.txt`` (``idx teff status niter T_tau23 t_pnlte t_formal``), the model
files with ``--keep model``, ``pnlte_tail.log`` for a failed model, and the extras: ``CONVERG``, ``MAXTCORR.dat`` and
``convergence.json`` (``full``, ~2 kB per point after gzip) or the one-line ``convergence.txt`` (``digest``). A packer
child process moves finished models into ``results/<tag>/part_*.tar.gz``; each archive's ledger ``part_*.idx`` (its
points' meta.txt lines) is written after the archive is complete, so a ledger line means "this point is safely
packed".

Restart and stop
----------------

* **Restart** = run the same command again. The union of the ledgers ``<tag>/*.idx`` is the done list (any status), so
  finished points are skipped; interrupted models recorded nothing and run again.
* **Stop**: SIGUSR1 (Slurm ``--signal=USR1@900``: 15 min before the time limit), SIGTERM, SIGINT, SIGHUP (a dropped
  ssh session) and SIGXCPU (the soft CPU-time limit, 3600 s on the Trillium login nodes) kill this runner's model
  process groups (never by name), pack what is finished and exit with status 3 (resumable).
* A runner killed outright (SIGKILL, out of memory) is cleaned up by its watchdog child: leftover models killed,
  finished results packed, the local root removed.
* Exit status of ``run``: 0 every point of the share is recorded (ok or failed), 1 errors (points neither recorded
  nor run, a failed final pack; run again), 3 stopped.
* Failed points: :func:`ppmpy.synspec.fwresults.combine` writes them to ``missing.txt`` with T_eff + 1 K per failed
  attempt (failures are deterministic: M424 point 571348 failed with "error in ne -- nlteopt" at iteration 19 twice
  with the same INDAT and converged at +1 K). Run ``run RUN_DIR missing.txt ...`` and give each retry round's list a
  new name (``missing2.txt``, ...): a list's tag directories skip the points they already hold.
* The packer is started as a fresh ``python -m ppmpy.synspec.fastwind.archive`` process every pack interval
  (900 s): do not change the PyPPM checkout while a run uses it.
* Login node: run in tmux (or with nohup), ~4 models at once for tests and at most ~20; heavy runs belong on compute
  nodes.

Status and convergence flags
----------------------------

pnlte ends every regular run with ``ESTO ES EL ACABOSE`` and exit status 0, **converged or not**, and also exits 0
on its error stops. The legacy status word (``meta.txt``, ``status`` of the merged files) is: ``ok`` (ACABOSE found
and pformalsol exited 0 in time with every expected OUT file complete), ``formal_failed``, ``pnlte_timeout``,
``pnlte_failed``. It says nothing about convergence.

:func:`ppmpy.synspec.fastwind.convergence` applies nlte.f90's own criteria to ``CONVERG`` and ``MAXTCORR.dat``: the
temperature has converged at the first correction with EMAXTC < 3e-3 and iteration > 21; after that, an iteration
with EMAX < 3e-3 or log MEANERR <= -4.5 marks the model converged and pnlte does one more iteration; without
convergence it stops at ITSTART + ITMORE. :func:`~ppmpy.synspec.fastwind.classify` turns this into flags:
``not_converged`` (stopped at the cap), ``temp_not_converged``, ``converged_at_cap``, ``inconsistent_converg``,
``no_converg``, ``error``, ``levels_not_ok``, ``timeout``, ``signal``, ``nonzero_exit`` and the ``formal_*`` flags.
With ``--extras full`` (the default) every new model records them in ``convergence.json``.

``niter`` in meta.txt counts the 'ITERATION NO' lines of pnlte.log, i.e. NLTE iterations + 2 for a fresh start, so a
capped model has niter = ITMORE + 2. The cap is a property of the run configuration and is never inferred from the
data: pass it (``niter_cap``) or get it with :func:`ppmpy.synspec.fwresults.niter_cap_from_indat` (M424:
ITMORE = 100, cap 102 = ``fwresults.NITER_CAP_M424``).

The M424 iteration-cap finding
------------------------------

* 230 649 of the 1 236 544 dump-3200 models (18.7 %) stopped at the cap (niter = 102), all at T_eff' = 38 301-38 598 K
  (48 % of the models there); their median pnlte time was 352 s against 179 s. CONVERG was not kept in the
  production, so their convergence was unknown.
* Check (2026-10-01): 40 capped and 8 uncapped models rerun from their archived INDATs with ITMORE = 300. The 8
  controls converged again after 57-60 iterations with profiles identical to the stored ones (FASTWIND is
  deterministic). 38 of the 40 capped models stopped at 300 too (the temperature correction never converges: EMAX
  0.014-0.07, log MEANERR -4.2 to -3.4, oscillating); 2 converged (100 and 112 iterations).
* Effect on the profiles (300 against 100 iterations): dEW mean +- rms -0.072 +- 0.093 / +0.014 +- 0.040 /
  +0.009 +- 0.047 mA for lambda 4026 / 4200 / 4922 (max 0.27 mA = 2.6e-4 of the EW), max abs(dF) 3.5e-3 / 2.6e-4 /
  8e-4 in single models: negligible against the model-to-model convergence noise (0.5 % rms in EW) and the EW(T_eff')
  sawtooth below. The models are formally unconverged; ITMORE for new runs is still to be decided. New runs flag
  them (``not_converged``), :meth:`ppmpy.synspec.fwresults.ProfileStore.usable` can exclude capped models
  (``cap_ok=False``) and :func:`~ppmpy.synspec.fwresults.status_summary` counts them.

Known artefact of the FASTWIND outputs
--------------------------------------

EW(T_eff') of lambda 4026 (and 4200) switches between branches ~1 % apart every ~170 K. It comes from pformalsol's
continuum under the line, interpolated between the two neighbouring continuum frequency points of ``CONT_FORMAL``
(4037 Angstrom for lambda 4026; 4172 / 4316 Angstrom for lambda 4200), which include background-line opacity that
switches with T_eff'. It contaminates the lambda 4026 EW time series (16 % of its rms) but the residual spectra with
velocities by <= 1 %. The library variants ``flux_sm335`` (nodes smoothed over one period) and ``flux_lamfix``
(correction for the 0.01 Angstrom wavelength rounding of the OUT files) quantify such output systematics.
