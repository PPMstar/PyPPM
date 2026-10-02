.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

ppmpy.synspec: synthetic line profiles
======================================

``ppmpy.synspec`` turns PPMstar moms data into disc-integrated, time-dependent spectral line profiles and
analyses their variability. Every point of an equal-area grid on a sphere inside the simulation gets a local
1D non-LTE model atmosphere (FASTWIND) whose effective temperature follows the local temperature fluctuation;
the local profiles are Doppler-shifted by the local flow and summed over the visible hemisphere for chosen
lines of sight, dump by dump.

The package holds the whole pipeline of the M424 project (25 M\ :sub:`sun` full-star run, lines He I 4026,
He II 4200, He I 4922). Its defaults reproduce the M424 products bit for bit: a full rerun of all 1601 dumps
(job 2483431, 2026-10-02) gave products identical to the production files.

* :doc:`pipeline`: the stages, their functions and files, a small-machine example, cluster commands.
* :doc:`conventions`: velocity grid, Doppler sign, frames and lines of sight, units, spectrum normalisation.
* :doc:`fastwind`: what ppmpy needs from an external FASTWIND install, the runner, restarts, convergence flags.
* :doc:`validation`: test tiers, the validation checks and their recorded M424 values, the bit-identity policy.
* :doc:`resources`: memory and time per stage on a laptop and on a Trillium node.

The API reference of each module (``ppmpy.synspec.moms``, ``.disc``, ...) is in the module index; the module
docstrings carry the full conventions, validation records and measurements that these pages summarise.

.. toctree::
   :maxdepth: 2

   pipeline
   conventions
   fastwind
   validation
   resources
