.. PPM documentation master file, created by
   sphinx-quickstart on Thu Oct 26 18:54:36 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPPM
===============================

NuGridPy is a Python package containing tools to access and analyse (e.g. plot) various output formats (including the hdf5-based se format) from NuGrid codes (mppnp and ppn) and from the MESA stellar evolution code.

NuGridPy has been developed, and is maintained and supported by the NuGrid collaboration. Feedback and input from the community is welcome. Examples of NuGridPy in action can be tried out on the NuGrid WENDI server. NuGridPy uses the future package so that it can be used with both python 2 and 3.

In principle the NuGridPy package can be used on any stellar evolution code output if the tools to write se hdf5 files available on the NuGrid web page are used. The mesa.py module will work with MESA ASCII output in the LOGS directory. These modules were written with an interactive work mode in mind, in particular taking advantage of the interactive ipython session or inside an ipython notebook.

NuGridPy has been made possible, in part, through support from JINA, Humboldt Foundation and NSERC to NuGrid and its members.

Documentation:
----------------------------
.. autosummary::
   :toctree: _autosummary

   ppmpy.ppm


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
