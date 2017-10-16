#
# ppm.py - Tools for accessing and visualising PPMstar data.
#          Depends on the nugridpy package developed by the
#          NuGrid collaboration
# (c) 2010 - 2013 Daniel Alexander Bertolino Conti
# (c) 2011 - 2013 Falk Herwig
# (c) 2014 - 2015 Sam Jones, Falk Herwig, Robert Andrassy
#

""" 
ppm.py

PPM is a Python module for reading Yprofile-01-xxxx.bobaaa files.
Simple session for working with ppm.py, here I assume user's working
directory contains some YProfile files.

If the user find any bugs or errors, please email us.

Yprofile files Assumptions
==========================

- labeled as YProfile-01-xxxx.bobaaa and the xxxx is the NDump that is
  located within each file.

- There can be a maximum of 9999 files in each directory.  The first 10
  lines of each file are the only place where header data is uniquely
  located.

- Header data is separated by five spaces and line breaks.

- An attribute cant be in the form of a number, ie if the user is able
  to 'float(attribute)' (in python) without an error attribute will not
  be returned.

- A row of column Names or Cycle Names is preceded and followed by a
  blank line.

- A block of data values are preceded and followed by a blank line
  except the last data block in the file, where it is not followed by a
  blank line.

- In the YProfile file, if a line of attributes contains the attribute
  Ndump, then that line of attributes and any other following lines of
  attributes in the file are cycle attributes.

Header Attribute Assumptions
============================

- Header attributes are separated by lines and instances of four spaces
  (    )

- Header attribute come in one of below here are things that do stuff
  with the data 6 types.

- The first type is the first attribute in the file. It is formatted in
  such a way that the name of the attribute is separated from its
  associated data with an equals sign ex.
  Stellar Conv. Luminosity =  1.61400E-02 x 10^43 ergs,

- The second type is when an attribute contains the sub string 'grid;'.
  It is formatted in such a way such that there are 3 numerical values
  separated by 'x' which are then followed by the string ' grid;' ex.
  384 x 384 x 384 grid;

- The third type is were name of the attribute is separated from its
  associated data with an equals sign.  Also that data may be followed
  by a unit of measurement. ex.
  Thickness (Mm) of heating shell =  1.00000E+00

- The fourth type is when an attribute contains a colon.  The String
  before the colon is the title of the attribute.  After the colon there
  is a list of n sub attributes, each with a sub title, and separated by
  its value with an equals sign.  Aslo each sub attribute is separated
  by a comma ex.
  At base of the convection zone:  R =  9.50000E+00,  g =  4.95450E-01,
  rho =  1.17400E+01,  p =  1.69600E+01

- The fifth is when an attribute starts with 'and'.  after the and, the
  next word after has to be the same as one word contained in the
  previous attribute ex.
  and of transition from convection to stability =  5.00000E-01  at
  R =  3.00000E+01 Mm.

- The sixth is when there is a string or attribute title followed by two
  spaces followed by one value followed by two spaces followed by an
  'and' which is then followed by a second Value ex.
  Gravity turns on between radii   6.00000E+00  and   7.00000E+00  Mm.

Examples
========
Here is an example runthrough.

>>> from ppm import *
>>> p=y_profile()
>>> head= p.hattri
>>> cols= p.dcols
>>> cyc=p.cattri
>>> print head
[['Stellar Conv. Luminosity', '1.61400E-02 x 10^43 ergs,'], ['384 x 384 x 384 grid;'], ... ]
>>> print cols
['j', 'Y', 'FVconv', 'UYconv', ... , 'Ek H+He']
>>> print cyc
['Ndump', 't', 'trescaled', 'Conv Ht', 'H+He Ht', ..., 'EkXZHHeMax']
>>> j2=p.getColData('j','Yprofile-01-0002',numType='file',resolution='a')
>>> print j2
[[384.0, 383.0, 382.0,..., 1.0],[192.0, 191.0, 190.0,...,1.0]]
>>> j2=p.get('j')
>>> print j2
[[384.0, 383.0, 382.0,..., 1.0],[192.0, 191.0, 190.0,...,1.0]]
>>> j55=p.getColData('j',55,numType='t',resolution='l')
The closest time is at Ndump = 2
>>> print j55
>>> y=p.getColData('Rho1 H+He',2, resolution='L')
>>> print y
[2.0420099999999999e-07, 5.4816300000000004e-07, ... , 0]

and

>>> p.plot('j','FVconv')

plots the data.

"""

# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
from math import *
#from nugridpy.data_plot import *
from nugridpy import utils
#import utils
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib import gridspec
import os
import re
import nugridpy.astronomy as ast
#import astronomy as ast
import scipy.interpolate as interpolate
from scipy import optimize
import copy
import sys
#sys.path.insert(0, '/data/ppm_rpod2/lib/lcse')
#import rprofile as rprof
import struct
import logging


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('rp_info')
log.setLevel(logging.DEBUG)

center_types = [(k, 'f8') for k in ('phi', 'theta', 'x', 'y', 'z')]
normal_types = [(k, 'f8') for k in ('x', 'y', 'z')]

# from rprofile import rprofile_reader

class rprofile_set(object):

  def __init__(self, source, **kwargs):
    """Load a set of RProfiles. Can pass a `path` to a set of files, or a list of `files`. Passing
    the `lazy=True` makes everything operate only from disk, without loading things into memory
    (for large datasets).

    `source` is the path to a directory containing profiles or a list of RProfile files to open

    `stride` the iteration increment. Every `stride` element is looked at during iteration. Default is 1.

    `first_dump` the first dump to iterate from or None

    `last_dump` the last dump to iterate through (inclusive)

    There is no `.get` method... you must iterate through the files like this:

    .. code-block:: python
      :linenos:

      rp_set = lcse.rprofile_set(path=targetpath, lazy=True, logging=True)
      for rp in rp_set:
        rp.get("fv_hi")
        rp.get_table("fv")
    """
    
    self.path = source if isinstance(source, str) else None
    self.files = source if isinstance(source, list) else []
    self.lazy = kwargs.get('lazy', True)
    self.stride = kwargs.get('stride', 1)
    self.first_dump = kwargs.get('first_dump')
    self.last_dump = kwargs.get('last_dump')

    self._logging = kwargs.get('logging')
    self.log = log if self._logging else None

    self.ray_profiles = {}

    self._current_ix = 0
    self._current = None

    if self.path:
      self.files = self.get_file_list_for_path(self.path) if os.path.isdir(self.path) else [self.path]

    dump_re = re.compile('(.*)-([\d]{4})\.bobaaa')

    self.dump_map = dict((int(dump_re.match(f).groups()[1]), f) for f in self.files if dump_re.match(f))
    self.file_map = dict((f, int(dump_re.match(f).groups()[1])) for f in self.files if dump_re.match(f))

    self.dumps = self.dump_map.keys()
    self.dumps.sort()

  def __iter__(self):
    self._current_ix = self.dumps.index(self.first_dump) if self.first_dump else 0
    return self

  # Python 3 bullshit
  def __next__(self):
    return self.next()

  def next(self):

    if self._current_ix < len(self.dumps):
      dump = self.dumps[self._current_ix]

      if self.last_dump and dump > self.last_dump:
        raise StopIteration()

      rp = self.ray_profiles.get(dump, rprofile(self.dump_map[dump], lazy=self.lazy, logging=self._logging))

      if not self.lazy and (dump not in self.ray_profiles):
          self.ray_profiles[dump] = rp

      self._current = rp
      self._current_ix += self.stride
      return rp

    else:
      raise StopIteration()

#  def reset(self):
#
#    if first_dump:
#      self.start = self.dumps.index(first_dump)
#
#    self._current_ix = self.start
#    self._current = None


  def get_file_list_for_path(self, path):
    """ Return a list of RProfiles at the given path"""

    filenames = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('RProfile') and f.endswith('.bobaaa')]
    filenames.sort()

    return filenames

#  def load_files(self, filenames):
#    """ Loads `filenames` """
#
#    # This should add to the existing
#
#    self.files = filenames
#    self.files.sort()
#    self.ray_profiles = [rprofile(f, lazy=self.lazy, logging=self._logging) for f in self.files]

  def check_for_new(self, path=None):
    """Check path for new files"""

    current_files = self.get_file_list_for_path(self.path or path)

    new_files = [f for f in current_files if f not in self.files]

    self.files.extend(new_files)
    self.files.sort()

    return len(new_files) > 0
  
  def get(self, attri, dump, globals_only = False):
    """ Get a new `rprofile` instance for `dump`. These are NOT cached internally."""

    if self.dumps and dump is None:
      dump = self.dumps[-1]
    elif dump not in self.dump_map:
      return None

    rpof = self.ray_profiles.get(dump, rprofile(self.dump_map[dump], lazy=self.lazy, logging=self._logging)) 
    
    return rpof.get(attri, globals_only)

class rprofile(object):
  """
   `rprofile.header_attrs` is a dictionary of header attributes

  """


  header_var_list = [
      dict(name='version', pos=0, type='i'),
      dict(name='cell_ct_low', pos=1, type='i'),
      dict(name='nbuckets', pos=2, type='i'),
      dict(name='dump', pos=3, type='i'),
      dict(name='sizeof_float', pos=4, type='i'),
      dict(name='has_centers', pos=5, type='i'),
      dict(name='has_corners', pos=6, type='i'),
      dict(name='has_normals', pos=7, type='i'),
      dict(name='isrestart', pos=8, type='i', min_ver=12),
      dict(name='var_ct_low', pos=9, type='i'),
      dict(name='var_ct_high', pos=10, type='i'),
      dict(name='cell_ct_high', pos=11, type='i'),
      dict(name='ncpucores', pos=12, type='i'),
      dict(name='ntxbricks', pos=13, type='i'),
      dict(name='ntybricks', pos=14, type='i'),
      dict(name='ntzbricks', pos=15, type='i'),
      dict(name='nxteams', pos=16, type='i'),
      dict(name='nyteams', pos=17, type='i'),
      dict(name='nzteams', pos=18, type='i'),
      dict(name='nx', pos=19, type='i'),
      dict(name='ny', pos=20, type='i'),
      dict(name='nz', pos=21, type='i'),
      dict(name='nsugar', pos=22, type='i'),
      dict(name='nbdy', pos=23, type='i'),
      dict(name='nfluids', pos=24, type='i'),
      dict(name='nvars', pos=25, type='i'),
      dict(name='nhalfwaves', pos=26, type='i'),
      dict(name='maxrad', pos=27, type='i'),
      dict(name='nteamsinbunch', pos=28, type='i'),
      dict(name='ndumps', pos=29, type='i'),
      dict(name='ndumpstodo', pos=30, type='i'),
      dict(name='nrminrad', pos=31, type='i'),
      dict(name='nrmaxrad', pos=32, type='i'),
      dict(name='iburn', pos=33, type='i', min_ver=12),
      dict(name='imuffledbdry', pos=34, type='i', min_ver=12),
      dict(name='ireflectbdry', pos=35, type='i', min_ver=12),

      # fheader (the offsets are in the fheader)
      dict(name='radin0', pos=0, type='f', help='Gravity completely off inside this radius'),
      dict(name='radinner', pos=1, type='f', help='Gravity starts turning off inside this radius'),
      dict(name='radbase', pos=2, type='f', help='Bot convect zone'),
      dict(name='radtop', pos=3, type='f', help='Top convect zone'),
      dict(name='radouter', pos=4, type='f', help='Grav starts turning off outside this radius'),
      dict(name='radout0', pos=5, type='f', help='Gravity completely off outside this radius'),
      dict(name='radmax', pos=6, type='f', help='distance from center of grid to nearest edge'),
      dict(name='dlayerbot', pos=7, type='f', help='thickness of flame zone'),
      dict(name='dlayertop', pos=8, type='f', help='thickness of transition @ top of convect zone'),
      dict(name='totallum', pos=9, type='f'),
      dict(name='grav00base', pos=10, type='f'),
      dict(name='rho00base', pos=11, type='f'),
      dict(name='prs00base', pos=12, type='f'),
      dict(name='gammaconv', pos=13, type='f'),
      dict(name='gammabelow', pos=14, type='f'),
      dict(name='gammaabove', pos=15, type='f'),
      dict(name='gravconst', pos=16, type='f'),
      dict(name='rhoconv', pos=17, type='f'),
      dict(name='rhoabove', pos=18, type='f'),
      dict(name='airmu', pos=19, type='f', min_ver=13),
      dict(name='cldmu', pos=20, type='f', min_ver=13),
      dict(name='fkair', pos=21, type='f', min_ver=13),
      dict(name='fkcld', pos=22, type='f', min_ver=13),
      dict(name='atomicnoair', pos=23, type='f', min_ver=13),
      dict(name='atomicnocld', pos=24, type='f', min_ver=13),

      # Global T-history
      dict(name='time', pos=31+0, type='f'),
      dict(name='timerescaled', pos=31+1, type='f'),
      dict(name='bubbleheight', pos=31+2, type='f'),
      dict(name='spikeheight', pos=31+3, type='f'),
      dict(name='cycl', pos=31+4, type='f', min_ver=12),
      dict(name='dt', pos=31+5, type='f'),
      dict(name='courmx', pos=31+6, type='f'),
      dict(name='urbubmx', pos=31+7, type='f'),
      dict(name='urspkmn', pos=31+8, type='f'),
      dict(name='ekmx', pos=31+9, type='f'),
      dict(name='ekrmx', pos=31+10, type='f'),
      dict(name='ektmx', pos=31+11, type='f'),
      dict(name='ekurmn', pos=31+12, type='f'),
      dict(name='ekurmx', pos=31+13, type='f'),
      dict(name='eiurmn', pos=31+14, type='f'),
      dict(name='eiurmx', pos=31+15, type='f'),
      dict(name='Hurmn', pos=31+16, type='f'),
      dict(name='Hurmx', pos=31+17, type='f'),
      dict(name='ekurspkmn', pos=31+18, type='f'),
      dict(name='ekurbubmx', pos=31+19, type='f'),
      dict(name='eiurspkmn', pos=31+20, type='f'),
      dict(name='eiurbubmx', pos=31+21, type='f'),
      dict(name='Hurspkmn', pos=31+22, type='f'),
      dict(name='Hurbubmx', pos=31+23, type='f'),
      dict(name='ekbubmx', pos=31+24, type='f'),
      dict(name='ekrbubmx', pos=31+25, type='f'),
      dict(name='ektbubmx', pos=31+26, type='f'),
      dict(name='ekspkmx', pos=31+27, type='f'),
      dict(name='ekrspkmx', pos=31+28, type='f'),
      dict(name='ektspkmx', pos=31+29, type='f'),

      # Args images
      dict(name='ai_vort', pos=64+0, type='f', len=2),
      dict(name='ai_divu', pos=64+2, type='f', len=2),
      dict(name='ai_s', pos=64+4, type='f', len=2),
      dict(name='ai_fv', pos=64+6, type='f', len=2),
      dict(name='ai_rho', pos=64+8, type='f', len=2),
      dict(name='ai_p', pos=64+10, type='f', len=2),
      dict(name='ai_ux', pos=64+12, type='f', len=2),
      dict(name='ai_uy', pos=64+14, type='f', len=2),
      dict(name='ai_uz', pos=64+16, type='f', len=2),
  ]

  def __init__(self, filename, lazy=True, **kwargs):
    """Create a ray profile reader object.

    `lazy` means only the header is loaded on open
    """

    logging = kwargs.get('logging')

    self._filename = filename
    self.lazy = lazy

    self.version = None
    self.bucket_count = 0

    self._centers = None
    self._corners = None
    self._normals = None

    self._cache = {}

    self._variable_map = {}

    self._names = []
    self._data = []

    self.header_attrs = {}

    if logging:
      import logging
      logging.basicConfig(level=logging.DEBUG)
      self.log = logging.getLogger('rp_info')
      self.log.setLevel(logging.DEBUG)
    else:
      self.log = None

    if str(filename).isdigit():
      filename = 'RProfile-01-%04i.bobaaa' % int(filename)

    if self.log: self.log.info("Opening %s" % filename)

    f = open(filename, 'rb')
    header = f.read(128)

    if header[:8] != 'LCSE:RPS':
      if self.log: self.log.warn('File %s is not a new Ray Profile, try an older rp_info.py' % filename)
      f.close()
      raise Exception('Unsupported file version')

    self.version = struct.unpack("i", header[8:12])[0]

    f.seek(0)

    if self.version < 8:
      raise Exception('Unsupported version %i' % self.version)
    elif self.version == 8:
      self._header_size = 128
      hlen = 8

      self.header_var_list = self.header_var_list[:8]

#      header = struct.unpack(hlen * "i", header[8:8+4*hlen])

#      self.header_attrs['version'] = header[0]
#      self.header_attrs['cell_ct_low'] = header[1]
#      self.header_attrs['nbuckets'] = header[2]
#      self.header_attrs['dump'] = header[3]
#      self.header_attrs['sizeof_float'] = header[4]
#      self.header_attrs['has_centers'] = header[5]
#      self.header_attrs['has_corners'] = header[6]
#      self.header_attrs['has_normals'] = header[7]

    elif self.version > 8:
      self._header_size = 1024
      hlen = 127

    header = f.read(self._header_size)
      # Bug fixes
      # Using the header info from v9
     # if self.version < 11:
     #   self._init_v9()
     #   self._init_legacy()

     #   raw_header = struct.unpack(hlen * "i", header[8:8+4*hlen])
     #   raw_fheader = struct.unpack(hlen * "f", header[8+4*hlen:8+8*hlen])
     #   self.header_attrs.update([(k, raw_header[i]) for i, k in enumerate(self._header_names)])
     #   self.header_attrs.update([(k, raw_fheader[i]) for i, k in enumerate(self._fheader_names)])
     #   self.header_attrs.update([(k, raw_fheader[32 + i]) for i, k in enumerate(self._fheader_names2)])
     #   self.header_attrs.update([(k, (raw_fheader[64 + 2*i], raw_fheader[64 + 2*i + 1] )) for i, k in enumerate(self._argsimg_names)])
     #elif self.version <= 12:

    hmap = dict(i=struct.unpack(hlen * "i", header[8 : 8 + 4 * hlen]),
                f=struct.unpack(hlen * "f", header[8 + 4 * hlen : 8 * (1 + hlen)]))

    for var in self.header_var_list:

      name = var['name']
      pos = var['pos']
      var_type = var['type']
      var_len = var.get('len', 1)
      min_ver = var.get('min_ver', 0)

      if self.version < min_ver:
        continue

      # A slight offset problem
      if self.version == 11 and var_type == 'f' and pos > 30: # and pos < 64:
        pos = pos + 1

      attr = hmap[var_type][pos] if var_len == 1 else hmap[var_type][pos : pos + var_len]
      self.header_attrs[name] = attr

    # Fix header problems
    if self.version == 8:
      self.header_attrs['cell_ct_high'] = 2 * self.header_attrs['cell_ct_low']
      self.header_attrs['var_ct_high'] = 1
      self.header_attrs['var_ct_low'] = 14

    if self.version == 9:
      self.header_attrs['cell_ct_low'] -= 2
      self.header_attrs['cell_ct_high'] -= 4

    if self.version < 12:
      self.header_attrs['isreflectbdry'] = 1
      self.header_attrs['ismuffledbdry'] = 0

    if self.version > 13:
      self.header_attrs['has_corners'] = False

    self.bucket_count = self.header_attrs['nbuckets']
    self.dump = self.header_attrs['dump']
    self.buckets = self.header_attrs['nbuckets']

    if self.version > 10:
      self._init_v11()

      if not self.lazy:
        f = open(self._filename, 'r')
        self._data = f.read()
        f.close()

    else:
      self._init_legacy()

      if self.version == 8:
        self._init_v8()
      else:
        self._init_v9()

      for k in ['has_centers', 'has_corners', 'has_normals']:
        self.header_attrs[k] = self.header_attrs.get(k, 0) == 1

      float_type = 'f8' if self.header_attrs.get('sizeof_float') == 8 else 'f4'

      self._dtypes_hi = [('j_hi', 'i4')]
      self._dtypes_hi.extend([(n, float_type) for n in self._names_hi])
      self._col_names_hi = ['j_hi'] + self._names_hi

      self._dtypes = [('j', 'i4')]
      self._dtypes.extend([(n, float_type) for n in self._names])

      self._col_names = ['j'] + self._names

      if self.lazy:
        log.warn("Lazy Loading not supported for v %i" % self.version)

      self._load(f)

    f.close()

  def _load(self, f):

    nbuckets = self.header_attrs.get('nbuckets')
    cell_ct_low = self.header_attrs.get('cell_ct_low')
    cell_ct_high = self.header_attrs.get('cell_ct_high')

    # Read the high resolution table
    self._data_hi = np.fromfile(f, dtype=self._dtypes_hi, count=cell_ct_high*(nbuckets+1))

    # Read the low resolution table
    self._data_low = np.fromfile(f, dtype=self._dtypes, count=cell_ct_low*(nbuckets+1))

    if self.header_attrs.get('has_centers'):
      vals = 3 if self.version > 12 else 5
      self._centers = np.fromfile(f, dtype=np.float64, count=5 * nbuckets).reshape((vals, -1), order='F')

    if self.header_attrs.get('has_normals'):
      self._normals = np.fromfile(f, dtype=np.float64, count=9*nbuckets).reshape((3, 3, -1), order='F')

    if self.header_attrs.get('has_corners'):
      self._corners = np.fromfile(f, dtype=np.float64, count=9*nbuckets).reshape((3, 3, -1), order='F')
 
    
  def get(self, var, global_only):
    """Get the global bucket for variable `var` or  get header attribute `var`.

    Use `get_table(self, var)` to get the same variable but for all buckets.

    If the global bucket is returned an array of dimension (4, ncells) is returned.
    The first dimension contains avg, min, max, sd.
    """

    if var in self.header_attrs:
      return self.header_attrs.get(var)

    if var in self._variable_names:
      return self._get_array(var, global_only=True)
    
  def _get_array(self, var, global_only):

    if var not in self._variable_map:
      return None

    if var in self._cache:
      return self._cache[var]

    offset, dtype, count, shape = self._variable_map[var]

#    print self._variable_map[var], global_only

    if global_only and len(shape) == 3 and shape[2] == self.bucket_count + 1:
      count = shape[0] * shape[1]
      shape = shape[:2]

    if self.lazy:
      f = open(self._filename, 'r')
      f.seek(offset)
      data = np.fromfile(f, dtype=dtype, count=count).reshape(shape, order='F')
      f.close()
    else:
      data = np.frombuffer(self._data[offset:], dtype=dtype,
                           count=count).reshape(shape, order='F')

    if not global_only:
      self._cache[var] = data

    return data
  def get_variables(self):
    return self._variable_names

  def _init_v11(self):

    cell_ct_high = self.header_attrs.get('cell_ct_high')
    cell_ct_low = self.header_attrs.get('cell_ct_low')

    buckets_total = 1 + self.bucket_count

    sizeof_float = self.header_attrs.get('sizeof_float')
    float_type = np.float64 if sizeof_float == 8 else np.float32
    int_type = np.int32

    vals = 3 if self.version > 12 else 5

    # name, size_in_bytes, <array dimensions>
    self._variable_list = [('centers', float_type, sizeof_float, (vals, self.bucket_count)),
                           ('normals', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('corners', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('j_hi', int_type, 4, (cell_ct_high,)),
                           ('y_hi', float_type, sizeof_float, (cell_ct_high,)),
                           ('fv_hi', float_type, sizeof_float, (4, cell_ct_high, buckets_total)),
                           ('j', int_type, 4, (cell_ct_low,)),
                           ('Y', float_type, sizeof_float, (cell_ct_low,)),
                           ('counts', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fv', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Rho', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhobubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhospike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourbubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourspike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('P', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ux', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uz', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ceul', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('mach', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('enuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fnuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('dy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Ekr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ekt', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Ek', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EkUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EiUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('HUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                          ]

#    present_vars = [v['name'] for v in self.header_var_list if self.version >= v.get('min_ver', self.version)]

    # TODO: HACK UGH
    skip_vars = []
    if self.version > 12:
      skip_vars.append('corners')

    offset = self._header_size

    for name, dtype, sizeof, shape in self._variable_list:

      if name in skip_vars:
        continue

#      print (name, offset, dtype, sizeof, shape)

      count = np.prod(shape)
      size = sizeof * count
      self._variable_map[name] = (offset, dtype, count, shape)
      offset += size
#      print (name, offset, dtype, count, shape, sizeof)

    self._variable_names = self._variable_map.keys()
    self._variable_names.sort()


@np.vectorize
def map_signed(x, p0, p1, s0, s1):
    '''
    This function emulates the mapping of signed variables in PPMstar.

    x: input value in code units; can be a single number or a vector
    p0, p1: mapping parameters used in the first scaling step
    s0, s1: mapping parameters used in the last scaling step (just before
            the conversion to an integer)
    '''

    thyng = (x - p0)*p1
    thang = thyng * thyng  +  1.
    thang = np.sqrt(thang)
    thyng = thyng + thang
    thyng = thyng * thyng
    thang = thyng + 1.
    thyng = thyng / thang
    y = s1 * thyng  +  s0

    return y

@np.vectorize
def inv_map_signed(y, p0, p1, s0, s1):
    '''
    This function inverts map_signed().
    '''

    if y <= s0:
        x = -np.inf
    elif y >= s0 + s1:
        x = np.inf
    else:
        def func(x):
            return y - map_signed(x, p0, p1, s0, s1)

        x = optimize.newton(func, 0.)

    return x

@np.vectorize
def map_posdef(x, p0, p1, s0, s1):
    '''
    This function emulates the mapping of positive definite variables in PPMstar.

    x: input value in code units; can be a single number or a vector
    p0, p1: mapping parameters used in the first scaling step
    s0, s1: mapping parameters used in the last scaling step (just before
            the conversion to an integer)
    '''

    thyng = (x - p0)*p1
    thang = thyng * thyng  +  1.
    thang = np.sqrt(thang)
    thyng = thyng + thang
    thyng = thyng * thyng
    thang = thyng + 1.
    thyng = (thyng - 1.) / thang
    y = s1 * thyng  +  s0

    return y

@np.vectorize
def inv_map_posdef(y, p0, p1, s0, s1):
    '''
    This function inverts map_posdef().
    '''

    if y <= s0:
        x = -np.inf
    elif y >= s0 + s1:
        x = np.inf
    else:
        def func(x):
            return y - map_posdef(x, p0, p1, s0, s1)

        x = optimize.newton(func, 0.)

    return x


def make_colourmap(colours, alphas=None):
    '''
    make a matplotlib colormap given a list of index [0-255], RGB tuple values
    (normalised 0-1) and (optionally) a list of alpha index [0-255] and alpha values, i.e.:
        
    colours = [[0, (0., 0., 0.)], [1, (1., 1., 1.)]]
    alphas = [[0, 0.], [1, 1.]]
    '''

    indices_normed = np.array([float(c[0])/255. for c in colours])
    # enforce [0-1]
    if indices_normed[-1] != 1.:
        print 'first/last colour indices:', indices_normed[-1]
        print 'correcting to 1.'
        indices_normed[-1] = 1.

    rgb = colours
    cdict = {'red': [], 'green': [], 'blue': []}
    for i in range(len(colours)):
        myrgb = rgb[i][1]
        cdict['red'].append([indices_normed[i], myrgb[0], myrgb[0]])
        cdict['green'].append([indices_normed[i], myrgb[1], myrgb[1]])
        cdict['blue'].append([indices_normed[i], myrgb[2], myrgb[2]])

    if alphas!=None:
        cdict['alpha'] = []
        indices_normed = np.array([float(a[0]) / 255. for a in alphas])
        alpha = alphas
        for i in range(len(alphas)):
            myalpha = alpha[i][1]
            cdict['alpha'].append([indices_normed[i], myalpha, myalpha])

    cmap = matplotlib.colors.LinearSegmentedColormap('ppm', cdict, N=1536)

    return cmap

def colourmap_from_str(str, segment=None):
    points = []
    for line in str.splitlines():
        parts = line.split()
        if (len(parts) == 5) and (parts[0] == 'Cnot:'):
            points.append(parts[1:])
    
    points = np.array(points, dtype=np.float)
    points = points[points[:,0].argsort()]

    if segment is not None:
        # Index of the first point with value > segment[0].
        idx0 = np.argmax(points[:, 0] > segment[0])
        if idx0 > 0:
            t = (float(segment[0]) - points[idx0 - 1, 0])/ \
                (points[idx0, 0] - points[idx0 - 1, 0])
                
            new_point = (1. - t)*points[idx0 - 1, :] + t*points[idx0, :]
            points = np.vstack([new_point, points[idx0:, :]])
            
        # Index of the first point with value > segment[1].
        idx1 = np.argmax(points[:, 0] > segment[1])
        if idx1 > 0:
            t = (float(segment[1]) - points[idx1 - 1, 0])/ \
                (points[idx1, 0] - points[idx1 - 1, 0])
            
            if t > 0.:
                new_point = (1. - t)*points[idx1 - 1, :] + t*points[idx1, :]
                points = np.vstack([points[0:idx1, :], new_point])
            else:
                points = points[0:idx1, :]

    p0 = points[0, 0]
    p1 = points[-1, 0]
    for i in range(points.shape[0]):
        points[i, 0] = (points[i, 0] - p0)/(p1 - p0)
    
    r = np.zeros((points.shape[0], 3))
    r[:, 0] = points[:, 0]
    r[:, 1] = points[:, 1]
    r[:, 2] = points[:, 1]
                 
    g = np.zeros((points.shape[0], 3))
    g[:, 0] = points[:, 0]
    g[:, 1] = points[:, 2]
    g[:, 2] = points[:, 2]
                 
    b = np.zeros((points.shape[0], 3))
    b[:, 0] = points[:, 0]
    b[:, 1] = points[:, 3]
    b[:, 2] = points[:, 3]

    cmap_points = {'red': r, 'green': g, 'blue': b}
    cmap = matplotlib.colors.LinearSegmentedColormap('my_cmap', cmap_points)
    
    return cmap

class LUT():
    def __init__(self, lutfile, p0, p1, s0, s1, posdef=False):
        '''given a LUT file from the PPMstar visualisation software and the
        colour compression variables s0, s1, p0 and p0 that were used to
        compress the values of the variable being visualised (e.g., radial
        velocity), this object contains the information needed to draw
        colourbars with matplotlib for a PPMstar volume rendering.
        
        The values of s0, s1, p0 and p1 should be stored in the file
        compression_variables.txt in the setup directory of the respective
        project.
        
        Examples:
        ---------
        import ppm
        lut = ppm.LUT('./LUTS/BW-1536-UR-3.lut', s0=5., s1=245.499,
                      p0=0., p1=1.747543E-02/8.790856E-03, posdef=False)
        cbar = lut.make_colourbar([-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1])
        cbar.set_label('$v_\mathrm{r}\,/\,1000\,km\,s^{-1}$',size=6)
        draw()
        '''

        self.colours = []
        self.alphas = []
        self.cmap = None
        self.s0 = s0
        self.s1 = s1
        self.p0 = p0
        self.p1 = p1
        # is the variable positive definite (otherwise signed):
        self.posdef = posdef
        with open(lutfile,'r') as f:
            nlines = len(f.readlines())
            f.seek(0)
            for i in range(nlines):
                spl = f.readline().split()
                index = int(spl[1])
                if spl[0][0] == 'C':
                    rgb_tuple = (float(spl[2]),
                                 float(spl[3]),
                                 float(spl[4]))
                    self.colours.append([index, rgb_tuple])
                elif spl[0][0] == 'A':
                    self.alphas.append([index, float(spl[2])])
                else:
                    raise IOError("unrecognised LUT file format")
        f.close()
        self.ncolours = len(self.colours)
        self.nalphas = len(self.alphas)

    def make_colourbar(self, ticks=[], horizontal=True, background=(0,0,0),
            scale_factor=1.):
        '''make a colourbar for a PPMstar volume rendering given the mapping
        between the [0-255] colour indices and the values to which they
        correspond. returns a matplotlib.pyplot.colorbar instance, for ease of
        editing the colorbar
        
        Parameters:
        -----------
        ticks: numpy array
            at which values of the variable you would like to have ticks on the
            colourbar
        scale_factor: float
            dividing your ticks by this number should give the real values in
            code units. so, if I give ticks in km/s, I should give
            scale_factor=1.e3

        Examples:
        ---------
        import ppm
        lut = ppm.LUT('./LUTS/BW-1536-UR-3.lut', s0=5., s1=245.499, p0=0., p1=1.747543E-02/8.790856E-03, posdef=False)
        cbar=lut.make_colourbar(np.linspace(-100,100,5),background=(0.4117647058823529,0.4117647058823529,0.4235294117647059),scale_factor=1.e3)
        cbar.set_label('$v_\mathrm{r}\,/\,km\,s^{-1}$',size=6)
        draw()
        '''

        if background != (1,1,1):
            fgcolor = (1, 1, 1)
            pl.rcParams['text.color'] = fgcolor
            pl.rcParams['xtick.color'] = fgcolor
            pl.rcParams['ytick.color'] = fgcolor
            pl.rcParams['axes.edgecolor'] = fgcolor
            pl.rcParams['axes.labelcolor'] = fgcolor
            pl.rcParams['figure.edgecolor'] = background
            pl.rcParams['figure.facecolor'] = background
            pl.rcParams['savefig.edgecolor'] = background
            pl.rcParams['savefig.facecolor'] = background
            pl.rcParams['axes.facecolor'] = background

        colours = copy.deepcopy(self.colours)
        ticks = np.array(ticks)/scale_factor

        # determine which codec to use
        self.map_values = map_posdef if self.posdef else map_signed
        self.inv_map_values = inv_map_posdef if self.posdef else inv_map_signed

        # make sure we have min and max for both colour index and values, and
        # also the locations that the ticks are going to be placed
        if ticks == []:
            minidx, maxidx = self.s0+1, self.s1-1
            minval,maxval = self.inv_map_values([minidx,maxidx],self.p0,self.p1,self.s0,self.s1)
            ticks = np.linspace(minval, maxval, 8)
        else:
            # ticks given
            minval = ticks[0]; maxval = ticks[-1]
            minidx = self.map_values(minval,self.p0,self.p1,self.s0,self.s1)
            maxidx = self.map_values(maxval,self.p0,self.p1,self.s0,self.s1)

        colour_index_ticks = [self.map_values(vt,self.p0,self.p1,self.s0,self.s1) for vt in ticks]
        colour_index_ticks = np.array(colour_index_ticks)

        if any(np.isinf(colour_index_ticks)):
            print 'ticks out of range'
            return

        print 'min/max ticks being set to:', minval, maxval
        print 'corresponding to colour indices:', minidx, maxidx
        print 'ticks being placed at:', ticks
        print 'with colour indices:', colour_index_ticks

        # OK, so now we have to make the colour map on the fly (because we are
        # having to sample the original LUT for the subsection we are
        # interested in.
        
        # This means normalising the appropriate interval to the interval
        # [0-255] and constructing new left and right edges, passing the colours (and maybe alphas)
        # to the make_colourmap function.

        # left:
        i0 = np.where(np.array([c[0] for c in colours]) <= minidx)[0][-1]
        i1 = i0 + 1
        ileft = i1
        idx0 = colours[i0][0]
        idx1 = colours[i1][0]
        r0, r1 = colours[i0][1][0], colours[i1][1][0]
        g0, g1 = colours[i0][1][1], colours[i1][1][1]
        b0, b1 = colours[i0][1][2], colours[i1][1][2]
        rl = r0 + (r1 - r0)/(idx1 - idx0) * (minidx - idx0)
        gl = g0 + (g1 - g0)/(idx1 - idx0) * (minidx - idx0)
        bl = b0 + (b1 - b0)/(idx1 - idx0) * (minidx - idx0)

        # right:
        i0 = np.where(np.array([c[0] for c in colours]) <= maxidx)[0][-1]
        i1 = i0 + 1
        iright = i1
        idx0 = colours[i0][0]
        idx1 = colours[i1][0]
        r0, r1 = colours[i0][1][0], colours[i1][1][0]
        g0, g1 = colours[i0][1][1], colours[i1][1][1]
        b0, b1 = colours[i0][1][2], colours[i1][1][2]
        rr = r0 + (r1 - r0)/(idx1 - idx0) * (maxidx - idx0)
        gr = g0 + (g1 - g0)/(idx1 - idx0) * (maxidx - idx0)
        br = b0 + (b1 - b0)/(idx1 - idx0) * (maxidx - idx0)

        print ileft, iright, minidx, maxidx

        to_splice = copy.deepcopy(colours)[ileft:iright]
        newcolours = [[minidx, (rl, gl, bl)]] + to_splice + [[maxidx, (rr, gr, br)]]

        # now normalise the indices to [0-255]
        indices = np.array([c[0] for c in newcolours])
        newindices = 255.*(indices - np.min(indices)) / indices.ptp()
        # renormalise index tick locations as well
        colour_index_ticks = 255.*(colour_index_ticks - np.min(colour_index_ticks)) / colour_index_ticks.ptp()
        print 'new colour indices:', newindices
        print 'ticks now at:', colour_index_ticks

        for i in range(len(newcolours)):
            newcolours[i][0] = newindices[i]

        self.cmap = make_colourmap(newcolours)

        x = np.linspace(0, 256, 257)
        y = x.copy()
        xx, yy = np.meshgrid(x, y)
        mat = xx.copy()

        pl.figure()
        pcol = pl.pcolor(x, y, mat.T, cmap=self.cmap)
        pl.gca().xaxis.set_visible(False)
        pl.ylabel('colour index')

        if horizontal:
            cbar = pl.colorbar(orientation='horizontal', ticks = colour_index_ticks)
            cbar.ax.set_xticklabels(ticks*scale_factor)
        else:
            cbar = pl.colorbar(ticks = colour_index_ticks)
            cbar.ax.set_yticklabels(ticks*scale_factor)
        cbar.solids.set_edgecolor('face')
        cbar.ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=6)
        pl.draw()

        return cbar

def cmap_from_str(str, segment=None):
    points = []
    for line in str.splitlines():
        parts = line.split()
        if (len(parts) == 5) and (parts[0] == 'Cnot:'):
            points.append(parts[1:])
    
    points = np.array(points, dtype=np.float)
    points = points[points[:,0].argsort()]

    if segment is not None:
        # Index of the first point with value > segment[0].
        idx0 = np.argmax(points[:, 0] > segment[0])
        if idx0 > 0:
            t = (float(segment[0]) - points[idx0 - 1, 0])/ \
                (points[idx0, 0] - points[idx0 - 1, 0])
                
            new_point = (1. - t)*points[idx0 - 1, :] + t*points[idx0, :]
            points = np.vstack([new_point, points[idx0:, :]])
            
        # Index of the first point with value > segment[1].
        idx1 = np.argmax(points[:, 0] > segment[1])
        if idx1 > 0:
            t = (float(segment[1]) - points[idx1 - 1, 0])/ \
                (points[idx1, 0] - points[idx1 - 1, 0])
            
            if t > 0.:
                new_point = (1. - t)*points[idx1 - 1, :] + t*points[idx1, :]
                points = np.vstack([points[0:idx1, :], new_point])
            else:
                points = points[0:idx1, :]

    p0 = points[0, 0]
    p1 = points[-1, 0]
    for i in range(points.shape[0]):
        points[i, 0] = (points[i, 0] - p0)/(p1 - p0)
    
    r = np.zeros((points.shape[0], 3))
    r[:, 0] = points[:, 0]
    r[:, 1] = points[:, 1]
    r[:, 2] = points[:, 1]
                 
    g = np.zeros((points.shape[0], 3))
    g[:, 0] = points[:, 0]
    g[:, 1] = points[:, 2]
    g[:, 2] = points[:, 2]
                 
    b = np.zeros((points.shape[0], 3))
    b[:, 0] = points[:, 0]
    b[:, 1] = points[:, 3]
    b[:, 2] = points[:, 3]

    cmap_points = {'red': r, 'green': g, 'blue': b}
    cmap = LinearSegmentedColormap('my_cmap', cmap_points)
    
    return cmap

###########################################################
# Additional plotting methods Jericho
###########################################################

def analyse_dump(rp, r1, r2):

    '''
    This function analyses ray profiles of one dump and returns

    r, ut, dutdr, r_ub,

    Parameters
    ----------
    rp: radial profile object
        radial profile
    r1: float
        minimum radius for the search for r_ub
    r2: float
        maximum radius for the search for r_ub\
        
    Output
    ------
    r: array
        radius
    ut: array
        RMS tangential velocity profiles for all buckets (except the 0th)
    dutdr: array
        radial gradient of ut for all buckets (except the 0th)
    r_ub: array
        radius of the upper boundary as defined by the minimum in dutdr
        for all buckets  (except the 0th).
        
    '''
    n_buckets = rp.get('nbuckets')

    r = rp.get_table('y')
    dr = 0.5*(np.roll(r, -1) - np.roll(r, +1))

    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    ekt = rp.get_table('ekt')
    ut = ekt[0, :, 1:n_buckets+1]**0.5

    dut = 0.5*(np.roll(ut, -1, axis = 0) - np.roll(ut, +1, axis = 0))
    dutdr = np.transpose(np.array([dut[:, i]/dr for i in range(n_buckets)]))

    idx_min_dutdr = [idx1 + np.argmin(dutdr[idx1:idx2 + 1, i]) \
                     for i in range(n_buckets)]
    r_ub = np.zeros(n_buckets)

    for bucket in range(n_buckets):
        idx = idx_min_dutdr[bucket]
        r_min = r[idx] # 0th-order estimate

        # try to fit a parabola around r_min
        r_fit = r[idx-1:idx+2]
        dutdr_fit = dutdr[idx-1:idx+2, bucket]
        coefs = np.polyfit(r_fit, dutdr_fit, 2)

        # hopefully we can determine the position of the minimum from the fit
        if coefs[0] != 0:
            r_min = -coefs[1]/(2.*coefs[0])
            # go back to 0th order if something has gone awry with the fit
            if r_min < r[idx -1] or r_min > r[idx + 1]:
                r_min = r[idx]

        r_ub[bucket] = r_min

    return r, ut, dutdr, r_ub

def upper_bound_ut(data_path, dump_to_plot, hist_dump_min, hist_dump_max, derivative = False, r1=7.4, r2=8.4):

    '''
    Finds the upper convective boundary as defined by the steepest decline in 
    tangential velocity. 

    Subpolot(1) plots the tangential velocity as a function of radius for a single dump and 
        displays the convective boundary
    Subplot(2) plots a histogram of the convective boundaries for a range of dumps specified by
        user and compares them to the selected dump

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    derivative = boolean
        True = plot dut/dr False = plot ut
    dump_To_plot = int
        The file number of the dump you wish to plot
    hist_dump_min/hist_dump_max = int
        Range of file numbers you want to use in the histogram
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2

    Example
    -------
    data_path = "/data/ppm_rpod2/RProfiles/O-shell-M25/D15/"
    dump_to_plot = 121
    hist_dump_min = 101; hist_dump_max = 135
    r1 = 7.4; r2 = 8.4
    upper_bound_ut(data_path, derivative, dump_to_plot,\
        hist_dump_min, hist_dump_max, r1, r2)

    '''
    cb = utils.colourblind
    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut = res[1]
        dutdr = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)

    for k in range(n_dumps):
        devp = dev[:, k]
        devp = devp[devp >= 0]
        if len(devp) > 0:
            sigmap_r_ub[k] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmam_r_ub[k] = None

        devm = dev[:, k]
        devm = devm[devm <= 0]
        if len(devm) > 0:
            sigmam_r_ub[k] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_ub[k] = None


    hist_bins = 0.5*(r + np.roll(r, -1))
    hist_bins[-1] = hist_bins[-2] + (hist_bins[-2] - hist_bins[-3])
    #hist_bins = np.insert(hist_bins, 0., 0.)       # #robert - this command throws an error?!?

    print "Dump {:d} (t = {:.2f} min).".format(dump_to_plot, t[dump_to_plot]/60.)
    print "Histogram constructed using dumps {:d} (t = {:.2f} min) to {:d} (t = {:.2f} min) inclusive."\
        .format(hist_dump_min, t[hist_dump_min]/60., hist_dump_max, t[hist_dump_max]/60.)

    fig = pl.figure( figsize = (2*3.39, 2*2.8))   
    #fig = pl.figure( figsize = (2*5, 2*4)) 
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
    ax0 = pl.subplot(gs[0])

    if derivative:
        temp = dutdr
        lims = (-0.49, 0.1)
    else:
        temp = 1e3*ut
        lims = (-9.99, 70)

    ax0.set_ylim(lims)

    for bucket in range(n_buckets):
        lbl = r'bucket data' if bucket == 0 else None

        ax0.plot(r, temp[:, bucket], ls = '-', lw = 0.5, color = cb(3), \
            label = lbl)

        lines = (min(lims) + (max(lims)- min(lims))/13.3 ,\
                 min(lims) + (max(lims)- min(lims))/13.3 + (max(lims)- min(lims))/30)
        lbl = r'steepest decline'
        lbl = lbl if bucket == 0 else None
        ax0.plot((r_ub[bucket, dump_to_plot], r_ub[bucket, dump_to_plot]), lines, \
                 ls = '-', lw = 0.5, color = cb(4), label = lbl)

    ax0.axvline(x = avg_r_ub[dump_to_plot], ls = '--', lw = 1., color = cb(4), label = 'average')
    ax0.axvline(x = avg_r_ub[dump_to_plot] - 2*sigmam_r_ub[dump_to_plot], ls = ':', lw = 1., \
                color = cb(4), label = '2$\sigma$ fluctuations')
    ax0.axvline(x = avg_r_ub[dump_to_plot] + 2*sigmap_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax0.set_xlim((r1 - 0.4, r2))

    ax0.set_ylabel(r'v$_{\!\perp}$ / km s$^{-1}$')
    yticks = ax0.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    ax0.legend(loc = 3, frameon = False)

    #ax0.autoscale(enable=True, axis='y', tight=True)
    ax1 = pl.subplot(gs[1])
    ax1.hist(r_ub[:, hist_dump_min:hist_dump_max+1].flatten(), bins = hist_bins, \
             log = True, color = cb(3), edgecolor = cb(4), lw = 0.5)
    ax1.axvline(x = avg_r_ub[dump_to_plot], ls = '--', lw = 1., color = cb(4))
    ax1.axvline(x = avg_r_ub[dump_to_plot] - 2*sigmam_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax1.axvline(x = avg_r_ub[dump_to_plot] + 2*sigmap_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax1.set_xlim((r1 - 0.4, r2))
    #ax1.set_ylim((4e-1, 4e3))
    ax1.set_xlabel(r'r / Mm')
    ax1.set_ylabel(r'N')
    ax1.minorticks_off()
    fig.subplots_adjust(hspace = 0)
    pl.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible = False)
    
def get_vr_max_evolution(prof, cycles, r1, r2):

    r = prof.get('Y', fname = cycles[0], resolution = 'l')
    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    t = np.zeros(len(cycles))
    vr_max = np.zeros(len(cycles))
    for k in range(len(cycles)):
        t[k] = prof.get('t', fname = cycles[k], resolution = 'l')[-1]
        vr_rms  = prof.get('EkY', fname = cycles[k], resolution = 'l')**0.5
        vr_max[k] = np.max(vr_rms[idx2:idx1])

    return t, vr_max

def vr_evolution(cases, ymin = 4., ymax = 8.,ifig = 10):

    '''
    Compares the time evolution of the max RMS velocity of different runs

    Plots Fig. 12 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    cases : string array
        directory names that contain the runs you wish to compare
        assumes ppm.set_YProf_path was used, thus only the directory name
        is necessary and not the full path ex. cases = ['D1', 'D2']
    ymin, ymax : float
        Boundaries of the range to look for vr_max in

    '''
    sparse = 1
    markevery = 25
    cb = utils.colourblind
    yy = 0
    
    pl.figure(ifig)
    
    for case in cases:

        try:
            prof = yprofile(ppm_path+case)
        except ValueError:
            print "have you set the yprofile filepath using ppm.set_YProf_path?"

        cycles = range(prof.cycles[0], prof.cycles[-1], sparse)
        t, vr_max = get_vr_max_evolution(prof, cycles, ymin, ymax)
        pl.plot(t/60.,  1e3*vr_max,  color = cb(yy),\
                 marker = 's', markevery = markevery, label = case)
        yy += 1

    pl.title('')
    pl.xlabel('t / min')
    pl.ylabel(r'v$_r$ / km s$^{-1}$')
    pl.legend(loc = 0)
   
def luminosity_for_dump(path, get_t = False):
    
    '''
    
    Takes a directory and returns luminosity and time vector with entry
    corresponding to each file in the dump
    
    !Can take both rprofile and yprofile filepaths
    
    Parameters
    ----------
    path : string
        Filepath for the dumps, rprofile or yprofile dumps.
    get_t: Boolean
        Return the time vector or not 

    Returns
    -------
    t : 1*ndumps array
        time vector [t/min]
    L_H : 1*ndumps array
        luminosity vector [L/Lsun]
    
    '''
    
    yprof = yprofile(path)
    
    try:
        dumps = np.arange(min(yprof.ndumpDict.keys()),\
                          #min(yprof.ndumpDict.keys())+100,1)
                          max(yprof.ndumpDict.keys()) + 1, 1)
        is_yprofile = True
    except:
        
        rp_set = rprof.rprofile_set(path)
        dumps = range(rp_set.dumps[0],\
                      #rp_set.dumps[0]+100,1)
                      rp_set.dumps[-1]+1,1)
        r_rp = rp_set.get_dump(dumps[0]).get('y')
        dV_rp = 4.*np.pi*r_rp**2*cdiff(r_rp)
        is_yprofile = False
        yprof = yprofile(path.replace('RProfiles','YProfiles'))

    airmu = 1.39165
    cldmu = 0.725
    fkair = 0.203606102635
    fkcld = 0.885906040268
    AtomicNoair = 6.65742024965
    AtomicNocld = 1.34228187919

    patience0 = 5
    patience = 10

    nd = len(dumps)
    t = np.zeros(nd)
        
    L_H = np.zeros(nd)
    
    t00 = time.time()
    t0 = t00
    k = 0
    for i in range(nd):
        if is_yprofile:
            if get_t:
                t[i] = yprof.get('t', fname = dumps[i], resolution = 'l')[-1]

            L_H[i] = yprof.get('L_C12pg', fname = dumps[i], resolution = 'l', airmu = airmu, \
                                  cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair, \
                                  AtomicNocld = AtomicNocld, corr_fact = 1.5)
        
        else:
            
            rp = rp_set.get_dump(dumps[i])
            enuc_rp = rp.get_table('enuc')
            if get_t:
                t[i] = rp.get('time')  
            # It looks like we do not need to make a correction for the heating bug here. Strange!!!
            L_H[i] = np.sum(enuc_rp[0, :, 0]*dV_rp)

        t_now = time.time()
        if (t_now - t0 >= patience) or \
           ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
            time_per_dump = (t_now - t00)/float(i + 1)
            time_remaining = (nd - i - 1)*time_per_dump
            print 'Processing will be done in {:.0f} s.'.format(time_remaining)
            t0 = t_now
            k += 1
    if get_t:        
        return t, L_H
    else:
        return L_H
    
def plot_luminosity(L_H_yp,L_H_rp,t):
    
    '''
    
    Plots two luminosity vectors against the same time vector
        
    Parameters
    ----------
    L_H_yp : 1 *ndumps vector
             Luminosity vector for yprofile can be generated by luminosity_for_dump
    L_H_rp : 1 *ndumps vector
             Luminosity vector for rprofile can be generated by luminosity_for_dump
    t : array size(L_H_rp)
             time vector to be plotted on the x-axis
    '''
    cb = utils.colourblind
    
    L_He = 2.25*2.98384E-03
 
    ifig = 1; pl.close(ifig); pl.figure(ifig)
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_yp, color = cb(6), \
                 zorder = 2, label = r'L$_\mathrm{H}$')
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

    ifig = 2; pl.close(ifig); pl.figure(ifig)
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_yp, color = cb(5), \
                 lw = 2., zorder = 2, label = r'L$_\mathrm{H,yp}$')
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_rp, color = cb(6), \
                 zorder = 4, label = r'L$_\mathrm{H,rp}$')
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

def L_H_L_He_comparison(cases, ifig=101):
    
    yprofs = {}
    res = {}
    
    for case in cases:
        
        try:
            yprofs[case] = yprofile(ppm_path+case)
        except ValueError:
            print "have you set the yprofile filepath using ppm.set_YProf_path?"
        
        r = yprofs[case].get('Y', fname=0, resolution='l')
        res[case] = 2*len(r)
        
    airmu = 1.39165
    cldmu = 0.725
    fkair = 0.203606102635
    fkcld = 0.885906040268
    AtomicNoair = 6.65742024965
    AtomicNocld = 1.34228187919

    cb = utils.colourblind

    patience0 = 5
    patience = 60

    sparse = 1
    dumps = {}
    nd = {}
    t = {}
    L_H = {}
    L_He = 2.25*2.98384E-03

    for this_case in cases:
        print 'Processing {:s}...'.format(this_case)

        dumps[this_case] = np.arange(min(yprofs[case].ndumpDict.keys()),\
           max(yprofs[case].ndumpDict.keys()) + 1, sparse)
        #dumps[this_case] = np.arange(min(yprofs[case].ndumpDict.keys()),\
        #   min(yprofs[case].ndumpDict.keys()) + 10, sparse)
        #n_dumps = len(rp_set.dumps)
        nd[this_case] = len(dumps[this_case])
        t[this_case] = np.zeros(nd[this_case])
        L_H[this_case] = np.zeros(nd[this_case])

        t00 = time.time()
        t0 = t00
        k = 0
        for i in range(nd[this_case]):
            t[this_case][i] = yprofs[this_case].get('t', fname = dumps[this_case][i], \
                              resolution = 'l')[-1]
            L_H[this_case][i] = yprofs[this_case].get('L_C12pg', fname = dumps[this_case][i], \
                                resolution = 'l', airmu = airmu, cldmu = cldmu, \
                                fkair = fkair, fkcld = fkcld,  AtomicNoair = AtomicNoair, \
                                AtomicNocld = AtomicNocld, corr_fact = 1.5)

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd[this_case] - i - 1)*time_per_dump
                print 'Processing will be done in {:.0f} s.'.format(time_remaining)
                t0 = t_now
                k += 1

    pl.close(ifig); pl.figure(ifig)
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
        label = r'L$_\mathrm{He}$')
    
    for this_case in cases:
        lbl = r'{:s} $\left({:d}^3\right)$'.format(this_case, res[this_case])
        pl.semilogy(t[this_case]/60., (1e43/ast.lsun_erg_s)*L_H[this_case], \
            ls = '-', color = cb(cases.index(this_case)), marker= 's', markevery=250/sparse, \
            label = case)
        
    pl.xlabel('t / min')
    pl.ylabel(r'L$_H$ / L$_\odot$')
    pl.legend(loc=0, ncol=2)
    pl.tight_layout()
    pl.savefig('L_H-L_He_F4_F5_F13.pdf')
    
def get_upper_bound(data_path, dump_to_plot, r1, r2):

    '''
    Returns information about the upper convective boundary
    
    Parameters
    ----------
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    
    Output
    ------
    [all arrays]
    avg_r_ub : average radius of upper boundary
    sigmam_r_ub/sigmap_r_ub: 2 \sigma fluctuations in upper boundary
    r_ub : upper boundary
    t: time
    
    '''

    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut = res[1]
        dutdr = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)

    for k in range(n_dumps):
        devp = dev[:, k]
        devp = devp[devp >= 0]
        if len(devp) > 0:
            sigmap_r_ub[k] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmam_r_ub[k] = None

        devm = dev[:, k]
        devm = devm[devm <= 0]
        if len(devm) > 0:
            sigmam_r_ub[k] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_ub[k] = None
            
    return(avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t)
    
def plot_boundary_evolution(data_path, dump_to_plot, show_fits = False, r1=7.4, r2=8.4, t_fit_start=700):
    
    '''

    Displays the time evolution of the convective boundary

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    data_path = string
        data path
    show_fits = boolean
        show the fits used in finding the upper boundary
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    t_fit_start = int
        The time to start the fit for upper boundary fit takes 
        range t[t_fit_start:-1] and computes average boundary
    '''
    cb = utils.colourblind
    
    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))
    avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t = get_upper_bound(data_path, dump_to_plot, r1, r2)
    
    idx_fit_start = np.argmin(np.abs(t - t_fit_start))
    t_fit_start = t[idx_fit_start]

    # fc = fit coefficients
    fc_avg = np.polyfit(t[idx_fit_start:-1], avg_r_ub[idx_fit_start:-1], 1)
    avg_fit = fc_avg[0]*t + fc_avg[1]
    fc_plus = np.polyfit(t[idx_fit_start:-1], 2.*sigmap_r_ub[idx_fit_start:-1], 1)
    plus_fit = fc_plus[0]*t + fc_plus[1]
    fc_minus = np.polyfit(t[idx_fit_start:-1], 2.*sigmam_r_ub[idx_fit_start:-1], 1)
    minus_fit = fc_minus[0]*t + fc_minus[1]

    ifig = 5; pl.close(ifig); fig = pl.figure(ifig)#, figsize = (6.0, 4.7))
    for bucket in range(n_buckets):
        lbl = 'bucket data' if bucket == 0 else None
        pl.plot(t/60., r_ub[bucket, :], ls = '-', lw = 0.5, color = cb(3), \
                 label = lbl)
    pl.plot(t/60., avg_r_ub, ls = '-', lw = 1., color = cb(4),\
             label = 'mean')
    pl.plot(t/60., avg_r_ub + 2*sigmap_r_ub, ls = '--', lw = 1., \
             color = cb(4), label = r'2$\sigma$ fluctuations')
    pl.plot(t/60., avg_r_ub - 2*sigmam_r_ub, ls = '--', lw = 1., \
             color = cb(4))
    if show_fits:
        pl.plot(t/60., avg_fit, ls = '-', lw = 0.5, color = cb(4), \
                label = r'$\mathregular{linear\ fits}$')
        pl.plot(t/60., avg_fit + plus_fit, ls = '-', lw = 0.5, color = cb(4))
        pl.plot(t/60., avg_fit - minus_fit, ls = '-', lw = 0.5, color = cb(4))
    pl.xlim((0., np.max(t)/60.))
    #plt.ylim((7.4, 8.6))
    pl.xlabel('t / min')
    pl.ylabel(r'r$_\mathrm{ub}$ / Mm')
    pl.legend(loc = 0, frameon = False)
    #fig.tight_layout()

    print 'The fitting starts at t = {:.1f} s = {:.1f} min.'.format(t_fit_start, t_fit_start/60.)
    print ''
    print 'Average:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_avg[1], fc_avg[0])
    print ''
    print 'Positive fluctuations:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_plus[1], fc_plus[0])
    print ''
    print 'Negative fluctuations:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_minus[1], fc_minus[0])
