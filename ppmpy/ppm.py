#
# ppm.py - Tools for accessing and visualising PPMstar data.
#          Depends on the nugridpy package developed by the
#          NuGrid collaboration
# (c) 2010 - 2013 Daniel Alexander Bertolino Conti
# (c) 2011 - 2013 Falk Herwig
# (c) 2014 - 2015 Sam Jones, Falk Herwig, Robert Andrassy
# (c) 2016 - 2018 Robert Andrassy, Jericho O'Connell, Falk Herwig
# (c) 2018 - 2019 David Stephens, Falk Herwig


# TBD:
# * add getting computable quantities from `compute` method directly to get method, and simplify `rp_plot` accordingly
# * check that compute_m adds the core mass correctly by pulling the appropriate boundary conditions and calculating the
#   mass at the inner boundary correctly

# updates Nov 26 (FH):
#     * rp_plot can now plot computable quantities
#     * rprofgui now can plot computable quantities
#     * there is a new plot called plot_vrad_prof that makes a plot of all velocity
#       components for a given dump, or list of dumps (or times) with km/s y unit
#       in publication ready format, and print a pdf if requested.
#       compute_g to compute_m as it applies to all cases where compute_m is used


"""
ppmpy.ppm

PPM is a Python module for reading Yprofile-01-xxxx.bobaaa files as well as
some analysis involving Rprofile files and Mesa stellar models.

Simple session for working with ppmpy, here I assume user's working on

`astrohub <https://astrohub.uvic.ca/>`_

which is the intended environment for ppmpy.

If the user find any bugs or errors, please email us.


Examples
=========

Here is an example runthrough.


.. ipython::

    In [136]: from ppmpy import ppm
       .....: !ls /data/ppm_rpod2/YProfiles/

    In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
       .....: project = 'O-shell-M25'
       .....: ppm.set_YProf_path(data_dir+project)

    In [136]: ppm.cases

and

.. ipython::

    In [136]: D2=ppm.yprofile('D2')
       .....: D2.vprofs(100)

plots the data.

.. plot::

    import ppmpy.ppm as ppm
    D2 = ppm.yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D2')
    D2.vprofs(100)

"""

from __future__ import (division,
                        print_function)

from builtins import zip
from builtins import input
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div

from numpy import *
import numpy as np
from math import *
from nugridpy.data_plot import *
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib import gridspec
from matplotlib import cm
from matplotlib import colors
import nugridpy.mesa as ms
import os
import re
import nugridpy.constants as nuconst
import scipy.interpolate
import scipy.stats
from scipy import optimize
from scipy import integrate as integrate
import copy
from . import rprofile as bprof
from nugridpy import utils
cb = utils.colourblind
from dateutil.parser import parse
import time
import glob
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from datetime import date
import pickle
import collections

# The unit of G in the code is 10^{-3} g cm^3 s^{-2}.
G_code = nuconst.grav_const*1000.

# from rprofile import rprofile_reader

def prep_Yprofile_data(user="Paul", run="BW-Sakurai-1536-N13"):
    '''
    for given user and run create YProfile dir in run dir and link all
    YProfile data from dump directories

    this is for use on BlueWaters
    '''
    import subprocess
    login_names = {'Paul': 'pwoodwar',\
                   'Stou': 'sandalsk',\
                   'Falk': 'fherwig'  }
    run_dir = '/scratch/sciteam/v_evol'+login_names[user]+'/'+run
    data_dir = run_dir+'/YProfiles'
    mkdir_command = 'mkdir '+data_dir
    subprocess.call([mkdir_command],shell=True)
    remove_broken_links = 'find -L '+data_dir+' -type l -delete'
    subprocess.call([remove_broken_links],shell=True)
    link_command = 'ln -fs '+run_dir+'/????/YProfile-01/* '+data_dir
    subprocess.call([link_command],shell=True)
    return data_dir

def index_nearest_value(a,afind):
    '''Return index of value in a which is closest to afind

    Parameters
    ----------

    a : array
    afind : scalar
    '''
    aabs = abs(a-afind)
    return where(min(aabs) == aabs)[0][0]

def set_nice_params():
    fsize=14

    params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    'font.family':        'Times New Roman',
    'figure.facecolor':  'white',
    'font.size':     fsize,
    'legend.fontsize':   fsize*0.75,
    'xtick.labelsize':   fsize*0.8,
    'ytick.labelsize':   fsize*0.8,
    'ytick.minor.pad': 8,
    'ytick.major.pad': 8,
    'xtick.minor.pad': 8,
    'xtick.major.pad': 8,
    'text.usetex':       False}
    pl.rcParams.update(params)

def set_YProf_path(path,YProf_fname='YProfile-01-0000.bobaaa'):
    '''
    Set path to location where YProfile directories can be found.

    For example, set path to the swj/PPM/RUNS_DIR VOSpace directory
    as a global variable, so that it need only be set once during
    an interactive session; instances can then be loaded by
    refering to the directory name that contains YProfile files.

    ppm.ppm_path: contains path
    ppm.cases: contains dirs in path that contain file with name
    YProf_fname usually used to determine dirs with
    YProfile files
    '''
    global ppm_path, cases
    ppm_path = path
    cases = []
    for thing in os.listdir(ppm_path):
        dir_thing = os.path.join(ppm_path,thing)
        if os.path.isdir(dir_thing) and \
           os.path.isfile(os.path.join(ppm_path,thing,YProf_fname)):
            cases.append(thing)

def prof_compare(cases,ndump=None,yaxis_thing='FV H+He',ifig=None,num_type='ndump',
                 labels=None,logy=True):
    """
    Compare profiles of quantities from multiple PPM Yprofile instances at a
    given time of nump number.

    Parameters
    ----------
    cases : list
        list containing the Yprofile instances that you want to compare
    ndump : string or int, optional
        The filename, Ndump or time, if None it defaults to the
        last NDump.  The default is None.
    yaxis_thing : string, optional
        What quantity to plot on the y-axis.
        The default is 'FV H+He'
    ifig : int, optional
        Figure number. If None, chose automatically.
        The default is None.
    num_type : string, optional
        Designates how this function acts and how it interprets
        fname.  If numType is 'file', this function will get the
        desired attribute from that file.  If numType is 'ndump'
        function will look at the cycle with that ndump.  If
        numType is 'T' or 'time' function will find the _cycle
        with the closest time stamp.  The default is "ndump".
    labels : list, optional
        List of labels; one for each of the cases.
        If None, labels are simply indices.
        The default is None.
    logy : boolean, optional
        Should the y-axis have a logarithmic scale?
        The default is True.

    Examples
    --------

    .. ipython::

        In [136]: from ppmpy import ppm
           .....: data_dir = '/data/ppm_rpod2/YProfiles/'
           .....: project = 'O-shell-M25'
           .....: ppm.set_YProf_path(data_dir+project)

        @savefig prof_compare.png width=6in
        In [136]: D2=ppm.yprofile('D2')
           .....: D1=ppm.yprofile('D1')
           .....: ppm.prof_compare([D2,D1],10,labels = ['D1','D2'])


    """

    fsize=14

    params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    'font.family':        'Times New Roman',
    'figure.facecolor':  'white',
    'text.fontsize':     fsize,
    'legend.fontsize':   fsize,
    'xtick.labelsize':   fsize*0.8,
    'ytick.labelsize':   fsize*0.8,
    'text.usetex':       False}
    pl.rcParams.update(params)

    jline_offset=6
    if labels is None:
        labels=[]*len(cases)
    if ifig is None:
        pl.figure()
    else:
        pl.figure(ifig)
    labels = zeros(len(cases))
    i=0
    for Y in cases:
        j=i+jline_offset
        if labels is None:
            labels[i] = str(i)
        Y.plot('Y',yaxis_thing,fname=ndump,numtype=num_type,legend=labels[i],\
               logy=logy,shape=utils.linestyle(j)[0],markevery=utils.linestyle(j)[1])
        i += 1

def cdiff(x):
    # compute 2nd order centred differences
    dx = (np.roll(x, -1) - np.roll(x, 1))/2.

    # 1st order differences to correct the boundaries
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]

    return dx

def interpolate(x, y, x_new, kind='linear'):
    inverse_order = (x[-1] < x[0])
    xx = x[::-1] if inverse_order else x
    yy = y[::-1] if inverse_order else y

    if kind == 'linear':
        int_func = scipy.interpolate.interp1d(xx, yy, fill_value='extrapolate')
        y_new = int_func(x_new)
    elif kind == 'cubic':
        cs = scipy.interpolate.CubicSpline(xx, yy, extrapolate=False)
        y_new = cs(x_new)
    else:
        print("Error: Unknown interpolation kind '{:s}'.".format(kind))
        return None

    return y_new

def any2list(arg):
    if isinstance(arg, str):
        return [arg, ]
    try:
        return list(arg)
    except TypeError:
        return [arg, ]


class PPMtools:
    def __init__(self, verbose=3):
        '''
        Init method.

        Parameters
        ----------
        verbose: integer
            Verbosity level as defined in class Messenger.
        '''

        self.__messenger = Messenger(verbose=verbose)
        self.__isyprofile = isinstance(self, yprofile)
        self.__isRprofSet= isinstance(self, RprofSet)

        # This sets which method computes which quantity.
        self.__compute_methods = {'enuc_C12pg':self.compute_enuc_C12pg, \
                                  'Hp':self.compute_Hp, \
                                  'nabla_rho':self.compute_nabla_rho, \
                                  'nabla_rho_ad':self.compute_nabla_rho_ad, \
                                  'prad':self.compute_prad, \
                                  'pgas_by_ptot':self.compute_pgas_by_ptot, \
                                  'g':self.compute_g, \
                                  'N2':self.compute_N2, \
                                  'm':self.compute_m, \
                                  'mt':self.compute_mt, \
                                  'r4rho2':self.compute_r4rho2, \
                                  'rhodot_C12pg':self.compute_rhodot_C12pg, \
                                  'T9':self.compute_T9, \
                                  'T9corr':self.compute_T9corr, \
                                  '|Ur|':self.compute_Ur, \
                                  'Xcld':self.compute_Xcld, \
                                  'Xdot_C12pg':self.compute_Xdot_C12pg}
        self.__computable_quantities = self.__compute_methods.keys()

    def isyprofile(self):
        return self.__isyprofile

    def isRprofSet(self):
        return self.__isRprofSet

    def get_computable_quantities(self):
        '''
        Returns a list of computable quantities.
        '''

        # Return a copy.
        return list(self.__computable_quantities)

    def compute(self, quantity, fname, num_type='ndump', extra_args={}):
        if quantity in self.__computable_quantities:
            m = self.__compute_methods[quantity]
            return m(fname, num_type=num_type, **extra_args)
        else:
            self.__messenger.error("Unknown quantity '{:s}'.".format(quantity))
            print('The following quantities can be computed:')
            print(self.get_computable_quantities())

    def compute_enuc_C12pg(self, fname, num_type='ndump', fkair=None, fkcld=None, \
                           atomicnoair=None, atomicnocld=None, airmu=None, cldmu=None, \
                           T9corr_params={}, Q=1.944, corr_fact=1.):
        if self.__isyprofile:
            fv = self.get('FV H+He', fname=fname, num_type=num_type, resolution='l')
            rho = self.get('Rho', fname=fname, num_type=num_type, resolution='l')
            rhocld = self.get('Rho H+He', fname=fname, num_type=num_type, \
                              resolution='l')
            rhoair = self.get('RHOconv', fname=fname, num_type = num_type, \
                              resolution='l')

            if fkair is None or fkcld is None or atomicnoair is None or \
               atomicnocld is None:
                self.__messenger.error('Yprofiles do not contain the values of fkair, '
                                       'fkcld, atomicnoair, and atomicnocld. You have '
                                       'to provide via optional parameters.')
                return None

        if self.__isRprofSet:
            fv = self.get('FV', fname=fname, num_type=num_type, resolution='l')
            rho = self.get('Rho0', fname=fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname=fname, num_type=num_type, resolution='l')

            # We allow the user to replace the following values read from the .rprof
            # files should those ever become wrong/unavailable.
            if fkair is None:
                fkair = self.get('fkair', fname, num_type=num_type)

            if fkcld is None:
                fkcld = self.get('fkcld', fname, num_type=num_type)

            if atomicnoair is None:
                atomicnoair = self.get('atomicnoair', fname, num_type=num_type)

            if atomicnocld is None:
                atomicnocld = self.get('atomicnocld', fname, num_type=num_type)

            if airmu is None:
                airmu = self.get('airmu', fname, num_type=num_type)

            if cldmu is None:
                cldmu = self.get('cldmu', fname, num_type=num_type)

            # Individual densities of the two ideal gases assuming thermal and
            # pressure equlibrium.
            rhocld = rho/(fv + (1. - fv)*airmu/cldmu)
            rhoair = rho/(fv*cldmu/airmu + (1. - fv))

        # compute_T9corr() returns the uncorrected temperature if no correction
        # parameters are supplied.
        T9 = self.compute_T9corr(fname=fname, num_type=num_type, airmu=airmu, \
                                 cldmu=cldmu, **T9corr_params)
        TP13 = T9**(1./3.)
        TP23 = TP13*TP13
        TP12 = np.sqrt(T9)
        TP14 = np.sqrt(TP12)
        TP32 = T9*TP12
        TM13 = 1./TP13
        TM23 = 1./TP23
        TM32 = 1./TP32

        T9inv = 1. / T9
        thyng = 2.173913043478260869565 * T9
        vc12pg = 20000000.*TM23 * np.exp(-13.692*TM13 - thyng*thyng)
        vc12pg = vc12pg * (1. + T9*(9.89-T9*(59.8 - 266.*T9)))
        thing2 = vc12pg + TM23 * 1.0e5 * np.exp(-4.913*T9inv) + \
                          TM32 * 4.24e5 * np.exp(-21.62*T9inv)

        thing2[np.where(T9 < .0059)] = 0.
        thing2[np.where(T9 > 0.75)] = 200.

        vc12pg = thing2 * rho * 1000.

        v = 1./ rho
        atomicnocldinv = 1./atomicnocld
        atomicnoairinv = 1./atomicnoair

        Y1 =  rhocld * fv * v * atomicnocldinv
        Y2 =  rhoair * (1. - fv) * v * atomicnoairinv

        CN = 96.480733
        # RA (6 Sep 2018): I have to comment out the following part of the code,
        # because the right instantaneous value of dt is currently unavailable.
        #
        #reallysmall=1e-14
        #if use_dt:
        #    # We want the average rate during the current time step.
        #    # If the burning is too fast all the stuff available burns
        #    # in a fraction of the time step. We do not allow to burn
        #    # more than what is available, so the average burn rate is
        #    # lower than then instantaneous one.
        #    thing3 = fkair * Y1 * Y2 * vc12pg * dt
        #    thing3[where(Y1 < reallysmall)] = 0.
        #    thing2 = np.min(np.array((thing3, Y1)), axis = 0)
        #
        #    #for i in range(len(Y1)):
        #    #    print '{:d}   {:.1e}   {:.1e}   {:.1e}'.format(i, Y1[i], thing3[i], Y1[i]/thing3[i])
        #
        #    DY = fkcld * thing2
        #    enuc = DY * rho * CN * Q / dt
        #else:

        # We want the instantaneous burning rate. This does not
        # depend on how much stuff is available.
        thing3 = fkair * Y1 * Y2 * vc12pg

        DY = fkcld * thing3
        enuc = DY * rho * CN * Q

        # This factor can account for the heating bug if present.
        enuc *= corr_fact

        return enuc

    def compute_Hp(self, fname, num_type='ndump'):
        if self.__isyprofile:
            r = self.get('Y', fname, num_type=num_type, resolution='l')
            p = self.get('P', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            r = self.get('R', fname, num_type=num_type, resolution='l')
            p = self.get('P0', fname, num_type=num_type, resolution='l') + \
                self.get('P1', fname, num_type=num_type, resolution='l')

        Hp = np.abs(cdiff(r))/(np.abs(cdiff(np.log(p))) + 1e-100)
        return Hp

    def compute_Ur(self, fname, num_type='ndump'):
        if self.__isyprofile:
            print("Nothing to compute for YProfile ....")
            return None
        if self.__isRprofSet:
            Ut = self.get('|Ut|', fname, num_type=num_type, resolution='l')
            U = self.get('|U|', fname, num_type=num_type, resolution='l')
        Ur = np.sqrt(U**2 - Ut**2)
        return Ur


    def compute_nabla_rho(self, fname, num_type='ndump'):
        if self.__isyprofile:
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')
            p = self.get('P', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            rho = self.get('Rho0', fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname, num_type=num_type, resolution='l')
            p = self.get('P0', fname, num_type=num_type, resolution='l') + \
                self.get('P1', fname, num_type=num_type, resolution='l')

        nabla_rho = cdiff(np.log(rho))/(cdiff(np.log(p)) + 1e-100)
        return nabla_rho

    def compute_nabla_rho_ad(self, fname, num_type='ndump', radeos=True):
        if radeos:
            beta = self.compute_pgas_by_ptot(fname, num_type=num_type)
            gamma3 = 1. + (2./3.)*(4. - 3.*beta)/(8. - 7.*beta)
            gamma1 = beta + (4. - 3.*beta)*(gamma3 - 1.)
            nabla_rho_ad = 1./gamma1
        else:
            if self.__isyprofile:
                r = self.get('Y', fname, num_type=num_type, resolution='l')

            if self.__isRprofSet:
                r = self.get('R', fname, num_type=num_type, resolution='l')

            nabla_rho_ad = (3./5.)*np.ones(len(r))

        return nabla_rho_ad

    def compute_prad(self, fname, num_type='ndump'):
        if self.__isyprofile:
            print('compute_prad() not implemented for YProfile input.')
            return None

        if self.__isRprofSet:
            T9 = self.get('T9', fname, num_type=num_type, resolution='l')

        # rad_const = 7.56577e-15 erg/cm^3/K^4
        rad_const = 7.56577e-15/1e43*(1e8)**3*(1e9)**4
        prad = (rad_const/3.)*T9**4
        return prad

    def compute_pgas_by_ptot(self, fname, num_type='ndump'):
        if self.__isyprofile:
            print('compute_pgas_by_ptot() not implemented for YProfile input.')
            return None

        if self.__isRprofSet:
            ptot = self.get('P0', fname, num_type=num_type, resolution='l') + \
                   self.get('P1', fname, num_type=num_type, resolution='l')

        prad = self.compute_prad(fname, num_type=num_type)
        pgas = ptot - prad
        return pgas/ptot

    def compute_g(self, fname, num_type='ndump'):
        if self.__isyprofile:
            r = self.get('Y', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            r = self.get('R', fname, num_type=num_type, resolution='l')

        m = self.compute_m(fname, num_type=num_type)

        g = G_code*m/r**2
        return g

    def compute_N2(self, fname, num_type='ndump', radeos=True):
        if self.__isyprofile:
            if radeos:
                print('radeos option not implemented for YProfile input.')
                return None
            r = self.get('Y', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            r = self.get('R', fname, num_type=num_type, resolution='l')

        g = self.compute_g(fname, num_type=num_type)
        Hp = self.compute_Hp(fname, num_type=num_type)
        nabla_rho = self.compute_nabla_rho(fname, num_type=num_type)
        nabla_rho_ad = self.compute_nabla_rho_ad(fname, num_type=num_type,
                       radeos=radeos)
        N2 = (g/Hp)*(nabla_rho - nabla_rho_ad)

        return N2

    def compute_m(self, fname, num_type='ndump'):
        if self.__isyprofile:
            r = self.get('Y', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            r = self.get('R', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho0', fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname, num_type=num_type, resolution='l')

        dm = -4.*np.pi*r**2*cdiff(r)*rho
        m = np.cumsum(dm)

        # We store everything from the surface to the core.
        m = m[-1] - m
        self.__messenger.warning('WARNING: PPMtools.compute_m() integrates mass from r = 0.\n'+\
              'This will not work for shell setups and wrong gravity will be returned.')
        return m

    def compute_mt(self, fname, num_type='ndump'):
        m = self.compute_m(fname, num_type=num_type)
        mt = m[0] - m

        return mt

    def compute_r4rho2(self, fname, num_type='ndump'):
        if self.__isyprofile:
            r = self.get('Y', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            r = self.get('R', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho0', fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname, num_type=num_type, resolution='l')

        return r**4*rho**2

    def compute_rhodot_C12pg(self, fname, num_type='ndump', fkair=None, fkcld=None, \
                             atomicnoair=None, atomicnocld=None, airmu=None, cldmu=None, \
                             T9corr_params={}):
        # We allow the user to replace the following values read from the .rprof
        # files should those ever become wrong/unavailable.
        if fkcld is None:
            fkcld = self.get('fkcld', fname, num_type=num_type)

        if atomicnocld is None:
            atomicnocld = self.get('atomicnocld', fname, num_type=num_type)

        # It does not matter what Q value we use, but we have to make sure that
        # compute_enuc_C12pg() uses the same Q as we use in this method.
        Q = 1.944
        corr_fact = 1.
        enuc = self.compute_enuc_C12pg(fname, num_type=num_type, fkair=fkair, \
                                       fkcld=fkcld, atomicnoair=atomicnoair, \
                                       atomicnocld=atomicnocld, airmu=airmu, \
                                       cldmu=cldmu, T9corr_params=T9corr_params, \
                                       Q=Q)

        # MeV in code units.
        MeV = 1.60218e-6/1e43

        # Reaction rate per unit volume.
        ndot = enuc/(Q*MeV)

        # Remember: atomicno is actually the mass number.
        # fkcld is the number fraction of H nucleii in the 'cloud' fluid.
        # X_H is the mass fraction of H in the 'cloud' fluid.
        atomicnoH = 1.
        X_H = fkcld*atomicnoH/atomicnocld

        # Atomic mass unit in PPMstar units.
        amu = 1.660539040e-24/1e27

        # The nominator is the mass burning rate of H per unit volume.
        # PPMstar cannot remove H from the 'cloud' fluid. When some mass M_H
        # of H is burnt, PPMstar actually removes M_H/X_H units of mass of
        # the 'cloud' fluid. The sign is negative, because mass is removed.
        rhodot = -atomicnoH*amu*ndot/X_H

        return rhodot

    def compute_T9(self, fname, num_type='ndump', airmu=None, cldmu=None):
        if self.__isyprofile:
            if (airmu is None or cldmu is None):
                self.__messenger.error('airmu and cldmu parameters are '
                                       'required to compute T9 for a '
                                       'yprofile.')

            fv = self.get('FV H+He', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')
            p = self.get('P', fname, num_type=num_type, resolution='l')

        muair = airmu
        mucld = cldmu

        if self.__isRprofSet:
            # We allow the user to replace the mu values read from the .rprof
            # files should those ever become wrong/unavailable.
            if muair is None:
                muair = self.get('airmu', fname, num_type=num_type)

            if mucld is None:
                mucld = self.get('cldmu', fname, num_type=num_type)

            fv = self.get('FV', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho0', fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname, num_type=num_type, resolution='l')
            p = self.get('P0', fname, num_type=num_type, resolution='l') + \
                self.get('P1', fname, num_type=num_type, resolution='l')

        # mu of a mixture of two ideal gases coexisting in thermal and
        # pressure equlibrium.
        mu = (1. - fv)*muair + fv*mucld

        # Gas constant as defined in PPMstar.
        Rgasconst = 8.314462

        # p = rho*Rgasconst*T9/mu
        T9 = (mu/Rgasconst)*(p/rho)

        return T9

    def compute_T9corr(self, fname, num_type='ndump', kind=0, params=None, \
                       airmu=None, cldmu=None):
        T9 = self.compute_T9(fname, num_type=num_type, airmu=airmu, \
                             cldmu=cldmu)

        T9corr = None
        if kind == 0:
            T9corr = T9
        elif kind == 1:
            if params is None or 'a' not in params or 'b' not in params:
                self.__messenger.error("Parameters 'a' and 'b' are required "
                                       "with kind = 1 (T9corr = a*T9**b).")
            else:
                T9corr = params['a']*T9**params['b']
        else:
            self.__messenger.error("T9 correction of kind = {:d} does "
                                   "not exist.".format(kind))

        return T9corr

    def compute_Xcld(self, fname, num_type='ndump'):
        # Different variables are available depending on data souce, so Xcld
        # is computed in different ways. Both results will be biased, although
        # in different ways, because the average of a product is not the
        # product of the corresponding averages.

        if self.__isyprofile:
            fv = self.get('FV H+He', fname, num_type=num_type, resolution='l')
            rho_cld = self.get('Rho H+He', fname, num_type=num_type, resolution='l')
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')
            Xcld = fv*rho_cld/rho

        if self.__isRprofSet:
            fv = self.get('FV', fname, num_type=num_type, resolution='l')
            airmu = self.get('airmu', fname, num_type=num_type)
            cldmu = self.get('cldmu', fname, num_type=num_type)
            Xcld = fv/((1. - fv)*(airmu/cldmu) + fv)

        return Xcld

    def compute_Xdot_C12pg(self, fname, num_type='ndump', fkair=None, fkcld=None, \
                           atomicnoair=None, atomicnocld=None, airmu=None, cldmu=None, \
                           T9corr_params={}):
        rhodot = self.compute_rhodot_C12pg(fname=fname, num_type=num_type, fkair=fkair, \
                                           fkcld=fkcld, atomicnoair=atomicnoair, \
                                           atomicnocld=atomicnocld, airmu=airmu, \
                                           cldmu=cldmu, T9corr_params=T9corr_params)

        if self.__isyprofile:
            rho = self.get('Rho', fname, num_type=num_type, resolution='l')

        if self.__isRprofSet:
            rho = self.get('Rho0', fname, num_type=num_type, resolution='l') + \
                  self.get('Rho1', fname, num_type=num_type, resolution='l')

        Xdot = rhodot/rho

        return Xdot

    def average_profiles(self, fname, var, num_type='ndump', func=None, \
                         lagrangian=False, data_rlim=None, extra_args={}):
        '''
        Method to compute time-averaged radial profiles of any quantity.
        Individual profiles can be stacked either in an Eulerian or in a
        Lagrangian frame.

        Parameters
        ----------
        fname : int or list
            Cycle number(s) or time value(s) to be used to in the time-
            averaging.
        var : str or list
            Name of the variable whose radial profiles are to be averaged.
            This can be anything if `func` is not None.
        num_type : str
            Type of fname, see yprofile.get() and RProf.get().
        func : function
            func(fname, num_type=num_type, **extra_args) can be any user-
            defined function that returns a radial profile. Any number of
            parameters can be passed to this function via `extra_args`.
            `var` should be a single string if func is not None.
        lagrangian : bool
            All radial profiles will be interpolated on the mass scale `mt`
            (integrated from the top inwards) that corresponds to t = 0.
        data_rlim : list
            List with two elements specifying what radial range should be
            used in the analysis. If `lagrangian` is True the mass range
            corresponding to this radial range at t = 0 will be used.
        extra_args : dictionary
            Extra arguments needed for either `func` or `self.compute` to
            work.

        Returns
        -------
        avg_profs : dictionary
            Dictionary containing the time-averaged profiles, the radial
            scale and, if `lagrangian` is True, also the mass scale `mt`
            (integrated from the top inwards).
        '''
        fname_list = any2list(fname)
        var_list = any2list(var)
        avg_profs = {}

        if self.__isyprofile:
            radius_variable = 'Y'
            r = self.get(radius_variable, fname_list[0], resolution='l')
            gettable_variables = self.getDCols()

        if self.__isRprofSet:
            radius_variable = 'R'
            r = self.get(radius_variable, fname_list[0], resolution='l')
            rp = self.get_dump(fname_list[0])
            gettable_variables = rp.get_anyr_variables()

        if radius_variable not in var_list:
            var_list.append(radius_variable)

        computable_quantities = self.__computable_quantities

        data_slice = range(0, len(r))
        if data_rlim is not None:
            idx0 = np.argmin(np.abs(r - data_rlim[0]))
            idx1 = np.argmin(np.abs(r - data_rlim[1]))

            # We store everything from the surface to the core.
            data_slice = range(idx1, idx0+1)

        if lagrangian:
            # Get the initial mass scale. Everything will be interpolated onto
            # this scale. We use mt, i.e. mass integrated from the top, because
            # the density stratification is extreme in some cases. There are
            # also some cases (PPMstar 1.0 run), in which some mass leaks in
            # through the inner crystal sphere and the total mass is not
            # conserved. There is no such issue with the outer crystal sphere.
            mt0 = self.compute('mt', 0, num_type='t')
            if data_rlim is not None:
                mt0 = mt0[data_slice]
            avg_profs['mt'] = mt0

        for v in var_list:
            avg_profs[v] = np.zeros(data_slice[-1] - data_slice[0] + 1)

            for i, fnm in enumerate(fname_list):
                if func is not None:
                    data = func(fnm, num_type=num_type, **extra_args)
                elif v in gettable_variables:
                    data = self.get(v, fnm, num_type=num_type, resolution='l')
                elif v in computable_quantities:
                    data = self.compute(v, fnm, num_type=num_type, \
                                        extra_args=extra_args)
                else:
                    self.__messenger.warning("Unknown quantity '{:s}'".\
                         format(v))
                    break

                if lagrangian:
                    # Interpolate everything on the initial mass scale.
                    mt = self.compute('mt', fnm, num_type=num_type)
                    data = interpolate(mt, data, mt0)

                avg_profs[v] += data

            avg_profs[v] /= float(len(fname_list))

        return avg_profs

    def DsolveLgr(self, cycles1, cycles2, var='Xcld', src_func=None, src_args={}, \
                  src_correction=False, integrate_upwards=False, data_rlim=None, \
                  show_plots=True, ifig0=1, run_id='', logmt=False, mtlim=None, \
                  rlim=None, plot_var=True, logvar=False, varlim=None, \
                  sigmalim=None, Dlim=None, fit_rlim=None):
        '''
        Method that inverts the Lagrangian diffusion equation and computes
        the diffusion coefficient based on radial profiles of the mass
        fraction Xcld of the 'cloud' fluid (although other variables can
        also be used) at two different points in time. The diffusion
        coefficient is assumed to vanish at the outermost data point (or
        innermost if integrate_upwards == True).

        Parameters
        ----------
        cycles1 : int or list
            Cycle number(s) to be used to construct the initial radial
            profile of `var`.
        cycles2 : int or list
            Cycle number(s) to be used to construct the final radial
            profile of `var`.
        var : str
            Name of the variable whose radial profiles are to be used in
            the analysis.
        src_func : function
            src_func(fname, num_type=num_type, **extra_args) should return
            the radial profile of the rate of change of `var` for the point
            in time specified by `fname` and `num_type`. This function will
            be called with any number of extra arguments specified in the
            dictionary `src_args`.
        src_args : dictionary
            Extra arguments needed for `src_func` to work.
        src_correction : bool
            Because `src_func` is only sampled at some number of discrete
            points in time, its discrete time integral will not exactly
            match the change in the radial profiles of `var`, which will
            appear in the analysis as some diffusive flux unaccounted for.
            This option allows the user to have `src_func` multiplied by
            a factor that guarantees perfect balance. This factor is
            reported in the output and a warning will be issued if it
            differs from unity significantly (<2./3. or > 3./2.).
        integrate_upwards : bool
            If True the usual integration for the downward diffusive flux
            starting from the outermost point inwards will be done from the
            innermost point outwards.
        data_rlim : list
            List with two elements specifying what radial range should be
            used in the analysis. The mass range corresponding to this
            radial range at t = 0 will be used.
        show_plots : bool
            Should any plots be shown?
        ifig0 : int
            The matplotlib figures used for graphical output will have
            indices of ifig0, ifig0+1, and ifig0+2.
        run_id : str
            Run identifier to be use in the title of the plots.
        logmt : bool
            Should the scale of mass integrated from the top be logarithmic?
        mtlim : list
            Plotting limits for the mass integrated from the top.
        rlim : list
            Plotting limits for the radius.
        plot_var : bool
            Should the profiles of `var` be shown in the plots?
        logvar : bool
            Should the scale of `var` be logarithmic?
        varlim : list
            Plotting limits for `var`.
        sigmalim : list
            Plotting limits for the Lagrangian diffusion coefficient `sigma`.
        Dlim : list
            Plotting limits for the Eulerian diffusion coefficient `D`.
        fit_rlim : list
            Radial range for the fitting of the f_CBM model.

        Returns
        -------
        res : dictionary
            Dictionary containing the Lagrangian and Eulerian diffusion
            coefficients (`sigma`, `D`), mass scale `mt`, time-averaged source
            function `xsrc`, times `t1` and `t2`, radial scales `r1` and `r2`,
            pressure scale height profiles `Hp1` and `Hp2`, and the profiles
            `x1` and `x2` of `var`, where 1 and 2 refer to the start and end
            points of the analysis.
        '''
        cycles1_list = any2list(cycles1)
        cycles2_list = any2list(cycles2)

        # Get average profiles and the average time corresponding to cycles1.
        res1 = self.average_profiles(cycles1_list, [var, 'Hp', 'r4rho2'], \
               lagrangian=True, data_rlim=data_rlim)
        mt = res1['mt']
        if self.__isyprofile:
            r1 = res1['Y']
        if self.__isRprofSet:
            r1 = res1['R']
        x1 = res1[var]
        Hp1 = res1['Hp']
        r4rho2_1 = res1['r4rho2']
        t1 = 0.
        for fn in cycles1_list:
            this_t = self.get('t', fn)
            t1 += this_t[-1] if self.__isyprofile else this_t
        t1 /= float(len(cycles1_list))

        # Get average profiles and the average time corresponding to cycles2.
        res2 = self.average_profiles(cycles2_list, [var, 'Hp', 'r4rho2'], \
               lagrangian=True, data_rlim=data_rlim)
        mt = res2['mt']
        if self.__isyprofile:
            r2 = res2['Y']
        if self.__isRprofSet:
            r2 = res2['R']
        x2 = res2[var]
        Hp2 = res2['Hp']
        r4rho2_2 = res2['r4rho2']
        t2 = 0.
        for fn in cycles2_list:
            this_t = self.get('t', fn)
            t2 += this_t[-1] if self.__isyprofile else this_t
        t2 /= float(len(cycles2_list))

        xsrc = np.zeros(len(mt))
        if src_func is not None:
            cyc1 = cycles1_list[len(cycles1_list)//2]
            cyc2 = cycles2_list[len(cycles2_list)//2]
            res = self.average_profiles(range(cyc1, cyc2+1), 'src_func', \
                                        func=src_func, extra_args=src_args, \
                                        lagrangian=True, data_rlim=data_rlim)
            xsrc = res['src_func']

        # We take a simple average whenever a single profile representative of
        # the whole solution interval is needed.
        r = 0.5*(r1 + r2)
        Hp = 0.5*(Hp1 + Hp2)
        r4rho2 = 0.5*(r4rho2_1 + r4rho2_2)

        # Technically speaking, the mass corrdinate mt[i] corresponds to
        # i-1/2, i.e. to the lower (outer) cell wall. We will make use of this
        # in calculating the cell mass dmt[i]. However, x1[i] and x2[i] are
        # cell averages, which correspond to cell-centred values to the 2nd
        # order of accuracy. This minute difference has probably little effect,
        # but let's get it right.
        dmt = np.roll(mt, -1) - mt
        dmt[-1] = dmt[-2] + (dmt[-2] - dmt[-3])
        mt = 0.5*(mt + np.roll(mt, -1))
        mt[-1] = mt[-2] + (mt[-2] - mt[-3])

        # We are going to compute the Lagrangian diffusion coefficient sigma
        # from the diffusion equation
        #
        # dx/dt = -d(-sigma*dx/dm)/dm + xsrc = -d(flux)/dm + xsrc,
        #
        # where `flux` is the diffusive flux and `xsrc` is a source term (to
        # represent e.g. nuclear burning). We integrate the equation
        # analytically from the outer (or inner if integrate_upwards == True)
        # crystal sphere, where we flux = 0, to a mass coordinate mt, which
        # gives us an explicit expression for sigma:
        #
        # sigma = (1./dxdm)*int_0^mt(dx/dt - xsrc)*dmt,
        #
        # where int_0^mt is a definite integral from 0 to mt.

        dxdt = (x2 - x1)/(t2 - t1)

        if src_correction:
            # If the source term is time dependent, we will never get the exact
            # time averages from a few samples and some mass will be unaccounted
            # for. We can correct for this by multiplying xsrc by the factor
            # src_corr_fact, which is computed by assuming that any change in the
            # sum of dxdt should be due to xsrc. This factor should be close to
            # unity and we will warn the user if it is not.
            dxdt_sum = np.sum(dxdt*dmt)
            xsrc_sum = np.sum(xsrc*dmt)
            src_corr_fact = dxdt_sum/xsrc_sum
            xsrc *= src_corr_fact
            print('Source term correction factor: ', src_corr_fact)
            if src_corr_fact < 2./3.:
                self.__messenger.warning('The value of the correction factor '
                                         'should be positive and close to '
                                         'unity. Please check your data and '
                                         'parameters.')
            if src_corr_fact > 3./2.:
                self.__messenger.warning('The value of the correction factor '
                                         'is suspiciously large. Please check '
                                         'your data and parameters.')

        # Downward diffusive flux at the top (inner) wall of the i-th cell.
        # iph = i + 0.5
        flux_iph = -np.cumsum((dxdt - xsrc)*dmt)

        if integrate_upwards:
            flux_iph -= flux_iph[-1]

        # flux_iph = sigma_iph*dx2dm_iph, where we take the gradient of x2,
        # because we want to solve an implicit diffusion equation.
        dx2dm_iph = (np.roll(x2, -1) - x2)/(np.roll(mt, -1) - mt)
        dx2dm_iph[-1] = dx2dm_iph[-2] + (dx2dm_iph[-2] - dx2dm_iph[-3])

        # dx2dm_iph will be zero in some places and we will not be able to
        # compute sigma_iph in those places. Let's avoid useless warnings.
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_iph = -flux_iph/dx2dm_iph

        # We should compute the cell-centred value of sigma, but let's not do
        # that, because it would increase the number of NaNs in the array if
        # there are any.
        sigma = sigma_iph

        # Convert the Lagrangian diffusion coefficient sigma to an Eulerian
        # diffusion coefficient D. The way we define r4rho2 may matter here.
        D = sigma/(16.*np.pi**2*r4rho2)
        if fit_rlim is not None:
            i0 = np.argmin(np.abs(r - fit_rlim[0]))
            i1 = np.argmin(np.abs(r - fit_rlim[1]))
            r_fit = r[i1:i0+1]
            D_data = D[i1:i0+1]
            fit_coeffs = np.polyfit(r_fit[D_data > 0], \
                        np.log(D_data[D_data > 0]), 1)
            D_fit = np.exp(r_fit*fit_coeffs[0] + fit_coeffs[1])
            f_CBM = -2./(fit_coeffs[0]*Hp[i0])

        if show_plots:
            var_lbl = var
            if var == 'Xcld':
                var_lbl = 'X'

            # Plot the diffusive flux.
            ifig=ifig0; pl.close(ifig); fig=pl.figure(ifig)
            ax1 = fig.gca()
            lns = []
            plt_func = ax1.semilogx if logmt else ax1.plot
            lns += plt_func((1e27/nuconst.m_sun)*mt, (1e27/nuconst.m_sun)*flux_iph, '-', \
                            color='k', label=r'flux')
            ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            if mtlim is not None:
                ax1.set_xlim(mtlim)
            ax1.set_xlabel(r'm$_\mathrm{top}$ / M$_\odot$')
            ax1.set_ylabel(r'Downward diffusive flux / M$_\odot$ s$^{-1}$')

            if plot_var:
                ax2 = ax1.twinx()
                plt_func = ax2.plot
                if logmt:
                    if logvar:
                        plt_func = ax2.loglog
                    else:
                        plt_func = ax2.semilogx
                elif logvar:
                        plt_func = ax2.semilogy

                lns += plt_func((1e27/nuconst.m_sun)*mt, x1, '-', color='b', \
                                label=var_lbl+r'$_1$')
                lns += plt_func((1e27/nuconst.m_sun)*mt, x2, '--', color='r', \
                                label=var_lbl+r'$_2$')

                if mtlim is not None:
                    ax2.set_xlim(mtlim)
                if varlim is not None:
                    ax2.set_ylim(varlim)
                else:
                    ax2.set_ylim((None, None))
                ax2.set_xlabel(r'm$_\mathrm{top}$ / M$_\odot$')
                ax2.set_ylabel(var_lbl)

            ttl = ''
            if run_id != '':
                ttl = run_id + ', '
            ttl += 'dumps '
            if len(cycles1_list) > 1:
                ttl += '[{:d}, {:d}]'.format(cycles1_list[0], cycles1_list[-1])
            else:
                ttl += '{:d}'.format(cycles1_list[0])
            ttl += ' - '
            if len(cycles2_list) > 1:
                ttl += '[{:d}, {:d}]'.format(cycles2_list[0], cycles2_list[-1])
            else:
                ttl += '{:d}'.format(cycles2_list[0])
            pl.title(ttl)
            lbls = [l.get_label() for l in lns]
            ncol = 2 if plot_var else 1
            pl.legend(lns, lbls, loc=0, ncol=ncol)

            # Plot the Lagrangian diffusion coefficient.
            ifig=ifig0+1; pl.close(ifig); fig=pl.figure(ifig)
            ax1 = fig.gca()
            lns = []
            plt_func = ax1.loglog if logmt else ax1.semilogy
            lns += plt_func((1e27/nuconst.m_sun)*mt, 1e54*sigma, '-', \
                            color='k', label=r'$\sigma$ > 0')
            lns += plt_func((1e27/nuconst.m_sun)*mt, -1e54*sigma, '--', \
                            color='k', label=r'$\sigma$ < 0')
            if mtlim is not None:
                ax1.set_xlim(mtlim)
            if sigmalim is not None:
                ax1.set_ylim(sigmalim)
            ax1.set_xlabel(r'm$_\mathrm{top}$ / M$_\odot$')
            ax1.set_ylabel(r'$\sigma$ / g$^2$ s$^{-1}$')

            if plot_var:
                ax2 = ax1.twinx()
                plt_func = ax2.plot
                if logmt:
                    if logvar:
                        plt_func = ax2.loglog
                    else:
                        plt_func = ax2.semilogx
                elif logvar:
                        plt_func = ax2.semilogy

                lns += plt_func((1e27/nuconst.m_sun)*mt, x1, '-', color='b', \
                                label=var_lbl+r'$_1$')
                lns += plt_func((1e27/nuconst.m_sun)*mt, x2, '--', color='r', \
                                label=var_lbl+r'$_2$')

                if mtlim is not None:
                    ax2.set_xlim(mtlim)
                if varlim is not None:
                    ax2.set_ylim(varlim)
                else:
                    ax2.set_ylim((None, None))
                ax2.set_xlabel(r'm$_\mathrm{top}$ / M$_\odot$')
                ax2.set_ylabel(var_lbl)

            pl.title(ttl)
            lbls = [l.get_label() for l in lns]
            ncol = 2 if plot_var else 1
            pl.legend(lns, lbls, loc=0, ncol=ncol)

            # Plot the Eulerian diffusion coefficient.
            ifig=ifig0+2; pl.close(ifig); fig=pl.figure(ifig)
            ax1 = fig.gca()
            lns = []
            # fit_rlim was here, FH moved it to before this plot section so
            # that f parameters can be returned even if show_plots = False
            if fit_rlim is not None:
                lbl = r'f$_\mathrm{{CBM}}$ = {:.3f}'.format(f_CBM)
                lns += ax1.semilogy(r_fit, 1e16*D_fit, '-', color='g', \
                            lw=4., label=lbl)
            lns += ax1.semilogy(r, 1e16*D, '-', color='k', label='D > 0')
            lns += ax1.semilogy(r, -1e16*D, '--', color='k', label='D < 0')
            if rlim is not None:
                ax1.set_xlim(rlim)
            if Dlim is not None:
                ax1.set_ylim(Dlim)
            ax1.set_xlabel('r / Mm')
            ax1.set_ylabel(r'D / cm$^2$ s$^{-1}$')

            if plot_var:
                ax2 = ax1.twinx()
                if logvar:
                    lns += ax2.semilogy(r1, x1, '-', color='b', \
                                label=var_lbl+r'$_1$')
                    lns += ax2.semilogy(r2, x2, '--', color='r', \
                                label=var_lbl+r'$_2$')
                else:
                    lns += ax2.plot(r1, x1, '-', color='b', \
                            label=var_lbl+r'$_1$')
                    lns += ax2.plot(r2, x2, '--', color='r', \
                            label=var_lbl+r'$_2$')
                if rlim is not None:
                    ax2.set_xlim(rlim)
                if varlim is not None:
                    ax2.set_ylim(varlim)
                else:
                    ax2.set_ylim((None, None))
                ax2.set_xlabel('r / Mm')
                ax2.set_ylabel(var_lbl)

            pl.title(ttl)
            lbls = [l.get_label() for l in lns]
            ncol = 2 if plot_var else 1
            pl.legend(lns, lbls, loc=0, ncol=ncol)

        if fit_rlim is not None:
            res = {'t1':t1, 't2':t2, 'mt':mt, 'r1':r1, 'r2':r2, 'Hp1':Hp1, \
             'Hp2':Hp2, 'x1':x1, 'x2':x2, 'xsrc':xsrc, 'sigma':sigma, \
             'D':1.e16*D, 'f_CBM':f_CBM}
        else:
            res = {'t1':t1, 't2':t2, 'mt':mt, 'r1':r1, 'r2':r2, 'Hp1':Hp1, \
             'Hp2':Hp2, 'x1':x1, 'x2':x2, 'xsrc':xsrc, 'sigma':sigma, \
             'D':1.e16*D}
        return res

    def bound_rad(self, cycles, r_min, r_max, var='ut', \
                  criterion='min_grad', var_value=None, \
                  return_var_scale_height=False, eps=1e-9):
        '''
        Method that finds the radius of a convective boundary.

        If the search is based on gradients, second-order centred finite
        differences are used to compute the gradient. The radius where
        the gradient reaches a local extreme (minimum or maximum depending
        on the value of criterion) is found. This radius is further
        refined by fitting a parabola to the three points around the
        local extreme and computing the radius at that the parabola
        reaches its extreme.

        If a certain value of var is searched for the method finds two
        cells between which var crosses the requested value and then it
        computes the radius of the actual crossing by linear interpolation.

        Parameters
        ----------
        cycles : list
            Cycle numbers to be used in the analysis.
        r_min : float
            Minimum radius to for the boundary search.
        r_max : float
            Maximum radius to for the boundary search.
        var : string
            Name of the variable to be used in the boundary search.
            This can be any variable contained in the data file or
            a special variable 'ut', which is the tangential velocity
            (computed in slightly different ways depending on the
            source of data).
        criterion : string
            Boundary definition criterion with the following options:
            'min_grad' : Search for a local minimum in d(var)/dr.
            'max_grad' : Search for a local maximum in d(var)/dr.
            'value' : Search for the radius where var == var_value.
        var_value : float
            Value of var to be searched for if criterion == 'value'.
        return_var_scale_height : bool
            Allows the user to have the scale height of var evaluated
            at the boundary radius to be returned as well.
        eps : float
            Smallest value considered non-zero in the search alorighm.

        Returns
        -------
        rb : 1D numpy array
            The boundary radius is returned if return_var_scale_height == False.
        rb, Hv: 1D numpy arrays
            The boundary radius rb and the scale height Hv of var evaluated
            at rb are returned if return_var_scale_height == True.
        '''
        cycle_list = any2list(cycles)
        rb = np.zeros(len(cycle_list))
        if return_var_scale_height:
            Hv = np.zeros(len(cycle_list))

        # The grid is assumed to be static, so we get the radial
        # scale only once at cycle_list[0].
        if self.__isyprofile:
            r = self.get('Y', cycle_list[0], resolution='l')

        if self.__isRprofSet:
            r = self.get('R', cycle_list[0], resolution='l')

        idx_r_min = np.argmin(np.abs(r - r_min))
        idx_r_max = np.argmin(np.abs(r - r_max))
        # Make sure that the range includes idx_r_min.
        idx_r_min = idx_r_min+1 if idx_r_min < len(r)-2 else len(r)-1

        for i, cyc in enumerate(cycle_list):
            if var == 'ut':
                if self.__isyprofile:
                    v = self.get('EkXZ', fname=cyc, resolution='l')**0.5

                if self.__isRprofSet:
                    v = self.get('|Ut|', fname=cyc, resolution='l')
            else:
                v = self.get(var, cyc, resolution='l')

            dvdr = cdiff(v)/cdiff(r)
            if criterion == 'min_grad' or criterion == 'max_grad':
                # The following code always looks for a local minimum in dv.
                # A local maximum is found by looking for a local minimum in
                # -dvdr.
                dvdr2 = dvdr
                if criterion == 'max_grad':
                    dvdr2 = -dvdr2

                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(dvdr2[idx_r_max:idx_r_min])
                r0 = r[idx0]

                # Try to pinpoint the radius of the local minimum by fitting
                # a parabola around r0.
                coefs = np.polyfit(r[idx0-1:idx0+2], dvdr2[idx0-1:idx0+2], 2)
                if np.abs(coefs[0]) > eps:
                    r00 = -coefs[1]/(2.*coefs[0])

                    # Only use the refined radius if it is within the three
                    # cells.
                    if r00 < r[idx0-1] and r00 > r[idx0+1]:
                        r0 = r00
            elif criterion == 'value':
                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(np.abs(v[idx_r_max:idx_r_min] - var_value[i]))
                r0 = r[idx0]

                if np.abs(v[idx0] - var_value[i]) > eps:
                    # 1st-order refinement.
                    if idx0 < idx_r_min and idx0 > idx_r_max:
                        if (v[idx0-1] < var_value[i] and v[idx0] > var_value[i]) or \
                           (v[idx0-1] > var_value[i] and v[idx0] < var_value[i]):
                            slope = v[idx0] - v[idx0-1]
                            t = (var_value[i] - v[idx0-1])/slope
                            r0 = (1. - t)*r[idx0-1] + t*r[idx0]
                        elif (v[idx0] < var_value[i] and v[idx0+1] > var_value[i]) or \
                            (v[idx0] > var_value[i] and v[idx0+1] < var_value[i]):
                            slope = v[idx0+1] - v[idx0]
                            t = (var_value[i] - v[idx0])/slope
                            r0 = (1. - t)*r[idx0] + t*r[idx0+1]
                        else:
                            r0 = r_max

            rb[i] = r0

            if return_var_scale_height:
                Hv_prof = np.abs(v)/(np.abs(dvdr) + 1e-100)

                idx0 = np.argmin(np.abs(r - r0))
                if r[idx0] > r0:
                    t = (r0 - r[idx0])/(r[idx0+1] - r[idx0])
                    Hv[i] = (1. - t)*Hv_prof[idx0] + t*Hv_prof[idx0+1]
                elif r[idx0] < r0:
                    t = (r0 - r[idx0-1])/(r[idx0] - r[idx0-1])
                    Hv[i] = (1. - t)*Hv_prof[idx0 - 1] + t*Hv_prof[idx0]
                else:
                    Hv[i] = Hv_prof[idx0]

        if return_var_scale_height:
            return rb, Hv
        else:
            return rb

    def entr_rate(self, cycles, r_min, r_max, var='ut', criterion='min_grad', \
                  var_value=None, offset=0., burn_func=None, burn_args={}, \
                  fit_interval=None, show_plots=True, ifig0=1, \
                  fig_file_name=None, return_time_series=False):
        '''
        Method for calculating entrainment rates.

        Parameters
        ----------
        cycles : list
            Cycle numbers to be used in the analysis.
        r_min : float
            Minimum radius to for the boundary search.
        r_max : float
            Maximum radius to for the boundary search.
        var : string
            Name of the variable to be used in the boundary search.
            See PPMtools.bound_rad().
        criterion : string
            Boundary definition criterion.
            See PPMtools.bound_rad() for allowed values.
        var_value : float
            Value of var to be searched for if criterion == 'value'.
            See PPMtools.bound_rad() for allowed values.
        offset : float
            Offset between the boundary radius and the upper integration
            limit for mass integration. The absolute value of the scale
            height of var evaluated at the boundary is used as a unit for
            the offset. Negative values shift the upper integration limit
            inwards and positive values outwards.
        burn_func : function
            burn_func(cycle, **burn_args) has to return the radial profile
            of the mass burning rate of the entrained fluid per unit volume
            for the cycle number passed as the first argument. For a data
            object obj, it can be e.g. obj.compute_rhodot_C12pg. The user
            can define a custom function and pass a reference to obj via
            burn_args.
        fit_interval : list
            List with the start and end time for the entrainment rate fit
            in seconds.
        burn_args : dictionary
            All arguments, with the exception of the cycle number, that
            burn_func() needs.
        show_plots : bool
            Should any plots be shown?
        ifig0 : int
            Figure index for the first plot shown.
        fig_file_name : string
            Name of the file, into which the entrainment rate plot will
            be saved.
        return_time_series : bool
            Switch to control what the method returns, see below.

        Returns
        -------
        mdot : float
            The entrainment rate is returned if return_time_series == False.
        time, mir, mb: 1D numpy arrays
            The entrained mass present and burnt inside the integration radius
            as a function of time. mb is an array of zeroes if burn_func is None.
        '''
        print('Computing boundary radius...')
        rb, Hv = self.bound_rad(cycles, r_min, r_max, var=var, criterion=criterion, \
                                var_value=var_value, return_var_scale_height=True)
        rt = rb + offset*np.abs(Hv)
        print('Done.')

        # The grid is assumed to be static, so we get the radial
        # scale only once at cycles[0].
        if self.__isyprofile:
            r = self.get('Y', cycles[0], resolution='l')

        if self.__isRprofSet:
            r = self.get('R', cycles[0], resolution='l')

        r_cell_top = 0.5*(np.roll(r, +1) + r)
        r_cell_top[0] = r_cell_top[1] + (r_cell_top[1] - r_cell_top[2])
        r3_cell_top = r_cell_top**3
        dr3 = r3_cell_top - np.roll(r3_cell_top, -1)
        dr3[-1] = dr3[-2] + (dr3[-2] - dr3[-3])
        dV = (4./3.)*np.pi*dr3

        t = np.zeros(len(cycles))
        burn_rate = np.zeros(len(cycles))
        # mir == mass inside radius (rt)
        mir = np.zeros(len(cycles))
        wct0 = time.time()
        print('Integrating entrained mass...')
        for i, cyc in enumerate(cycles):
            if self.__isyprofile:
                t[i] = self.get('t', fname=cyc, resolution='l')[-1]

            if self.__isRprofSet:
                t[i] = self.get('t', fname=cyc, resolution='l')

            Xcld = self.compute('Xcld', cyc)

            if self.__isyprofile:
                rho = self.get('Rho', cyc, resolution='l')

            if self.__isRprofSet:
                rho = self.get('Rho0', cyc, resolution='l') + \
                      self.get('Rho1', cyc, resolution='l')

            if burn_func is not None:
                rhodot = burn_func(cyc, **burn_args)

            # We assume that Xlcd and rho are cell averages, so we can
            # integrate all but the last cell using the rectangle rule.
            # We do piecewise linear reconstruction in the last cell and
            # integrate only over the portion of the cell where r < rt[i].
            # We should be as accurate as possible here, because Xcld is
            # usually much larger in this cell than in the bulk of the
            # convection zone.
            idx_top = np.argmax(r_cell_top < rt[i])
            dm = Xcld*rho*dV
            # mir == mass inside radius (rt)
            mir[i] = np.sum(dm[idx_top:])

            # Linear reconstruction. The integrand f = Xcld*rho is
            # assumed to take the form
            #
            #   f(r) = f0 + s*r
            #
            # within a cell. The slope s is computed using cell averages
            # in the cell's two next-door neighbours and the integral of
            # f(r)*dV over the cell's volume V_j is enforced to be
            # f_j*V_j, where j is the index of the cell.
            #
            j = idx_top - 1

            # jmo = j - 1
            # jpo = j + 1
            r_jmo = r[j-1]
            r_j   = r[j]
            r_jpo = r[j+1]
            f_jmo = Xcld[j-1]*rho[j-1]
            f_j   = Xcld[j  ]*rho[j  ]
            f_jpo = Xcld[j+1]*rho[j+1]

            # jmh = j - 0.5
            # jph = j + 0.5
            r_jmh = 0.5*(r_jmo + r_j)
            r_jph = 0.5*(r_j + r_jpo)
            f_jmh = 0.5*(f_jmo + f_j)
            f_jph = 0.5*(f_j + f_jpo)

            s = (f_jph - f_jmh)/(r_jph - r_jmh)
            V_j = (4./3.)*np.pi*(r_jmh**3 - r_jph**3)
            f0 = f_j - np.pi*(r_jmh**4 - r_jph**4)/V_j*s
            mir[i] += (4./3.)*np.pi*(rt[i]**3 - r_jph**3)*f0 + \
                      np.pi*(rt[i]**4 - r_jph**4)*s

            if burn_func is not None:
                # We have to use the minus sign, because rhodot < 0.
                burn_rate[i] = -np.sum(rhodot[idx_top:]*dV[idx_top:])

            wct1 = time.time()
            if wct1 - wct0 > 5.:
                print('{:d}/{:d} cycles processed.'.format(\
                      i+1, len(cycles)), end='\r')
                wct0 = wct1

        print('{:d}/{:d} cycles processed.'.format(i+1, len(cycles)))
        print('Done.')

        mir *= 1e27/nuconst.m_sun
        # mb = mass burnt
        mb = integrate.cumtrapz(burn_rate, x=t, initial=0.)
        mb *= 1e27/nuconst.m_sun
        mtot = mir + mb

        fit_idx0 = 0
        fit_idx1 = len(t)
        if fit_interval is not None:
            fit_idx0 = np.argmin(np.abs(t - fit_interval[0]))
            fit_idx1 = np.argmin(np.abs(t - fit_interval[1])) + 1
            if fit_idx1 > len(t):
                fit_idx1 = len(t)
        fit_range = range(fit_idx0, fit_idx1)

        # fc = fit coefficients
        rb_fc = np.polyfit(t[fit_range], rb[fit_range], 1)
        rb_fit = rb_fc[0]*t[fit_range] + rb_fc[1]
        rt_fc = np.polyfit(t[fit_range], rt[fit_range], 1)
        rt_fit = rt_fc[0]*t[fit_range] + rt_fc[1]

        # fc = fit coefficients
        mtot_fc = np.polyfit(t[fit_range], mtot[fit_range], 1)
        mtot_fit = mtot_fc[0]*t[fit_range] + mtot_fc[1]
        mdot = mtot_fc[0]

        if show_plots:
            cb = utils.colourblind
            pl.close(ifig0); fig1=pl.figure(ifig0)
            pl.plot(t/60., rt, color=cb(5), ls='-', label=r'r$_\mathrm{top}$')
            pl.plot(t/60., rb, color=cb(8), ls='--', label=r'r$_\mathrm{b}$')
            pl.plot(t[fit_range]/60., rt_fit, color=cb(4), ls='-')
            pl.plot(t[fit_range]/60., rb_fit, color=cb(4), ls='-')
            pl.xlabel('t / min')
            pl.ylabel('r / Mm')
            xfmt = ScalarFormatter(useMathText=True)
            pl.gca().xaxis.set_major_formatter(xfmt)
            pl.legend(loc = 0)
            fig1.tight_layout()

            print('rb is the radius of the convective boundary.')
            print('drb/dt = {:.2e} km/s\n'.format(1e3*rb_fc[0]))
            print('rt is the upper limit for mass integration.')
            print('drt/dt = {:.2e} km/s'.format(1e3*rt_fc[0]))

            min_val = np.min([np.min(mtot), np.min(mb), np.min(mtot_fit)])
            max_val = np.max([np.max(mtot), np.max(mb), np.max(mtot_fit)])
            max_val *= 1.1 # allow for some margin at the top of the plot
            oom = int(np.floor(np.log10(max_val)))

            pl.close(ifig0+1); fig2=pl.figure(ifig0+1)
            mdot_str = '{:e}'.format(mdot)
            parts = mdot_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            lbl = (r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} '
                   r'\times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$').\
                   format(mantissa, exponent)
            pl.plot(t[fit_range]/60., mtot_fit/10**oom, color=cb(4), \
                    ls='-', label=lbl, zorder=100)

            lbl = ''
            if burn_func is not None:
                pl.plot(t/60., mir/10**oom, ':', color=cb(3), label='present')
                pl.plot(t/60., mb/10**oom, '--', color=cb(6), label='burnt')
                lbl = 'total'
            pl.plot(t/60., mtot/10**oom, color=cb(5), label=lbl)


            pl.ylim((min_val/10**oom, max_val/10**oom))
            pl.xlabel('t / min')
            sub = 'e'
            ylbl = r'M$_{:s}$ / 10$^{{{:d}}}$ M$_\odot$'.format(sub, oom)
            if oom == 0.:
                ylbl = r'M$_{:s}$ / M$_\odot$'.format(sub)

            pl.ylabel(ylbl)
            yfmt = FormatStrFormatter('%.1f')
            fig2.gca().yaxis.set_major_formatter(yfmt)
            fig2.tight_layout()
            pl.legend(loc=2)
            if fig_file_name is not None:
                fig2.savefig(fig_file_name)

            print('Resolution: {:d}^3'.format(2*len(r)))
            print('mtot_fc = ', mtot_fc)
            print('Entrainment rate: {:.3e} M_Sun/s'.format(mdot))

        if return_time_series:
            return t, mir, mb
        else:
            return mdot

class yprofile(DataPlot, PPMtools):
    """
    Data structure for holding data in the  YProfile.bobaaa files.

    Parameters
    ----------
    sldir : string
        which directory we are working in.  The default is '.'.

    """

    def __init__(self, sldir='.', filename_offset=0, silent=False):
        """
        init method

        Parameters
        ----------
        sldir : string
            which directory we are working in.  The default is '.'.

        """

        PPMtools.__init__(self)
        self.files = []  # List of files in this directory
        self.cycles= []  # list of cycles in this directory
        self.hattrs = [] # header attributes
        self.dcols = []  # list of the column attributes
        self.cattrs= []  # List of the attributes of the y profiles
        self._cycle=[]    # private var
        self._top=[]      # privite var
        self.sldir = sldir #Standard Directory

        if not os.path.isdir(sldir): # then try if ppm_path has been set
            try:
                sldir = ppm_path+'/'+sldir
            except:
                print('ppm_path not correctly set: '+sldir+' is not directory.')
        self.sldir = sldir
        if not os.path.isdir(sldir):  # If the path still does not exist
            print('error: Directory, '+sldir+ ' not found')
            print('Now returning None')
            return None
        else:
            f=os.listdir(sldir) # reads the directory
            for i in range(len(f)):  # Removes any files that are not YProfile files
                if re.search(r"^YProfile-01-[0-9]{4,4}.bobaaa$",f[i]):
                    self.files.append(f[i])
            self.files.sort()
            if len(self.files)==0: # If there are no YProfile files in the directory
                print('Error: no YProfile named files exist in Directory')
                print('Now returning None')
                return None
            slname=self.files[len(self.files)-1] #
            self.slname = slname
            if not silent:
                print("Reading attributes from file ",slname)
            self.hattrs,self.dcols, self._cycle=self._readFile()
            # split the header into header attributes and top attributes
            self._splitHeader()
            # return the header attributes as a dictionary
            self.hattrs=self._formatHeader()
            # returns the concatination of cycle and top attributes
            self.cattrs=self.getCattrs()

            self.ndumpDict=self.ndumpDict(self.files, filename_offset=filename_offset)
            self.radbase = float(self.hattrs['At base of the convection zone R'])
            self.radtop  = float(self.hattrs['Thickness (Mm) of transition from convection to stability '].split()[4])
            if not silent:
                print('There are '+str(len(self.files))+' YProfile files in the ' +self.sldir+' directory.')
                print('Ndump values range from '+str(min(self.ndumpDict.keys()))+' to '+str(max(self.ndumpDict.keys())))
            t=self.get('t',max(self.ndumpDict.keys()))
            t1=self.get('t',min(self.ndumpDict.keys()))
            if not silent:
                print('Time values range from '+ str(t1[-1])+' to '+str(t[-1]))
            self.cycles=list(self.ndumpDict.keys())
        return None

    def ndumpDict(self, fileList, filename_offset=0):
        """
        Method that creates a dictionary of Filenames with the
        associated key of the filename's Ndump.

        Parameters
        ----------
        fileList : list
            A list of yprofile filenames.

        Returns
        -------
        dictionary : dict
            the filename, ndump dictionary

        """
        ndumpDict={}
        for i in range(len(fileList)):
            ndump=fileList[i].split("-")[-1]
            ndump=int(ndump.split(".")[0])
            ndump-=filename_offset
            ndumpDict[ndump]=fileList[i]

        return ndumpDict

    def getHattrs(self):
        """ returns the list of header attributes"""
        h=self.hattrs.sorted()
        return h.sort()

    def getDCols(self):
        """ returns the list of column attributes"""
        return self.dcols

    def getCattrs(self):
        """ returns the list of cycle attributes"""
        dupe=False
        data=[]
        for i in range(len(self._cycle)):
            data.append(self._cycle[i])

        for i in range(len(self._top)): # checking for dublicate attributes
                                        # ie if an Atribute is both a top attri and a cycle attri
            for k in range(len(self._cycle)):
                if self._top[i]==self._cycle[k]:
                    dupe=True
            if not dupe:
                data.append(self._top[i])
            dupe=False

        return data

    def get(self, attri, fname=None, numtype='ndump', num_type=None, \
            resolution='H', silent=True, **kwargs):
        """
        Method that dynamically determines the type of attribute that is
        passed into this method.  Also it then returns that attribute's
        associated data.

        Parameters
        ----------
        attri : string
            The attribute we are looking for.
        fname : string, optional
            The filename, Ndump or time, if None it defaults to the
            last NDump.  The default is None.
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        num_type : string, optional
            Overrides numtype if specified.
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'h' will return the
            largest, 'l' will return the lowest. The default is 'H'.

        """
        isCyc=False #If Attri is in the Cycle Atribute section
        isCol=False #If Attri is in the Column Atribute section
        isHead=False #If Attri is in the Header Atribute section

        if fname==None:
            fname=max(self.ndumpDict.keys())
            if not silent:
                print("Warning at yprofile.get(): fname is None, "\
                      "the last dump (%d) will be used." \
                      % max(self.ndumpDict.keys()))

        if attri in self.cattrs: # if it is a cycle attribute
            isCyc = True
        elif attri in self.dcols:#  if it is a column attribute
            isCol = True
        elif attri in self.hattrs:# if it is a header attribute
            isHead = True

        if num_type is not None:
            numtype = num_type

        # directing to proper get method
        if isCyc:
            return self.getCycleData(attri,fname, numtype, resolution=resolution, \
                                     silent=silent)
        if isCol:
            return self.getColData(attri,fname, numtype, resolution=resolution, \
                                   silent=silent)
        if isHead:
            return self.getHeaderData(attri, silent=silent)
        else:
            res = self.computeData(attri, fname, numtype, silent=silent, **kwargs)
            if res is None:
                if not silent:
                    print('That Data name does not appear in this YProfile Directory')
                    print('Returning none')
            return res

    def getHeaderData(self, attri, silent=False):
        """
        Method returns header attributes.

        Parameters
        ----------
        attri : string
            The name of the attribute.

        Returns
        -------
        data : string or integer
            Header data that is associated with the attri.

        Notes
        -----
        To see all possable options in this instance type
        instance.getHattrs().

        """
        isHead = False
        if attri in self.hattrs:
            isHead = True
        if not isHead:# Error checking
            if not silent:
                print('The attribute '+attri+' does not appear in these YProfiles')
                print('Returning None')
            return None
        data=self.hattrs[attri] #Simple dictionary access
        return data

    def getCycleData(self, attri, FName=None, numType='ndump',
                     Single=False, resolution='H', silent=False):
        """
        Method that returns a Datalist of values for the given attribute or a
        single attribute in the file FName.

        Parameters
        ----------
        attri : string
            What we are looking for.
        FName : string, optional
            The filename, Ndump or time, if None it defaults to the last
            NDump.  The default is None.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        Single : boolean, optional
            Determining whether the user wants just the Attri contained
            in the specified ndump or all the dumps below that ndump.
            The default is False.
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'h' will return the
            largest, 'l' will return the lowest.  The defalut is 'H'.

        Returns
        -------
        list
            A Datalist of values for the given attribute or a
            single attribute in the file FName.

        """

        if FName==None: #By default choose the last YProfile
            FName=max(self.ndumpDict.keys())
            if not silent:
                print("Warning at yprofile.getCycleData(): FName is None, "\
                      "the last dump (%d) will be used." % \
                      max(self.ndumpDict.keys()))

        isCyc= False #If Attri is in the Cycle Atribute section
        boo=True
        filename=self.findFile(FName, numType, silent=silent)
        data=0

        if attri in self._cycle: #if attri is a cycle attribute rather than a top attribute
            isCyc = True

        if attri not in self._cycle and isCyc:# Error checking
            if not silent:
                print('Sorry that Attribute does not appear in the fille')
                print('Returning None')
            return None

        if not Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)

            return data
        if Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)
            if data==None:
                return None
            index=len(data)-1
            return data[index]
        if not Single and not isCyc:

            data=[]
            i=0
            while boo: #Here we basically open up each YProfile File and append the required top attribute
                       # to our data variable
                data.append(self.readTop(attri,self.files[i],self.sldir))
                if self.files[i]==filename: #if we have reached the final file that the user wants.
                    boo=False
                i+=1
            for j in range(len(data)): #One top attribute hase a value of '*****' we cant float that, so we ignore it here
                if '*' not in data[j]:
                    data[j]=float(data[j])

            data=array(data)
        if Single and not isCyc:

            data=self.readTop(attri,filename,self.sldir)
            data=float(data)

        return data

    def getColData(self, attri, FName, numType='ndump', resolution='H',
                   cycle=False, silent=False):
        """
        Method that returns column data.

        Parameters
        ----------
        attri : string
            Attri is the attribute we are loking for.
        FName : string
            The name of the file, Ndump or time we are looking for.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'H' will return the
            largest, 'l' will return the lowest.  The defalut is 'H'.
        cycle : boolean, optional
            Are we looking for a cycle or column attribute.
            The default is False.

        Returns
        -------
        list
            A Datalist of values for the given attribute.

        Notes
        -----
        To ADD: options for a middle length of data.

        """
        def reduce_h(r):
            '''
            Function for reducing hi-res arrays to low-res arrays.
            To be called for FV profiles at present because of
            averaging issues.

            Parameters
            ----------
            r : array
                array to be reduced

            Returns
            --------
            r : array
                reduced array
            '''
            return (r[0::2]+r[0::2])/2.

        num=[]       #temp list that holds the line numbers where the
                     # attribute is found in
        dataList=[]  # holds final data
        attriLine='' # hold a line that the attribute is found in
        attriList=[] # holds a list of attributes that the attribute is
                     # found in
        numList=[]   # holds a single column of data
        boo=False     #temp boolean
        tmp=''

        FName=self.findFile(FName, numType, silent=silent)
        #print FName
        stddir=self.sldir
        resolution=resolution.capitalize()
        if stddir.endswith('/'):  # Makeing sure that the standard dir ends with a slash
                                  # So we get something like /dir/file instead of /dirfile
            FName = str(stddir)+str(FName)
        else:
            FName = str(stddir)+'/'+str(FName)


        boo=True
        try:
            f=open(FName, 'r')
        except IOError:
            if not silent:
                print("That File, "+FName+ ", does not exist.")
                print('Returning None')
            return None
        List=f.readlines()
        f.close()
        for i in range(len(List)):#finds in what lines the attribute is
                                  #located in. This is stored in num, a list of line numbers
            tmpList=List[i].split('  ')
            for k in range(len(tmpList)):
                tmpList[k]=tmpList[k].strip()

            for k in range(len(tmpList)):
                if not cycle:
                    if attri == tmpList[k] and not('Ndump' in List[i]): #if we are looking for a column attribute
                        num.append(i)
                else:
                    if attri == tmpList[k] and ('Ndump' in List[i]): #if we are looking for a cycle attribute
                        num.append(i)


        if i==(len(List) -1) and len(num)==0: #error checking
            if not silent:
                print("Attribute DNE in file, Returning None")
            return None

        for j in range(len(num)): #for each line that attri appears in
            attriList=[]
            rowNum=num[j] #the line in the file that the attribute
                          #appears at
            attriLine=List[rowNum]

            tmpList=attriLine.split('  ') #tmplist will be a list of
                                          #attributes
            for i in range(len(tmpList)):#formating tmplist
                tmpList[i]=tmpList[i].strip()
                if tmpList[i]!='':
                    attriList.append(tmpList[i])

            for i in range(len(attriList)):
                # here we find at what element in the list attri
                # appears at, this value bacomes colNum
                if attri == attriList[i]:
                    break
            colNum=i

            rowNum+=2 #Since the line of Data is two lines after
                      #the line of attributes


            #print List, rowNum
            while rowNum<len(List) and List[rowNum]!= '\n': #and rowNum<len(List)-1:
                # while we are looking at a line with data in it
                # and not a blank line and not the last line in
                # the file
                tmpList=List[rowNum].split(None) #split the line
                                         #into a list of data
                #print tmpList, colNum
                numList.append(tmpList[colNum])
                #append it to the list of data
                rowNum+=1

            #Because an attributes column of data may appear more
            #than onece in a file, must make sure not to add it twice
            if len(dataList)==0: #No data in dataList yet, no need to check
                dataList.append(numList)

            else:
                for k in range(len(dataList)):
                    #if a list of data is allready in the
                    #dataList with the same length of numList
                    #it means the data is allready present, do not add
                    if len(numList)== len(dataList[k]):
                        boo=False

                if boo:
                    dataList.append(numList)


                boo = True

            numList=[]

        tmp=''


        tmpList=[]



        #here we format the data if the user wants higher or the lower resolution of data
        if resolution.startswith('H'):
            for i in range(len(dataList)):
                if len(dataList[i])>len(tmp):
                    tmp=dataList[i]
            dataList=array(tmp,dtype=float)
        elif resolution.startswith('L'):
            for i in range(len(dataList)):
                if len(dataList[i])<len(tmp) or len(tmp)==0:
                    tmp=dataList[i]
            dataList=array(tmp,dtype=float)
        else:
            for i in range(len(dataList)):
                for k in range(len(dataList[i])):
                    dataList[i][k]=float(dataList[i][k])
                    dataList[i][k]=float(dataList[i][k])
                dataList[i]=array(dataList[i])

            try: # If dataList is a list of lists that has one element [[1,2,3]]
            # reformat dataList as [1,2,3]
                j=dataList[1][0]
            except IndexError:
                tmp = True
            except TypeError:
                tmp = False
            if tmp:
                dataList=dataList[0]

        tmp = False

#        print resolution, len(dataList), int(self.hattrs['gridX'])
        # reduce FV arrays if resolution:
        if resolution == 'L' and len(dataList)==int(self.hattrs['gridX']):
            #print 'reducing array for low resolution request'
            dataList = reduce_h(dataList)

        return dataList

    def computeData(self, attri, fname, numtype = 'ndump', silent=False,\
                    **kwargs):
        '''
        Method for computing data not in Yprofile file

        Parameters
        ----------
        attri : str
            What you are looking for
        fname : int
            what dump or time you are look for attri at
            time or dump specifies by numtype.
        numtype : str, optional
            'ndump' or 'time' depending on which you are
            looking for

        Returns
        -------
        attri : array
            An array for the attirbute you are looking for
        '''
        def get_missing_args(required_args, **kwargs):
            missing_args = []

            for this_arg in required_args:
                if not this_arg in kwargs:
                    missing_args.append(this_arg)

            return missing_args

        nabla_ad = 0.4

        if attri == 'T9' or attri == 'mu':
            required_args = ('airmu', 'cldmu')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                          missing_args)
                return None

            airmu = kwargs['airmu']
            cldmu = kwargs['cldmu']

            if 'T9_corr_params' in kwargs:
                T9_corr_params = kwargs['T9_corr_params']
            else:
                T9_corr_params = None

            rho = self.get('Rho', fname, numtype = numtype, resolution = 'l', \
                           silent = silent)
            p = self.get('P', fname, numtype = numtype, resolution = 'l', \
                         silent = silent)
            fv = self.get('FV H+He', fname, numtype = numtype, resolution = 'l', \
                          silent = silent)

            mu = fv*cldmu + (1. - fv)*airmu

            if attri == 'mu':
                return mu
            else:
                # gas constant in code units
                RR = 8.3144598

                T9 = mu*p/(RR*rho)

                if T9_corr_params is not None:
                    T9 = T9_corr_params[0]*T9**T9_corr_params[1]

                return T9
        elif attri == 'Hp':
            r = self.get('Y', fname, numtype, resolution = 'l', silent = silent)
            p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

            logp = np.log(p)

            dr = cdiff(r)
            dlogp = cdiff(logp)

            Hp = -dr/dlogp

            return Hp
        elif attri == 'g':
            r = self.get('Y', 0., 'time', resolution = 'l')
            r_bot = float(self.get('At base of the convection zone R', 0))
            rho = self.get('Rho', 0., 'time', resolution = 'l')
            g_bot = float(self.get('At base of the convection zone g', 0))

            # Centre r_bot on the nearest cell.
            idx_bot = np.argmin(np.abs(r - r_bot))

            # Mass below r_bot (using r[idx_bot] instead of r_bot makes
            # g[idx_bot] == g_bot).
            m_bot = g_bot*(r[idx_bot]**2)/G_code

            dm = -4.*np.pi*(r**2)*cdiff(r)*rho
            m = -np.cumsum(dm) # Get the mass profile by integration.

            # Shift the mass profile to make sure that m[idx_bot] == m_bot.
            # The mass profile at small radii won't make sense, because
            # the core is artificial with no gravity.
            m += m_bot - m[idx_bot]

            g = G_code*m/(r**2) # Gravity profile (see the note above).

            return g
        elif attri == 'nabla':
            T9 = self.get('T9', fname, numtype, resolution = 'l', silent = silent, \
                          **kwargs)
            p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

            logT9 = np.log(T9)
            logp = np.log(p)

            dlogT9 = cdiff(logT9)
            dlogp = cdiff(logp)

            nabla = dlogT9/dlogp

            return nabla
        elif attri == 'nabla_rho':
            rho = self.get('Rho', fname, numtype, resolution = 'l', silent = silent)
            p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

            logrho = np.log(rho)
            logp = np.log(p)

            dlogrho = cdiff(logrho)
            dlogp = cdiff(logp)

            nabla_rho = dlogrho/dlogp

            return nabla_rho
        elif attri == 'N2':
            g = self.get('g', 0, resolution = 'l')
            Hp = self.get('Hp', fname, numtype, resolution = 'l', silent = silent)
            nabla_rho = self.get('nabla_rho', fname, numtype, resolution = 'l', silent = silent)

            # Ideal gas assumed.
            N2 = (g/Hp)*(nabla_ad - 1. + nabla_rho)

            return N2
        elif attri == 'enuc_C12pg':
            required_args = ('airmu', 'cldmu', 'fkair', 'fkcld', \
                             'AtomicNoair', 'AtomicNocld')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                    missing_args)
                return None

            airmu = kwargs['airmu']
            cldmu = kwargs['cldmu']
            fkair = kwargs['fkair']
            fkcld = kwargs['fkcld']
            AtomicNoair = kwargs['AtomicNoair']
            AtomicNocld = kwargs['AtomicNocld']

            kwargs2 = {}
            if 'Q' in kwargs:
                kwargs2['Q'] = kwargs['Q']

            if 'corr_fact' in kwargs:
                kwargs2['corr_fact'] = kwargs['corr_fact']

            if 'use_dt' in kwargs:
                kwargs2['use_dt'] = kwargs['use_dt']

            if 'T9_corr_params' in kwargs:
                kwargs2['T9_corr_params'] = kwargs['T9_corr_params']

            enuc_C12pg = self._get_enuc_C12pg(fname, airmu, cldmu, fkair, \
                                              fkcld, AtomicNoair, AtomicNocld, \
                                              numtype = numtype, silent = silent, \
                                              **kwargs2)

            return enuc_C12pg
        elif attri == 'L_C12pg':
            enuc_C12pg = self.get('enuc_C12pg', fname, numtype, resolution = 'l', \
                                  silent = silent, **kwargs)
            r = self.get('Y', fname, numtype, resolution = 'l', silent = silent)
            dV = -4.*np.pi*r**2*cdiff(r)
            L_C12pg = np.sum(enuc_C12pg*dV)

            return L_C12pg
        elif attri == 'enuc_C12C12':
            required_args = ('airmu', 'cldmu', 'fkcld', 'AtomicNocld')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                    missing_args)
                return None

            airmu = kwargs['airmu']
            cldmu = kwargs['cldmu']
            fkcld = kwargs['fkcld']
            AtomicNocld = kwargs['AtomicNocld']

            kwargs2 = {}
            if 'Q' in kwargs:
                kwargs2['Q'] = kwargs['Q']

            if 'corr_fact' in kwargs:
                kwargs2['corr_fact'] = kwargs['corr_fact']

            if 'corr_func' in kwargs:
                kwargs2['corr_func'] = kwargs['corr_func']

            if 'T9_func' in kwargs:
                kwargs2['T9_func'] = kwargs['T9_func']

            enuc_C12C12 = self._get_enuc_C12C12(fname, airmu, cldmu, \
                          fkcld, AtomicNocld, numtype=numtype, \
                          silent=silent, **kwargs2)

            return enuc_C12C12
        elif attri == 'enuc_O16O16':
            required_args = ('airmu', 'cldmu', 'XO16conv')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                    missing_args)
                return None

            airmu = kwargs['airmu']
            cldmu = kwargs['cldmu']
            XO16conv = kwargs['XO16conv']

            kwargs2 = {}
            if 'Q' in kwargs:
                kwargs2['Q'] = kwargs['Q']

            if 'corr_fact' in kwargs:
                kwargs2['corr_fact'] = kwargs['corr_fact']

            if 'T9_func' in kwargs:
                kwargs2['T9_func'] = kwargs['T9_func']

            enuc_O16O16 = self._get_enuc_O16O16(fname, airmu, cldmu, \
                          XO16conv, numtype=numtype, silent=silent, \
                          **kwargs2)

            return enuc_O16O16
        elif attri == 'enuc_C12O16':
            required_args = ('AtomicNocld', 'AtomicNoair')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                    missing_args)
                return None

            AtomicNoair = kwargs['AtomicNoair']
            AtomicNocld = kwargs['AtomicNocld']

            kwargs2 = {}
            if 'AtomicNoair' in kwargs:
                kwargs2['AtomicNoair'] = kwargs['AtomicNoair']

            if 'AtomicNocld' in kwargs:
                kwargs2['AtomicNocld'] = kwargs['AtomicNocld']

            enuc_C12O16 = self._get_enuc_O16O16(fname, AtomicNocld, AtomicNoair, \
                          XO16conv, numtype=numtype, silent=silent, \
                          **kwargs2)

            return enuc_O16O16
        elif attri == 'L_C12C12':

            required_args = ('airmu', 'cldmu', 'fkcld', 'AtomicNocld','rp_set','r_top')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                    missing_args)
                return None

            airmu = kwargs['airmu']
            cldmu = kwargs['cldmu']
            fkcld = kwargs['fkcld']
            AtomicNocld = kwargs['AtomicNocld']
            rp_set = kwargs['rp_set']
            r_top = kwargs['r_top']

            kwargs2 = {}
            if 'Q' in kwargs:
                kwargs2['Q'] = kwargs['Q']

            if 'corr_fact' in kwargs:
                kwargs2['corr_fact'] = kwargs['corr_fact']

            if 'corr_func' in kwargs:
                kwargs2['corr_func'] = kwargs['corr_func']

            if 'T9_func' in kwargs:
                kwargs2['T9_func'] = kwargs['T9_func']

            if 'rp_set' in kwargs:
                kwargs2['rp_set'] = kwargs['rp_set']

            if 'r_top' in kwargs:
                kwargs2['r_top'] = kwargs['r_top']

            r = self.get('Y', fname=dumps[0], resolution='l')
            idx_top = np.argmin(np.abs(r - r_top))
            # integration from 0 to r_top
            idx = list(range(idx_top, len(r)))
            dV = -4.*np.pi*r**2*cdiff(r)

            enuc_C12C12 = self._get_enuc_C12C12(fname, airmu, cldmu, \
                          fkcld, AtomicNocld, numtype=numtype, \
                          silent=silent, **kwargs2)

            rp = rp_set.get_dump(dumps[i])
            avg_fv = rp.get_table('fv')[0, ::-1, 0]
            sigma_fv = rp.get_table('fv')[3, ::-1, 0]

            avg_fv[avg_fv < 1e-9] = 1e-9
            eta = 1. + (sigma_fv/avg_fv)**2

            # limit eta where avg_fv --> 0
            eta[avg_fv < 1e-6] = 1.

            L_C12C12 = np.sum(eta[idx]*enuc_C12C12[idx]*dV[idx])

            return L_C12C12
        else:
            return None

    def _get_enuc_C12pg(self, fname, airmu, cldmu, fkair, fkcld, \
                        AtomicNoair, AtomicNocld, numtype = 'ndump', \
                        Q = 1.944, corr_fact = 1., use_dt = False, \
                        T9_corr_params = None, silent = False):
        T9 = self.get('T9', fname = fname, numtype = numtype, \
                      resolution = 'l', airmu = airmu, cldmu = cldmu, \
                      T9_corr_params = T9_corr_params, silent = silent)
        fv = self.get('FV H+He', fname = fname, numtype = numtype, \
                      resolution = 'l', silent = silent)
        rho = self.get('Rho', fname = fname, numtype = numtype, \
                       resolution = 'l', silent = silent)
        rhocld = self.get('Rho H+He', fname = fname, numtype = numtype, \
                          resolution = 'l', silent = silent)
        rhoair = self.get('RHOconv', fname = fname, numtype = numtype, \
                          resolution = 'l', silent = silent)
        dt = float(self.get('dt', fname = fname, numtype = numtype, \
                      silent = silent))

        TP13 = T9**(1./3.)
        TP23 = TP13*TP13
        TP12 = np.sqrt(T9)
        TP14 = np.sqrt(TP12)
        TP32 = T9*TP12
        TM13 = 1./TP13
        TM23 = 1./TP23
        TM32 = 1./TP32

        T9inv = 1. / T9
        thyng = 2.173913043478260869565 * T9
        vc12pg = 20000000.*TM23 * np.exp(-13.692*TM13 - thyng*thyng)
        vc12pg = vc12pg * (1. + T9*(9.89-T9*(59.8 - 266.*T9)))
        thing2 = vc12pg + TM32*(1.0e5 * np.exp(-4.913*T9inv) + \
                                4.24e5 * np.exp(-21.62*T9inv))

        thing2[np.where(T9 < .0059)] = 0.
        thing2[np.where(T9 > 0.75)] = 200.

        vc12pg = thing2 * rho * 1000.

        v = 1./ rho
        atomicnocldinv = 1./AtomicNocld
        atomicnoairinv = 1./AtomicNoair

        Y1 =  rhocld * fv * v * atomicnocldinv
        Y2 =  rhoair * (1. - fv) * v * atomicnoairinv

        smaller = .0000001
        reallysmall = smaller * smaller
        CN = 96.480733
        if use_dt:
            # We want the average rate during the current time step.
            # If the burning is too fast all the stuff available burns
            # in a fraction of the time step. We do not allow to burn
            # more than what is available, so the average burn rate is
            # lower than then instantaneous one.
            thing3 = fkair * Y1 * Y2 * vc12pg * dt
            thing3[where(Y1 < reallysmall)] = 0.
            thing2 = np.min(np.array((thing3, Y1)), axis = 0)

            #for i in range(len(Y1)):
            #    print '{:d}   {:.1e}   {:.1e}   {:.1e}'.format(i, Y1[i], thing3[i], Y1[i]/thing3[i])

            DY = fkcld * thing2
            enuc = DY * rho * CN * Q / dt
        else:
            # We want the instantaneous burning rate. This does not
            # depend on how much stuff is available.
            thing3 = fkair * Y1 * Y2 * vc12pg

            DY = fkcld * thing3
            enuc = DY * rho * CN * Q

        # This factor can account for the heating bug if present.
        enuc *= corr_fact

        return enuc

    def _get_enuc_C12C12(self, fname, airmu, cldmu, fkcld, AtomicNocld,\
                        numtype='ndump', silent=False, Q=9.35, \
                        corr_fact=1., corr_func=None, T9_func=None):
        if T9_func is None:
            print('Corrected T9 profile not supplied, using uncorrected T9.')
            T9 = self.get('T9', fname=fname, numtype=numtype, \
                          resolution='l', airmu=airmu, cldmu=cldmu, \
                          silent=silent)
        else:
            T9 = T9_func(self, fname=fname, numtype=numtype, resolution='l')

        fv = self.get('FV H+He', fname=fname, numtype=numtype, \
                      resolution='l', silent=silent)
        rho = self.get('Rho', fname=fname, numtype=numtype, \
                       resolution='l', silent=silent)
        rhocld = self.get('Rho H+He', fname=fname, numtype=numtype, \
                          resolution='l', silent=silent)

        TP13 = T9**(1./3.)
        TP23 = TP13*TP13
        thyng = np.sqrt(T9)
        TP14 = np.sqrt(thyng)
        TP12 = TP14*TP14
        TP32 = T9*TP12
        TM13 = 1./TP13
        TM23 = 1./TP23
        TM32 = 1./TP32

        T9B = T9/(1.+0.0396*T9)
        T9B13 = T9B**(1./3.)
        T9B56 = T9B**(5./6.)
        T932 = T9**(3./2.)

        # C12(C12,a)ne20,60, VITAL
        BRCCA = 0.65*np.ones(len(T9))

        # BRCCN according to Dayras et al. 1977 NUC. PHYS. A 279
        BRCCN = np.zeros(len(T9))

        tmp = -(0.766e0/T9**3.e0)
        tmp = (1.e0 + 0.0789e0*T9 + 7.74e0*T9**3.e0)*tmp
        tmp = 0.859e0*np.exp(tmp)
        idx = where((T9 >= 0.5) & (T9 <= 1.5))
        BRCCN[idx] = tmp[idx]

        idx = where((T9 > 1.5) & (T9 <= 5.0))
        BRCCN[idx] = 0.055e0*(1.e0 - np.exp(-(0.789e0*T9[idx] - 0.976e0)))

        idx = where(T9 > 5.0)
        BRCCN[idx] = 0.02e0

        # Rate from CF88 for C12+C12 (MG24):
        C12C12 = 4.27e26*T9B56/T932*np.exp(-84.165/T9B13-2.12e-03*T9**3)

        # Channel modifcation to get (c12,a):
        vc12c12 = (C12C12 - C12C12*BRCCN)*BRCCA

        vc12c12 = vc12c12 * rho * 1000.

        v = 1./ rho
        atomicnocldinv = 1./AtomicNocld

        Y1 =  rhocld * fv * v * atomicnocldinv

        thing2 = fkcld * Y1 * Y1 * vc12c12

        DY = 0.5 * fkcld * thing2

        CN = 96.480733
        enuc = DY * rho * CN * Q

        # This factor can account for the heating bug if present.
        enuc *= corr_fact

        if corr_func is not None:
            cf = corr_func(self, fname=fname, numtype=numtype, resolution='l')
            enuc *= cf

        return enuc

    def _get_enuc_O16O16(self, fname, airmu, cldmu, XO16conv, \
                         numtype='ndump', silent=False, corr_fact=1., \
                         T9_func=None):
        if T9_func is None:
            print('Corrected T9 profile not supplied, using uncorrected T9.')
            T9 = self.get('T9', fname=fname, numtype=numtype, \
                          resolution='l', airmu=airmu, cldmu=cldmu, \
                          silent=silent)
        else:
            T9 = T9_func(self, fname=fname, numtype=numtype, resolution='l')

        rho = self.get('Rho', fname=fname, numtype=numtype, \
                       resolution=  'l', silent=silent)

        TP13 = T9**(1./3.)
        TP23 = TP13*TP13
        TP43=TP23*TP23
        thyng = np.sqrt(T9)
        TP14 = np.sqrt(thyng)
        TP12 = TP14*TP14
        TP32 = T9*TP12
        TM13 = 1./TP13
        TM23 = 1./TP23
        TM32 = 1./TP32

        # Equation 18.75 of Kippenhahn+12 with electron screening neglected.
        enuc = 2.14e37*(1.e3*rho)*XO16conv*XO16conv*TM23
        thyng = -135.93*TM13 - 0.629*TP23 - 0.445*TP43 + 0.0103*T9*T9
        enuc = enuc*np.exp(thyng)
        enuc = enuc*rho

        # This factor can account for the heating bug if present.
        enuc *= corr_fact

        return enuc
    def _get_enuc_C12O16(self, fname, AtomicNoCld, AtomicNoair):
        T9 = T9_func(yprof, dump)
        rho = yprof.get('Rho', fname=dump, resolution='l')
        rhoair = yprof.get('RHOconv', fname=dump, resolution='l')
        rhocld = yprof.get('Rho H+He', fname=dump, resolution='l')
        fv = yprof.get('FV H+He', fname=dump, resolution='l')

        TM32 = T9**(-3./2.)
        T9A = T9/(1. + 0.055*T9)
        T9A13 = T9A**(1./3.)
        T9A23 = T9A13*T9A13
        T9A56 = T9A**(5./6.)

        # Rate from Caughlan & Fowler (1998) for C12+O16 (Si28):
        vc12o16 = np.exp(-0.180*T9A*T9A) + 1.06e-3*np.exp(2.562*T9A23)
        vc12o16 = 1.72e31*T9A56*TM32*np.exp(-106.594/T9A13)/vc12o16

        vc12o16 = vc12o16 * rho * 1000.

        atomicnocldinv = 1./AtomicNocld
        atomicnoairinv = 1./AtomicNoair

        Y1 =  rhocld * fv / rho * atomicnocldinv
        Y2 =  rhoair * (1.-fv) / rho * atomicnoairinv

        CN = 96.480733
        enuc_c12o16 = fkcld * fkair * Y1 * Y2 * vc12o16 * rho * CN * Q

        return enuc_c12o16
    def findFile(self, FName, numType='FILE', silent=False):
        """
        Function that finds the associated file for FName when Fname
        is time or NDump.

        Parameters
        ----------
        FName : string
            The name of the file, Ndump or time we are looking for.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desird attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the cycle with
            the closest time stamp.  The default is "FILE".

        Returns
        -------
        FName : int
            The number corresponding to the file.

        """

        numType=numType.upper()
        boo=False
        indexH=0
        indexL=0
        if numType=='FILE':

            #do nothing
            return str(FName)

        elif numType=='NDUMP':
            try:    #ensuring FName can be a proper NDump, ie no letters
                FName=int(FName)
            except:
                if not silent:
                    print('Improper value for NDump, choosing 0 instead')
                FName=0
            if FName < 0:
                if not silent:
                    print('User Cant select a negative NDump')
                    print('Reselecting NDump as 0')
                FName=0
            if FName not in list(self.ndumpDict.keys()):
                if not silent:
                    print('NDump '+str(FName)+ ' Does not exist in this directory')
                    print('Reselecting NDump as the largest in the Directory')
                    print('Which is '+ str(max(self.ndumpDict.keys())))
                FName=max(self.ndumpDict.keys())
            boo=True

        elif numType=='T' or numType=='TIME':
            try:    #ensuring FName can be a proper time, ie no letters
                FName=float(FName)
            except:
                if not silent:
                    print('Improper value for time, choosing 0 instead')
                FName=0
            if FName < 0:
                if not silent:
                    print('A negative time does not exist, choosing a time = 0 instead')
                FName=0
            timeData=self.get('t',self.ndumpDict[max(self.ndumpDict.keys())],numtype='file')
            keys=list(self.ndumpDict.keys())
            keys.sort()
            tmp=[]
            for i in range(len(keys)):
                tmp.append(timeData[keys[i]])

            timeData=tmp
            time= float(FName)
            for i in range(len(timeData)): #for all the values of time in the list, find the Ndump that has the closest time to FName
                if timeData[i]>time and i ==0:
                    indexH=i
                    indexL=i
                    break
                if timeData[i]>time:
                    indexH=i
                    indexL=i-1
                    break
                if i == len(timeData)-1:
                    indexH=i
                    indexL=i-1
            high=float(timeData[indexH])
            low= float(timeData[indexL])
            high=high-time
            low=time-low

            if high >=low:
                if not silent:
                    print('The closest time is at Ndump = ' +str(keys[indexL]))
                FName=keys[indexL]
            else:
                if not silent:
                   print('The closest time is at Ndump = ' +str(keys[indexH]))
                FName=keys[indexH]
            boo=True
        else:
            if not silent:
                print('Please enter a valid numType Identifyer')
                print('Returning None')
            return None

        if boo:#here i assume all yprofile files start like 'YProfile-01-'
            FName=self.ndumpDict[FName]
        return FName

    def _splitHeader(self):
        """
        Private function that splits up the data in the header section of the YProfile
        into header attributes and top attributes, where top attributes are just
        cycle attributes located in the header section of the YProfile

        """
        tmp=[]
        tmp2=[]
        slname=self.files[0]
        # Find the header section from another YProfile
        header, tm, tm1=self._readFile()


        if len(header)!=len(self.hattrs): #error checking
            print('Header atribute error, directory has two YProfiles that have different header sections')
            print('Returning unchanged header')
            return None
        for i in range(len(header)):
            if header[i]==self.hattrs[i]: #if the headers are bothe the same, that means its a
                tmp.append(header[i]) #header attribute
            else:                         #Else it changes cycle to cycle and therfore
                                          #its a cycle attribute
                tmp2.append(header[i])

        for i in range(len(tmp2)):            #Formats the top attributes
            tmp2[i]=tmp2[i][0]            #Ie splits the attributes from its associated data.


        self.hattrs=tmp
        self._top=tmp2

    def _formatHeader(self):
        """
        Private function that takes in a set of header attributes and
        then Formats them into a dictionary.

        Input -> A List of headers in the proper format

        Assumptions:

        The first element in the list is Stellar Conv. Luminosity header

        The output in the dictionary looks like
        {'Stellar Conv. Luminosity':The associated data}.

        If an element contains the string 'grid;' it is the grid size
        header and the first, second and third are the x, y and z grid
        sizes respectively.

        The output in the dictionary looks like
        {'gridX':9,'gridY':9,'gridZ':9}.

        If an element is size two the first item is the header name and
        the second will be its associated value.

        The output in the dictionary looks like {'First Item':value}

        If an element contains a colon, The string preceding the colon
        is one part of the header name.  The string after the colon
        will be a list of associations in the form of the name followed
        by an equals sign followed by its value.

        for example a line like this would look like:

        """
        headers=self.hattrs
        dic={}
        i=0
        #print "Analyzing headers ..."
        while i < len(headers):
            if i ==0: # If it is the Stellar Luminosity attribute
                tmp=headers[i][1]
                """
                if '^' in tmp[2]:


                        j=tmp[2].split('^')
                        j=float(j[0])**float(j[1])
                else:
                        j=tmp[2]
                tmp=float(tmp[0])*float(j)
                """

                dic[headers[i][0]]=tmp
                i+=1

            elif 'grid;' in headers[i][0]: # If its the grid header attribute
                tmp1=[]
                tmp= headers[i][0].split()

                for j in range(len(tmp)):
                    tmp[j]=tmp[j].strip('x')
                    tmp[j]=tmp[j].strip('')
                for j in range(len(tmp)):
                    if tmp[j]!='':
                        tmp1.append(tmp[j])
                tmp=tmp1
                dic['gridX']=tmp[0]
                dic['gridY']=tmp[1]
                dic['gridZ']=tmp[2]
                i+=1

            elif len(headers[i])==2: # If its the header attribute that is seperated by a single = sign
                tmp=headers[i][1]

                dic[headers[i][0]]=tmp
                i+=1

            elif ':' in headers[i][0]: #If its the header attribute like 'Title: a=2, b=3
                tmp=headers[i][0].split(':')
                tmp2=tmp[1].split(',')
                for j in range(len(tmp2)):
                    tmp3=tmp2[j].split('=')
                    for k in range(len(tmp3)):
                        tmp3[k]=tmp3[k].strip()
                    dic[tmp[0]+' '+tmp3[0]]=tmp3[1]
                i+=1

            elif headers[i][0].startswith('and '):

                tmp=headers[i][0].split('=',1)

                tmp2=tmp[0].lstrip('and')
                tmp2=tmp2.lstrip()
                prev=headers[i-1][0].split()
                curr=tmp2.split()
                curr=curr[0]
                tmp3=''
                for j in range(len(prev)):
                    if prev[j]== curr:
                        break;
                    tmp3+=prev[j]+' '
                tmp2=tmp3+tmp2
                dic[tmp2]=tmp[1].strip()
                i+=1

            else:
                tmp=headers[i][0].split('  ')
                for j in range(len(tmp)):
                    tmp[j]=tmp[j].strip()
                dic[tmp[0]+' Low']=tmp[1]
                dic[tmp[0]+' High']=tmp[3]
                i+=1
        return dic

# below are some plotting functions integrated from the PPMstar_svn
# server utils/YProfPy directory

    def prof_time(self,fname,yaxis_thing='vY',num_type='ndump',logy=False,
                  radbase=None,radtop=None,ifig=101,ls_offset=0,label_case=" ",markevery = None,
                  **kwargs):
        """
        Plot the time evolution of a profile from multiple
        dumps of the same run (...on the same figure).

        Velocities 'v', 'vY' and/or 'vXZ' may also be plotted.

        Parameters
        ----------
        fname : int or list
            Cycle or list of cycles to plot. Could also be time
            in minutes (see num_type).
        yaxis_thing : string
            What should be plotted on the y-axis?
            In addition to the dcols quantities, you may also choose
            'v', 'vY', and 'vXZ' -- the RMS 'total', radial and
            tangential velocities.
            The default is 'vY'.
        logy : boolean
            Should the velocity axis be logarithmic?
            The default value is False
        radbase : float
            Radii of the base of the convective region,
            respectively. If not None, dashed vertical lines will
            be drawn to mark these boundaries.
            The default value is None.
        radtop : float
            Radii of the top of the convective region,
            respectively. If not None, dashed vertical lines will
            be drawn to mark these boundaries.
            The default value is None.
        num_type : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        ifig : integer
            figure number
        ls_offset : integer
            linestyle offset for argument in utils.linestyle for plotting
            more than one case
        label_case : string
            optional extra label for case

        Examples
        --------

        .. ipython::

            In [136]: from ppmpy import ppm
               .....: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'O-shell-M25'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig prof_time.png width=6in
            In [136]: D2 = ppm.yprofile('D2')
               .....: D2.prof_time([0,5,10],logy=False,num_type='time',ifig=78)

        """

        #fsize=14

        #params = {'axes.labelsize':  fsize,
        #    'font.family':       'serif',
        #'font.family':        'Times New Roman',
        #'figure.facecolor':  'white',
        #'text.fontsize':     fsize,
        #'legend.fontsize':   fsize,
        #'xtick.labelsize':   fsize*0.8,
        #'ytick.labelsize':   fsize*0.8,
        #'text.usetex':       False}
        #pl.rcParams.update(params)
        if type(fname) is not list:
            fname = [fname]
        if num_type is 'time':
            fname = [f * 60 for f in fname]

        pl.figure(ifig)
        i=0
        for dump in fname:
            # FH: I am changing this. The double resolution data
            # in YProfiles is broken and needs to be reduced.
            # if yaxis_thing in ['j','Y','FVconv','UYconv','FV H+He',\
            #                    'UY H+He','Rho','Rho1','A']:
            #     Y=self.get('Y',fname=dump)

            # else:
            Y=self.get('Y',fname=dump,resolution='L')
            cb = utils.colourblind

            if yaxis_thing is 'v':
                Ek = self.get('Ek',fname=dump,numtype=num_type,resolution='l')
                v = np.sqrt(2.*array(Ek,dtype=float))
                y = v*1000
                if logy:
                    ylab = '$\log <u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            elif yaxis_thing is 'vY':
                EkY  = self.get('EkY',fname=dump,numtype=num_type,resolution='l')
                vY = np.sqrt(array(EkY,dtype=float))  # no factor 2 for v_Y and v_XZ
                y = vY*1000
                if logy:
                    ylab = '$\log <u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            elif yaxis_thing is 'vXZ':
                EkXZ = self.get('EkXZ',fname=dump,numtype=num_type,resolution='l')
                vXZ = np.sqrt(array(EkXZ,dtype=float))  # no factor 2 for v_Y and v_XZ
                y = vXZ*1000
                if logy:
                    ylab = '$\log <u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'

            elif yaxis_thing is 'FV':
                y = self.get('FV H+He',fname=dump,numtype=num_type,resolution='l')
                if logy:
                    ylab = '$\log_{10}$ fractional volume'
                else:
                    ylab = 'fractional volume'

            else:
                y = self.get(yaxis_thing,fname=dump,numtype=num_type,resolution='L', **kwargs)
                ylab = yaxis_thing
                if logy: ylab = 'log '+ylab
            if num_type is 'ndump':
                lab = label_case+', '+str(dump)
                leg_tit = num_type
            elif num_type is 'time':
                idx = np.abs(self.get('t')-dump).argmin()
                time = self.get('t')[idx]
                time_min = time/60.
                lab=label_case+', '+str("%.3f" % time_min)
                leg_tit = 'time / min'

            if markevery is not None:
                markevery = markevery
            else:
                markevery = utils.linestyle(i+ls_offset)[1]
            if logy:
                pl.plot(Y,np.log10(y),utils.linestyle(i+ls_offset)[0],
                        markevery= markevery,label=lab,
                       color = cb(i))
            else:
                pl.plot(Y,y,utils.linestyle(i+ls_offset)[0],
                        markevery= markevery,label=lab,
                        color = cb(i))

            if radbase is not None and dump is fname[0]:
                pl.axvline(radbase,linestyle='dashed',color='k')
            if radtop is not None and dump is fname[0]:
                pl.axvline(radtop,linestyle='dashed',color='k')

            i+=1

        pl.xlabel('Radius $[1000\mathrm{km}]$')
        pl.ylabel(ylab)
        pl.legend(loc='best',title=leg_tit).draw_frame(False)

    def get_mass_fraction(fluid,fname,resolution):
        '''
        Get mass fraction profile of fluid 'fluid' at fname with resolution
        'resolution'.
        '''
        y = self.get(fluid,fname=fname,resolution=resolution)
        if fluid == 'FV H+He':
            rhofluid = self.get('Rho H+He',fname=fname,resolution=resolution)
        else:
            rhofluid = self.get('RHOconv',fname=fname,resolution=resolution)
        rho = self.get('Rho',fname=fname,resolution=resolution)
        y = rhofluid * y / rho
        return y

    def vprofs(self,fname,fname_type = 'discrete',log_logic=False,lims=None,save=False,
               prefix='PPM',format='pdf',initial_conv_boundaries=True,
               lw=1., label=True, ifig = 11, which_to_plot = [True,True,True],run=None):
        """
        Plot velocity profiles v_tot, v_Y and v_XZ for a given cycle number
        or list of cycle numbers (fname).
        If a list of cycle number is given, separate figures are made for
        each cycle. If one wishes to compare velocity profiles for two or
        more cycles, see function vprof_time.

        Parameters
        ----------

        fname : int or list
            Cycle number or list of cycle numbers to plot

        fname_type : string
            'discrete' or 'range' whether to average over a range
            to find velocities or plot entries discretely
            default is 'discrete'
        log_logic : boolean
            Should the velocity axis be logarithmic?
            The default value is False
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : logical
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        lw : float, optional
            line width of the plot
        label : string, optional
            label for the line, if multiple models plotted
        which_to_plot : boolean array, optional
            booleans as to whether to plot v_tot v_Y and v_XZ
            respectively True to plot
        run : string, optional
            the name of the run

        Examples
        --------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'O-shell-M25'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig vprofs.png width=6in
            In [136]: D2 = ppm.yprofile('D2')
               .....: D2.vprofs(100,ifig = 111)
        """

        ## fsize=14

        ## params = {'axes.labelsize':  fsize,
        ## #    'font.family':       'serif',
        ## 'font.family':        'Times New Roman',
        ## 'figure.facecolor':  'white',
        ## 'text.fontsize':     fsize,
        ## 'legend.fontsize':   fsize,
        ## 'xtick.labelsize':   fsize*0.8,
        ## 'ytick.labelsize':   fsize*0.8,
        ## 'text.usetex':       False}
        ## pl.rcParams.update(params)

        pl.figure(ifig)
        if type(fname) is not list:
            fname = [fname]
        Y=self.get('Y',fname=fname[0],resolution='l')
        nn = 0
        vXZtot = np.zeros(size(Y))
        vtot = np.zeros(size(Y))
        vYtot =  np.zeros(size(Y))

        for dump in fname:
#            if save or dump == fname[0]:
#                pl.close(ifig),pl.figure(ifig)
#            if not save and dump != fname[0]:
#                pl.figure()
            Ek   = self.get('Ek',fname=dump,resolution='l')
            EkY  = self.get('EkY',fname=dump,resolution='l')
            EkXZ = self.get('EkXZ',fname=dump,resolution='l')
            v =   np.sqrt(2.*array(Ek,dtype=float))
            vY =  np.sqrt(array(EkY,dtype=float))  # no factor 2 for v_Y and v_XZ
            vXZ = np.sqrt(array(EkXZ,dtype=float))  # no factor 2 for v_Y and v_XZ
            if run is not None:
                line_labels = ['$v$ '+run,'$v_\mathrm{r}$ '+run,'$v_\perp$ '+run]
            else:
                line_labels = ['$v$ ','$v_\mathrm{r}$ ','$v_\perp$ ']
            styles = ['-','--',':']
            cb = utils.colourblind
            if fname_type == 'discrete':
                if log_logic:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                        ylab='log v$_\mathrm{rms}$ [km/s]'
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                        ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
                else:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                    ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
                if initial_conv_boundaries:
                    pl.axvline(self.radbase,linestyle='dashed',color='k')
                    pl.axvline(self.radtop,linestyle='dashed',color='k')
                if lims is not None:
                    pl.axis(lims)
                pl.xlabel('r / Mm')
                pl.ylabel(ylab)
                #pl.title(prefix+', Dump '+str(dump))
                if label:
                    pl.legend(loc=8).draw_frame(False)
                number_str=str(dump).zfill(11)
                if save:
                    pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)
            else:
                vtot = vtot + v
                vYtot = vYtot + vY
                vXZtot = vXZtot + vXZ
                nn += 1
        if fname_type == 'range':
            v = vtot/nn
            vY = vYtot/nn
            #vXZ = VXZtot/nn
            if log_logic:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                        ylab='log v$_\mathrm{rms}$ [km/s]'
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                        ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
            else:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                    ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
            if initial_conv_boundaries:
                    pl.axvline(self.radbase,linestyle='dashed',color='k')
                    pl.axvline(self.radtop,linestyle='dashed',color='k')
            if lims is not None:
                    pl.axis(lims)
            pl.xlabel('r / Mm')
            pl.ylabel(ylab)
            pl.title(prefix+', Dump '+str(dump))
            if label:
                    pl.legend(loc=0).draw_frame(False)
            number_str=str(dump).zfill(11)
            if save:
                    pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)

    def vprof_time(self,dumps,Np,comp = 'r',lims=None,save=False,
               prefix='PPM',format='pdf',initial_conv_boundaries=True,lw=1., ifig = 12):
        '''
        Plots the same velocity profile at different times

        Parameters
        ----------
        dumps : array
            dumps to plot
        Np : int
            This function averages over a range of Np points centered at
            the dump number, preferable an even number or else the range
            will not be centered around the dump number
        comp : str
            'r' 't' or 'tot' for the velocity component that will be plotted
            see vprof for comparing all three
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : bool
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : str
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : bool, optional
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        ifig : int
            figure number to plot into

        '''

        avg_rms_v = {}
        for d in dumps:
            # Always take an average except the very end of the run.
            v = get_avg_rms_velocities(self, list(range(d-int(Np/2), d+int(Np/2))),comp)

            avg_rms_v[d] = v

        r = self.get('Y', fname=0, resolution='l')
        markers = ['v', '^', '<', '>', 'o', 's']
        colours = [9, 3, 5, 8, 1, 6]

        ifig = ifig; pl.close(ifig); pl.figure(ifig)
        for i in range(len(dumps)):
            pl.semilogy(r, 1e3*avg_rms_v[dumps[i]], '-', color=cb(colours[i]), \
                         marker=markers[i], markevery=50, \
                         label=r'$\tau_{{{:d}}}$'.format(i+1))
        pl.xlabel('r / Mm')
        pl.ylabel(r'v$_\mathrm{%s}$ / km s$^{-1}$' % comp)
        pl.legend(loc = 0)
        if initial_conv_boundaries:
            pl.axvline(self.radbase,linestyle='dashed',color='k')
            pl.axvline(self.radtop,linestyle='dashed',color='k')
        if lims is not None:
            pl.axis(lims)
        pl.legend(loc=0, ncol=2)
        if save:
            number_str=str(dump).zfill(11)
            pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)

    def Aprof_time(self,tau,Nl,lims=None,save=False,silent = True,
                   prefix='PPM',format='pdf',initial_conv_boundaries=True,lw=1., ifig = 12):
        '''
        Plots the same velocity profile at different times

        Parameters
        ----------
        tau : array
            times to plot
        Nl : range
            Number of lines to plot, eg. range(0,10,1) would plot ten
            dotted contour lines for every minute from 0 to 10
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        silent: boolean,optional
            Should plot display output?
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : logical
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        ifig : int
            figure number to plot into

        Examples
        --------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig Aprof_time.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: F4.Aprof_time(np.array([807., 1399.]),range(0,30,5),lims =[10., 30.,0., 2.5e-2],silent = True)
        '''

        r = self.get('Y', fname = 0, resolution = 'l')
        A0 = self.get('A', fname = 0., numtype = 'time', resolution = 'l', silent = True)
        markers = ['v', '^', '<', '>', 'o', 's']
        colours = [9, 3, 5, 8, 1, 6]

        ifig = ifig; pl.close(ifig); pl.figure(ifig)
        for i in Nl:
            t = self.get('t', fname = 60.*100.*i, numtype = 'time', resolution = 'l', silent = True)[-1]
            A = self.get('A', fname = 60.*100.*i, numtype = 'time', resolution = 'l', silent = True)
            pl.plot(r, A/A0 - 1., ':', lw = 0.5, color = cb(4))
        for i in range(len(tau)):
            t = self.get('t', fname = 60.*tau[i], numtype = 'time', resolution = 'l', silent = False)[-1]
            A = self.get('A', fname = 60.*tau[i], numtype = 'time', resolution = 'l', silent = True)
            pl.plot(r, A/A0 - 1., '-', marker = markers[i], color = cb(colours[i]), \
                     markevery = 50, label = r'$\tau_{{{:d}}}$'.format(i+1))
            if not silent:
                print('You wanted tau = {:.1f} min. yprofile.get() found the closest dump at t = {:.1f} min.\n'.\
                  format(tau[i], t/60.))
        pl.xlabel('r / Mm')
        pl.ylabel('A(t)/A(0) - 1')
        if initial_conv_boundaries:
            pl.axvline(self.radbase,linestyle='dashed',color='k')
            pl.axvline(self.radtop,linestyle='dashed',color='k')
        if lims is not None:
            pl.axis(lims)
        pl.legend(loc=0)
        pl.tight_layout()
        if save:
            number_str=str(dump).zfill(11)
            pl.savefig(prefix+'-A-'+number_str+'.'+format,format=format)

    def tEkmax(self,ifig=None,label=None,save=False,prefix='PPM',format='pdf',
               logy=False,id=0):
        """
        Plot maximum kinetic energy as a function of time.

        Parameters
        ----------
        ifig : int, optional
            Figure number. If None, chose automatically.
            The default is None.
        label : string, optional
            Label for the model
            The default is None.
        save : boolean, optional
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-t-EkMax.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string, optional
            see 'save' above
        format : string, optional
            see 'save' above
        logy : boolean, optional
            Should the y-axis have a logarithmic scale?
            The default is False
        id : int, optional
            An id for the model, which esures that the lines are
            plotted in different colours and styles.
            The default is 0

        Examples
        --------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'O-shell-M25'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig tEKmax.png width=6in
            In [136]: D2 = ppm.yprofile('D2')
               .....: D2.tEkmax(ifig=77,label='D2',id=0)

        """

        fsize=14

        params = {'axes.labelsize':  fsize,
        #    'font.family':       'serif',
        'font.family':        'Times New Roman',
        'figure.facecolor':  'white',
        'text.fontsize':     fsize,
        'legend.fontsize':   fsize,
        'xtick.labelsize':   fsize*0.8,
        'ytick.labelsize':   fsize*0.8,
        'text.usetex':       False}
        pl.rcParams.update(params)

        if ifig is None:
            pl.figure()
        else:
            pl.figure(ifig)

        t = self.get('t')
        EkMax = self.get('EkMax') # Ek in 10^43 erg

        if logy:
            y = np.log10(EkMax*1.e43)
            ylab = '$\log E_{\\rm k,max}/ {\\rm erg}$'
        else:
            y = EkMax * 1.e5 # EkMax in 10^38 erg
            ylab = '$E_{\\rm k,max} / 10^{38}{\\rm \, erg}$'
        if label is not None:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=label)
        else:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=str(id))
        pl.legend(loc='best').draw_frame(False)
        pl.xlabel('t/min')
        pl.ylabel(ylab)
        if save:
            pl.savefig(prefix+'-t-EkMax.'+format,format=format)

    def tvmax(self,ifig=None,label=None,save=False,prefix='PPM',format='pdf',
              logy=False,id=0):
        """
        Plot maximum velocity as a function of time.

        Parameters
        ----------
        ifig : int, optional
            Figure number. If None, chose automatically.
            The default is None.
        label : string, optional
            Label for the model
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-t-vMax.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        logy : boolean, optional
            Should the y-axis have a logarithmic scale?
            The default is False
        id : int, optional
            An id for the model, which esures that the lines are
            plotted in different colours and styles.
            The default is 0

        Examples
        --------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'O-shell-M25'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig tvmax.png width=6in
            In [136]: D2 = ppm.yprofile('D2')
               .....: D2.tvmax(ifig=11,label='D2',id=0)

        """

        fsize=14

        params = {'axes.labelsize':  fsize,
        #    'font.family':       'serif',
        'font.family':        'Times New Roman',
        'figure.facecolor':  'white',
        'text.fontsize':     fsize,
        'legend.fontsize':   fsize,
        'xtick.labelsize':   fsize*0.8,
        'ytick.labelsize':   fsize*0.8,
        'text.usetex':       False}
        #pl.rcParams.update(params)

        if ifig is None:
            pl.figure()
        else:
            pl.figure(ifig)

        t=self.get('t')
        EkMax=self.get('EkMax')

        vMax = 1000*np.sqrt(2.*EkMax)   # velocity in km.s
        if logy:
            y = np.log10(vMax)
            ylab = '$\log v_{\\rm max}/ {\\rm km\,s}^{-1}$'
        else:
            y = vMax
            ylab = '$v_{\\rm max} / {\\rm km\,s}^{-1}$'
        if label is not None:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=label)
        else:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=str(id))
        pl.legend(loc='best').draw_frame(False)
        pl.xlabel('t/min')
        pl.ylabel(ylab)
        if save:
            pl.savefig(prefix+'-t-vMax.'+format,format=format)

    def Richardson_plot(self, fname1 = 0, fname2 = 2, R_low = None, R_top = None, \
                        do_plots = False, logRi_levels = [-1., -0.6, 0., 1., 2., 3.], \
                        ylim_max = 2.02, compressible_fluid = True, plot_type = 0, \
                        ifig = 101):

        '''
        Make a plot of radius vs tangential velocity in the vicinity of the
        boundary and draw on lines of constant Richardson number. Compared to
        the function that produced Fig. 9 of Woodward+ (2015) this one takes
        into account the compressibility of the gas. Several bugs have been
        removed, too.


        Parameters
        ----------
        fname1 : int
            Which dump do you want to take the stratification from?
        fname2 : int
            Which dump do you want to take the velocities from?
        R_low : float
            The minimum radius in the plot. If invalid or None it will be set
            to R_top - 1.
        R_top : float
            Radius of top of convection zone. If invalid or None it will be set
            to the radius at which FV H+He = 0.9.
        do_plots : logical
            Do you want to do some intermittent plotting?
        logRi_levels : list, optional
            Values of Ri for which to draw contours.
        ylim_max : float
            Max of ylim (min is automatically determined) in the final plot.
        compressible_fluid : logical
            You can set it to False to use the Richardson criterion for
            an incompressible fluid.
        plot_type : int
            plot_type = 0: Use a variable lower endpoint and a fixed upper endpoint of
            the radial interval, in which Ri is calculated. Ri is plotted
            for a range of assumed velocity differences with respect to
            the upper endpoint.
            plot_type = 1: Compute Ri locally assuming that the local velocities vanish
            on a certain length scale, which is computed from the radial
            profile of the RMS horizontal velocity.
        ifig : int
            Figure number for the Richardson plot (a new window must be opened).

        Examples
        ---------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig richardson.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: F4.Richardson_plot()

        '''

        # the whole calculation is done in code units

        if fname1 < 0 or fname1 > np.max(list(self.ndumpDict.keys())):
            raise IOError("fname1 out of range.")

        if fname2 == 0:
            raise IOError("Velocities at fname2=0 will be 0; please "+\
                          "make another choice")

        if fname2 < 0 or fname2 > np.max(list(self.ndumpDict.keys())):
            raise IOError("fname2 out of range.")

        if plot_type != 0 and plot_type != 1:
            print("plot_type = %s is not implemented." % str(plot_type))
            return

        # get some header attributes
        R_bot = float(self.hattrs['At base of the convection zone R'])
        g_bot = float(self.hattrs['At base of the convection zone g'])

        # get the stratification at fname = fname1
        # radius is called 'Y' in the YProfiles
        r = self.get('Y', fname = fname1, resolution = 'l')
        fv_H_He = self.get('FV H+He', fname = fname1, resolution = 'l')
        rho = self.get('Rho', fname = fname1, resolution = 'l')
        p = self.get('P', fname = fname1, resolution = 'l')

        # get the rms horizontal velocities at fname = fname2
        ek_xz = self.get('EkXZ', fname = fname2, resolution = 'l')
        rms_u_xz = np.sqrt(ek_xz) # no factor of 2 for ek_xz (known bug)

        # pre-compute logarithms of some variables to speed up the code
        logrho = np.log(rho)
        logp = np.log(p)

        min_r = np.min(r)
        max_r = np.max(r)
        dr = cdiff(r)

        if R_top is not None:
            if R_top < min_r:
                print("R_top too low.")
                return
            elif R_top > max_r:
                print("R_top too high.")
                return
            else:
                # centre R_top on the nearest cell
                idx_top = np.argmin(np.abs(r - R_top))
                R_top = r[idx_top]
                print("R_top centred on the nearest cell: R_top = %.3f." % R_top)
        else:
            # put R_top where fv_H_He is the closest to 0.9
            idx_top = np.argmin(np.abs(fv_H_He - 0.9))
            R_top = r[idx_top]
            print("R_top set to %.3f." % R_top)

        if R_low is not None:
            if R_low < min_r:
                print("R_low too low.")
                return
            elif R_low >= R_top:
                print("R_low too high.")
                return
            else:
                # centre R_low on the nearest cell
                idx_low = np.argmin(np.abs(r - R_low))
                R_low = r[idx_low]
                print("R_low centred on the nearest cell: R_low = %.3f." % R_low)
        else:
            # the default setting
            R_low = R_top - 1.
            if R_low < min_r:
                R_low = min_r

            # find the point nearest to r = R_low
            idx_low = np.argmin(np.abs(r - R_low))
            R_low = r[idx_low]
            print("R_low centred on the cell nearest to R_top - 1: R_low = %.3f." % R_low)

        # centre R_bot on the nearest cell
        idx_bot = np.argmin(np.abs(r - R_bot))
        # mass below R_bot
        # (using r[idx_bot] instead of R_bot makes g[idx_bot] == g_bot)
        M_bot = g_bot*(r[idx_bot]**2)/G_code

        dm = 4.*np.pi*(r**2)*dr*rho
        m = np.cumsum(dm) # get the mass profile by integration
        # shift the mass profile to make sure that m[idx_bot] == M_bot
        # the mass profile at small radii won't make sense, because
        # the core is artificial with no gravity
        m += M_bot - m[idx_bot]

        g = G_code*m/(r**2) # gravity profile (see the note above)
        H_p = p/(rho*g) # pressure scale height (assuming hydrostatic equilibrium)

        nabla_ad = 0.4 # adiabatic temperature gradient
        nabla_rho_ad = 1. - nabla_ad # adiabatic density gradient

        # compute the Richardson number for the shearing flow
        # between r[idx_1] and r[idx_2] (it's faster when using
        # indices instead of radii)
        # arbitrary du or dudr can be supplied
        def Richardson(idx_1, idx_2, du = None, dudr = None):
            # average g and H_p between idx_1 and idx_2
            g_avg = (g[idx_1] + g[idx_2])/2.
            H_p_avg = (H_p[idx_1] + H_p[idx_2])/2.

            # approximate density gradient between idx_1 and idx_2
            dlogrho = logrho[idx_2] - logrho[idx_1]
            dlogp = logp[idx_2] - logp[idx_1]
            nabla_rho = dlogrho/dlogp

            # buoyancy frequency squared
            if compressible_fluid:
                N2 = (g_avg/H_p_avg)*(nabla_rho - nabla_rho_ad)
            else:
                N2 = (g_avg/H_p_avg)*nabla_rho

            # compute the velocity difference we have no information
            # about the velocity difference or gradient
            if du is None and dudr is None:
                du = rms_u_xz[idx_2] - rms_u_xz[idx_1]

            # compute the velocity gradient if none was supplied
            if dudr is None:
                dr = r[idx_2] - r[idx_1]
                dudr = du/dr

            # velocity gradient squared
            dudr2 = (dudr)**2

            # Richardson number
            Ri = N2/dudr2

            return Ri

        if plot_type == 0:
            # grid of indices for a sequence of intervals, in which Ri will be computed
            idx_grid = np.arange(idx_top + 1, idx_low + 1)

            # the corresponding grid of radii
            r_grid = np.array([r[i] for i in idx_grid])

            # reference velocity
            u_0 = rms_u_xz[idx_top]

            # velocity limit for the plot
            # the factor of 1e-3 converts the velocity from km/s to
            # code units (10^8 cm s^{-1})
            u_max = 1e-3*(10**ylim_max)

            # construct grids of assumed velocities and velocity differences
            # logarithmic spacing between (u_0 + log10u_step) and u_max
            log10u_step = 0.02
            u_grid = 10**np.arange(np.log10(rms_u_xz[idx_top]) + log10u_step, \
                     np.log10(u_max), log10u_step)
            du_grid = u_grid - u_0

            # perhaps the loops could be optimised, but it runs fast enough as is
            Ri = np.zeros((du_grid.size, idx_grid.size))
            for i in range(0, du_grid.size):
                for j in range(0, idx_grid.size):
                    Ri[i, j] = Richardson(idx_grid[j], idx_top, du = du_grid[i])

            pl.close(ifig)
            pl.figure(ifig)

            # pl.contour() fails if np.log10(Ri) is undefined everywhere
            if any(Ri > 0):
                cs = pl.contour(r_grid, np.log10(1e3*u_grid), np.log10(Ri), \
                                logRi_levels, linestyles = '-')
                pl.clabel(cs)

            # pl.contour() fails if np.log10(Ri) is undefined everywhere
            if any(Ri < 0):
                cs2 = pl.contour(r_grid, np.log10(1e3*u_grid), np.log10(-Ri), \
                                 logRi_levels, linestyles = '--')
                pl.clabel(cs2)

            pl.plot(r, np.log10(1e3*rms_u_xz + 1e-100), \
                    marker = 'o', markevery = utils.linestyle(0)[1], \
                    label = ("velocity at dump %d" % fname2))

            pl.xlabel(r'$r\ \mathrm{[Mm]}$', fontsize = 16)
            pl.ylabel(r"$\log\ u_\mathrm{hor,rms}\ \mathrm{[km/s]}$", fontsize = 16)

            pl.xlim(R_low, R_top)
            pl.ylim(np.log10(1e3*rms_u_xz[idx_top]), np.log10(1e3*u_max))
            pl.legend(loc = 3)
        elif plot_type == 1:
            # the steepest velocity gradient between R_low and R_top
            dudr = cdiff(rms_u_xz)/dr
            max_dudr = np.max(np.abs(dudr[idx_top:(idx_low + 1)]))

            # characteristic length scale on which velocities decrease at the boundary
            l_u = np.max(rms_u_xz[idx_top:(idx_low + 1)])/max_dudr
            print("Velocities decrease on a characteristic length scale of %.2f Mm" % l_u)

            # grid of positions, at which Ri will be computed
            idx_grid = np.arange(idx_top, idx_low + 1)

            # the corresponding grid of radii
            r_grid = np.array([r[i] for i in idx_grid])

            # estimate the local velocity gradient assuming that the local velocities
            # vanish on the length scale l_u
            dudr_estimate = rms_u_xz[idx_top:(idx_low + 1)]/l_u

            # compute the local Richardson numbers
            Ri = np.zeros(idx_grid.size)
            for i in range(0, idx_grid.size):
                Ri[i] = Richardson(idx_grid[i] - 1, idx_grid[i] + 1, dudr = dudr_estimate[i])

            # determine the upper limit for Ri in the plot
            min_log10absRi = np.min(np.log10(np.abs(Ri)))
            # always keep some margin
            if min_log10absRi - np.floor(min_log10absRi) > 0.1:
                pl_Ri_min = np.floor(min_log10absRi)
            else:
                pl_Ri_min = np.floor(min_log10absRi) - 1.

            # determine the upper limit for Ri in the plot
            max_log10absRi = np.max(np.log10(np.abs(Ri)))
            # always keep some margin
            if np.ceil(max_log10absRi) - max_log10absRi > 0.1:
                pl_Ri_max = np.ceil(max_log10absRi)
            else:
                pl_Ri_max = np.ceil(max_log10absRi) + 1.

            # FV values smaller than 10^{-8} are not interesting
            min_log10fv_H_He = np.min(np.log10(fv_H_He[idx_top:(idx_low + 1)] + 1e-100))
            if min_log10fv_H_He < -8.:
                min_log10fv_H_He = -8.

            # do we need to shift the FV curve in the plot?
            fv_offset = 0
            if pl_Ri_min > min_log10fv_H_He:
                fv_offset = pl_Ri_max

            pl.close(ifig)
            fig = pl.figure(ifig)
            ax = fig.add_subplot(111)
            lns = [] # array of lines to be put into a joint legend

            if any(Ri < 0):
                # temporarily suppress numpy warnings
                old_settings = np.seterr()
                np.seterr(invalid = 'ignore')
                lns += ax.plot(r_grid, np.log10(Ri), linestyle = '-', linewidth = 2, \
                               label = r'$Ri > 0$')
                lns += ax.plot(r_grid, np.log10(-Ri), linestyle = '--', linewidth = 2, \
                               label = r'$Ri < 0$')
                np.seterr(**old_settings)
            else:
                lns += ax.plot(r_grid, np.log10(Ri), linestyle = '-', linewidth = 2, \
                               label = r'$Ri$')

            lns += ax.plot(r_grid, np.log10(fv_H_He[idx_top:(idx_low + 1)] + 1e-100) + \
                           fv_offset, linestyle = '-', label = r'$FV$')

            ax.set_xlim(R_low, R_top)
            ax.set_ylim(pl_Ri_min, pl_Ri_max)

            ax.set_xlabel(r'$\mathrm{radius\ [Mm]}$', fontsize = 16)
            if fv_offset == 0:
                ax.set_ylabel(r'$\log Ri; \log FV$', fontsize = 16)
            else:
                ax.set_ylabel(r'$\log Ri;\ \log FV + %d$' % fv_offset, fontsize = 16)

            ax2 = ax.twinx() # get a new set of axes
            lns += ax2.plot(r_grid, 1e3*rms_u_xz[idx_top:(idx_low + 1)], linestyle = '-.', \
                            label = r'$u_\mathrm{hor,rms}$')
            ax2.set_ylabel(r'$u_\mathrm{hor,rms}\ \mathrm{[km/s]}$', fontsize = 16)

            lbls = [l.get_label() for l in lns]
            ax2.legend(lns, lbls, loc = 2)

        # show some plots to inspect more variables
        if do_plots:
            i1 = idx_bot
            i2 = np.argmax(r)
            if i1 > i2: tmp = i1; i1 = i2; i2 = tmp

            pl.close(102); pl.figure(102)
            pl.plot(r[i1:i2], 5.025e-07*m[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("mass [M_Sun]")
            pl.title("enclosed mass")

            pl.close(103); pl.figure(103)
            pl.plot(r[i1:i2], 1e8*g[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("gravity [cm/s^2]")
            pl.title("gravity")

            pl.close(104); pl.figure(104)
            pl.plot(r[i1:i2], np.log10(1e3*rho[i1:i2]))
            pl.xlabel("radius [Mm]")
            pl.ylabel("log10(rho [g/cm^3])")
            pl.title("log10 density")

            pl.close(105); pl.figure(105)
            pl.plot(r[i1:i2], np.log10(1e19*p[i1:i2]))
            pl.xlabel("radius [Mm]")
            pl.ylabel("log10(p [g/(cm s^2)])")
            pl.title("log10 pressure")

            pl.close(106); pl.figure(106)
            pl.plot(r[i1:i2], H_p[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("Hp [Mm]")
            pl.title("pressure scale height")

            pl.close(107); pl.figure(107)
            nabla_rho = cdiff(logrho)/cdiff(logp)
            pl.plot(r[i1:i2], nabla_rho[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("nabla_rho")
            pl.title("density gradient")

            pl.close(108); pl.figure(108)
            nabla_rho = cdiff(logrho)/cdiff(logp)
            N2 = (g/H_p)*(nabla_rho - nabla_rho_ad)
            pl.plot(r[i1:i2], N2[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("N^2 [1/s^2]")
            pl.title("buoyancy frequency")

        pl.show() # show everything we have plotted

    def Dov(self,r0,D0,f,fname=1):
        '''
        Calculate and plot an exponentially decaying diffusion coefficient
        given an r0, D0 and f.

        Dov is given by the formula D0*exp(-2*(r-r0)/f*Hp)

        Parameters
        ----------
        r0 : float
            radius (Mm) at which the decay will begin
        D0 : float
            diffusion coefficient at r0
        f : float
            what f-value (parameter, where f*Hp is the e-folding length
            of the diffusion coefficient) should we decay with?
        fname : int
            which dump do you want to take r and P from?

        Returns
        --------
        r : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
            the decays exponentially
        D : array
            Exponentially decaying diffusion coefficient (cm^2/s)
        '''
        #    Hp = 1 / d lnP/dr = P / (d)
        r = self.get('Y',fname=fname,resolution='l')[::-1] * 1.e8  # cm, centre to surface
        P = self.get('P',fname=fname)[::-1] * 1.e19 # barye, centre to surface
        Hp = - P[1:] * np.diff(r) / np.diff(P)
        Hp = np.insert(Hp,0,0)
        idx = np.abs(r - r0*1.e8).argmin()
        r0 = r[idx]
        Hp0 = Hp[idx]

        print(r0, Hp0, idx)

        D = D0 * np.exp(-2. * (r[idx:] - r0) / f / Hp0)
        return r[idx:] / 1.e8, D

    def Dov2(self,r0,D0,f1=0.2,f2=0.05,fname=1, silent = True):
        '''
        Calculate and plot an 2-parameter exponentially decaying diffusion coefficient
        given an r0, D0 and f1 and f2.

        Dov is given by the formula:
        D = 2. * D0 * 1. / (1. / exp(-2*(r-r0)/f1*Hp) + 1. / exp(-2*(r-r0)/f2*Hp))

        Parameters
        ----------
        r0 : float
            radius (Mm) at which the decay will begin
        D0 : float
            diffusion coefficient at r0
        f1,f2,A : float
            parameters of the model
        fname : int
            which dump do you want to take r and P from?

        Returns
        --------
        r : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
            the decays exponentially
        D : array
            Exponentially decaying diffusion coefficient (cm^2/s)
        '''
        #    Hp = 1 / d lnP/dr = P / (d)
        r = self.get('Y',fname=fname,resolution='l')[::-1] * 1.e8  # cm, centre to surface
        P = self.get('P',fname=fname)[::-1] * 1.e19 # barye, centre to surface
        Hp = - P[1:] * np.diff(r) / np.diff(P)
        Hp = np.insert(Hp,0,0)
        idx = np.abs(r - r0*1.e8).argmin()
        r0 = r[idx]
        Hp0 = Hp[idx]
        if not silent:
            print(r0, Hp0, idx)

        D = 2. * D0 * 1./(1./(np.exp(-2. * (r - r0) / f1 / Hp0)) +
                         1./(np.exp(-2. * (r - r0) / f2 / Hp0))
                         )
        return r / 1.e8, D

    def Dinv(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,
             niter=3,debug=False,grid=False,FVaverage=False,tauconv=None,
             returnY=False,plot_Dlt0=True, silent = True, initial_conv_boundaries = True,
             approx_D=False,linelims=None, r0 = None):
        '''
        Solve inverted diffusion equation to see what diffusion coefficient
        profile would have been appropriate to mimic the mixing of species
        seen in the Yprofile dumps.

        In the present version, we only solve for D in the region where
        'FV H+He' is not 0 or 1 in dump fname 2.

        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.
        plot_Dlt0 : boolean, optional
            whether or not to plot the diffusion coefficient when it is
            negative
        approx_D : boolean, optional
            whether or not to plot an approximate diffusion coefficient
            for an area of the line
        linelims : range, optional
            limits of the radius to approximate diffusion coefficient
            default is none (whole vector)
        r0 : None, optional
            Start of exponential diffusion decay, necessary
            for approx_D

        Returns
        --------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)

        Examples
        ---------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig Dinv.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: res = F4.Dinv(1,640)

        '''


        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        x = x * 1.e8

        def mf(fluid,fname):
            '''
            Get mass fraction profile of fluid 'fluid' at fname.
            '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y

        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting

            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)

            y0long = y0
            y1long = y1

        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if debug: print(len(xlong), len(y0long))

        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0

        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        if not silent:
            print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)

        # in the current formulation for the inner boundary condition,
        # y1[0] != 0:
        while y1[0] == 0.:
            x = x[1:]
            y0 = y0[1:]
            y1 = y1[1:]

        # Try moving left boundary one over to allow for "exact"
        # boundary condition and saving the now ghost cell value
        xl = x[100]
        y0l = y0[100]
        y1l = y1[100]
        x = x[101:]
        y0 = y0[101:]
        y1 = y1[101:]

        if debug : print(y0, y1, deltat)
        if not silent:
            print('deltat = ', deltat, 's')
        p = np.zeros(len(x))
        q = np.zeros(len(x))

        xdum = np.zeros(3) # our workhorse array for differencing

        dt = float(deltat)

        # calculate matrix elements for intermediate mesh points:
        def matrixdiffus(x, y0, y1, dt):
            m = len(x) - 1
            for i in range(1,m):
                xdum[0] = x[i-1]
                xdum[1] = x[i]
                xdum[2] = x[i+1]

                xl = xdum[1] - xdum[0]
                xr = xdum[2] - xdum[1]

                xm = 0.5 * (xdum[2] - xdum[0])

                alpha = dt / xm

                p[i] = (y1[i] - y1[i-1]) * alpha / xl
                q[i] = (y1[i] - y1[i+1]) * alpha / xr

            # central (left) boundary:
            xdum[1] = x[0]
            xdum[2] = x[1]
            xr = xdum[2] - xdum[1]
            alpha = dt / (xr * xr)
#            p[0] = y1[i] * alpha
#            q[0] = (y1[i] - y1[i+1]) * alpha
#            p[0] = y1[0] * alpha
#            q[0] = (y1[0] - y1[1]) * alpha
            # Trying new boundary conditions:
            p[0] = (y1[0] - y1l) * alpha
            q[0] = (y1[0] - y1[1]) * alpha
            if not silent:
                print('p0, q0 = ', p[0],q[0])

            # surface (right) boundary:
            xdum[0] = x[m-1]
            xdum[1] = x[m]
            xl = xdum[1] - xdum[0]
            alpha = dt / (xl * xl)
#            p[m] = (y1[i] - y1[i-1]) * alpha
#            q[m] = 0
            p[m] = (y1[m] - y1[m-1]) * alpha
            q[m] = 0.
            if not silent:
                print('pm, qm = ', p[m],q[m])

            G = np.zeros([len(x),len(x)])

            # set up matrix:

            for i in range(len(x)):
                if not silent:
                    print('p[i] = ', p[i])
                G[i,i] = p[i]
                if debug : print(G[i,i])
                if i != len(x)-1 :
                    G[i,i+1] = q[i]
                    if debug : print(G[i,i+1])

            A = np.array( [ G[i,:] for i in range(len(x)) ] )
            if not silent:
                print(A[0])
                print('determinant = ', np.linalg.det(A))
            return A


        # Direct solution (initial guess if moving on to do Newton-
        # Raphson refinement:
        A = matrixdiffus(x,y0,y1,dt)
        B = y0 - y1
        D = np.linalg.solve(A,B)

        if newton:
            x0 = D
            xn = np.zeros(len(x0))
            xn = x0
            xnp1 = np.zeros(len(x0))
            J = np.linalg.inv(A) # Jacobian matrix
            # refinement loop:
            for i in range(1,niter+1):
                f = np.dot(A,xn) - B
                xnp1 = xn - np.dot(J,f)
                corr = np.abs(xnp1 - xn) / xn
                cmax = max(corr)
                cmin = min(corr)
                if not silent:
                    print('NEWTON: iter '+str(i))
                    print('max. correction = ', cmax)
                    print('min. correction = ', cmin)
                xn = xnp1

            D = xnp1

        cb = utils.colourblind
        lsty = utils.linestyle

        def safe_log10(x, minval=0.0000000001):
            return np.log10(x.clip(min=minval))

        pl.figure()

        pl.plot(xlong,safe_log10(y0long),
                marker='o',
                color=cb(8),
                markevery=6,
                label='$X_{'+str(fname1)+'}$')

        pl.plot(xlong,safe_log10(y1long),
                marker='o',
                color=cb(9),
                markevery=6,
                label='$X_{'+str(fname2)+'}$')
#        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.ylabel('$\log\,X$ ')
        pl.xlabel('r / Mm')

        pl.legend(loc='center right').draw_frame(False)
        if grid:
            pl.grid()

        pl.twinx()

        pl.plot(x/1.e8,safe_log10(D),'k-',
                label='$D$') #'$D > 0$')
        if initial_conv_boundaries:
            pl.axvline(self.radbase,linestyle='dashed',color='k')
            pl.axvline(self.radtop,linestyle='dashed',color='k')
        if plot_Dlt0:
            pl.plot(x/1.e8,np.log10(-D),'k--',
                    label='$D < 0$')
        pl.ylabel('$\log(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
        if approx_D:
            if linelims is not None:
                indx1 = np.abs(x/1.e8 - linelims[0]).argmin()
                indx2 = np.abs(x/1.e8 - linelims[1]).argmin()
                m,b = pyl.polyfit(x[indx1:indx2]/1.e8,np.log10(D[indx1:indx2]),1)
            else:
                m,b = pyl.polyfit(x/1.e8,np.log10(D),1)
            if r0 is None:
                print('Please define r0')
            rr = self.get('Y',fname=1,resolution='l')[::-1]
            P = self.get('P',fname=fname1)[::-1] * 1.e19 # barye, centre to surface
            Hp = - P[1:] * np.diff(rr) / np.diff(P)
            Hp = np.insert(Hp,0,0)
            idxhp = np.abs(rr - r0).argmin()
            idx = np.abs(x - r0*1.e8).argmin()
            Hp0 = Hp[idxhp]
            D0 = D[idx]
            f = (-2. * np.abs(np.mean(x[indx1:indx2])/1.e8 - r0))\
                /( np.log( (np.mean(D[indx1:indx2]))/D0) *Hp0)
            lab='f= '+str("%.3f" % f)
            pl.plot(x[indx1:indx2]/1.e8,m*x[indx1:indx2]/1.e8+b,
                    linestyle='dashed',color='r',label=lab)
        pl.legend(loc='upper right').draw_frame(False)
        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolve(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,niter=3,
             debug=False,grid=False,FVaverage=False,tauconv=None,returnY=False):
        '''
        Solve inverse diffusion equation sequentially by iterating over the spatial
        domain using a lower boundary condition (see MEB's thesis, page
        223, Eq. B.15).


        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
            numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.

        Returns
        --------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)

        '''


        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        x = x * 1.e8

        def mf(fluid,fname):
            '''
                Get mass fraction profile of fluid 'fluid' at fname.
                '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y

        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting

            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)

            y0long = y0
            y1long = y1

        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if debug: print(len(xlong), len(y0long))

        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0

        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)

        # in the current formulation for the inner boundary condition,
        # y1[0] != 0:
        while y1[0] == 0.:
            x = x[1:]
            y0 = y0[1:]
            y1 = y1[1:]

        # Try moving left boundary one over to allow for "exact"
        # boundary condition and saving the now ghost cell value
        xl = x[0]
        y0l = y0[0]
        y1l = y1[0]
        x = x[1:]
        y0 = y0[1:]
        y1 = y1[1:]

        if debug : print(y0, y1, deltat)
        print('deltat = ', deltat, 's')
        p = np.zeros(len(x))
        q = np.zeros(len(x))

        xdum = np.zeros(3) # our workhorse array for differencing

        dt = float(deltat)

        # Calculate D starting from inner boundary:
        D = np.zeros(len(x))
        # inner boundary:
        xr = x[1] - x[0]
        xl = xr
        xm = (xl + xr) / 2.
        p = (y1[0] - y1l) / xl
        s = xm * (y1[0] - y0[0]) / dt
        u = xr / (y1[1] - y1[0])
        Dghost = 0. # is this OK?
        D[0] = u * (s + Dghost * p)
        # now do the rest:
        for i in range(1,len(x)-1):
            xr = x[i+1] - x[i]
            xl = x[i] - x[i-1]
            xm = (xl + xr) / 2.
            p = (y1[i] - y1[i-1]) / xl
            s = xm * (y1[i] - y0[i]) / dt
            u = xr / (y1[i+1] - y1[i])
            D[i] = u * (s + D[i-1] * p)
        # outer boundary:
        m = len(x) - 1
        xl = x[m] - x[m-1]
        xr = xl
        xm = (xl + xr) / 2.
        p = (y1[m] - y1[m-1]) / xl
        s = xm * (y1[m] - y0[m]) / dt
        u = xr / (1. - y1[m]) # assuming here that y[m+1] - 1.. is this OK??
        D[m] = u * (s + D[m-1] * p)

        pl.figure()
        #        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
        #                markevery=utils.linestyle(1)[1],\
        #                label=fluid.replace('FV','')+' '+str(fname1))
        #        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
        #                markevery=utils.linestyle(2)[1],\
        #                label=fluid.replace('FV','')+' '+str(fname2))
        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
                markevery=utils.linestyle(1)[1],\
                label='fluid above'+' '+str(fname1))
        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
                markevery=utils.linestyle(2)[1],\
                label='fluid above'+' '+str(fname2))
        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.xlabel('r / Mm')
        pl.ylim(-8,0.1)
        pl.legend(loc='lower left').draw_frame(False)
        if grid:
            pl.grid()
        pl.twinx()
        pl.plot(x/1.e8,np.log10(D),'k-',\
                label='$D$') #'$D > 0$')
        pl.plot(x/1.e8,np.log10(-D),'k--',\
                label='$D < 0$')
        pl.ylabel('$\log D\,/\,{\\rm cm}^2\,{\\rm s}^{-1}$')
        pl.legend(loc='upper right').draw_frame(False)

        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolvedown(self,fname1,fname2,fluid='FV H+He',numtype='ndump',
       newton=False,niter=3,debug=False,grid=False,FVaverage=False,
       tauconv=None,returnY=False,smooth=False,plot_Dlt0=True,
       sinusoidal_FV=False, log_X=True, Xlim=None, Dlim=None,
       silent=True,showfig=True):
        '''
        Solve diffusion equation sequentially by iterating over the spatial
        domain inwards from the upper boundary.

        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.
        smooth : boolean, optional
            Smooth the abundance profiles with a spline fit, enforcing their
            monotonicity. Only works for FV H+He choice of fluid
        plot_Dlt0 : boolean, optional
            whether or not to plot D where it is <0
            the default value is True

        Returns
        --------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)

        '''


        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong

        def mf(fluid,fname):
            '''
            Get mass fraction profile of fluid 'fluid' at fname.
            '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y

        def make_monotonic(r,x):
            '''
            function for making x monotonic, as the solution to the diffusion
            equation should be. Only works when considering FV H+He (not conv)
            '''
            from scipy.interpolate import UnivariateSpline as uvs

            xorig = x
            rorig = r
            xnew0 = np.zeros(len(x))

            # find top and botom of convection zone

            ib = np.where(x!=0)[0][-1]
            it = np.where(x!=1)[0][0]

            x = x[it:ib]
            r = r[it:ib]
            x = np.maximum(x,1.e-12)
            x = np.log10(x)

            # is there a local min?

            dx = np.diff( x )
            c = len( np.where( dx > 0 )[0] ) > 0

            # find midpoint, left and right sides of linear reconstruction

            if c:
                # there is a local minimum, but also a local maximum
                # find the min by going top inwards and seeing where
                # difference changes sign
                # continue on to find the max, with which we set the width of the linear
                # reconstruction region
#                for i in range(len(r)):
#                    if dx[i] > 0.:
#                        idxmin = i
#                        break
# ask user for now to identify the local min (noisy data)
                pl.plot(r,x)
                usr_rmin = input("Please input local min in Mm: ")
                usr_rmin = float(usr_rmin)
                idxmin = np.abs( r - usr_rmin ).argmin()
                for i in range(idxmin,len(r)):
                    if x[i] == max(x[idxmin:]):
                        idxmax = i
                        break
            else:
                # everything was OK-ish anyway
                return xorig

            # to take local max as left side as interval:
            if False:
                il = idxmax

            # set left side of interval to radius where x drops below the local minimum:
            if True:
                for i in range(idxmin,len(r)):
                    if x[i] < x[idxmin]:
                        il = i
                        break

            rmid = r[idxmin]
            width = ( rmid - r[il] ) * 2.
            rl = rmid - width / 2.
            rr = rmid + width / 2.

            il = np.abs( r - rl ).argmin()
            ir = np.abs( r - rr ).argmin()

            # just sort the two ends

            rt = r[:ir]
            rb = r[il:]

            xt = np.array(sorted(x[:ir])[::-1])
            xb = np.array(sorted(x[il:])[::-1])

            # now we fit the reconstruction region

            def expfunc(x, a, c, d):
                return a*np.exp(c*x)+d

            if True:

                rm = r[ir:il]
                xms = sorted(x[ir:il])[::-1]
                xms = np.array(xms)
                from scipy.optimize import curve_fit
                # fit an exponential
                #popt, pcov = curve_fit(expfunc, rm, xms, p0=(1, 1, 1))
                # linear reconstruction
                m = ( xms[-1] - xms[0] ) / ( rm[-1] - rm[0] )
                c = xms[0] - m * rm[0]
                # now extend it <bw> Mm beyond the l and r bounds of the reconstruction
                # region so that we can do a blend
                bw = 0.1
                idxr = np.abs( r - ( rr + bw ) ).argmin()
                idxl = np.abs( r - ( rl - bw ) ).argmin()
                rm = r[idxr:idxl]
                # exponential:
                #xm = func(rm, *popt)
                # linear:
                xm = m * rm + c

            #now combine back the results with a sinusoidal blend at each overlap of the
            # reconstruction region with the top and bottom components

            xnew = np.zeros(len(x))
            # top
            itr = np.abs(rt-(rr+bw)).argmin()
            xnew[:idxr] = xt[:itr]
            # bottom
            ibl = np.abs(rb-(rl-bw)).argmin()
            xnew[idxl:] = xb[ibl:]
            # upper middle
            imrbb = np.abs( rm - rr ).argmin()
            xnew[idxr:ir] = xt[itr:] * np.sin( np.abs(rt[itr:] - r[ir]) / bw * np.pi / 2. ) ** 2 + \
                            xm[:imrbb] * np.cos( np.abs(rm[:imrbb] - r[ir]) / bw * np.pi / 2. ) ** 2
            # lower middle
            imlbt = np.abs( rm - rl ).argmin()
            xnew[il:idxl] = xb[:ibl] * np.sin( np.abs(rb[:ibl] - r[il]) / bw * np.pi / 2. ) ** 2 + \
                            xm[imlbt:] * np.cos( np.abs(rm[imlbt:] - r[il]) / bw * np.pi / 2. ) ** 2
            # middle
            xnew[ir:il] = xm[imrbb:imlbt]

            xnew0[it:ib] = xnew[:]
            xnew0 = 10. ** xnew0
            xnew0[:it] = 1.
            xnew0[ib:] = 0.
            return xnew0


        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting

            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)

            y0long = y0
            y1long = y1


        if debug: print(len(xlong), len(y0long))

        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0

        if smooth:
            y1 = make_monotonic(x,y1)
            #y0 = make_monotonic(x,y0)

        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if sinusoidal_FV:
            ru = float(self.get('Gravity turns off between radii Low', fname=fname1))
            idxu = np.argmin(np.abs(x - ru))
            rl = float(self.get('Gravity turns on between radii High', fname=fname1))
            idxl = np.argmin(np.abs(x - rl))
        else:
            # restrict the computational domain to only where the regions are mixed
            idxu = np.where( y1 != 1. )[0][-1] + 1
            idxl = np.where( y1 != 0. )[0][0] - 1
        if not silent:
            print(idxl, idxu)
            print(y1)
        y1 = y1[idxl:idxu]
        y0 = y0[idxl:idxu]
        x = x[idxl:idxu]
        if not silent:
            print(x[0], x[-1])

        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        if not silent:
            print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)

        dt = float(deltat)

        # Calculate D starting from outer boundary:
        D = np.zeros(len(x))
        m = len(x) - 1
        # now do the solution:
        for i in range(m,1,-1):
            xl = np.float64(x[i] - x[i-1])
            r = np.float64(y0[i] - y1[i])
            p = np.float64(dt * (y1[i] - y1[i-1]) / (xl * xl))
            if i == m:
                D[i] = np.float64(r / p)
            else:
                xr = np.float64(x[i+1] - x[i])
                xm = np.float64(xl + xr) / 2.
                q = np.float64(dt * (y1[i] - y1[i+1]) / (xr * xm))
                D[i] = np.float64((r - q * D[i+1]) / p)


        D = D * 1.e16 # Mm^2/s ==> cm^2/s
        if not silent:
            print(D)
        x = x * 1e8   # Mm ==> cm

        cb = utils.colourblind
        lsty = utils.linestyle
        if showfig:
            pl.figure()
            yplot = np.log10(y0long) if log_X else y0long
            pl.plot(xlong,yplot,\
                    marker='o',
                    color=cb(8),\
                    markevery=lsty(1)[1],\
                    mec = cb(8),
                    mew = 1.,
                    mfc = 'w',
                    label='$X_{'+str(fname1)+'}$')
            yplot = np.log10(y1long) if log_X else y1long
            pl.plot(xlong,yplot,\
                    marker='o',\
                    color=cb(9),\
                    lw=0.5,
                    markevery=lsty(2)[1],\
                    label='$X_{'+str(fname2)+'}$')
            lbl = '$\log_{10}\,X$ ' if log_X else '$X$ '
            pl.ylabel(lbl)
            pl.xlabel('$\mathrm{r\,/\,Mm}$')
            if Xlim is not None:
                pl.ylim(Xlim)
            else:
               pl.ylim(-8,0.1)
            pl.legend(loc='center right').draw_frame(False)
            if grid:
                pl.grid()
            pl.twinx()
            pl.plot(x/1.e8,np.log10(D),'k-',\
                    label='$D$') #'$D > 0$')

            if plot_Dlt0:
                pl.plot(x/1.e8,np.log10(-D),'k--',\
                        label='$D < 0$')
            pl.xlim((3.5, 9.5))
            if Dlim is not None:
                pl.ylim(Dlim)
            else:
                pl.ylim((8., 18.))
            pl.ylabel('$\log_{10}(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
            pl.legend(loc='upper right').draw_frame(False)

        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolvedownexp(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,niter=3,
                   debug=False,grid=False,FVaverage=False,tauconv=None,returnY=False):
        '''
        Solve diffusion equation sequentially by iterating over the spatial
        domain inwards from the upper boundary. This version of the method is
        explicit.

        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.

        Returns
        --------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)

        '''


        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        #        x = x * 1.e8

        def mf(fluid,fname):
            '''
                Get mass fraction profile of fluid 'fluid' at fname.
                '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y

        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting

            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)

            y0long = y0
            y1long = y1

        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if debug: print(len(xlong), len(y0long))

        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0

        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y0) == 0)[0]
        print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)

        dt = float(deltat)

        # Calculate D starting from outer boundary:
        D = np.zeros(len(x))
        m = len(x) - 1
        # now do the solution:
        for i in range(m,1,-1):
            xl = np.float64(x[i] - x[i-1])
            r = np.float64(y0[i] - y1[i])
            p = np.float64(dt * (y0[i] - y0[i-1]) / (xl * xl))
            if i == m:
                D[i] = np.float64(r / p)
            else:
                xr = np.float64(x[i+1] - x[i])
                xm = np.float64(xl + xr) / 2.
                q = np.float64(dt * (y0[i] - y0[i+1]) / (xr * xm))
                D[i] = np.float64((r - q * D[i+1]) / p)


        D = D * 1.e16 # Mm^2/s ==> cm^2/s
        x = x * 1e8   # Mm ==> cm

        pl.figure()
        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
                markevery=utils.linestyle(1)[1],\
                label='fluid above'+' '+str(fname1))
        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
                markevery=utils.linestyle(2)[1],\
                label='fluid above'+' '+str(fname2))
        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.xlabel('r / Mm')
        pl.ylim(-8,0.1)
        pl.legend(loc='lower left').draw_frame(False)
        if grid:
            pl.grid()
        pl.twinx()
        pl.plot(x/1.e8,np.log10(D),'k-',\
                label='$D$') #'$D > 0$')
        pl.plot(x/1.e8,np.log10(-D),'k--',\
                label='$D < 0$')
        pl.ylabel('$\log D\,/\,{\\rm cm}^2\,{\\rm s}^{-1}$')
        pl.legend(loc='upper right').draw_frame(False)

        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def plot_entrainment_rates(self,dumps,r1,r2,fit=False,fit_bounds=None,save=False,lims=None,ifig=4,
                              Q = 1.944*1.60218e-6/1e43,RR = 8.3144598,amu = 1.66054e-24/1e27,
                              airmu = 1.39165,cldmu = 0.725,fkair = 0.203606102635,
                              fkcld = 0.885906040268,AtomicNoair = 6.65742024965,
                              AtomicNocld = 1.34228187919):
        '''
        Plots entrainment rates for burnt and unburnt material

        Parameters
        ----------
        data_path : str
            data path
        r1 : float
            This function will only search for the convective
            boundary in the range between r1/r2
        r2 : float
        fit : boolean, optional
            show the fits used in finding the upper boundary
        fit_bounds : array
            The time to start and stop the fit for average entrainment
            rate units in minutes
        save : bool, optional
            save the plot or not
        lims : list, optional
            axes lims [xl,xu,yl,yu]

        Examples
        ---------

        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig plot_entrainment_rates.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: dumps = np.array(range(0,1400,100))
               .....: F4.plot_entrainment_rates(dumps,27.7,28.5)
        '''
        atomicnocldinv = 1./AtomicNocld
        atomicnoairinv = 1./AtomicNoair
        patience0 = 5
        patience = 10

        nd = len(dumps)
        t = np.zeros(nd)
        L_H = np.zeros(nd)

        t00 = time.time()
        t0 = t00
        k = 0
        for i in range(nd):
            t[i] = self.get('t', fname = dumps[i], resolution = 'l')[-1]

            if dumps[i] >= 620:
                L_H[i] = self.get('L_C12pg', fname = dumps[i], resolution = 'l', airmu = airmu, \
                                   cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair, \
                                   AtomicNocld = AtomicNocld, corr_fact = 1.0)
            else:
                L_H[i] = 0.

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd - i - 1)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

        ndot = L_H/Q
        X_H = fkcld*1./AtomicNocld
        mdot_L = 1.*amu*ndot/X_H
        dt = cdiff(t)
        m_HHe_burnt = (1e27/nuconst.m_sun)*np.cumsum(mdot_L*dt)

        m_HHe_present = self.entrainment_rate(dumps,r1,r2, var='vxz', criterion='min_grad', offset=-1., \
                        integrate_both_fluids=False, show_output=False, return_time_series=True)

        m_HHe_total = m_HHe_present + m_HHe_burnt
        if fit_bounds is not None:
            idx2 = list(range(np.argmin(t/60. < fit_bounds[0]), np.argmin(t/60. < fit_bounds[1])))
            print(idx2)
            m_HHe_total_fc2 = np.polyfit(t[idx2], m_HHe_total[idx2], 1)
            m_HHe_total_fit2 = m_HHe_total_fc2[0]*t[idx2] + m_HHe_total_fc2[1]

            mdot2 = m_HHe_total_fc2[0]
            mdot2_str = '{:e}'.format(mdot2)
            parts = mdot2_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            lbl2 = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                  format(mantissa, exponent)

        pl.close(ifig); pl.figure(ifig)
        if fit:
            pl.plot(t[idx2]/60., m_HHe_total_fit2, '-', color =  'k', lw = 0.5, \
                     zorder = 102, label = lbl2)
        pl.plot(t/60., m_HHe_present, ':', color = cb(3), label = 'present')
        pl.plot(t/60., m_HHe_burnt, '--', color = cb(6), label = 'burnt')
        pl.plot(t/60., m_HHe_total, '-', color = cb(5), label = 'total')
        pl.xlabel('t / min')
        pl.ylabel(r'M$_\mathrm{HHe}$ [M_Sun]')
        if lims is not None:
            pl.axis(lims)
        pl.gca().ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        pl.legend(loc = 0)
        pl.tight_layout()
        if save:
            pl.savefig('entrainment_rate.pdf')

    '''
    def plot_entrainment_rates(self,rp_set,dumps,r1,r2,burning_on_from=0,fit=False,fit_bounds=None,
                               save=False,lims=None,T9_func=None,return_burnt=False,Q = 1.944,
                               airmu = 1.39165,cldmu = 0.725,fkcld = 0.885906040268,
                               AtomicNocld = 1.34228187919):

        Plots entrainment rates for burnt and unburnt material

        Parameters
        ----------
        rp_set : rp_set instance
        r1/r2 : float
            This function will only search for the convective
            boundary in the range between r1/r2
        fit : boolean, optional
            show the fits used in finding the upper boundary
        fit_bounds : [int,int]
            The time to start and stop the fit for average entrainment
            rate units in minutes
        save : bool, optional
            save the plot or not
        lims : list, optional
            axes lims [xl,xu,yl,yu]


        amu = 1.66054e-24/1e27
        atomicnocldinv = 1./AtomicNocld
        patience0 = 5
        patience = 10

        nd = len(dumps)
        t = np.zeros(nd)
        L_C12C12 = np.zeros(nd)

        r = self.get('Y', fname=dumps[0], resolution='l')
        idx_top = np.argmin(np.abs(r - r1))
        idx = range(idx_top, len(r))
        dV = -4.*np.pi*r**2*cdiff(r)

        t00 = time.time()
        t0 = t00
        k = 0
        for i in range(nd):
            t[i] = self.get('t', fname = dumps[i], resolution = 'l')[-1]

            if dumps[i] >= burning_on_from:
                enuc_C12C12 = self.get('enuc_C12C12', dumps[i], airmu=airmu, \
                              cldmu=cldmu, fkcld=fkcld, AtomicNocld=AtomicNocld, \
                              Q=Q, T9_func=T9_func)

                rp = rp_set.get_dump(dumps[i])
                avg_fv = rp.get_table('fv')[0, ::-1, 0]
                sigma_fv = rp.get_table('fv')[3, ::-1, 0]

                avg_fv[avg_fv < 1e-9] = 1e-9
                eta = 1. + (sigma_fv/avg_fv)**2

                # limit eta where avg_fv --> 0
                eta[avg_fv < 1e-6] = 1.

                L_C12C12[i] = np.sum(eta[idx]*enuc_C12C12[idx]*dV[idx])

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and \
                (t_now - t00 >= patience0) and \
                (k ==0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd - i - 1)*time_per_dump
                print 'Processing will be done in {:.0f} s.'.format(time_remaining)
                t0 = t_now
                k += 1

        # Mass fraction of C12 in the lighter fluid.
        # N*fkcld*12*amu is the mass of C12 atoms in N atoms of the lighter fluid.
        # N*AtomicNocld*amu is the mass of all of the N atoms.
        X_C12 = fkcld*12./AtomicNocld

        # Destruction rate of C12 atoms.
        ndot = 2.*L_C12C12/(Q*1.60218e-6/1e43)
        mdot_L = 12.*amu*ndot/X_C12

        m_HHe_burnt = (1e27/nuconst.m_sun)*integrate.cumtrapz(mdot_L, x=t, initial=0.)

        m_HHe_present = self.entrainment_rate(dumps,r1,r2, var='vxz', criterion='min_grad', offset=-1., \
                            integrate_both_fluids=False, show_output=False, return_time_series=True)

        m_HHe_total = m_HHe_present + m_HHe_burnt

        if fit_bounds is not None:
            idx2 = range(np.argmin(t/60. < fit_bounds[0]), np.argmin(t/60. < fit_bounds[1]))
            print(idx2)
            m_HHe_total_fc2 = np.polyfit(t[idx2], m_HHe_total[idx2], 1)
            m_HHe_total_fit2 = m_HHe_total_fc2[0]*t[idx2] + m_HHe_total_fc2[1]

            mdot2 = m_HHe_total_fc2[0]
            mdot2_str = '{:e}'.format(mdot2)
            parts = mdot2_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            lbl2 = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                  format(mantissa, exponent)

        ifig = 1; pl.close(ifig); pl.figure(ifig)
        if fit:
            pl.plot(t[idx2]/60., m_HHe_total_fit2, '-', color =  'k', lw = 0.5, \
                     zorder = 102, label = lbl2)
        pl.plot(t/60., m_HHe_present, ':', color = cb(3), label = 'present')
        pl.plot(t/60., m_HHe_burnt, '--', color = cb(6), label = 'burnt')
        pl.plot(t/60., m_HHe_total, '-', color = cb(5), label = 'total')
        pl.xlabel('t / min')
        pl.ylabel(r'M$_\mathrm{HHe}$ [M_Sun]')
        if lims is not None:
            pl.axis(lims)
        pl.gca().ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        pl.legend(loc = 0)
        pl.tight_layout()
        if save:
            pl.savefig('entrainment_rate.pdf')

        if return_burnt:
            return m_HHe_burnt
    '''
    def entrainment_rate(self, cycles, r_min, r_max, var='vxz', criterion='min_grad', \
                         offset=0., integrate_both_fluids=False,
                         integrate_upwards=False, show_output=True, ifig0=1, \
                         silent=True, mdot_curve_label=None, file_name=None,
                         return_time_series=False):
        '''
        Function for calculating entrainment rates.

        Parameters
        ----------
        cycles : range
            cycles to get entrainment rate for
        r_min : float
            minimum radius to look for boundary
        r_max : float
            maximum radius to look for boundary

        Examples
        ---------
        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig entrainment_rate.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: dumps = np.array(range(0,1400,100))
               .....: F4.entrainment_rate(dumps,27.7,28.5)

        '''

        def regrid(x, y, x_int):
            int_func = scipy.interpolate.CubicSpline(x[::-1], y[::-1])
            return int_func(x_int)

        r = self.get('Y', fname = cycles[0], resolution='l')

        idx_min = np.argmin(np.abs(r - r_max))
        idx_max = np.argmin(np.abs(r - r_min))

        r_min = r[idx_max]
        r_max = r[idx_min]

        r = r[idx_min:(idx_max + 1)]
        r_int = np.linspace(r_min, r_max, num = 20.*(idx_max - idx_min + 1))
        dr_int = cdiff(r_int)

        time = np.zeros(len(cycles))
        r_b = np.zeros(len(cycles))
        r_top = np.zeros(len(cycles))
        for i in range(len(cycles)):
            time[i] = self.get('t', fname = cycles[i], resolution='l')[-1]

            if var == 'vxz':
                q = self.get('EkXZ', fname = cycles[i], resolution='l')[idx_min:(idx_max + 1)]**0.5
            else:
                q = self.get(var, fname = cycles[i], resolution='l')[idx_min:(idx_max + 1)]

            q_int = regrid(r, q, r_int)
            grad = cdiff(q_int)/dr_int

            if criterion == 'min_grad':
                idx_b = np.argmin(grad)
            elif criterion == 'max_grad':
                idx_b = np.argmax(grad)
            else:
                idx_b = np.argmax(np.abs(grad))

            r_b[i] = r_int[idx_b]
            r_top[i] = r_b[i]

            # Optionally offset the integration limit by a multiple of q's
            # scale height.
            if np.abs(grad[idx_b]) > 0.:
                H_b = q_int[idx_b]/np.abs(grad[idx_b])
                r_top[i] += offset*H_b

        timelong = time
        delta = 0.05*(np.max(time) - np.min(time))
        timelong = np.insert(timelong,0, timelong[0] - delta)
        timelong = np.append(timelong, timelong[-1] + delta)

        # fc = fit coefficients
        r_b_fc = np.polyfit(time, r_b, 1)
        r_b_fit = r_b_fc[0]*timelong + r_b_fc[1]
        r_top_fc = np.polyfit(time, r_top, 1)
        r_top_fit = r_top_fc[0]*timelong + r_top_fc[1]

        m_ir = np.zeros(len(cycles))
        r = self.get('Y', fname = cycles[0], resolution='l')
        r_int = np.linspace(np.min(r), np.max(r), num = 20.*len(r))
        dr_int = cdiff(r_int)
        for i in range(len(cycles)):
            if integrate_both_fluids:
                rho = self.get('Rho', fname = cycles[i], resolution='l')
            else:
                rho_HHe = self.get('Rho H+He', fname = cycles[i], resolution='l')
                FV_HHe = self.get('FV H+He', fname = cycles[i], resolution='l')
                rho = rho_HHe*FV_HHe

            rho_int = regrid(r, rho, r_int)

            idx_top = np.argmin(np.abs(r_int - r_top[i]))
            dm = 4.*np.pi*r_int**2*dr_int*rho_int

            if integrate_upwards:
                m_ir[i] = np.sum(dm[(idx_top + 1):-1])
            else:
                m_ir[i] = np.sum(dm[0:(idx_top + 1)])

        # fc = fit coefficients
        m_ir *= 1e27/nuconst.m_sun
        m_ir_fc = np.polyfit(time, m_ir, 1)
        m_ir_fit = m_ir_fc[0]*timelong + m_ir_fc[1]
        if integrate_upwards:
            mdot = -m_ir_fc[0]
        else:
            mdot = m_ir_fc[0]

        if show_output:
            cb = utils.colourblind
            pl.close(ifig0); fig1 = pl.figure(ifig0)
            pl.plot(time/60., r_top, color = cb(5), ls = '-', label = r'r$_\mathrm{top}$')
            pl.plot(time/60., r_b, color = cb(8), ls = '--', label = r'r$_\mathrm{b}$')
            pl.plot(timelong/60., r_top_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.plot(timelong/60., r_b_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.xlabel('t / min')
            pl.ylabel('r / Mm')
            xfmt = ScalarFormatter(useMathText = True)
            pl.gca().xaxis.set_major_formatter(xfmt)
            pl.legend(loc = 0)
            fig1.tight_layout()

            if not silent:
                print('r_b is the radius of the convective boundary.')
                print('r_b_fc = ', r_b_fc)
                print('dr_b/dt = {:.2e} km/s\n'.format(1e3*r_b_fc[0]))
                print('r_top is the upper limit for mass integration.')
                print('dr_top/dt = {:.2e} km/s'.format(1e3*r_top_fc[0]))

            max_val = np.max(m_ir)
            #if show_fits:
            max_val = np.max((max_val, np.max(m_ir_fit)))
            max_val *= 1.1 # allow for some margin at the top
            oom = int(np.floor(np.log10(max_val)))

            pl.close(ifig0 + 1); fig2 = pl.figure(ifig0 + 1)
            pl.plot(time/60., m_ir/10**oom, color = cb(5))
            mdot_str = '{:e}'.format(mdot)
            parts = mdot_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            #if show_fits:
            if integrate_upwards:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{a}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(-mantissa, exponent)
            else:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(mantissa, exponent)
            pl.plot(timelong/60., m_ir_fit/10**oom, color = cb(4), ls = '-', lw = 0.5, label = lbl)
            pl.xlabel('t / min')
            if integrate_upwards:
                sub = 'a'
            else:
                sub = 'e'
            ylbl = r'M$_{:s}$ / 10$^{{{:d}}}$ M$_\odot$'.format(sub, oom)
            if oom == 0.:
                ylbl = r'M$_{:s}$ / M$_\odot$'.format(sub)

            pl.ylabel(ylbl)
            yfmt = FormatStrFormatter('%.1f')
            fig2.gca().yaxis.set_major_formatter(yfmt)
            fig2.tight_layout()
            if integrate_upwards:
                loc = 1
            else:
                loc = 2
            pl.legend(loc = loc)
            if file_name is not None:
                fig2.savefig(file_name)

            if not silent:
                print('Resolution: {:d}^3'.format(2*len(r)))
                print('m_ir_fc = ', m_ir_fc)
                print('Entrainment rate: {:.3e} M_Sun/s'.format(mdot))

        if return_time_series:
            return m_ir
        else:
            return mdot

    def boundary_radius(self, cycles, r_min, r_max, var='vxz', \
                        criterion='min_grad', var_value=None):
        '''
        Calculates the boundary of the yprofile.

        Parameters
        ----------
        cycles : range
            range of yprofiles to calculate boundary for
        r_min : float
            min rad to look for boundary
        r_max : float
            max rad to look for boundary

        Returns
        --------
        rb : array
            boundary of cycles

        '''
        eps = 1e-9
        n_cycles = len(cycles)
        rb = np.zeros(n_cycles)

        r = self.get('Y', cycles[0], resolution='l')
        idx_r_min = np.argmin(np.abs(r - r_min))
        idx_r_max = np.argmin(np.abs(r - r_max))

        for i in range(n_cycles):
            if var == 'vxz':
                v = self.get('EkXZ', fname = cycles[i], resolution='l')**0.5
            else:
                v = self.get(var, cycles[i], resolution='l')

            if criterion == 'min_grad' or criterion == 'max_grad':
                # The following code always looks for a local minimum in dv.
                dvdr = cdiff(v)/cdiff(r)

                # A local maximum is found by looking for a local minimum in
                # -dvdr.
                if criterion == 'max_grad':
                    dvdr = -dvdr

                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(dvdr[idx_r_max:idx_r_min])
                r0 = r[idx0]

                # Try to pinpoint the radius of the local minimum by fitting
                # a parabola around r0.
                coefs = np.polyfit(r[idx0-1:idx0+2], dvdr[idx0-1:idx0+2], 2)

                if np.abs(coefs[0]) > eps:
                    r00 = -coefs[1]/(2.*coefs[0])

                    # Only use the refined radius if it is within the three
                    # cells.
                    if r00 < r[idx0-1] and r00 > r[idx0+1]:
                        r0 = r00
            elif criterion == 'value':
                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(np.abs(v[idx_r_max:idx_r_min] - var_value))
                r0 = r[idx0]

                if np.abs(v[idx0] - var_value) > eps:
                    # 1st-order refinement.
                    if idx0 > idx_r_max and idx0 < idx_r_min:
                        if (v[idx0-1] < var_value and v[idx0] > var_value) or \
                           (v[idx0-1] > var_value and v[idx0] < var_value):
                            slope = v[idx0] - v[idx0-1]
                            t = (var_value - v[idx0-1])/slope
                            r0 = (1. - t)*r[idx0-1] + t*r[idx0]
                        elif (v[idx0] < var_value and v[idx0+1] > var_value) or \
                            (v[idx0] > var_value and v[idx0+1] < var_value):
                            slope = v[idx0+1] - v[idx0]
                            t = (var_value - v[idx0])/slope
                            r0 = (1. - t)*r[idx0] + t*r[idx0+1]
                        else:
                            r0 = r_max

            rb[i] = r0

        return rb

    def vaverage(self,vi='v',transient=0.,sparse=1,showfig = False):
        '''
        plots and returns the average velocity profile for a given
        orientation (total, radial or tangential) over a range of dumps
        and excluding an initial user-specified transient in seconds.
        '''

        cycs = self.cycles
        time = self.get('t')
        istart = np.abs(time-transient).argmin()
        cycs = cycs[istart::sparse]

        Y    = self.get('Y',fname=1,resolution='l')

        if vi == 'v':
            Ei ='Ek'
            ylab='$\log~v_\mathrm{tot}$'
        if vi == 'vY':
            Ei = 'EkY'
            ylab='$\log~v_\mathrm{Y}$'
        if vi == 'vXZ':
            Ei = 'EkXZ'
            ylab='$\log~v_\mathrm{XZ}$'

        vav = np.zeros(len(Y))

        for cyc in cycs:
            Ek   = self.get(Ei,fname=cyc,resolution='l')
            if vi == 'v':
                v    = np.sqrt(2.*array(Ek,dtype=float))
            else:
                v    = np.sqrt(array(Ek,dtype=float))

            vav += v

        vav = vav * 1.e8 / len(cycs) # average in cm / s

        if showfig:
            pl.figure()
            pl.plot(Y,np.log10(vav),'r-')
            pl.ylabel(ylab)
            pl.xlabel('r / Mm')

        return vav

# below are some utilities that the user typically never calls directly
    def readTop(self,atri,filename,stddir='./'):
        """
        Private routine that Finds and returns the associated value for
        attribute in the header section of the file.

        Input:
        atri, what we are looking for.
        filename where we are looking.
        StdDir the directory where we are looking, Defaults to the
        working Directory.

        """
        if stddir.endswith('/'):
            filename = str(stddir)+str(filename)
        else:
            filename = str(stddir)+'/'+str(filename)
        f=open(filename,'r')
        headerLines=[]
        header=[]
        headerAttri=[]
        for i in range(0,10): # Read out the header section of the file.
            line = f.readline()
            line=line.strip()
            if line != '':
                headerLines.append(line)
        f.close()
        for i in range(len(headerLines)): #for each line of header data split up the occurances of '    '
            header.extend(headerLines[i].split('     '))
            header[i]=header[i].strip()

        for i in range(len(header)):
            tmp=header[i].split('=')# for each line split up on occurances of =
            if len(tmp)!=2: # If there are not two parts, add the unsplit line to headerAttri
                tmp=[]
                tmp.append(header[i].strip())
                headerAttri.append(tmp)
            elif len(tmp)==2: # If there are two parts, add the list of the two parts to headerAttri
                tmp[0]=tmp[0].strip()
                tmp[1]=tmp[1].strip()
                headerAttri.append([tmp[0],tmp[1]])

        for i in range(len(headerAttri)):
            if atri in headerAttri[i]: # if the header arrtibute equals atri, return its associated value
                value=headerAttri[i][1]

        value =value.partition(' ')
        value=value[0]
        return value

    def _readFile(self):
        """
        private routine that is not directly called by the user.
        filename is the name of the file we are reading
        stdDir is the location of filename, defaults to the
        working directory
        Returns a list of the header attributes with their values
        and a List of the column values that are located in this
        particular file and a list of directory attributes.

        Assumptions:
        An attribute can't be in the form of a num, if
        the user can float(attribute) without an error
        attribute will not be returned
        Lines of attributs are followd and preceded by
        *blank lines

        """
        filename = os.path.join(self.sldir,self.slname)
        f=open(filename,'r')
        line=''
        headerLines=[] # List of lines in the header section of the YProfile
        header=[]      # Single line of header data
        tmp=[]
        tmp2=[]
        headerAttri=[] # Final list of header Data to be retruned
        colAttri=[]    # Final list of column attributes to be returned
        cycAttri=[]    # Final list of cycle attributes to be returned
        for i in range(0,10): # read the first 10 lines of the YProfile
                              # Add the line to headerLines if the line is not empty
            line = f.readline()
            line=line.strip()
            if line != '':
                headerLines.append(line)

        for i in range(len(headerLines)): # For each line split on occurances of '    '
                                          # And then clean up any extra whitespace.
            header.extend(headerLines[i].split('     '))
            header[i]=header[i].strip()

        for i in range(len(header)):# for each line split up on occurances of =
            tmp=header[i].split('=')
            if len(tmp)!=2: # If there are not two parts, add the unsplit line to headerAttri
                tmp=[]
                tmp.append(header[i].strip())
                headerAttri.append(tmp)
            elif len(tmp)==2: # If there are two parts, add the list of the two parts to headerAttri
                tmp[0]=tmp[0].strip()
                tmp[1]=tmp[1].strip()
                headerAttri.append([tmp[0],tmp[1]])

        lines= f.readlines()
        boo = True
        ndump=False
        attri=[]
        for i in range(len(lines)-2): #for the length of the file
            if lines[i] =='\n'and lines[i+2]=='\n': # If there is a blank line,
                #that is followed by some line and by another blank line
                # it means the second line is a line of attributes
                line = lines[i+1] # line of attributes

                line=line.split('  ') # split it up on occurances of '  '

                for j in range(len(line)): #Clean up any excess whitespace
                    if line[j]!='':    #And add it to a list of attributes
                        attri.append(line[j].strip())

                for j in range(len(attri)):
                    """
                    if attri[j]=='Ndump':
                            i = len(lines)
                            break
                            """
                    for k in range(len(colAttri)):  #If it is not allready in the list of Attributes
                                                    # add it
                        if colAttri[k]==attri[j]:
                            boo = False
                            break
                    if boo :
                        colAttri.append(attri[j])
                    boo=True

        tmp=[]
        for i in range(len(colAttri)):#gets rid of blank lines in the list
            if colAttri[i]!='':
                tmp.append(colAttri[i])
        colAttri=tmp
        tmp=[]
        for i in range(len(colAttri)):#gets rid of numbers in the list
            try:
                float(colAttri[i])
            except ValueError:
                tmp.append(colAttri[i])
        colAttri=tmp
        tmp=[]
        # NOTE at this point in the program colAttri is a unified list of Column attributes and Cycle Attributes


        for i in range(len(colAttri)): #Here we split up our list into Column attributes and Cycle Attributes
            if colAttri[i]=='Ndump':
                # If we get to Ndump in our list of attributes, then any following attributes are cycle attributes
                ndump=True
            if not ndump:
                tmp.append(colAttri[i])
            else:
                cycAttri.append(colAttri[i])

        colAttri=tmp
        f.close()
        return headerAttri,colAttri, cycAttri

    def spacetime_diagram(self, var_name, nt, fig, tlim=None, rlim=None, vlim=None, logscale=True, \
                  cmap='viridis', aspect=1./3., zero_intervals=None, patience0 = 5, patience = 30, \
                  **kwargs):
        '''
        Creates a spacetime diagram.

        Parameters
        -----------
        var_name : str
            variable to plot
        nt : int
            size of time vector, t = np.linspace(t0,tf,nt)

        Examples
        --------
        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig space_time.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: import matplotlib.pyplot as plt
               .....: fig2 = plt.figure(19)
               .....: F4.spacetime_diagram('Ek',5,fig2)

        '''
        if var_name == 'Ek':
            cbar_lbl = r'e$_\mathrm{k}$ / erg g$^{-1}$'
            unit = 1e43/1e27
        elif var_name == 'enuc_C12pg':
            cbar_lbl = r'$\epsilon_\mathrm{C12pg}$ / erg cm$^{-3}$ s$^{-1}$'
            unit = 1e43/1e24
        elif var_name == 'enuc_C12C12':
            cbar_lbl = r'$\epsilon_\mathrm{C12C12}$ / erg cm$^{-3}$ s$^{-1}$'
            unit = 1e43/1e24
        else:
            cbar_lbl = var_name
            unit = 1.

        r = self.get('Y', fname = 0, resolution = 'l')
        if rlim is None:
            rlim = [r[-1], r[0]]

        ridx0 = np.argmin(np.abs(r - rlim[1]))
        ridx1 = np.argmin(np.abs(r - rlim[0]))
        nr = ridx1-ridx0+1
        ridx = np.linspace(ridx0, ridx1, num=nr, dtype=np.int32)

        r = r[ridx]

        if tlim is None:
            tlim = [0., 0.]
            tlim[0] = self.get('t', silent = True)[0]
            tlim[1] = self.get('t', silent = True)[-1]
        t = np.linspace(tlim[0], tlim[1], nt)

        zero = np.zeros(nt, dtype = bool)
        if zero_intervals is not None:
            for i in range(len(zero_intervals)/2):
                idx = where((t >= zero_intervals[2*i]) & \
                            (t <= zero_intervals[2*i + 1]))
                zero[idx] = True

        var = np.zeros((nr, nt))

        t00 = time.time()
        t0 = t00
        n = 0
        k = 0
        n_nonzero = nt - np.count_nonzero(zero)
        for i in range(nt):
            if zero[i]:
                continue

            var[:, i] = unit*self.get(var_name, fname = t[i], numtype = 'time', \
                                      resolution = 'l', silent = True, **kwargs)[ridx]

            n += 1
            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(n)
                time_remaining = (n_nonzero - n - 1.)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

        if vlim is None:
            if logscale:
                vlim = [np.min(var[where(var > 0)]), \
                        np.max(var[where(var > 0)])]
            else:
                vlim = [np.min(var), np.max(var)]

            print('vlim = [{:.3e}, {:.3e}]'.format(vlim[0], vlim[1]))

        var[where(var < vlim[0])] = vlim[0]
        var[where(var > vlim[1])] = vlim[1]
        ax1 = fig.add_subplot(111)

        extent = (t[0]/60., t[-1]/60., r[-1], r[0])
        aspect *= (extent[1] - extent[0])/(extent[3] - extent[2])

        if logscale:
            norm = colors.LogNorm(vmin=vlim[0], vmax=vlim[1], clip=True)
        else:
            norm = colors.Normalize(vmin=vlim[0], vmax=vlim[1], clip=True)

        ax1i = ax1.imshow(var, aspect = aspect, cmap = cmap, extent = extent, \
                        norm=norm, interpolation = 'spline16')
        ax1.get_yaxis().set_tick_params(which='both', direction='out')
        ax1.get_xaxis().set_tick_params(which='both', direction='out')
        ax1.set_xlabel('t / min')
        ax1.set_ylabel('r / Mm')

        cbar = fig.colorbar(ax1i, orientation='vertical')
        cbar.set_label(cbar_lbl)


##########################################################
# mapping of visualisation variables from Robert Andrassy:
##########################################################

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
        print('first/last colour indices:', indices_normed[-1])
        print('correcting to 1.')
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

        Examples
        ---------
        import ppm
        lut = ppm.LUT('./ /BW-1536-UR-3.lut', s0=5., s1=245.499,
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

        Parameters
        -----------
        ticks: numpy array
            at which values of the variable you would like to have ticks on the
            colourbar
        scale_factor: float
            dividing your ticks by this number should give the real values in
            code units. so, if I give ticks in km/s, I should give
            scale_factor=1.e3

        Examples
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
            print('ticks out of range')
            return

        print('min/max ticks being set to:', minval, maxval)
        print('corresponding to colour indices:', minidx, maxidx)
        print('ticks being placed at:', ticks)
        print('with colour indices:', colour_index_ticks)

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

        print(ileft, iright, minidx, maxidx)

        to_splice = copy.deepcopy(colours)[ileft:iright]
        newcolours = [[minidx, (rl, gl, bl)]] + to_splice + [[maxidx, (rr, gr, br)]]

        # now normalise the indices to [0-255]
        indices = np.array([c[0] for c in newcolours])
        newindices = 255.*(indices - np.min(indices)) / indices.ptp()
        # renormalise index tick locations as well
        colour_index_ticks = 255.*\
            (colour_index_ticks - np.min(colour_index_ticks)) / colour_index_ticks.ptp()
        print('new colour indices:', newindices)
        print('ticks now at:', colour_index_ticks)

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
        cbar.ax.tick_params(axis='both', which='both',length=0,labelsize=6)
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
    rp : radial profile
        radial profile
    r1 : float
        minimum radius for the search for r_ub
    r2 : float
        maximum radius for the search for r_ub\

    Returns
    ------
    r : array
        radius
    ut : array
        RMS tangential velocity profiles for all buckets (except the 0th)
    dutdr : array
        radial gradient of ut for all buckets (except the 0th)
    r_ub : array
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

def upper_bound_ut(data_path, dump_to_plot, hist_dump_min,
                   hist_dump_max, r1, r2, ylims = None, derivative = False, silent = True):

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
    derivative : bool
        True = plot dut/dr False = plot ut
    dump_To_plot : int
        The file number of the dump you wish to plot
    hist_dump_min/hist_dump_max = int
        Range of file numbers you want to use in the histogram
    r1/r2 : float
        This function will only search for the convective
        boundary in the range between r1/r2

    Examples
    --------

    .. ipython::

        In [136]: data_path = '/data/ppm_rpod2/RProfiles/AGBTP_M2.0Z1.e-5/F4/'
           .....: dump = 560; hist_dmin = dump - 1; hist_dmax = dump + 1
           .....: r_lim = (27.0, 30.5)

        @savefig upper_bound.png width=6in
        In [136]: ppm.upper_bound_ut(data_path,dump, hist_dmin, hist_dmax,r1 = r_lim[0],r2 = 31, derivative = False,ylims = [1e-3,19.])

    '''
    cb = utils.colourblind
    rp_set = bprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)
    nr = len(rp.get('y'))

    sparse = 1
    dumps = np.array([rp_set.dumps[i] for i in range(0, len(rp_set.dumps), sparse)])

    n_dumps = len(dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))
    ut = np.zeros((nr, n_buckets, n_dumps))
    dutdr = np.zeros((nr, n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut[:, :, k] = res[1]
        dutdr[:, :, k] = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)
    idx = np.argmin(np.abs(dumps - dump_to_plot))

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
    if not silent:
        print("Dump {:d} (t = {:.2f} min).".format(dump_to_plot, t[dump_to_plot]/60.))
        print("Histogram constructed using dumps {:d} (t = {:.2f} min) to {:d} (t = {:.2f} min) inclusive."\
            .format(hist_dump_min, t[hist_dump_min]/60., hist_dump_max, t[hist_dump_max]/60.))

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

        ax0.plot(r, temp[:, bucket, idx], ls = '-', lw = 0.5, color = cb(3), \
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
    if ylims is not None:
        ax0.set_ylim(ylims)
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

def get_avg_rms_velocities(prof, dumps, comp):
    '''
    Finds an average velocity vector as a function of radius for a given
    range of dumps. To find a velocity vector as a function of radius see
    get_v_evolution().

    Parameters
    ----------
    yprof : yprofile instance
        profile to examine
    dumps : range
        dumps to average over
    comp : string
        component to use 'r':'radial','t':'tangential','tot':'total'
    '''
    avg_rms_v = 0.

    for d in dumps:
        if comp == 'tot':
            rms_v = np.sqrt(2.*prof.get('Ek', fname=d, resolution='l'))
        elif comp == 'r':
            rms_v = np.sqrt(prof.get('EkY', fname=d, resolution='l'))
        elif comp == 't':
            rms_v = np.sqrt(prof.get('EkXZ', fname=d, resolution='l'))

        avg_rms_v += rms_v

    avg_rms_v /= float(len(dumps))

    return avg_rms_v

def get_v_evolution(prof, cycles, r1, r2, comp, RMS):
    '''
    Finds a velocity vector and a time vector for a range of cycles

    Parameters
    ----------
    prof : yprofile object
        prof to look at
    cycles : range
        cycles to look at
    r1/r1 : float
        boundaries of the range to look for v max in
    comp : string
        velocity component to look for 'radial' or 'tangential'
    RMS : string
        'mean' 'min' or 'max velocity
    '''

    r = prof.get('Y', fname = cycles[0], resolution = 'l')
    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    t = np.zeros(len(cycles))
    v = np.zeros(len(cycles))
    for k in range(len(cycles)):
        t[k] = prof.get('t', fname = cycles[k], resolution = 'l')[-1]
        if comp == 'radial':
            v_rms  = prof.get('EkY', fname = cycles[k], resolution = 'l')**0.5
        elif comp == 'tangential':
            v_rms  = prof.get('EkXZ', fname = cycles[k], resolution = 'l')**0.5
        elif comp == 'tot':
            v_rms = (2.*prof.get('Ek', fname = cycles[k], resolution = 'l'))**0.5
        if RMS == 'mean':
            v[k] = np.mean(v_rms[idx2:idx1])
        elif RMS == 'min':
            v[k] = np.min(v_rms[idx2:idx1])
        elif RMS == 'max':
            v[k] = np.max(v_rms[idx2:idx1])
    return t, v

def v_evolution(cases, ymin, ymax, comp, RMS, sparse = 1, markevery = 25, ifig = 12,
                dumps = [0,-1], lims = None):

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
    comp : string
        component that the velocity is in. option: 'radial', 'tangential'
    RMS : string
        options: 'min', 'max' or 'mean'. What aspect of the RMS velocity
        to look at
    dumps : int, optional
        Gives the range of dumps to plot if larger than last dump
        will plot up to last dump
    lims : array
        axes lims [xl,xu,yl,yu]

    Examples
    ---------

    ppm.set_YProf_path('/data/ppm_rpod2/YProfiles/O-shell-M25/',YProf_fname='YProfile-01-0000.bobaaa')
    ppm.v_evolution(['D15','D2','D1'], 4., 8.,'max','radial')

    '''
    pl.close(ifig),pl.figure(ifig)
    cb = utils.colourblind
    ls = utils.linestylecb
    yy = 0
    for case in cases:

        try:
            prof = yprofile(os.path.join(ppm_path,case))
        except ValueError:
            print("have you set the yprofile filepath using ppm.set_YProf_path?")
        if dumps[1] > prof.cycles[-1]:
            end = -1
        else:
            end = dumps[1]
        cycles = list(range(prof.cycles[dumps[0]], prof.cycles[end], sparse))
        t, vr_max = get_v_evolution(prof, cycles, ymin, ymax, comp, RMS)
        pl.plot(t/60.,  1e3*vr_max,  color = cb(yy),\
                 marker = ls(yy)[1], markevery = markevery, label = case)
        yy += 1

    pl.xlabel('t / min')
    pl.ylabel(r'v$_r$ / km s$^{-1}$')
    if lims is not None:
        pl.axis(lims)
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

        rp_set = bprof.rprofile_set(path)
        dumps = list(range(rp_set.dumps[0],\
                      #rp_set.dumps[0]+100,1)
                      rp_set.dumps[-1]+1,1))
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
                                  cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair,
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
            print('Processing will be done in {:.0f} s.'.format(time_remaining))
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
    pl.semilogy(t/60., (1e43/nuconst.l_sun)*L_H_yp, color = cb(6), \
                 zorder = 2, label = r'L$_\mathrm{H}$')
    pl.axhline((1e43/nuconst.l_sun)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

    ifig = 2; pl.close(ifig); pl.figure(ifig)
    pl.semilogy(t/60., (1e43/nuconst.l_sun)*L_H_yp, color = cb(5), \
                 lw = 2., zorder = 2, label = r'L$_\mathrm{H,yp}$')
    pl.semilogy(t/60., (1e43/nuconst.l_sun)*L_H_rp, color = cb(6), \
                 zorder = 4, label = r'L$_\mathrm{H,rp}$')
    pl.axhline((1e43/nuconst.l_sun)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

def L_H_L_He_comparison(cases, sparse = 1, ifig=101,L_He = 2.25*2.98384E-03,airmu=1.39165,cldmu=0.725,
    fkair=0.203606102635,fkcld=0.885906040268,AtomicNoair=6.65742024965,
    AtomicNocld=1.34228187919,markevery=1,lims=None,save=False):

    '''
    Compares L_H to L_He, optional values are set to values from O-shell
    burning paper.

    Parameters
    ----------
    cases : string array
        names of yprofile instances i.e['D1','D2'...]
    sparse :int
        what interval in the range to calculate
        1 calculates every dump 2 every second dump ect
    lims : array
        plot limits
    save : boolean
        save figure
    '''
    yprofs = {}
    res = {}

    for case in cases:

        try:
            yprofs[case] = yprofile(os.path.join(ppm_path,case))
        except ValueError:
            print("have you set the yprofile filepath using ppm.set_YProf_path?")

        r = yprofs[case].get('Y', fname=0, resolution='l')
        res[case] = 2*len(r)

    patience0 = 5
    patience = 60

    dumps = {}
    nd = {}
    t = {}
    L_H = {}

    for this_case in cases:
        print('Processing {:s}...'.format(this_case))

        dumps[this_case] = np.arange(min(yprofs[this_case].ndumpDict.keys()),\
           max(yprofs[this_case].ndumpDict.keys()) + 1, sparse)
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
            L_H[this_case][i] = yprofs[this_case].get('L_C12pg', fname = dumps[this_case][i],
                                resolution = 'l', airmu = airmu, cldmu = cldmu,
                                fkair = fkair, fkcld = fkcld,  AtomicNoair = AtomicNoair,
                                AtomicNocld = AtomicNocld, corr_fact = 1.5)

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd[this_case] - i - 1)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

    pl.close(ifig); pl.figure(ifig)
    pl.axhline((1e43/nuconst.l_sun)*L_He, ls = '--', color = cb(4), \
        label = r'L$_\mathrm{He}$')

    markers = ['o','v', '^', '<', '>', 's']
    colours = [5, 8, 1, 6, 9, 3]

    i =0;j =0;nn=0 # super hacky hack for nice colours

    for this_case in cases:
        lbl = r'{:s} $\left({:d}^3\right)$'.format(this_case, res[this_case])
        pl.semilogy(t[this_case]/60., (1e43/nuconst.l_sun)*L_H[this_case], \
            ls = '-', color = cb(colours[i]), marker= markers[j], \
            label = this_case)
        i+=1
        j+=1
        if j == 6:
            nn+=1
            i = 0
            j = nn
    if lims is not None:
        pl.axis(lims)
    pl.xlabel('t / min')
    pl.ylabel(r'L$_H$ / L$_\odot$')
    pl.legend(loc=0, ncol=2)
    pl.tight_layout()
    if save:
        pl.savefig('L_H-L_He_'+cases[0]+cases[1]+cases[2]+'.pdf')

def get_upper_bound(data_path, r1, r2, sparse = 10):

    '''
    Returns information about the upper convective boundary

    Parameters
    ----------
    data_path : string
        path to rprofile set
    r1/r2 = float
        This function will only search for the convective
        boundary in the range between r1/r2
    sparse : int
        What interval to plot data at

    Returns
    ------
    [all arrays]
    avg_r_ub : average radius of upper boundary
    sigmam_r_ub/sigmap_r_ub: 2 \sigma fluctuations in upper boundary
    r_ub : upper boundary
    t: time

    '''

    rp_set = bprof.rprofile_set(data_path)

    n_dumps = len(rp_set.dumps)
    nt = len(list(range(0,n_dumps,sparse)))
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')

    t = np.zeros(nt)
    r_ub = np.zeros((n_buckets, nt))
    ll = 0
    for k in range(0,n_dumps,sparse):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[ll] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut = res[1]
        dutdr = res[2]
        r_ub[:, ll] = res[3]
        ll +=1

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(nt)
    sigmam_r_ub = np.zeros(nt)

    for k in range(nt):
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

def get_r_int(data_path, r_ref, gamma, sparse = 1):
    '''
    Returns values for radius of interface for a given
    reference radius.

    Parameters
    ----------
    data_path : string
        data path
    r_ref : float
        r_int is the radius of the interface at which the relative
        [A(t) - A0(t)]/A0(t) with respect to a reference value A0(t)
        exceeds a threshold. The reference value A0(t) is taken to be
        the average A at a reference radius r_ref at time t.
        We need to put r_ref to the bottom of the convection zone,
        because most of the convection zone is heated in the
        final explosion. The threshold should be a few times
        larger than the typical bucket-to-bucket fluctuations
        in A due to convection. A value of 1e-3 is a good trade-off
        between sensitivity and the amount of noise.
    sparse: int, optional
        What interval to have between data points

    Returns
    ------
    [all arrays]
    avg_r_int : average radius of interface
    sigmam_r_int/sigmap_r_int: 2 \sigma fluctuations in r_int
    r_int : interface radius
    t: time
    '''
    rp_set = bprof.rprofile_set(data_path)
    rp = rp_set.get_dump(rp_set.dumps[0])
    n_buckets = rp.get('nbuckets')
    r = rp.get('y')
    idx_ref = np.argmin(np.abs(r - r_ref))

    dumps = np.arange(1, 1960, sparse)
    nd = len(dumps)

    dr = r[1] - r[0]
    r_int = np.zeros((n_buckets, nd))
    avg_r_int = np.zeros(nd)
    sigmap_r_int = np.zeros(nd)
    sigmam_r_int = np.zeros(nd)
    t = np.zeros(nd)

    for i in range(nd):
        rp = rp_set.get_dump(dumps[i])
        p = rp.get_table('p')[0, :, :]
        rho = rp.get_table('rho')[0, :, :]
        A = p/rho**gamma
        A0 = A[idx_ref, 0]
        t[i] = rp.get('time')

        for bucket in range(1, n_buckets + 1):
            rel_diff = (A[:, bucket] - A0)/A0

            threshold = 1e-3
            # 0-th order estimate first
            idx_top = np.argmax(rel_diff > threshold)
            r_int[bucket - 1, i] = r[idx_top]

            # refine to 1-st order now
            slope = (rel_diff[idx_top] - rel_diff[idx_top - 1])/dr
            r_int[bucket - 1, i] -= (rel_diff[idx_top] - threshold)/slope

        avg_r_int[i] = np.sum(r_int[:,i])/float(n_buckets)
        dev = np.array([r_int[b,i] - avg_r_int[i] for b in range(n_buckets)])
        devp = dev[dev >= 0]
        if len(devp) > 0:
            sigmap_r_int[i] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmap_r_int[i] = None

        devm = dev[dev <= 0]
        if len(devm) > 0:
            sigmam_r_int[i] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_int[i] = None

    return avg_r_int, sigmam_r_int, sigmap_r_int, r_int, t

def plot_boundary_evolution(data_path, r1, r2, t_fit_start=700,
                            silent = True, show_fits = False, ifig = 5,
                            r_int = False, r_ref = None, gamma = None,
                            sparse = 10, lims = None, insert = False):

    '''

    Displays the time evolution of the convective boundary or Interface radius

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    data_path : string
        data path
    r1/r2 : float
        This function will only search for the convective
        boundary in the range between r1/r2
    show_fits : boolean, optional
        show the fits used in finding the upper boundary
    t_fit_start : int
        The time to start the fit for upper boundary fit takes
        range t[t_fit_start:-1] and computes average boundary
    r_int : bool, optional
        True plots interface radius, False plots convective boundary
        !If true r_ref must have a value
    r_ref : float
        r_int is the radius of the interface at which the relative
        [A(t) - A0(t)]/A0(t) with respect to a reference value A0(t)
        exceeds a threshold. The reference value A0(t) is taken to be
        the average A at a reference radius r_ref at time t.
        We need to put r_ref to the bottom of the convection zone,
        because most of the convection zone is heated in the
        final explosion. The threshold should be a few times
        larger than the typical bucket-to-bucket fluctuations
        in A due to convection. A value of 1e-3 is a good trade-off
        between sensitivity and the amount of noise.
    sparse: int, optional
        What interval to have between data points
    lims : array
        limits of the insert axes [xl,xu,yl,yu]
    insert: bool
        whether or not to include a second inserted subplot

    '''
    cb = utils.colourblind

    rp_set = bprof.rprofile_set(data_path)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')

    if not r_int:
        res = get_upper_bound(data_path, r1, r2,sparse = sparse)
    else:
        res = get_r_int(data_path, r_ref, gamma, sparse = sparse)

    avg_r_ub = res[0]
    sigmam_r_ub = res[1]
    sigmap_r_ub = res[2]
    r_ub = res[3]
    t = res[4]

    if show_fits:
        idx_fit_start = np.argmin(np.abs(t - t_fit_start))
        t_fit_start = t[idx_fit_start]

        # fc = fit coefficients
        fc_avg = np.polyfit(t[idx_fit_start:-1], avg_r_ub[idx_fit_start:-1], 1)
        avg_fit = fc_avg[0]*t + fc_avg[1]
        fc_plus = np.polyfit(t[idx_fit_start:-1], 2.*sigmap_r_ub[idx_fit_start:-1], 1)
        plus_fit = fc_plus[0]*t + fc_plus[1]
        fc_minus = np.polyfit(t[idx_fit_start:-1], 2.*sigmam_r_ub[idx_fit_start:-1], 1)
        minus_fit = fc_minus[0]*t + fc_minus[1]
    if not insert:
        pl.close(ifig); fig = pl.figure(ifig)#, figsize = (6.0, 4.7))
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
        pl.xlabel('t / min')
        if r_int:
            pl.ylabel(r'r$_\mathrm{int}$ / Mm')
        else:
            pl.ylabel(r'r$_\mathrm{ub}$ / Mm')
        pl.legend(loc = 0, frameon = False)
        if not silent:
            print('The fitting starts at t = {:.1f} s = {:.1f} min.'.format(t_fit_start, t_fit_start/60.))
            print('')
            print('Average:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_avg[1], fc_avg[0]))
            print('')
            print('Positive fluctuations:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_plus[1], fc_plus[0]))
            print('')
            print('Negative fluctuations:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_minus[1], fc_minus[0]))
    else:
        def plot_data(ax):
            for i in range(n_buckets):
                lbl=''
                if i == 0:
                    lbl='bucket data'
                ax.plot(t/60., r_ub[i,:], '-', lw=0.5, color=cb(3), label=lbl)

            ax.plot(t/60., avg_r_ub, '-', lw=1.0, color=cb(4), label='mean')
            ax.plot(t/60., avg_r_ub + 2*sigmap_r_ub, '--', lw=1., color=cb(4), \
                     label = r'2$\sigma$ fluctuations')
            ax.plot(t/60., avg_r_ub - 2*sigmam_r_ub, '--', lw=1., color=cb(4))

        ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
        ax1 = fig.add_subplot(111)
        plot_data(ax1)
        ax1.set_xlim((0., t[-1]/60))
        ax1.set_ylim((12., 30.))
        ax1.set_xlabel('t / min')
        ax1.set_ylabel(r'r$_\mathrm{int}$ / Mm')
        ax1.legend(loc=0)

        left, bottom, width, height = [0.29, 0.44, 0.35, 0.35]
        ax2 = fig.add_axes([left, bottom, width, height])
        plot_data(ax2)
        ax2.set_xlim((lims[0], lims[1]))
        ax2.set_ylim((lims[2], lims[3]))
        fig.tight_layout()

def compare_entrained_material(yps, labels, fname, ifig = 1):

    '''
    Compares the entrainment rate of two seperate yprofile objects

    Parameters
    ----------
    yps: yprofile objects
        yprofiles to compare
    labels : string array matching yp array
        labels for yp objects
    fname
        specific dump to access information from

    Examples
    --------
    import ppm
    yp1 = yprofile('/data/ppm_rpod2/YProfiles/agb-entrainment-convergence/H1')
    yp2 = yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D2')
    compare_entrained_material([yp1,yp2],['O shell','AGB'], fname = 271)
    '''

    pl.close(ifig); fig = pl.figure(ifig)
    cb = utils.colourblind
    ls = utils.linestylecb
    me = 0.1
    yy = 0
    for yp in yps:
        pl.plot( yp.get('j',resolution='L', fname = fname),
              yp.get('FV H+He',resolution='L', fname = fname), \
              color = cb(yy), label=labels[yy],
              marker=ls(yy)[1], markevery = me,
              markeredgecolor = 'w')
        yy += 1
    pl.legend(loc=(.6,.1))

    pl.ylabel('$\mathrm{FV}\,\mathcal{F}_1$')
    pl.xlabel('grid cell number')


    ax=pl.gca()
    ax.set_yscale('log')
    #yticks = ax.yaxis.get_major_ticks()
    pl.yticks(10.**np.linspace(-12,0,7))
    fig.subplots_adjust( left = 0.17 )

########################################################################
# Plotting funcitons that dont have supported dependencies
########################################################################

def get_power_spectrum_RProfile(yprof_path, rprof_path, r0, t_lim=None, t_res=None, l_max=6):
    '''
    Calculates power spectra of radial velocity

    Parameters
    ----------
    yprof_path/rprof_path: strings
        paths for matching rprofile and yprofile files
    r0: float
        radius to examine
    t_lim: 2 index array
        [t_min, t_max], data to look at
    t_res: int
        set number of data points between [t_min, t_max], None = ndumps
    l_max:
        maximum spherical harmonic degree l

    Returns
    --------
    t,l,power : time, spherical harmonic degree l, and power vectors
    '''
    yprof = ppm.yprofile(yprof_path, filename_offset=-1)
    n_buckets = 80
    rp_set = rprofile.rprofile_set(rprof_path)
    n_dumps = rp_set.dumps[-1]

    if t_lim is None:
        t_min = rp_set.get_dump(1).get('time')
        t_max = rp_set.get_dump(n_dumps).get('time')
        t_lim = [t_min, t_max]

    if t_res is None:
        t_res = n_dumps

    t = np.linspace(t_lim[0], t_lim[1], t_res)

    l = np.arange(0, l_max + 1)
    power = np.zeros((len(t), l_max + 1))

    for i in range(len(t)):
        dump = int(yprof.get('Ndump', fname=t[i], numtype='t', silent=True)[-1])
        if dump > n_dumps:
            dump = n_dumps
        rp = rp_set.get_dump(dump)
        r = rp.get_table('y')
        idx0 = np.argmin(np.abs(r - r0))

        vx = rp.get_table('ux')
        vy = rp.get_table('uy')
        vz = rp.get_table('uz')
        centers = rp.get_centers()

        lat = np.zeros(n_buckets)
        lon = np.zeros(n_buckets)
        vr = np.zeros(n_buckets)
        for bucket in range(n_buckets):
            x = centers[0, bucket]
            y = centers[1, bucket]
            z = centers[2, bucket]
            bucket_r = (x**2 + y**2 + z**2)**0.5
            lat[bucket] = 90. - (180./np.pi)*np.arccos(z/bucket_r)
            lon[bucket] = (180./np.pi)*np.arctan2(y, x)

            r_norm = np.array([x, y, z])/bucket_r
            v = np.array([vx[0, idx0, bucket+1], \
                          vy[0, idx0, bucket+1], \
                          vz[0, idx0, bucket+1]])
            vr[bucket] = np.dot(v, r_norm)

        coeffs, _ = SHExpandLSQ(vr, lat, lon, l_max)
        power[i, :] = spectrum(coeffs, convention='power', unit='per_l')

    return t, l, power

def plot_power_spectrum_RProfile(t, l, power, ifig=1, title='', vmin=1e-2, vmax=1.):
    '''
    Plots power spectra of radial velocity

    Parameters
    ----------
    t,l,power : arrays
        time, sperical harmonic degree and power generated by
        get_power_spectrum_RProfile()
    title : string
        title
    '''
    ifig = ifig; pl.close(ifig); pl.figure(ifig, figsize=(8., 5.), dpi=125)
    extent = (t[0]/60., t[-1]/60., l[0] - 0.5, l[-1] + 0.5)
    aspect = 0.5*(extent[1] - extent[0])/(extent[3] - extent[2])
    max_power = np.max(power)
    norm = LogNorm(vmin=vmin*max_power, vmax=vmax*max_power, clip=True)
    #norm = Normalize(vmin=0., vmax=max_power, clip=True)
    pl.imshow(np.transpose(np.abs(power)), origin='lower', extent=extent, aspect=aspect, \
               interpolation='nearest', norm=norm, cmap='viridis')
    cb = pl.colorbar()
    cb.set_label('m$^2$ (s$^2$ l)$^{-1}$')
    pl.xlabel('t / min')
    pl.ylabel('l')
    pl.title(title, y=1.025)
    ax0= pl.gca()
    ax0.get_yaxis().set_tick_params(direction='out')
    ax0.get_xaxis().set_tick_params(direction='out')

def bucket_map(rprofile, quantity, limits = None, ticks = None, file_name = None, time = None):

    '''
    Plots a Mollweide projection of the rprofile object using the mpl_toolkits.basemap package

    Parameters
    -----------
    rprofile: rprofile object
        rprofile dump used just for geometry
    quantity: array
        data to be passed into the projection
    limits: 2 index array
        cmap limits, scale the colormap for limit[0] =min to
        limit[1] =max
    ticks:
        passed into matplotlib.colors.ColorbarBase see ColourbarBase
    file_name: string
        file name: '/path/filename' to save the image as
    time: float
        time to display as the title

    '''
    q = quantity#rp.get_table(quantity)[0, :, :]
    #r = rp.get('y')
    #idx_r0 = np.argmin(np.abs(r - r0))

    corners = rprofile.get_corners()
    corners_per_bucket = corners.shape[1]
    n_buckets = corners.shape[2]
    points_per_side = 10
    points_per_bucket = corners_per_bucket*points_per_side

    x = np.zeros((n_buckets, points_per_bucket))
    y = np.zeros((n_buckets, points_per_bucket))
    z = np.zeros((n_buckets, points_per_bucket))
    t = np.linspace(1., 0., num = points_per_side)
    for i in range(n_buckets):
        for k in range(corners_per_bucket):
            idx_range = list(range(points_per_side*k, points_per_side*(k + 1)))
            x[i, idx_range] = t*corners[0, k - 1, i] + (1. - t)*corners[0, k, i]
            y[i, idx_range] = t*corners[1, k - 1, i] + (1. - t)*corners[1, k, i]
            z[i, idx_range] = t*corners[2, k - 1, i] + (1. - t)*corners[2, k, i]

    radius = (x**2 + y**2 + z**2)**0.5
    phi = np.arctan2(y, x)
    theta = np.pi/2. - np.arccos(z/radius)

    eps = 1e-3
    for i in range(phi.shape[0]):
        for k in range(phi.shape[1] - 1):
            # if the vertex k is at one of the poles
            if (np.abs(theta[i, k] - 0.5*np.pi) < eps or
                np.abs(theta[i, k] + 0.5*np.pi) < eps):
                if (theta[i, k] == theta[i, k - 1] and
                    phi[i, k] == phi[i, k - 1]):
                    phi[i, k - 1] = phi[i, k - 2]
                    phi[i, k] = phi[i, k + 1]

    # A first estimate of how many rows will be needed. We need more
    # than n_buckets, because we have to slice the polygons that
    # lie at the boundaries of the plot.
    n_rows_est = int(np.round(1.25*n_buckets))
    phi2 = np.zeros((n_rows_est, points_per_bucket))
    theta2 = np.zeros((n_rows_est, points_per_bucket))
    value = np.zeros(n_rows_est)
    n_rows = 0
    for i in range(n_buckets):
        # Add more rows if necessary.
        if n_rows >= phi2.shape[0]:
            n_rows_add = int(np.round(0.25*phi2.shape[0]))
            phi2 = np.vstack((phi2, np.zeros((n_rows_add, points_per_bucket))))
            theta2 = np.vstack((theta2, np.zeros((n_rows_add, points_per_bucket))))
            value = np.append(value, np.zeros(n_rows_add))

        this_phi = np.copy(phi[i, :])
        this_theta = np.copy(theta[i, :])
        this_value = q[i]# np.log10(q[idx_r0, i])

        if not (np.min(this_phi) < -0.5*np.pi and np.max(this_phi) > 0.5*np.pi):
            # This polygon doesn't touch the boundaries of the plot. Original
            # coordinates can be used directly.
            phi2[n_rows, :] = this_phi
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1
        else:
            # This polygon lies on the boundary of the plot. We have to slice into
            # two polygons -- one on the left side of the plot and on on the right.
            # First add the one on the right.
            this_phi2 = np.copy(this_phi)
            for k in range(points_per_bucket):
                if this_phi2[k] <= -0.:
                    this_phi2[k] = np.pi

            phi2[n_rows, :] = this_phi2
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1

            # Now add the one on the left.
            this_phi2 = np.copy(this_phi)
            for k in range(points_per_bucket):
                if this_phi2[k] >= 0.:
                    this_phi2[k] = -np.pi

            phi2[n_rows, :] = this_phi2
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1

    # Trim the arrays to the actual size of the data.
    if n_rows < phi2.shape[0]:
        phi2 = phi2[0:n_rows, :]
        theta2 = theta2[0:n_rows, :]
        value = value[0:n_rows]

    #ifig = 1; plt.close(ifig); fig = plt.figure(ifig, figsize = (9, 4))
    #ifig = 1; plt.close(ifig); fig = plt.figure(ifig, figsize = (3.39, 2.4))
    pl.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios = [12, 1])
    ax0 = pl.subplot(gs[0])
    ax1 = pl.subplot(gs[1])
    m = Basemap(projection = 'moll', lon_0 = 0., ax = ax0)

    cmap_min = np.min(quantity)
    cmap_max = np.max(quantity)
    cmap_avg = np.sum(quantity)/float(len(quantity))

    if limits is not None:
        cmap_min = limits[0]
        cmap_max = limits[1]

        cmap_avg = 0.5*(cmap_min + cmap_max)

        if len(limits) > 2:
            cmap_avg = limits[2]

    cmap_avg_rel = (cmap_avg - cmap_min)/(cmap_max - cmap_min)
    gamma = 1.0
    c1 = np.array([95, 158, 209])/255.
    c2 = np.array([255, 255, 255])/255.
    c3 = np.array([200, 82, 0])/255.
    cmap_points = {'red':   ((0.0, 0.0, c1[0]),
                            (cmap_avg_rel**gamma, c2[0], c2[0]),
                            (1.0, c3[0], 0.0)),

                   'green': ((0.0, 0.0, c1[1]),
                            (cmap_avg_rel**gamma, c2[1], c2[1]),
                            (1.0, c3[1], 0.0)),

                   'blue':  ((0.0, 0.0, c1[2]),
                            (cmap_avg_rel**gamma, c2[2], c2[2]),
                            (1.0, c3[2], 0.0))
                  }
    cmap = LinearSegmentedColormap('my_cmap', cmap_points, gamma = gamma)
    #cmap_name = 'gist_earth_r'
    #cmap = plt.get_cmap(cmap_name)
    for i in range(phi2.shape[0]):
        t = (value[i] - cmap_min)/(cmap_max - cmap_min)
        if t < 0: t = 0.
        if t > 1: t = 1.
        facecolor = cmap(t)
        x, y = m((180./np.pi)*phi2[i, :], (180./np.pi)*theta2[i, :])
        xy = list(zip(x, y))
        poly = Polygon(xy, facecolor = facecolor, edgecolor = facecolor, lw = 0.25)
        ax0.add_patch(poly)
    #m.drawmapboundary(color = 'k', linewidth = 1.5)
    m.drawmapboundary(color = 'k', fill_color = 'none', zorder = 10000)
    #ax0.set_title(cmap_name)
    if time is not None:
        ax0.set_title('t = {:.0f} min'.format(time/60.))

    def fmt(x, pos):
        return '{: .2f}'.format(x)

    norm = matplotlib.colors.Normalize(vmin = cmap_min, vmax = cmap_max)
    cb = ColorbarBase(ax1, cmap = cmap, norm = norm, ticks = ticks, \
                      format = ticker.FuncFormatter(fmt), orientation='horizontal')
    cb.set_label(r'$\Delta$r$_\mathrm{ub}$ / Mm')
    #ropplt.tight_layout(h_pad = 2.)
    pl.show()
    if file_name is not None:
        #plt.savefig(file_name + '_' + cmap_name + '.pdf', bbox_inches = 'tight', facecolor = 'w', dpi = 332.7)
        pl.savefig(file_name, bbox_inches = 'tight', facecolor = 'w', dpi = 332.7)

def plot_Mollweide(rp_set, dump_min, dump_max, r1, r2, output_dir = None, Filename = None, ifig = 2):

    '''
    Plot Mollweide spherical projection plot

    Parameters
    ----------
    dump_min/dump_max = int
        Range of file numbers you want to use in the histogram
    r1/r2 = float
        This function will only search for the convective
        boundary in the range between r1/r2
    ouput_dir: string
        path to output directory
    filename: string
        name for output file, None: no output

    Examples
    --------
    data_path = "/rpod2/PPM/RProfiles/AGBTP_M2.0Z1.e-5/F4"
    rp_set = rprofile.rprofile_set(data_path)
    plot_Mollweide(rp_set, 100,209,7.4,8.4)
    '''

    pl.close(ifig); fig = pl.figure(ifig, figsize = (3.384, 2.))
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1.)
    dr_ub_avg = np.zeros(80)

    n = 0
    for dump in range(dump_min, dump_max + 1):
        rp = rp_set.get_dump(dump)
        res = analyse_dump(rp, r1, r2)
        r_ub = res[3]
        dr_ub = r_ub - sum(r_ub)/float(len(r_ub))
        dr_ub_avg += dr_ub
        n += 1

    dr_ub_avg /= float(n)
    if Filename is not None:
        filename = output_dir + Filename
    else:
        filename = None

    bucket_map(rp, dr_ub_avg, file_name = filename)

def get_mach_number(rp_set,yp,dumps,comp,filename_offset = 1):
    '''
    Returns max Mach number and matching time vector

    Parameters
    ----------
    yp: yprofile instance
    rp_set: rprofile set instance
    dumps: range
        range of dumps to include
    comp: string
        'mean' 'max' or 'min' mach number

    Returns
    -------
    Ma_max : vector
        max mach nunmber
    t : vector
        time vector
    '''
    nd = len(dumps)
    t = np.zeros(nd)
    Ma_max = np.zeros(nd)

    for i in range(nd):
        rp = rp_set.get_dump(dumps[i])
        t[i] = yp.get('t', fname = dumps[i] - filename_offset, resolution = 'l')[-1]
        if comp == 'max':
            Ma_max[i] = np.max(rp.get_table('mach')[2, :, 0])
        if comp == 'mean':
            Ma_max[i] = np.mean(rp.get_table('mach')[0, :, 0])
        if comp == 'min':
            Ma_max[i] = np.min(rp.get_table('mach')[1, :, 0])

    return Ma_max, t

def plot_mach_number(rp_set,yp,dumps,comp = 'max',ifig = 1,lims =None,insert=False,save=False,\
                      prefix='PPM',format='pdf',lims_insert =None,f_offset=1):
    '''
    A function for geterating the time evolution of the mach number.

    Parameters
    ----------
    yp: yprofile instance
    rp_set: rprofile set instance
    dumps: range
        range of dumps to include
    lims : list
        Limits for the plot, i.e. [xl,xu,yl,yu].
        If None, the default values are used.
        The default is None.
    save : boolean
        Do you want the figures to be saved for each cycle?
        Figure names will be <prefix>-Vel-00000000001.<format>,
        where <prefix> and <format> are input options that default
        to 'PPM' and 'pdf'.
        The default value is False.
    prefix : string
        see 'save' above
    format : string
        see 'save' above
    '''
    try:
        Ma_max,t = get_mach_number(rp_set,yp,dumps,comp,filename_offset = f_offset)
    except:
        print('Dumps range must start at a value of 1 or greater due to filename offset between rprofiles and yprofiles')
    ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
    ax1 = fig.add_subplot(111)
    ax1.plot(t/60., Ma_max, color=cb(3))
    if lims is not None:
        ax1.set_xlim((lims[0],lims[1]))
        ax1.set_ylim((lims[2],lims[3]))
    ax1.set_xlabel('t / min')
    if comp == 'max':
        ax1.set_ylabel(r'Ma$_\mathrm{max}$')
    if comp == 'mean':
        ax1.set_ylabel(r'Ma$_\mathrm{av}$')
    if comp == 'min':
        ax1.set_ylabel(r'Ma$_\mathrm{min}$')

    if insert:
        left, bottom, width, height = [0.27, 0.55, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(t/60., Ma_max, color=cb(3))
        if lims_insert is not None:
            ax2.set_xlim((lims_insert[0],lims_insert[1]))
            ax2.set_ylim((lims_insert[2],lims_insert[3]))
    fig.tight_layout()

    if save:
        pl.savefig(prefix+'_Ma_max_evolution.'+format,format=format)

def get_p_top(prof,dumps,r_top):
    '''
    Returns p_top and matching time vector

    Parameters
    ----------
    yp: yprofile instance
    rtop: float
        boundary radius
    dumps: range
        range of dumps to include can be sparse

    Returns
    -------
    Ma_max : vector
        max mach nunmber
    t : vector
        time vector
    '''
    nd = len(dumps)
    t = np.zeros(nd)
    p_top = np.zeros(nd)

    r = prof.get('Y', fname = 0, resolution = 'l')
    idx_top = np.argmin(np.abs(r - r_top))

    for i in range(nd):
        t[i] = prof.get('t', fname = dumps[i], resolution = 'l')[-1]
        p_top[i] = prof.get('P', fname = dumps[i], resolution = 'l')[idx_top]

    return p_top,t

def plot_p_top(yp,dumps,r_top,ifig = 1,lims = None,insert = False,save=False,\
                      prefix='PPM',format='pdf',lims_insert =None):
    '''
    Parameters
    ----------
    yp: yprofile instance
    r_top: float
        boundary radius
    dumps: range
        range of dumps to include can be sparse
    lims : list
        Limits for the plot, i.e. [xl,xu,yl,yu].
        If None, the default values are used.
        The default is None.
    save : boolean
        Do you want the figures to be saved for each cycle?
        Figure names will be <prefix>-Vel-00000000001.<format>,
        where <prefix> and <format> are input options that default
        to 'PPM' and 'pdf'.
        The default value is False.
    prefix : string
        see 'save' above
    format : string
        see 'save' above
    '''

    p_top,t = get_p_top(yp,dumps,r_top)
    ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
    ax1 = fig.add_subplot(111)
    ax1.plot(t/60., p_top/p_top[0] - 1., color=cb(3))
    if lims is not None:
        ax1.set_xlim((lims[0],lims[1]))
        ax1.set_ylim((lims[2],lims[3]))
    ax1.set_xlabel('t / min')
    ax1.set_ylabel(r'p$_\mathrm{top}$(t) / p$_\mathrm{top}$(0) - 1')

    if insert:
        left, bottom, width, height = [0.31, 0.55, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(t/60., p_top/p_top[0] - 1., color=cb(3))
        if lims_insert is not None:
            ax2.set_xlim((lims_insert[0],lims_insert[1]))
            ax2.set_ylim((lims_insert[2],lims_insert[3]))
    fig.tight_layout()

    if save:
        pl.savefig(prefix+'_p_top_evolution.'+format,format=format)


#####################################################
# Files with mesa dependancy
#####################################################

def get_N2(yp, dump):
    '''
    squared Brunt-Vaisala frequency N^2

    Parameters
    ----------
    yp = yprofile object
        yprofile to use
    dump = int
        dump to analyze

    Returns
    -------
    N2 = vector
        Brunt Vaisala frequency [rad^2 s^-1]
    '''

    nabla_ad = 0.4

    R_bot = float(yp.hattrs['At base of the convection zone R'])
    g_bot = float(yp.hattrs['At base of the convection zone g'])

    r = yp.get('Y', fname = dump, resolution = 'l')
    p = yp.get('P', fname = dump, resolution = 'l')
    rho = yp.get('Rho', fname = dump, resolution = 'l')

    # centre R_bot on the nearest cell
    idx_bot = np.argmin(np.abs(r - R_bot))

    # mass below R_bot
    # (using r[idx_bot] instead of R_bot makes g[idx_bot] == g_bot)
    M_bot = g_bot*(r[idx_bot]**2)/G_code
    dr = 0.5*(np.roll(r, -1) - np.roll(r, +1))
    dm = 4.*np.pi*(r**2)*dr*rho
    m = np.cumsum(dm) # get the mass profile by integration
    # shift the mass profile to make sure that m[idx_bot] == M_bot
    # the mass profile at small radii won't make sense, because
    # the core is artificial with no gravity
    m += M_bot - m[idx_bot]

    g = G_code*m/(r**2) # gravity profile (see the note above)
    H_p = p/(rho*g) # pressure scale height (assuming hydrostatic equilibrium)

    logrho = np.log(rho)
    dlogrho = 0.5*(np.roll(logrho, -1) - np.roll(logrho, +1))
    logp = np.log(p)
    dlogp = 0.5*(np.roll(logp, -1) - np.roll(logp, +1))
    nabla_rho = dlogrho/dlogp

    # N^2 for an ideal gas
    N2 = (g/H_p)*(nabla_ad - 1. + nabla_rho)

    return N2

def plot_N2(case, dump1, dump2, lims1, lims2,mesa_logs_path, mesa_model_num):

    '''
        plots squared Brunt-Vaisala frequency N^2 with zoom window

        Parameters
        ----------
        case = string
            Name of run eg. 'D1'
        dump1/ dump2 = int
            dump to analyze
        mesa_A_model_num = int
            number for mesa model
        lims1/lims2 = 4 index array
            axes limits [xmin xmax ymin ymax] lims1 = smaller window

        Examples
        --------
        import ppm
        set_YProf_path('/data/ppm_rpod2/YProfiles/O-shell-M25/')
        plot_N2('D1', 0, 132, mesa_model_num = 29350, mesa_B_model_num = 28950)
        '''
    ppm_run= case
    yp = yprofile(os.path.join(ppm_path + case))
    mesa_A_prof = ms.mesa_profile(mesa_logs_path, mesa_model_num)
    # convert the mesa variables to code units
    mesa_A_r = (nuconst.r_sun/1e8)*mesa_A_prof.get('radius')
    mesa_A_N2 = mesa_A_prof.get('brunt_N2')
    mesa_A_N2_mu = mesa_A_prof.get('brunt_N2_composition_term')
    mesa_A_N2_T = mesa_A_prof.get('brunt_N2_structure_term')
    mesa_A_mu = mesa_A_prof.get('mu')

    # get the PPM models
    ppm_prof = yp
    ppm_r = ppm_prof.get('Y', fname = 0, resolution = 'l')
    ppm_dump_num_A = dump1
    ppm_t_A = ppm_prof.get('t', fname = ppm_dump_num_A, resolution = 'l')[-1]
    ppm_N2_A = get_N2(ppm_prof, ppm_dump_num_A)
    ppm_dump_num_B = dump2
    ppm_t_B = ppm_prof.get('t', fname = ppm_dump_num_B, resolution = 'l')[-1]
    ppm_N2_B = get_N2(ppm_prof, ppm_dump_num_B)

    cb = utils.colourblind

    ifig = 1; pl.close(ifig); fig = pl.figure(ifig)

    ax1 = fig.add_subplot(111)
    ax1.plot(mesa_A_r, mesa_A_N2, ls = '-', lw = 0.5, color = cb(4), label = "MESA")
    lbl = "{:s}, t = {:.0f} min.".format(ppm_run, ppm_t_A/60.)
    ax1.plot(ppm_r, ppm_N2_A, ls = ':', color = cb(5), label = lbl)
    lbl = "{:s}, t = {:.0f} min.".format(ppm_run, ppm_t_B/60.)
    ax1.plot(ppm_r, ppm_N2_B, ls = '-', color = cb(8), label = lbl)
    ax1.set_xlim((lims1[0], lims1[1]))
    ax1.set_ylim((lims1[2],lims1[3]))
    ax1.set_xlabel('r / Mm')
    ax1.set_ylabel(r'N$^2$ / rad$^2\,$s$^{-2}$')
    ax1.legend(loc = 0)

    left, bottom, width, height = [0.34, 0.3, 0.4, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(mesa_A_r, mesa_A_N2, ls = '-', lw = 0.5, color = cb(4))
    ax2.plot(ppm_r, ppm_N2_A, ls = ':', color = cb(5))
    ax2.plot(ppm_r, ppm_N2_B, ls = '-', color = cb(8))
    ax2.set_xlim((lims2[0], lims2[1]))
    ax2.set_ylim((lims2[2],lims2[3]))

def energy_comparison(yprof,mesa_model, xthing = 'm',ifig = 2, silent = True,\
                     range_conv1 = None , range_conv2 = None,\
                     xlim = [0.5,2.5] , ylim = [8,13], radbase = 4.1297,\
                     dlayerbot = 0.5, totallum = 20.153):
    '''
    Nuclear  energy  generation  rate (enuc) thermal  neutrino
    energy   loss   rate (enu) and   luminosity   profiles (L) for MESA
    plotted with ppmstar energy estimation (eppm)

    Parameters
    ----------
    yprof : yprofile object
        yprofile to examine
    mesa_model : mesa_model
        mesa model to examine
    xthing : str
        x axis as mass, 'm' or radius, 'r'
    silent : bool
        suppress output or not
    range_conv1 : range, optional
        range to shade for convection zone 1
    range_conv2 : range, optional
        range to shade for convection zone 2
    xlim: range
    ylim: range

    values from setup.txt
    radbase = 4.1297
    dlayerbot = 0.5
    totallum = 20.153

    Examples
    --------
    import ppm
    import nugrid.mesa as ms
    mesa_A_model_num = 29350
    mesa_logs_path = '/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/LOGS_N2b'
    mesa_model = ms.mesa_profile(mesa_logs_path, mesa_A_model_num)
    yprof = yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D1')
    energy_comparison(yprof,mesa_model)

    '''
    xthing1 = xthing
    p = mesa_model
    # function giving PPM heating curve given PPM profile

    # figure vs mass
    ifig = 2; pl.close(ifig); fig = pl.figure(ifig)
    m = p.get('mass')

    eps_nuc = np.log10(p.get('eps_nuc'))
    eps_neu = np.log10(p.get('non_nuc_neu'))
    lum = p.get('logL')

    # plot conv zone boundaries
    if range_conv1 is not None:
        pl.fill_between( range_conv1 , 0, 100, color= '#fffdd8' )

    if range_conv2 is not None:
        pl.fill_between( range_conv2 , 0, 100, color='#b39eb5' )

    res = get_heat_source(yprof, radbase = radbase, dlayerbot = dlayerbot, totallum = totallum)
    r_ppm = res[:, 0]
    m_ppm = res[:, 1]
    m_ppm = m_ppm * 5.025e-07 # code unit --> solar masses
    eps_ppm = res[:, 2]
    eps_ppm = eps_ppm * 1.e16 # code units --> erg/g/s
    eps_ppm = np.log10(eps_ppm)

    # trim mess off eps_nuc below O-shell
    idx = np.abs(m - 1.0).argmin()

    # plot simulation boundaries
    rad1 = 3.0 # Mm
    rad2 = 9.8 # Mm
    rad  = 10. ** p.get('logR') * nuconst.r_sun / 1.e8
    if not silent:
        print(rad)

    idx1 = np.abs(rad - rad1).argmin()
    idx2 = np.abs(rad - rad2).argmin()
    m1   = m[idx1]
    m2   = m[idx2]

    if not silent:
        print(m1, m2)

    if xthing1 is 'm':
        xthing = m
        xthing_ppm = m_ppm
    else:
        xthing = r
        xthing_ppm = r_ppm

    #semilogy(m[:idx],eps_nuc[:idx],
    pl.plot(xthing[:idx],eps_nuc[:idx],
        label='$\epsilon_\mathrm{nuc}$',
        color=utils.colourblind(8),
        linestyle='-')

    #semilogy(m,eps_neu,
    pl.plot(xthing,eps_neu,
        label='$\epsilon_\\nu$',
        color=utils.colourblind(4),
        linestyle='--')

    #semilogy(m,lum,
    pl.plot(xthing,lum,
        label='$L$',
        color=utils.colourblind(0),
        linestyle='-.')

    #semilogy(m_ppm,eps_ppm,
    pl.plot(xthing_ppm,eps_ppm,
        label='$\epsilon_\mathrm{PPM}$',
        color=utils.colourblind(4),
        linestyle='-',
        marker='.',
        markevery=.3)

    #pl.ylabel('$\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot$')
    if xthing1 is 'm':
        pl.ylabel('$\log_{10}(\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot)$')
        pl.xlabel('$\mathrm{Mass\,/\,M}_\odot$')
    else:
        pl.ylabel('$\log_{10}(\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot)$')
        pl.xlabel('$\mathrm{r\,/\,Mm}$')

    pl.legend(loc='upper left')

    #pl.xlim(1.,2.5)
    pl.xlim(xlim)
    #pl.ylim(1.e8,1.e13)
    pl.ylim(ylim)

def get_heat_source(yprof, radbase = 4.1297, dlayerbot = 0.5, totallum = 20.153):
    '''
    returns estimation of yprofiles energy

    # values from setup.txt
    radbase = 4.1297
    dlayerbot = 0.5
    totallum = 20.153

    Parameters
    ----------
    yprof: yprofile object
        yprofile to examine

    Returns
    -------
    array
        array with vectors [radius, mass, energy estimate]
    '''

    r = yprof.get('Y', fname = 0, resolution = 'l')
    rho = yprof.get('Rho', fname = 0, resolution = 'l')
    r_bot = float(yprof.hattrs['At base of the convection zone R'])
    g_bot = float(yprof.hattrs['At base of the convection zone g'])

    dr = -0.5*(np.roll(r, -1) - np.roll(r, +1))
    dr[0] = dr[1]; dr[-1] = dr[-2]

    # centre r_bot on the nearest cell
    idx_bot = np.argmin(np.abs(r - r_bot))
    # mass below r_bot
    m_bot = g_bot*(r[idx_bot]**2)/G_code

    dV = 4.*np.pi*(r**2)*dr
    dm = rho*dV
    m = -np.cumsum(dm) # get the mass profile by integration
    # shift the mass profile to make sure that m[idx_bot] == m_bot
    m += m_bot - m[idx_bot]

    radminbase = radbase + 0.5*dlayerbot
    radmaxbase = radbase + 1.5*dlayerbot

    # PPMstar computes this integral numerically
    heatsum = 4.*pi*(-(radmaxbase**5 - radminbase**5)/5. + \
                     (radminbase + radmaxbase)*(radmaxbase**4 - radminbase**4)/4. - \
                     radminbase*radmaxbase*(radmaxbase**3 - radminbase**3)/3.)
    dist = np.maximum(0.*r, r - radminbase)*np.maximum(0.*r, radmaxbase - r)*dV/heatsum

    # the tiny offset of 1e-100 makes sure lines don't end suddenly in a logarithmic plot
    # 2.25 is the correction factor to account for the heating bug in PPMstar
    eps = 2.25*totallum*dist/dm + 1e-100

    return np.transpose(np.array([r, m, eps]))

def get_mesa_time_evo(mesa_path,mesa_logs,t_end,save = False):
    '''
    Function to generate data for figure 5 in O-shell paper.

    Parameters
    ----------
    mesa_path : string
        path to the mesa data
    mesa_logs : range
        cycles you would like to include in the plot
    t_end : float
        time of core collapse
    save : boolean,optional
        save the output into data files

    Returns
    -------
    agearr : array
        age yrs
    ltlarr  : array
        time until core collapse
    rbotarr : array
        radius of lower convective boundary
    rtoparr : array
        radius of top convective boundary
    muarr : array
        mean molecular weight in convective region
    peakLarr : array
        peak luminosity
    peakL_Lsunarr : array
        peak luminosity units Lsun
    peakepsgravarr : array
        peak sepperation

    '''
    tag = 'shell1'
    #tag = 'shell2'

    s = ms.history_data(mesa_path)

    #logs=range(825,1000)

    agearr         = []
    ltlarr         = []
    rbotarr        = []
    rtoparr        = []
    muarr          = []
    peakLarr       = []
    peakL_Lsunarr  = []
    peakepsgravarr = []

    # latex table file:
    if save:
        f=open('table.tex','w')
    for log in mesa_logs:
        p=ms.mesa_profile(mesa_path,log,num_type='profile_num')
        mass = p.get('mass')
        idxl = np.abs( mass - 1. ).argmin()
        idxu = np.abs( mass - 2. ).argmin()
        mass = mass[idxu:idxl]
        rad = 10.**p.get('logR')[idxu:idxl]
        mt = p.get('mixing_type')[idxu:idxl]
        L = 10.**p.get('logL')[idxu:idxl]
        epsgrav = p.get('eps_grav')[idxu:idxl]
        peakL_Lsun = np.max(L)/1.e10
        peakL = peakL_Lsun*1.e10*nuconst.l_sun/1.e44
        peakepsgrav = np.max(epsgrav)
        ipL = L.argmax()
        mu = p.get('mu')[idxu:idxl]
        try:
            itop = np.where(mt[:ipL]!=1)[0][-1]
        except:
            continue
        rtop = rad[:ipL][itop]*nuconst.r_sun/1.e8
        mtop = mass[itop]
        ibot = np.where(mt==1)[0][-1]
        rbot = rad[ibot]*nuconst.r_sun/1.e8
        mbot = mass[ibot]
        mu = mu[int((itop+ibot)/2)]
        # time from end of core O burning
        iaoe = np.where(s.get('center_o16')>1e-3)[0][-1] #was 's'?
        aoe = s.get('star_age')[iaoe] #was 's'?
        age = ( p.header_attr['star_age'] - aoe ) * 365.
        ltl = np.log10( t_end - p.header_attr['star_age'] )

        agearr.append( age )
        ltlarr.append( ltl )
        rbotarr.append( rbot )
        rtoparr.append( rtop )
        muarr.append( mu )
        peakLarr.append( peakL )
        peakepsgravarr.append( peakepsgrav )
        peakL_Lsunarr.append( peakL_Lsun )
        if save:
            for x in [age,ltl,rbot,rtop,mu,peakL,peakL_Lsun]:
                f.write("{0:.3f}".format(x) + ' & ')
            f.write(' \\ \n')
    if save:
        f.close()

        #save arrays for making some plots
        np.save( tag+'_age.npy', np.array( agearr ) )
        np.save( tag+'_ltl.npy', np.array( ltlarr ) )
        np.save( tag+'_rbot.npy', np.array( rbotarr ) )
        np.save( tag+'_rtop.npy', np.array( rtoparr ) )
        np.save( tag+'_mu.npy', np.array( muarr ) )
        np.save( tag+'_peakL.npy', np.array( peakLarr ) )
        np.save( tag+'_peakL_Lsun.npy', np.array( peakL_Lsunarr ) )
        np.save( tag+'_peakepsgrav.npy', np.array( peakepsgravarr ) )

    return agearr,ltlarr,rbotarr,rtoparr,muarr,peakLarr,peakL_Lsunarr,peakepsgravarr

def plot_mesa_time_evo(mesa_path,mesa_logs,t_end,ifig=21):
    '''
    Function to plot data for figure 5 in O-shell paper.

    Parameters
    ----------
    mesa_path : string
        path to the mesa data
    mesa_logs : range
        cycles you would like to include in the plot
    t_end : float
        time of core collapse
    save : boolean, optional
        save the output into data files

    Examples
    --------
    plot_mesa_time_evo('/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/LOGS',
        range(550,560),7.5829245098141646E+006,ifig=21)
    '''
    cb = utils.colourblind
    #pl.close(ifig),pl.figure(ifig)
    f, (ax1, ax2, ax3) = pl.subplots(3, sharex=True, figsize=(3.39,5))
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0.1)
    pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # load arrays

    data = get_mesa_time_evo(mesa_path,mesa_logs,t_end)

    age = data[0]
    ltl = data[1]
    mu = data[4]
    peakL_Lsun = data[6]
    peakL = data[5]
    rbot = data[2]
    rtop = data[3]
    epsgrav = data[7]

    xax = ltl
    #xax = age

    #xlim(-1.4,-1.9)

    # make plots
    pl.subplot(ax1)
    pl.plot(xax,mu,color=cb(5))
    pl.ylabel('$\mu$')
    #pl.ylim(1.82,1.88)
    #pl.yticks(linspace(1.82,1.88,4))

    pl.subplot(ax2)
    pl.plot(xax, rbot, color=cb(5), label='$r_\mathrm{lb}$')
    pl.plot(xax, rtop, ls='--', color=cb(8), label='$r_\mathrm{ub}$')
    #pl.ylim(3,9)
    #pl.yticks(np.linspace(3,9,4))
    pl.ylabel('$\mathrm{r\,/\,Mm}$')
    pl.legend(loc='best')

    pl.subplot(ax3)
    pl.plot(xax, peakL_Lsun, color=cb(5) )
    pl.ylabel('$L\,/\,10^{10}L_\odot$')
    pl.xlabel('$\log_{10}(t^*\,/\,\mathrm{yr})$')
    #pl.ylim(4,8)
    #pl.yticks(linspace(4,8,5))

    pl.subplots_adjust(bottom=0.1)

def plot_entr_v_ut(cases,c0, Ncycles,r1,r2, comp,metric,label,ifig = 3,
                  integrate_both_fluids = False):
    '''
    Plots entrainment rate vs max radial or tangential velocity.

    Parameters
    ----------
    cases : string list
        list of cases i.e.D1
    c0 : list
        cycle to start on for each case
    Ncycles: int
        number of cycles to average to find v
    r1 :float
        radius range to search for v, also r2
    comp : str
        component of velocity 'tangential' or 'radial'
    metric : str
        metric of veloctiy 'min' max' or 'mean'

    Examples
    --------
    cases = ('D1', 'D8', 'D5', 'D6', 'D9', 'D10', 'D20', 'D23', 'D2')
    c0 = (241,154,142,155,123,78,355,241,124)
    global ppm_path
    ppm_path = '/data/ppm_rpod2/YProfiles/O-shell-M25/'
    plot_entr_v_ut(cases,c0,10,7.5,8.5,ifig = 3,integrate_both_fluids = False)
    '''
    mdot = np.zeros(len(cases))
    vt = np.zeros(len(cases))
    vr = np.zeros(len(cases))

    for i in range(len(cases)):
        prof = yprofile(os.path.join(ppm_path,cases[i]))
        cycles = list(range(c0[i], c0[i] + Ncycles, 1))
        #vt[i], vr[i] = find_max_velocities(prof, cycles, 7.5, 8.5, 4., 8., label = cases[i], ifig = i)
        t, v = get_v_evolution(prof, cycles, r1,r2,comp = comp, RMS = metric)
        vt[i] = np.mean(1e3*v)
        mdot[i] = prof.entrainment_rate(cycles, r1, r2, var = 'vxz',
                                        criterion = 'min_grad', offset = -1.,
                                        integrate_both_fluids = integrate_both_fluids,
                                        ifig0 = ifig,
                                        show_output = False)

    fc = np.polyfit(np.log(vt[0:len(cases)]/30.), np.log(mdot[0:len(cases)]), 1)
    vt_fit = np.array((1e0, 1e3))
    mdot_fit = np.exp(fc[0]*np.log(vt_fit/30.) + fc[1])

    mdot0_str = '{:9e}'.format(np.exp(fc[1]))
    tmp = mdot0_str.split('e')
    mantissa = float(tmp[0])
    exponent = int(tmp[1])
    fit_label = r'${:.2f} \times 10^{{{:d}}}$ (v$_\perp$ / 30)$^{{{:.2f}}}$'.\
            format(mantissa, exponent, fc[0])
    nn = len(cases)
    cb = utils.colourblind
    pl.close(ifig); pl.figure(ifig)
    pl.plot(np.log10(vt[0:nn]), np.log10(mdot[0:nn]),
             ls = 'none', color = cb(5), marker = 'o', \
             label = label)
    pl.plot(np.log10(vt_fit), np.log10(mdot_fit),
             ls = '-', lw = 0.5, color = cb(4), \
             label = fit_label)
    pl.plot(np.log10(59.), np.log10(1.1e-4), ls = 'None',
             color = cb(8), marker = 's', \
             label = 'MA07')
    pl.xlabel(r'log$_{10}$(v$_\perp$ / km s$^{-1}$)')
    pl.ylabel(r'log$_{10} ( \dot{\mathrm{M}}_\mathrm{e}$ / M$_\odot$ s$^{-1}$])')
    #plt.xlim((0.9, 2.3))
    #plt.ylim((-9., -3.))
    pl.legend(loc = 4)
    pl.tight_layout()

def plot_diffusion_profiles(run,mesa_path,mesa_log,rtop,Dsolve_range,tauconv,r0,D0,f1,f2,
                            alpha,fluid = 'FV H+He',markevery = 25):
    '''
    This is a function that generates various diffusion profiles
    it is dependant on the functions Dsolvedown and Dov2

    Parameters
    ----------
    run : string
        yprofile case to use i.e. 'D2'
    mesa_path : string
        path to the mesa data
    mesa_logs : range
        cycles you would like to include in the plot
    Dsolve_range : int array
        cycles from which to take initial and final abundance profiles
        for the diffusion step we want to mimic.
    fluid : string
        Which fluid do you want to track?
    tauconv : float, optional
        If averaging the abundance profiles over a convective turnover
        timescale, give the convective turnover timescale (seconds).
        The default value is None.
    r0 : float
        radius (Mm) at which the decay will begin
    D0 : float
        diffusion coefficient at r0
    f1,f2 : float
        parameters of the model

    Examples
    --------
    plot_diffusion_profiles('D2','/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/',28900,
                        7.8489,(1,160),2.*460.,7.8282,10.**12.27,0.21,0.055,1.6)
    pl.xlim(3,9)
    pl.ylim(11,16)
    '''

    run = 'D2'
    cb = utils.colourblind
    ls = utils.linestyle
    model = mesa_log
    dir  = mesa_path
    yy = 0
    YY = yprofile(ppm_path+run)

    [rrc, DDc, r0c, rc] = YY.Dsolvedown(Dsolve_range[0],Dsolve_range[1], fluid = fluid,
                                        tauconv = tauconv,returnY = True, newton = True,
                                        smooth = False,showfig = False)

    rrf2, DDf2 = YY.Dov2(r0,D0,f1,f2) # r0, D0, f1, f2

    r = YY.get('Y',fname=1,resolution='l')[::-1]
    vav = YY.vaverage('vY')[::-1] # cm / s

    P = YY.get('P',fname=1)[::-1] * 1.e19 # barye, centre to surface
    Hp = - P[1:] * np.diff(r) * 1.e8 / np.diff(P)
    Hp = np.insert(Hp,0,0)
    Dav = (1./3.) * vav * alpha * Hp

    p=ms.mesa_profile(dir+'/LOGS',model)
    rm = p.get('radius') * nuconst.r_sun / 1.e8
    idx = np.abs(rm-rtop).argmin()
    rm = rm[idx:]
    Dm = p.get('log_mlt_D_mix')
    Dm = Dm[idx:]

    v_mlt = 10.**p.get('log_conv_vel')[idx:]
    Hpmes = p.get('pressure_scale_height')[idx:]*nuconst.r_sun
    Davmes = (1./3.) * v_mlt * np.minimum(alpha * Hpmes,rtop*1.e8-rm*1.e8)

    Dav2 = (1./3.) * vav * np.minimum(alpha * Hp,rtop*1.e8-r*1.e8)

    fig = pl.figure(21)
    pl.plot(rrc,np.log10(DDc),color = cb(yy),markevery=markevery,\
            label='$D_{\\rm conv}$')
    yy +=1
    pl.plot(rm,Dm,color = cb(yy),marker = ls(yy)[1] ,markevery=markevery,\
         label='$D_{\\rm MLT}$')
    yy +=1

    pl.plot(r,np.log10(Dav),color = cb(yy),marker = ls(yy)[1], markevery=markevery,\
         label='$\\frac{1}{3}v\ell$')
    yy +=1

    pl.plot(r,np.log10(Dav2),color = cb(yy),marker = ls(yy)[1],markevery=markevery,linewidth=2.5,alpha=0.5,\
         label='$\\frac{1}{3}v{\\rm min}(\ell,r_0-r)$')
    yy +=1
    pl.plot(rm,np.log10(Davmes),color = cb(yy),marker = ls(yy)[1],markevery=markevery,linewidth=2.5,alpha=0.5,\
         label='$\\frac{1}{3}v_{\\rm MLT}{\\rm min}(\ell,r_0-r)$')
    pl.legend(loc='center left',numpoints=1).draw_frame(False)
    pl.ylabel('$\log(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
    pl.xlabel('r / Mm')



class Messenger:
    '''
    Messenger reports messages, warnings, and errors to the user.
    '''

    def __init__(self, verbose=3):
        '''
        Init method.

        Parameters
        ----------
        verbose: integer
            Verbosity level: no output (0), errors only (1), errors and
            warnings (2), everything (3). Negative values are replaced
            with 0 and any values greater than 3 are considered to be 3.
        '''
        # Print everything by default (verbose=3).
        self.__print_messages = True
        self.__print_warnings = True
        self.__print_errors = True

        vrbs = int(verbose)
        if vrbs > 3:
            self.__verbose = 3
        elif vrbs < 0:
            self.__verbose = 0
        else:
            self.__verbose = vrbs

        if self.__verbose < 3:
            self.__print_messages = False
        if self.__verbose < 2:
            self.__print_warnings = False
        if self.__verbose < 1:
            self.__print_errors = False

    def message(self, message):
        '''
        Reports a message to the user. The message is currently printed to
        stdout.

        Parameters
        ----------
        message: string
        '''

        if self.__print_messages:
            print(message)

    def warning(self, warning):
        '''
        Reports a warning to the user. The warning is currently printed to
        stdout.

        Parameters
        ----------
        warning: string
        '''

        if self.__print_warnings:
            print('Warning: ' + warning)

    def error(self, error):
        '''
        Reports an error to the user. The error is currently printed to
        stdout.

        Parameters
        ----------
        error: string
        '''

        if self.__print_errors:
            print('Error: ' + error)



class RprofHistory:
    '''
    RprofHistory reads .hstry files produced by PPMstar 2.0.
    '''

    def __init__(self, file_path, verbose=3):
        '''
        Init method.

        Parameters
        ----------
        file_path: string
            Path to the .hstry file to be read.
        verbose: integer
            Verbosity level as defined in class Messenger.
        '''

        self.__is_valid = False
        self.__messenger = Messenger(verbose=verbose)

        # __read_history_file() will set the following variables,
        # if successful:
        self.__file_path = ''
        self.__vars = []
        self.__data = {}
        if not self.__read_history_file(file_path):
            return

        # Instance successfully initialized.
        self.__is_valid = True

    def __read_history_file(self, file_path):
        '''
        Reads a .hstry file written by PPMstar 2.0.

        Parameters
        ----------
        file_path: string
            Path to the .hstry file to be read.

        Returns
        -------
        True when the file has been successfully read. False otherwise.
        '''
        try:
            with open(file_path, 'r') as fin:
                #msg = "Reading history file '{:s}'.".format(file_path)
                #self.__messenger.message(msg)
                lines = fin.readlines()
            self.__file_path = file_path
        except FileNotFoundError as e:
            err = "History file '{:s}' not found.".format(file_path)
            self.__messenger.error(err)
            return False
        except OSError as e:
            err = 'I/O error({0}): {1}'.format(e.errno, e.strerror)
            self.__messenger.error(err)
            return False

        n_lines = len(lines)
        l = 0 # line index

        # Find the header.
        while not lines[l].startswith('NDump'):
            l += 1
        # Get column names.
        self.__vars = lines[l].split()
        l += 1

        for this_var in self.__vars:
            self.__data[this_var] = []

        # Read the data table.
        while l < n_lines:
            if lines[l].strip() == '':
                l += 1
                continue

            sline = lines[l].split()

            # The 1st variable is always NDump.
            self.__data['NDump'].append(int(sline[0]))

            # Another len(self.__vars) - 2 columns contain floats.
            for i in range(1, len(self.__vars) - 1):
                self.__data[self.__vars[i]].append(float(sline[i]))

            # The last column is always TimeStamp (which contains spaces,
            # so we have to put it back together). We parse the datetime
            # string and convert it to a UNIX time stamp (float). We do
            # not care what time zone the datetime string corresponds to,
            # because we only care about time differences.
            timestamp_str = ' '.join(sline[len(self.__vars)-1:])
            timestamp = time.mktime(parse(timestamp_str).timetuple())
            self.__data['TimeStamp'].append(timestamp)
            l += 1

        return True

    def is_valid(self):
        '''
        Checks if the instance is valid, i.e. fully initialised.

        Returns
        -------
        True if the instance is valid. False otherwise.
        '''

        return self.__is_valid

    def get_variables(self):
        '''
        Returns a list of variables available.
        '''

        # Return a copy.
        return list(self.__vars)

    def get(self, var_name):
        '''
        Returns variable var_name if it exists.

        Parameters
        ----------
        var_name: string
            Name of the variable.

        Returns
        -------
        numpy.ndarray
            Variable var_name if it exists.
        NoneType
            If variable var_name does not exist.
        '''

        if var_name in self.__vars:
            # Return a copy.
            return np.array(self.__data[var_name])
        else:
            err = "Variable '{:s}' does not exist.".format(var_name)
            self.__messenger.error(err)

            msg = 'Available variables: \n'
            msg += str(self.__vars)
            self.__messenger.message(msg)

            return None

    def plot_wct_per_dump(self):
        '''
        Plots wall-clock time (WCT) per dump as a function of dump
        number.
        '''

        dumps = self.get('NDump')
        timestamps = self.get('TimeStamp')

        wct_per_dump = (np.roll(timestamps, -1) - timestamps)/\
                       (np.roll(dumps, -1) - dumps)

        pl.plot(dumps[:-1], wct_per_dump[:-1])
        pl.xlabel('NDump')
        pl.ylabel('WCT per dump / s')



class RprofSet(PPMtools):
    '''
    RprofSet holds a set of .rprof files from a single run
    of PPMstar 2.0.
    '''

    def __init__(self, dir_name, verbose=3, cache_rprofs=True):
        '''
        Init method.

        Parameters
        ----------
        dir_name: string
            Name of the directory to be searched for .rprof files.
        verbose: integer
            Verbosity level as defined in class Messenger.
        '''

        PPMtools.__init__(self,verbose=verbose)
        self.__is_valid = False
        self.__messenger = Messenger(verbose=verbose)

        # __find_dumps() will set the following variables,
        # if successful:
        self.__dir_name = ''
        self.__run_id = ''
        self.__dumps = []
        if not self.__find_dumps(dir_name):
            return

        # To speed up repeated requests for data from the same dump, we
        # will cache every Rprof that has already been read from disk.
        # The user can still turn this off in case that they analyse a
        # large number of long runs on a machine with a small RAM.
        self.__cache_rprofs = cache_rprofs
        self.__rprof_cache = {}

        history_file_path = '{:s}{:s}-0000.hstry'.format(self.__dir_name, \
                                                         self.__run_id)
        self.__history = RprofHistory(history_file_path, verbose=verbose)
        if not self.__history.is_valid():
            wrng = ('History not available. You will not be able to access '
                   'rprof data by simulation time.')
            self.__messenger.warning(wrng)

        self.__is_valid = True

    def __find_dumps(self, dir_name):
        '''
        Searches for .rprof files and creates an internal list of dump numbers
        available.

        Parameters
        ----------
        dir_name: string
            Name of the directory to be searched for .rprof files.

        Returns
        -------
        boolean
            True when a set of .rprof files has been found. False otherwise.
        '''

        if not os.path.isdir(dir_name):
            err = "Directory '{:s}' does not exist.".format(dir_name)
            self.__messenger.error(err)
            return False

        # join() will add a trailing slash if not present.
        self.__dir_name = os.path.join(dir_name, '')

        rprof_files = glob.glob(self.__dir_name + '*.rprof')
        if len(rprof_files) == 0:
            err = "No rprof files found in '{:s}'.".format(self.__dir_name)
            self.__messenger.error(err)
            return False

        rprof_files = [os.path.basename(rprof_files[i]) \
                       for i in range(len(rprof_files))]

        # run_id is what is before the last dash, check that it is
        # followed by 4 numeric characters
        _ind   = rprof_files[0].rindex('-')                # index of last dash
        self.__run_id = rprof_files[0][0:_ind]

        for i in range(len(rprof_files)):
            _ind   = rprof_files[i].rindex('-')
            if rprof_files[i][0:_ind] == self.__run_id:     # record dump number
                dump_number = rprof_files[i][_ind+1:_ind+5]
                if dump_number.isnumeric():
                    self.__dumps.append(int(dump_number))
                else:
                    self.__messenger.error("rprof filename does not have 4 digits after last dash: "+\
                                    rprof_files[i])
                    return False
            else:                                           # exclude files with non-matching runid
                wrng = ("rprof files with multiple run ids found in '{:s}'."
                        "Using only those with run id '{:s}'.").\
                       format(self.__dir_name, self.__run_id)
                self.__messenger.warning(wrng)
                continue

        self.__dumps = sorted(self.__dumps)
        msg = "{:d} rprof files found in '{:s}.\n".format(len(self.__dumps), \
              self.__dir_name)
        msg += "Dump numbers range from {:d} to {:d}.".format(\
               self.__dumps[0], self.__dumps[-1])
        self.__messenger.message(msg)
        if (self.__dumps[-1] - self.__dumps[0] + 1) != len(self.__dumps):
            wrng = 'Some dumps are missing.'
            self.__messenger.warning(wrng)

        return True

    def is_valid(self):
        '''
        Checks if the instance is valid, i.e. fully initialised.

        Returns
        -------
        boolean
            True if the instance is valid. False otherwise.
        '''

        return self.__is_valid

    def get_run_id(self):
        '''
        Returns the run identifier that precedes the dump number in the names
        of .rprof files.
        '''

        return str(self.__run_id)

    def get_history(self):
        '''
        Returns the RprofHistory object associated with this instance if available
        and None otherwise.
        '''

        if self.__history is not None:
            return self.__history
        else:
            self.__messenger.error('History not available.')
            return None

    def get_dump_list(self):
        '''
        Returns a list of dumps available.
        '''

        return list(self.__dumps)

    def get_dump(self, dump):
        '''
        Returns a single dump.

        Parameters
        ----------
        dump: integer

        Returns
        -------
        Rprof
            Rprof object corresponding to the selected dump.
        '''

        if dump not in self.__dumps:
            err = 'Dump {:d} is not available.'.format(dump)
            self.__messenger.error(err)
            return None

        file_path = '{:s}{:s}-{:04d}.rprof'.format(self.__dir_name, \
                                                   self.__run_id, dump)

        if self.__cache_rprofs:
            if dump in self.__rprof_cache:
                rp = self.__rprof_cache[dump]
            else:
                rp = Rprof(file_path)
                self.__rprof_cache[dump] = rp
        else:
            rp = Rprof(file_path)

        return rp

    def get(self, var, fname, num_type='NDump', resolution='l'):
        '''
        Returns variable var at a specific point in the simulation's time
        evolution.

        Parameters
        ----------
        var: string
            Name of the variable.
        fname: integer/float
            Dump number or time in seconds depending on the value of
            num_type.
        num_type: string (case insensitive)
            If 'ndump' fname is expected to be a dump number (integer).
            If 't' fname is expected to be a time value in seconds; run
            history file (.hstry) must be available to search by time value.
        resolution: string (case insensitive)
            'l' (low) or 'h' (high). A few variables are available at
            double resolution ('h'), the rest correspond to the resolution
            of the computational grid ('l').


        Returns
        -------
        numpy.ndarray
            Variable var as given by Rprof.get() if the Rprof corresponding
            to fname exists.
        NoneType
            If the Rprof corresponding to fname does not exist.
        '''

        if num_type.lower() == 'ndump':
            rp = self.get_dump(fname)
        elif num_type.lower() == 't':
            if self.__history is None:
                err = 'History not available. Cannot search by t.'
                self.__messenger.error(err)
                return None

            t = self.__history.get('time(secs)')
            ndump = self.__history.get('NDump')

            closest_idx = np.argmin(np.abs(t - fname))
            closest_dump = ndump[closest_idx]
            closest_t = t[closest_idx]
            msg = ('Dump {:d} at t = {:.2f} min is the closest to '
                   't = {:.2f} min.').format(closest_dump, \
                  closest_t/60., fname/60.)
            self.__messenger.message(msg)
            rp = self.get_dump(closest_dump)
        else:
            self.__messenger.error("'{:s}' is not a valid value of "
                                   "num_type.".format(num_type))
            return None

        if rp is not None:
            return rp.get(var, resolution=resolution)
        else:
            return None

    def rprofgui(self,ifig=11):
        def w_plot(dump1,dump2,ything,log10y=False,ifig=ifig):
            self.rp_plot([dump1,dump2],ything,logy=log10y,ifig=ifig)
        rp_hst = self.get_history()
        dumpmin, dumpmax = rp_hst.get('NDump')[0],rp_hst.get('NDump')[-1]
        dumpmean = int(2*(-dumpmin + dumpmax)/3.)
        things_list = self.get_dump(dumpmin).get_lr_variables()+\
                          self.get_dump(dumpmin).get_hr_variables()+\
                    self.get_computable_quantities()
        interact(w_plot,dump1=widgets.IntSlider(min=dumpmin,\
                max=dumpmax,step=1,value=int(dumpmin+0.05*(dumpmean-dumpmin))),\
                     dump2=widgets.IntSlider(min=dumpmin,\
                max=dumpmax,step=1,value=int(dumpmax-0.05*(dumpmax-dumpmean))),\
                     ything=things_list,ifig=fixed(ifig))

    def rp_plot(self, dump, ything, xthing='R', num_type='NDump', ifig=11, runlabel=None,\
                xxlim=None, yylim=None, logy=False,newfig=True,idn=0):
        '''
        Plot one thing or list for a line profile

        Parameters
        ----------

        ything : string
            name of y quantity to plot, print(rp.get_hr_variables())
            print(rp.get_lr_variables()) prints available options

        dump : integer or list of integers
            dump number or list of dump numbers

        num_type: string (case insensitive)
            If 'ndump' fname is expected to be a dump number (integer).
            If 't' fname is expected to be a time value in seconds; run
            history file (.hstry) must be available to search by time value.

        xthing : string
          name of x quantity to plot, default is 'R'

        runlabel : str
           label of this rp_set/case to appear in legend, defaults
           to run_id pulled from rprof header

        xxlim, yylim : float
           x and y lims, tuple

        logy : boolean
           log10 of ything for plotting; defaults to False

        newfig : boolean
           close and create new figure if True, this is the default

        idn : integer
           set to some value >0 to generate new series of line selection integers
        '''
        if runlabel is None: runlabel = self.__run_id
        # Ensure that dump is list type
        if type(dump) is not list:
            dump = [dump]
        len_dump = len(dump)
        # Get dump and assign it to a variable
        rp = self.get_dump(self.get_dump_list()[0])

        # Define resolution and throw errors if they don't match;
        # throw error if ything is not defined
        ything_computable = False
        if ything in rp.get_hr_variables():
            if xthing not in rp.get_hr_variables():
                print('ERROR: If ything is high resolution xthing must be too.')
                return
            res = 'h'
        elif ything in rp.get_lr_variables():
            if xthing not in rp.get_lr_variables():
                print('ERROR: If ything is low resolution xthing must be too.')
                return
            res = 'l'
        elif ything in self.get_computable_quantities():
            if xthing not in rp.get_lr_variables():
                # it seems all computable things are low res
                print('ERROR: If ything is computable xthing must be low res.')
                return
            ything_computable = True
            res = 'l'
        else:
            print("ERROR: Don't know ything")
            return

        # Define x- and y-values to be plotted, determine if logy is
        # necessary, generate plot
        if newfig:
            pl.close(ifig)
            pl.figure(ifig)
        for i,thisdump in enumerate(dump):
            xquantity = self.get(xthing,thisdump,num_type=num_type,resolution=res)
            if ything_computable:
                yquantity = self.compute(ything,thisdump,num_type=num_type)
            else:
                yquantity = self.get(ything,thisdump,num_type=num_type,resolution=res)
            if logy: yquantity = np.log10(yquantity)
            if num_type == 't':
                time_thing = " t/[min]="; d_num = thisdump/60.
            else:
                time_thing = " dump ="; d_num = thisdump
            pl.plot(xquantity,yquantity,label=runlabel+time_thing+str(d_num),\
                 color=utils.linestylecb(i+len_dump*idn)[2],\
                    linestyle=utils.linestylecb(i+len_dump*idn)[0],\
                 marker=utils.linestylecb(i+len_dump*idn)[1],\
                    markevery=utils.linestyle(i+len_dump*idn,a=25,b=7)[1])

        # Plot detailing
        pl.legend()
        pl.xlabel(xthing)

        if logy == False:
            pl.ylabel(ything)
        else:
            pl.ylabel('log '+ything)

        if xxlim is not None:
            pl.xlim(xxlim)

        if yylim is not None:
            pl.ylim(yylim)

    def plot_FV(self, fname, num_type='NDump', resolution='l', idec=3, xxlim=None, \
                yylim=None,legend='', ylog=True):
        print("Warning: plot_FV is deprecated. Use rp_plot instead.")
        return
        np.warnings.filterwarnings('ignore')
        R = self.get('R', fname=fname, num_type=num_type, resolution='l')
        FV = self.get('FV', fname=fname, num_type=num_type, resolution='l')
        t = self.get('t', fname=fname, num_type=num_type)
        yy = FV
        if ylog: yy = np.log10(FV)
        pl.plot(R, yy, utils.linestylecb(idec)[0], \
                color=utils.linestylecb(idec)[2], \
                label=legend+', {:.1f} min'.format(t/60.))
        pl.xlim(xxlim)
        pl.ylim(yylim)
        pl.xlabel('r / Mm')
        pl.ylabel('FV')
        pl.legend(loc=2)
        pl.tight_layout()

    def plot_A(self, fname, num_type='NDump', resolution='l', idec=3,xxlim=None, \
               yylim=None, legend=''):
        print("Warning: plot_FV is deprecated. Use rp_plot instead.")
        print("(Method not yet deleted, but marked for deleation.)")
        return

        R = self.get('R', fname=fname, num_type=num_type, resolution='l')
        A = self.get('A', fname=fname, num_type=num_type, resolution='l')
        t = self.get('t', fname=fname, num_type=num_type)
        pl.plot(R, A,  utils.linestylecb(idec)[0], \
                color=utils.linestylecb(idec)[2], \
                label=legend+', {:.1f} min'.format(t/60.))
        pl.xlim(xxlim)
        pl.ylim(yylim)
        pl.xlabel('r / Mm')
        pl.ylabel('A')
        pl.legend(loc=2)
        pl.tight_layout()

    def plot_vrad_prof(self,fname,num_type='NDump', vel_comps=['|U|','|Ut|','|Ur|'],\
                       plot_title=None,ifig=102,save_fig=True,logy=True,close_fig=True,\
                      id=0):
        '''Plot velocity profiles for one dump

        fname : int, list
          dump or list of dumps or times to be plotted

        num_type : str
          defaults to 'NDump' for fname to be dump number, set to
          't' for fname to be time in s

        vel_comps : list of str
          list of velocity components to be plotted

        '''

        if type(fname) is not list:
            fname = [fname]
        if close_fig: pl.close(ifig)
        ymax = 0.
        for j,dump in enumerate(fname):
            Ut = self.get('|Ut|',dump,num_type=num_type)
            U = self.get('|U|',dump,num_type=num_type)
            Ur = np.sqrt(U**2 - Ut**2)
            vels = [U, Ut, Ur]
            vel_keys=['|U|','|Ut|','|Ur|']
            labels = ['$U$', '$ U_\mathrm{h} $', '$U_\mathrm{r}$ ']
            vel_dict = {}; vel_dict['vel'] = {}; vel_dict['label'] = {}
            for i,vel_key in enumerate(vel_keys):
                vel_dict['vel'][vel_key] =  vels[i]
                vel_dict['label'][vel_key] =  labels[i]

            ifig=ifig
            if not pl.fignum_exists(ifig): pl.figure(ifig)

            cb = utils.colourblind
            R = self.get('R',dump,num_type=num_type)
            for i,vel in enumerate(vel_comps):
                if num_type == 't':
                    time_thing = " t/[min]="; d_num = dump/60.
                else:
                    time_thing = " dump ="; d_num = dump
                ything = vel_dict['vel'][vel]*1000.
                if logy:
                    ything = np.log10(ything)
                    ymax = max(ymax,ything.max())
                pl.plot(R,ything,utils.linestylecb(j+id)[0],\
                    color = utils.linestylecb(i+id)[2],label=vel_dict['label'][vel]\
                        +time_thing+str(d_num))
        ylab = '$U/\mathrm{[km/s]}$'
        if logy:
            ylab = '$log_\mathrm{10}$ '+ylab
            pl.ylim(ymax-2.5,ymax+0.1)
        pl.legend(loc=0); pl.xlabel('$R/\mathrm{[Mm]}$');pl.ylabel(ylab)
        if plot_title is not None: pyl.title(plot_title+", "+str(dump))

        if save_fig:
            if plot_title is not None:
                pl.savefig("v-profiles_"+plot_title+"_"+str(dump)+".pdf")
            else:
                pl.savefig("v-profiles_"+str(dump)+".pdf")

        Ur_max = np.max(Ur)            # max radial velocity in Mm
        return Ur_max*1000.  # return Ur_max in km/s

    def entrainment_rate(self, cycles, r_min, r_max, airmu, cldmu, var='vxz', criterion='min_grad', \
                         offset=0., integrate_both_fluids=False,
                         integrate_upwards=False, show_output=True, ifig0=1, \
                         silent=True, mdot_curve_label=None, file_name=None,
                         return_time_series=False):
        '''
        Function for calculating entrainment rates.

        Parameters
        ----------
        cycles : range
            cycles to get entrainment rate for
        r_min : float
            minimum radius to look for boundary
        r_max : float
            maximum radius to look for boundary

        Examples
        ---------
        .. ipython::

            In [136]: data_dir = '/data/ppm_rpod2/YProfiles/'
               .....: project = 'AGBTP_M2.0Z1.e-5'
               .....: ppm.set_YProf_path(data_dir+project)

            @savefig entrainment_rate.png width=6in
            In [136]: F4 = ppm.yprofile('F4')
               .....: dumps = np.array(range(0,1400,100))
               .....: F4.entrainment_rate(dumps,27.7,28.5)

        '''

        def regrid(x, y, x_int):
            int_func = scipy.interpolate.CubicSpline(x[::-1], y[::-1])
            return int_func(x_int)

        r = self.get('R', fname = cycles[0], resolution='l')

        idx_min = np.argmin(np.abs(r - r_max))
        idx_max = np.argmin(np.abs(r - r_min))

        r_min = r[idx_max]
        r_max = r[idx_min]

        r = r[idx_min:(idx_max + 1)]
        r_int = np.linspace(r_min, r_max, num = 20.*(idx_max - idx_min + 1))
        dr_int = cdiff(r_int)

        time = np.zeros(len(cycles))
        r_b = np.zeros(len(cycles))
        r_top = np.zeros(len(cycles))
        for i in range(len(cycles)):
            time[i] = self.get('t', fname = cycles[i], resolution='l')

            if var == 'vxz':
                q = (self.get('|Ut|', fname = cycles[i], resolution='l'))[idx_min:(idx_max + 1)]**0.5
            else:
                q = (self.get(var, fname = cycles[i], resolution='l'))[idx_min:(idx_max + 1)]

            q_int = regrid(r, q, r_int)
            grad = cdiff(q_int)/dr_int

            if criterion == 'min_grad':
                idx_b = np.argmin(grad)
            elif criterion == 'max_grad':
                idx_b = np.argmax(grad)
            else:
                idx_b = np.argmax(np.abs(grad))

            r_b[i] = r_int[idx_b]
            r_top[i] = r_b[i]

            # Optionally offset the integration limit by a multiple of q's
            # scale height.
            if np.abs(grad[idx_b]) > 0.:
                H_b = q_int[idx_b]/np.abs(grad[idx_b])
                r_top[i] += offset*H_b

        timelong = time
        delta = 0.05*(np.max(time) - np.min(time))
        timelong = np.insert(timelong,0, timelong[0] - delta)
        timelong = np.append(timelong, timelong[-1] + delta)

        # fc = fit coefficients
        r_b_fc = np.polyfit(time, r_b, 1)
        r_b_fit = r_b_fc[0]*timelong + r_b_fc[1]
        r_top_fc = np.polyfit(time, r_top, 1)
        r_top_fit = r_top_fc[0]*timelong + r_top_fc[1]

        m_ir = np.zeros(len(cycles))
        r = self.get('R', fname = cycles[0], resolution='h')
        r_int = np.linspace(np.min(r), np.max(r), num = 20.*len(r))
        dr_int = cdiff(r_int)
        for i in range(len(cycles)):
            rho0 = self.get('Rho0', fname = cycles[i], resolution='h')
            rho1 = self.get('Rho1', fname = cycles[i], resolution='h')
            rho = rho0 + rho1
            if not integrate_both_fluids:
                FV_HHe = self.get('FV', fname = cycles[i], resolution='h')
                rhocldtoair = cldmu/airmu
                rhoair = rho/((1. - FV_HHe) + FV_HHe*rhocldtoair)
                rhocld = rhocldtoair*rhoair
                rho_HHe = rhocld
                rho = rho_HHe*FV_HHe

            rho_int = regrid(r, rho, r_int)

            idx_top = np.argmin(np.abs(r_int - r_top[i]))
            dm = 4.*np.pi*r_int**2*dr_int*rho_int

            if integrate_upwards:
                m_ir[i] = np.sum(dm[(idx_top + 1):-1])
            else:
                m_ir[i] = np.sum(dm[0:(idx_top + 1)])

        # fc = fit coefficients
        m_ir *= 1e27/nuconst.m_sun
        m_ir_fc = np.polyfit(time, m_ir, 1)
        m_ir_fit = m_ir_fc[0]*timelong + m_ir_fc[1]
        if integrate_upwards:
            mdot = -m_ir_fc[0]
        else:
            mdot = m_ir_fc[0]

        if show_output:
            cb = utils.colourblind
            pl.close(ifig0); fig1 = pl.figure(ifig0)
            pl.plot(time/60., r_top, color = cb(5), ls = '-', label = r'r$_\mathrm{top}$')
            pl.plot(time/60., r_b, color = cb(8), ls = '--', label = r'r$_\mathrm{b}$')
            pl.plot(timelong/60., r_top_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.plot(timelong/60., r_b_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.xlabel('t / min')
            pl.ylabel('r / Mm')
            xfmt = ScalarFormatter(useMathText = True)
            pl.gca().xaxis.set_major_formatter(xfmt)
            pl.legend(loc = 0)
            fig1.tight_layout()

            if not silent:
                print('r_b is the radius of the convective boundary.')
                print('r_b_fc = ', r_b_fc)
                print('dr_b/dt = {:.2e} km/s\n'.format(1e3*r_b_fc[0]))
                print('r_top is the upper limit for mass integration.')
                print('dr_top/dt = {:.2e} km/s'.format(1e3*r_top_fc[0]))

            max_val = np.max(m_ir)
            #if show_fits:
            max_val = np.max((max_val, np.max(m_ir_fit)))
            max_val *= 1.1 # allow for some margin at the top
            oom = int(np.floor(np.log10(max_val)))

            pl.close(ifig0 + 1); fig2 = pl.figure(ifig0 + 1)
            pl.plot(time/60., m_ir/10**oom, color = cb(5), label=mdot_curve_label)
            mdot_str = '{:e}'.format(mdot)
            parts = mdot_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            #if show_fits:
            if integrate_upwards:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{a}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(-mantissa, exponent)
            else:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(mantissa, exponent)
            pl.plot(timelong/60., m_ir_fit/10**oom, color = cb(4), ls = '-', lw = 0.5, label = lbl)
            pl.xlabel('t / min')
            if integrate_upwards:
                sub = 'a'
            else:
                sub = 'e'
            ylbl = r'M$_{:s}$ / 10$^{{{:d}}}$ M$_\odot$'.format(sub, oom)
            if oom == 0.:
                ylbl = r'M$_{:s}$ / M$_\odot$'.format(sub)

            pl.ylabel(ylbl)
            yfmt = FormatStrFormatter('%.1f')
            fig2.gca().yaxis.set_major_formatter(yfmt)
            fig2.tight_layout()
            if integrate_upwards:
                loc = 1
            else:
                loc = 2
            pl.legend(loc = loc)
            if file_name is not None:
                fig2.savefig(file_name)

            if not silent:
                print('Resolution: {:d}^3'.format(2*len(r)))
                print('m_ir_fc = ', m_ir_fc)
                print('Entrainment rate: {:.3e} M_Sun/s'.format(mdot))

        if return_time_series:
            return m_ir
        else:
            return mdot


class Rprof:
    '''
    Rprof reads and holds the contents of a single .rprof file written by
    PPMstar 2.0.
    '''

    def __init__(self, file_path, verbose=3):
        '''
        Init method.

        Parameters
        ----------
        file_path: string
            Path to the .rprof file.
        verbose: integer
            Verbosity level as defined in class Messenger.
        '''

        self.__is_valid = False
        self.__messenger = Messenger(verbose=verbose)
        if self.__read_rprof_file(file_path) != 0:
            return

        self.__is_valid = True

    def __read_rprof_file(self, file_path):
        '''
        Reads a single .rprof file written by PPMstar 2.0.

        Parameters
        ----------
        file_path: string
            Path to the .rprof file.

        Returns
        -------
        int
            0 on success.
        NoneType
            Something failed.
        '''

        try:
            with open(file_path, 'r') as fin:
                lines = fin.readlines()
            self.__file_path = file_path
        except IOError as e:
            err = 'I/O error({0}): {1}'.format(e.errno, e.strerror)
            self.__messenger.error(err)
            return None

        self.__header_vars = []
        self.__header_data = {}
        self.__lr_vars = [] # low-resolution data columns
        self.__lr_data = {} # low-resolution data columns
        self.__hr_vars = [] # high-resolution data columns
        self.__hr_data = {} # high-resolution data columns

        l = 0 # line index

        # Find simulation time.
        while not lines[l].startswith('DUMP'):
            l += 1
        self.__header_vars.append('t')
        self.__header_data['t'] = float(lines[l].split('=')[1].split(',')[0])

        # Find resolution.
        while not lines[l].startswith('Nx ='):
            l += 1
        self.__header_vars.append('Nx')
        self.__header_data['Nx'] = int(lines[l].split()[-1])

        eof = False
        footer_reached = False
        while l < len(lines):
            # Find the next table.
            while True:
                sline = lines[l].split()
                if len(sline) > 0:
                    if sline[0] == 'IR':
                        col_names = sline
                        break

                    if sline[0] == 'DATE:':
                        footer_reached = True
                        # The footer always starts with 16 lines of text. The
                        # idea was to keep a brief run description there, but it
                        # is hardcoded in PPMstar and no one seems to bother to
                        # go and ever change it. We will skip those 16 lines to
                        # get to an array of parameters, which we want to read.
                        l += 16
                        break

                l += 1
                if l == len(lines):
                   eof = True
                   break

            if footer_reached or eof:
                break

            # Go to the table's body.
            while True:
                stripped_line = lines[l].strip()
                if len(stripped_line) > 0 and stripped_line[0].isdigit():
                    break
                l += 1

            sline = lines[l].split()
            # The first number is always the number of radial zones.
            n = int(sline[0])

            # n should equal Nx/2 for standard-resolution columns
            # and Nx for high-resolution ones.
            is_hr = False
            if n > self.__header_data['Nx']/2:
                is_hr = True

            # Register the columns, skipping the 1st one (IR) and any
            # that are already registered (some columns appear in
            # multiple places in rprof files, probably to make them
            # easier for humans to read).
            for i in range(1, len(col_names)):
                if is_hr and col_names[i] not in self.__hr_vars:
                    self.__hr_vars.append(col_names[i])
                    self.__hr_data[col_names[i]] = np.zeros(n)
                elif col_names[i] not in self.__lr_vars:
                    self.__lr_vars.append(col_names[i])
                    self.__lr_data[col_names[i]] = np.zeros(n)

            for j in range(n):
                sline = lines[l].split()
                idx = int(sline[0])
                for i in range(1, len(col_names)):
                    val = float(sline[i])
                    if is_hr:
                        self.__hr_data[col_names[i]][n-idx] = val
                    else:
                        self.__lr_data[col_names[i]][n-idx] = val
                l += 1

        if footer_reached:
            while l < len(lines):
                sline = lines[l].split()
                # The two consecutive lists of parameters that appear in the
                # footer are formatted this way:
                #
                # line_index   par1_name   par1_value   par2_name   par2_value
                #
                # Most of them are probably constant, but some may change upon a
                # restart.
                if len(sline) == 5:
                    for i in (1, 3):
                        par_name = sline[i]
                        if '.' in sline[i+1]:
                            par_value = float(sline[i+1])
                        else:
                            par_value = int(sline[i+1])

                        # Although we are technically in the file's footer, we
                        # call these parameters "header variables".
                        self.__header_vars.append(par_name)
                        self.__header_data[par_name] = par_value
                l += 1

        self.__lr_vars = sorted(self.__lr_vars, key=lambda s: s.lower())
        self.__hr_vars = sorted(self.__hr_vars, key=lambda s: s.lower())
        self.__header_vars = sorted(self.__header_vars, key=lambda s: s.lower())

        # The list(set()) construct removes duplicate names.
        self.__anyr_vars = sorted(list(set(self.__lr_vars+self.__hr_vars)), \
                                  key=lambda s: s.lower())
        self.__all_vars = sorted(list(set(self.__header_vars+self.__anyr_vars)), \
                                  key=lambda s: s.lower())

        return 0

    def is_valid(self):
        '''
        Checks if the instance is valid, i.e. fully initialised.

        Returns
        -------
        boolean
            True if the instance is valid. False otherwise.
        '''

        return self.__is_valid

    def get_header_variables(self):
        '''
        Returns a list of header variables available.
        '''

        # Return a copy.
        return list(self.__header_vars)

    def get_lr_variables(self):
        '''
        Returns a list of low-resolution variables available.
        '''

        # Return a copy.
        return list(self.__lr_vars)

    def get_hr_variables(self):
        '''
        Returns a list of high-resolution variables available.
        '''

        # Return a copy.
        return list(self.__hr_vars)

    def get_anyr_variables(self):
        '''
        Returns a list of any-resolution variables available.
        '''

        # Return a copy.
        return list(self.__anyr_vars)

    def get_all_variables(self):
        '''
        Returns a list of all variables available.
        '''

        # Return a copy.
        return list(self.__all_vars)

    def get(self, var, resolution='l'):
        '''
        Returns variable var if it exists. The method first searches for
        a variable name match in this order: header variables, low-
        resolution variables, and high-resolution variables.

        Parameters
        ----------
        var: string
            Variable name.
        resolution: string (case insensitive)
            'l' (low) or 'h' (high). A few variables are available at
            double resolution ('h'), the rest correspond to the resolution
            of the computational grid ('l').

        Returns

        -------
        np.ndarray
            Variable var, if found.
        '''

        if var in self.__header_vars:
            return self.__header_data[var]
        else:
            if resolution.lower() == 'l':
                if var in self.__lr_vars:
                    return np.array(self.__lr_data[var])
                elif var in self.__hr_vars:
                    data = np.array(self.__hr_data[var])
                    # This will only work for arrays of even length, but
                    # we always get such arrays from PPMstar.
                    data = 0.5*(data[::2] + data[1::2])
                    return data
                else:
                    err = ("Variable '{:s}' does not exist.").format(var)
                    self.__messenger.error(err)

                    msg = 'Available variables:\n'
                    msg += str(self.__anyr_vars)
                    self.__messenger.message(msg)
            elif resolution.lower() == 'h':
                if var in self.__hr_vars:
                    return np.array(self.__hr_data[var])
                else:
                    err = ("High-resolution data not available for variable "
                          "{:s}.").format(var)
                    self.__messenger.error(err)

                    msg = 'Available high-resolution variables:\n'
                    msg += str(self.__hr_vars)
                    self.__messenger.message(msg)
            else:
                err = "Unknown resolution setting '{:s}'.".format(resolution)
                self.__messenger.error(err)

class MomsData():
    '''
    MomsData reads in a single briquette averaged data cube which contains up to 10 user-defined
    variables. In the MomentsDataSet it is assumed that either whatever(0) = xc OR an rprof is
    supplied so that a suitable grid can be made.

    Parameters
    ----------
    file_path: string
        Path to the .aaa file.
    verbose: integer
        Verbosity level as defined in class Messenger.
    '''

    # we have a couple of "constant" class variables
    def __init__(self, file_path, verbose=3):

        # initialize a few quantities
        self.__is_valid = False
        self.__messenger = Messenger(verbose=verbose)

        # how many extra "ghost" indices are there for each dimension?
        self.__ghost = 2

        # number of whatevers, probably won't change anytime soon
        self.__number_of_whatevers = 10

        # set blanks for these at the start
        self.ngridpoints = 0
        self.run_ngridpoints = 0

        # read a moments data binary file
        if not self.read_moms_cube(file_path):
            return

        # all is good, initialize bools
        self.__is_valid = True

    def read_moms_cube(self, file_path):
        '''
        Reads a single .aaa uncompressed data cube file written by PPMstar 2.0.

        Parameters
        ----------
        file_path: string
            Path to the .aaa data cube file

        Returns
        -------
        Boolean
            True on success.
            False on failure.
        '''

        # we need to first try out if we can read the data...
        try:
            ghostdata = np.fromfile(file_path,dtype='float32',count=-1,sep="")

        except IOError as e:
            err = 'I/O error({0}): {1}'.format(e.errno, e.strerror)
            self.__messenger.error(err)
            return False


        # just by knowing how paul layed out the data I can infer these quantities
        self.run_ngridpoints = int(np.ceil(4. * (np.power(np.shape(ghostdata)[0] / 10.,1/3.) - self.__ghost)))
        self.ngridpoints = int(np.ceil(self.run_ngridpoints/4.))

        # ok, I can reshape without reallocating an array
        ghostview = ghostdata.view()
        size = int(np.ceil(self.ngridpoints + self.__ghost))
        ghostview.shape = (self.__number_of_whatevers, size, size, size)

        # my removed "ghost" values is quite easy! Now in an intuitive format
        # self.var[z,y,x]
        self.var = ghostview[0:,(self.__ghost-1):(self.ngridpoints+self.__ghost-1),
                              (self.__ghost-1):(self.ngridpoints+self.__ghost-1),
                              (self.__ghost-1):(self.ngridpoints+self.__ghost-1)]

        return True

    def is_valid(self):
        '''
        Checks if the instance is valid

        Returns
        -------
        boolean
            True if the data cube was read.
            False if there is an error in reading.
        '''

        return self.__is_valid


    def get(self, varloc):
        '''
        Returns a 3d array of the variable that is defined at whatever(varloc).

        Parameters
        ----------
        varloc: integer
            integer index of the quantity that is defined under whatever(varloc)

        Returns
        -------
        np.ndarray
            Variable at whatever(varloc)
        '''

        # self.var contains a shaped array that has no ghosts
        return self.var[varloc]

class MomsDataSet:
    '''
    MomsDataSet tracks a set of "dumps" of MomsData which are single "moments" datacubes
    from PPMStar 2.0

    Parameters
    ----------
    dir_name: string
        Name of the directory to be searched for .aaa uncompressed moms datacubes
    init_dump_read: integer
        The initial dump to read into memory when object is initialized
    dumps_in_mem: integer
        The number of dumps to be held into memory. These datacubes can be large (~2Gb for 384^3)
    var_list: list
        This is a list that can be filled with strings that will reference data. E.g element 0 is
        'xc' which will refer to the variable location data[0] in the large array of data
    rprofset: RprofSet
        Instead of constructing the grid with moments data, use the rprofset
    verbose: integer
        Verbosity level as defined in class Messenger.
    '''

    def __init__(self, dir_name, init_dump_read=0, dumps_in_mem=2, var_list=[], rprofset=None, verbose=3):

        self._is_valid = False
        self._messenger = Messenger(verbose=verbose)

        # __find_dumps() will set the following variables,
        # if successful:
        self._dir_name = ''
        self._dumps = []
        if not self._find_dumps(dir_name):
            return

        # we do not create the grid, we only do that if it is needed
        # bools track what has happened
        self._cgrid_exists = False
        self._sgrid_exists = False
        self._mollweide_exists = False

        # we also check if we have our unit vectors
        self._grid_jacobian_exists = False

        # setup some useful dictionaries/lists
        self._varloc = {}
        self._interpolation_methods = ['trilinear','moments']

        # options for MomsData
        self._number_of_whatevers = 10

        # For storing multiple momsdata we use a dictionary so that it can be referenced
        # I will store in a list the dumps that are in there
        self._many_momsdata = {}
        self._many_momsdata_keys = []
        self._dumps_in_mem = dumps_in_mem

        # initialize the dictionaries
        if not self._set_dictionaries(var_list):
            return

        # get the initial dump momsdata
        self._get_dump(init_dump_read)

        # hold the initial dump in attribute
        self.what_dump_am_i = init_dump_read

        # set objects ngridpoints and original run ngridpoints
        # these are deep copies to ensure no reference back on momsdata
        self.moms_ngridpoints = copy.deepcopy(self._many_momsdata[str(init_dump_read)].ngridpoints)
        self.run_ngridpoints = copy.deepcopy(self._many_momsdata[str(init_dump_read)].run_ngridpoints)

        # do we have an rprofset?
        self._rprofset = rprofset

        # On instantiation we create cartesian ALWAYS
        if not self._cgrid_exists:
            self._get_cgrid()

        # we now have grid, can easily get the following
        self.moms_gridresolution = np.mean(np.diff(self._unique_coord))
        self.run_gridresolution = np.mean(np.diff(self._unique_coord))/4.

        # alright we are now a valid instance

    def _find_dumps(self, dir_name):
        '''
        Searches for .aaa files and creates an internal list of dump numbers available.

        Parameters
        ----------
        dir_name: string
            Name of the directory to be searched for .aaa files.

        Returns
        -------
        boolean
            True: when a set of .aaa files has been found.
            False: otherwise.
        '''

        if not os.path.isdir(dir_name):
            err = "Directory '{:s}' does not exist.".format(dir_name)
            self._messenger.error(err)
            return False

        # join() will add a trailing slash if not present.
        self._dir_name = os.path.join(dir_name, '')

        # ok this directory contains bobs like directory structure, search in
        # sub-directories for actual files, grab dumps from file names
        moms_files = []

        for dirpath, dirnames, filenames in os.walk(self._dir_name):
            for filename in [f for f in filenames if f.endswith('.aaa')]:
                moms_files.append(os.path.join(dirpath,filename))

        if len(moms_files) == 0:
            err = "No .aaa files found in '{:s}'.".format(self._dir_name)
            self._messenger.error(err)
            return False

        moms_files = [os.path.basename(moms_files[i]) \
                       for i in range(len(moms_files))]

        # run_id is always separated from the rest of the file name
        # by a dash at the END of the file
        _ind = moms_files[0].rindex('-')
        self._run_id = moms_files[0][0:_ind]

        for i in range(len(moms_files)):

            _ind = moms_files[i].rindex('-')
            check_runId = moms_files[i][0:_ind]
            if check_runId != self._run_id:
                wrng = (".aaa files with multiple run ids found in '{:s}'."
                        "Using only those with run id '{:s}'.").\
                       format(self._dir_name, self._run_id)
                self._messenger.warning(wrng)
                continue

            # Get rid of the extension and try to parse the dump number.
            # Skip files that do not fit the momsdata naming pattern.
            split = moms_files[i][_ind+1:].split('.')

            # there is always BQav prefix before dump
            dump_num = split[0][4:]
            if dump_num.isnumeric():
                self._dumps.append(int(dump_num))
            else:
                self.__messenger.error("moms filename does not have 4 digits after -BQav"+\
                                moms_files[i])
                return False

        self._dumps = sorted(self._dumps)
        msg = "{:d} .aaa files found in '{:s}.\n".format(len(self._dumps), \
              self._dir_name)
        msg += "Dump numbers range from {:d} to {:d}.".format(\
               self._dumps[0], self._dumps[-1])
        self._messenger.message(msg)
        if (self._dumps[-1] - self._dumps[0] + 1) != len(self._dumps):
            wrng = 'Some dumps are missing!'
            self._messenger.warning(wrng)

        return True

    def _get_dump(self, dump):
        '''
        Gets a new dump of MomsData that is in memory or instantiates a new MomsData

        Parameters
        ----------
        dump: integer
            The dump number you want access to
        '''

        if dump not in self._dumps:
            err = 'Dump {:d} is not available.'.format(dump)
            self._messenger.error(err)
            return None

        file_path = '{:s}{:04d}/{:s}-BQav{:04d}.aaa'.format(self._dir_name, \
                                                             dump, self._run_id, dump)

        # we first check if we can add a new moments data to memory
        # without removing another
        if len(self._many_momsdata) < self._dumps_in_mem:

            # add it to our dictionary!
            self._many_momsdata.update(zip([str(dump)],[MomsData(file_path)]))

            # append the key. This keeps track of order of read in
            self._many_momsdata_keys.append(str(dump))

        else:

            # we gotta remove one of them, this will be index 0 of a list
            del self._many_momsdata[str(self._many_momsdata_keys[0])]
            self._many_momsdata_keys.remove(self._many_momsdata_keys[0])

            # now add a new momsdata object to our dict
            self._many_momsdata.update(zip([str(dump)],[MomsData(file_path)]))

            # append the key. This keeps track of order of read in
            self._many_momsdata_keys.append(str(dump))

        # all is good. update what_dump_am_i
        self.what_dump_am_i = dump

    def _get_igrid(self, radius, theta, phi, npoints):
        """
        For interpolation methods I need to convert from spherical coordinates to cartesian. To
        support scipy.interpolate.RegularGridInterpolator igrid is formatted as:

        igrid.shape = [npoints,3]
        igrid[:,0] = z, igrid[:,1] = y, igrid[:,2] = x

        Parameters
        ----------
        radius: np.ndarray
            The radius values for the interpolated quantity
        theta: np.ndarray
            The "physics" theta values for the interpolated quantity
        phi: np.ndarray
            The "physics" phi values for the interpolated quantity
        npoints: int
            The number of interpolated points to be used for each "radius"

        Returns
        -------
        igrid: np.ndarray
            The grid of x,y,z points to be interpolated to
        """

        # we need to hold our coordinate values
        igrid = np.zeros((len(radius)*npoints,3))

        # using the spherical coordinates grid of theta and phi (and r) we can get x,y,z
        igrid[:npoints, 0] = radius[0] * np.cos(theta)                  # z
        igrid[:npoints, 1] = radius[0] * np.sin(theta) * np.sin(phi)    # y
        igrid[:npoints, 2] = radius[0] * np.sin(theta) * np.cos(phi)    # x

        # we can use a shortcut to get the coordinates at other radii
        if len(radius) > 1:
            for i in range(len(radius)-1):

                # since x = r * const, y = r * const, z = r * const for any ray we can...
                igrid[npoints*(i+1):npoints*(i+2)] = igrid[:npoints] * radius[i+1] / radius[0]

        return igrid

    def _interpolation_moments(self, var, igrid, x_idx, y_idx, z_idx):
        '''
        The interpolation form is quadratic:

        f(xi,yi,zi) = a000 + a100x + a010y + a001z + a200x^2 + a110xy + a101xz + a020y^2
                      + a011yz + a002z^2

        Through moments averaging, we can determine these coefficients based on the value of f at
        various points. Similar to the convention above, the subscripts for coefficients below
        refer to a displacement of central cell points from our closest central cell to where our
        interpolation is actually happening in an (x,y,z) format. So...

        a000 = 5/4c000 - 1/24(c100 + c-100 + c010 + c0-10 + c001 + c00-1)
        a100 = 1/2(c100 - c-100)
        a010 = 1/2(c010 - c0-10)
        a001 = 1/2(c001 - c00-1)
        a110 = 1/4(c110 + c-1-10 - c-110 - c1-10)
        a101 = 1/4(c101 + c-10-1 - c-101 - c10-1)
        a011 = 1/4(c011 + c0-1-1 - c0-11 - c01-1)
        a200 = 1/2(c100 + c-100) - c000
        a020 = 1/2(c010 + c0-10) - c000
        a002 = 1/2(c001 + c00-1) - c000

        Parameters
        ----------
        var: np.ndarray
            The quantity on the grid
        igrid: np.ndarray
            The array that contains all of the points that are to be interpolated to
            igrid.shape = [nset,ninterpolation_points,3]
            igrid[nset,:,0] = z, igrid[nset,:,1] = y, igrid[nset,:,2] = z
        x_idx: np.ndarray
            The x indices that the interpolation points are closest to
        y_idx: np.ndarray
            The y indices that the interpolation points are closest to
        z_idx: np.ndarray
            The z indices that the interpolation points are closest to
        '''

        # flatten igrid, now every 3 is either x,y or z values
        iflat = igrid.flatten()

        # Grab all of the c values needed to get the a coefficients. Note var[z,y,x]

        # a000, a100, a010, a001
        c000 = var[z_idx, y_idx, x_idx]
        c100 = var[z_idx, y_idx, x_idx+1]
        c_100 = var[z_idx, y_idx, x_idx-1]
        c010 = var[z_idx, y_idx+1, x_idx]
        c0_10 = var[z_idx, y_idx-1, x_idx]
        c001 = var[z_idx+1, y_idx, x_idx]
        c00_1 = var[z_idx-1, y_idx, x_idx]

        # a110
        c110 = var[z_idx, y_idx+1, x_idx+1]
        c_1_10 = var[z_idx, y_idx-1, x_idx-1]
        c1_10 = var[z_idx, y_idx-1, x_idx+1]
        c_110 = var[z_idx, y_idx+1, x_idx-1]

        # a101
        c101 = var[z_idx+1, y_idx, x_idx+1]
        c_10_1 = var[z_idx-1, y_idx, x_idx-1]
        c_101 = var[z_idx+1, y_idx, x_idx-1]
        c10_1 = var[z_idx-1, y_idx, x_idx+1]

        # a011
        c011 = var[z_idx+1, y_idx+1, x_idx]
        c0_1_1 = var[z_idx-1, y_idx-1, x_idx]
        c0_11 = var[z_idx+1, y_idx-1, x_idx]
        c01_1 = var[z_idx-1, y_idx+1, x_idx]

        # now compute a's with my c values
        a000 = 5./4. * c000 - 1/24. * (c100 + c_100 + c010 + c0_10 + c001 + c00_1)
        a100 = 1/2. * (c100 - c_100)
        a010 = 1/2. * (c010 - c0_10)
        a001 = 1/2. * (c001 - c00_1)
        a110 = 1/4. * (c110 + c_1_10 - c_110 - c1_10)
        a101 = 1/4. * (c101 + c_10_1 - c_101 - c10_1)
        a011 = 1/4. * (c011 + c0_1_1 - c0_11 - c01_1)
        a200 = 1/2. * (c100 + c_100) - c000
        a020 = 1/2. * (c010 + c0_10) - c000
        a002 = 1/2. * (c001 + c00_1) - c000

        # using the flattened igrid...
        xiflat = iflat[2::3]
        yiflat = iflat[1::3]
        ziflat = iflat[::3]

        # the formula uses CELL CENTERED COORDINATES, I will have to subtract off the cell
        # centered x,y,z from the "flats" and then scale it with the fact that: 1 cell width = 1
        xiflat = (xiflat - self._unique_coord[x_idx]) / np.mean(abs(np.diff(self._unique_coord)))
        yiflat = (yiflat - self._unique_coord[y_idx]) / np.mean(abs(np.diff(self._unique_coord)))
        ziflat = (ziflat - self._unique_coord[z_idx])  / np.mean(abs(np.diff(self._unique_coord)))

        # interpolate the quantity using the definition of f
        var_interp = (a000 + a100*xiflat + a010*yiflat + a001*ziflat + a110*xiflat*yiflat +
                         a101*xiflat*ziflat + a011*yiflat*ziflat + a200*xiflat*xiflat +
                         a020*yiflat*yiflat + a002*ziflat*ziflat)

        return var_interp

    def _set_dictionaries(self, var_list):
        '''
        This function will setup the dictionaries that will house multiple MomsData objects and
        a convenient dictionary to refer to variables by a string

        Parameters
        ---------
        var_list: list
            A list that may contain strings to associate the integer locations "varloc"

        Returns
        -------
        Boolean
            True if successful
            False if failure
        '''

        # check if the list is empty
        if not var_list:

            # ok it is empty, construct default dictionary
            var_keys = [str(i) for i in range(self._number_of_whatevers)]
            var_vals = [i for i in range(self._number_of_whatevers)]

        else:

            # first we check that var_list is the correct length
            if len(var_list) != self._number_of_whatevers:

                # we use the default
                var_keys = [str(i) for i in range(self._number_of_whatevers)]
                var_vals = [i for i in range(self._number_of_whatevers)]

            else:

                # ok we are in the clear
                var_keys = [str(i) for i in var_list]
                var_vals = [i for i in range(self._number_of_whatevers)]

                # I will also allow for known internal varloc to point to the same things
                # with this dictionary, i.e xc: varloc = 0 ALWAYS
                var_keys2 = [str(i) for i in range(self._number_of_whatevers)]

                self._varloc.update(zip(var_keys2,var_vals))

        # construct the variable dictionary
        self._varloc.update(zip(var_keys,var_vals))

        return True

    def _transform_mollweide(self, theta, phi):
        '''
        Transforms a "physics" spherical coordinates array into the spherical coordinates
        that matplotlib uses for projection plots. This creates a copy of the input arrays

        Parameters
        ----------
        theta: np.ndarray
            An array of theta with the "physics" defined angle to be converted
        phi: np.ndarray
            An array of phi with the "physics" defined angle to be converted

        Returns
        -------
        theta: np.ndarray
            theta transformed to the "mollweide" defined angle
        phi: np.ndarray
            phi transformed to the "mollweide" defined angle
        '''

        # create a copy
        phi_copy = phi.copy()
        theta_copy = theta.copy()

        # phi instead goes from -pi to pi with -pi/2 being the -y axis and
        # the x axis defines phi = 0

        phi_copy[np.where(phi_copy > np.pi)] = phi_copy[np.where(phi_copy > np.pi)] - 2*np.pi

        # theta instead goes from -pi/2 to pi/2 with pi/2 being the positive
        # z axis and the xy plane defines theta = 0
        theta_copy[np.where(theta_copy <= np.pi/2.)] = abs(theta_copy[np.where(theta_copy <= np.pi/2.)] - np.pi/2.)
        theta_copy[np.where(theta_copy > np.pi/2.)] = -(theta_copy[np.where(theta_copy > np.pi/2.)] - np.pi/2.)

        return theta_copy, phi_copy

    def _transform_spherical(self, theta, phi):
        '''
        Transforms a "mollweide" spherical coordinates array into the "physics" spherical coordinates.
        This creates a copy of the input arrays

        Parameters
        ----------
        theta: np.ndarray
            An array of theta with the "mollweide" defined angle to be converted
        phi: np.ndarray
            An array of phi with the "mollweide" defined angle to be converted

        Returns
        -------
        theta: np.ndarray
            theta transformed to the "physics" defined angle
        phi: np.ndarray
            phi transformed to the "physics" defined angle
        '''

        # create a copy
        phi_copy = phi.copy()
        theta_copy = theta.copy()

        # mollweide arrays have the following which we will invert!:
        # phi instead goes from -pi to pi with -pi/2 being the -y axis and
        # the x axis defines phi = 0
        phi_bool = phi_copy < 0
        phi_copy[phi_bool] = phi_copy[phi_bool] + 2*np.pi

        # theta instead goes from -pi/2 to pi/2 with pi/2 being the positive
        # z axis and the xy plane defines theta = 0
        theta_bool = theta_copy >= 0
        theta_copy[theta_bool] = abs(theta_copy[theta_bool] - np.pi/2.)
        theta_copy[~theta_bool] = abs(theta_copy[~theta_bool]) + np.pi/2.

        return theta_copy, phi_copy

    def _sphericalHarmonics_grid(self, radius, N):
        """
        Create a uniformly spaced (in spherical coordinates) grid of points in which
        the interpolation is done on to get a value of the quantity of radius r which
        you intend to compute the spherical harmonics coefficients on. This requires a
        special grid of points

        Parameters
        ----------
        radius: np.ndarray
            The radius values for the interpolated quantity
        N: int
            The number of subdivisions of theta between np.pi and 0

        Returns
        -------
        igrid: np.ndarray
            The grid of x,y,z points to be interpolated to
        theta_interp: np.ndarray
            The theta spherical coordinates (physics) of the "npoints"
        phi_interp: np.ndarray
            The phi spherical coordinates (physics) of the "npoints"
        """

        # to conform with spherical harmonics, start at 90 deg N, skip 90 deg S!
        dangle = np.pi / (N)

        theta_points = np.arange(0, N) * dangle
        phi_points = np.arange(0, 2*N) * dangle

        # theta changes with axis=1, phi changes wtih axis=0
        phi, theta = np.meshgrid(phi_points, theta_points)

        # now I need to flatten for igrid
        npoints = N * 2*N
        igrid = self._get_igrid(radius, theta.flatten(), phi.flatten(), npoints)

        # return these quantities
        return igrid, theta_points, phi_points

    def _constantArea_spherical_grid(self, radius, npoints):
        '''
        Create a uniformly spaced (in spherical coordinates) grid of points in which
        the interpolation is done on to get a value of quantity at radius r

        Parameters
        ----------
        radius: np.ndarray
            The radius values for the interpolated quantity
        npoints: int
            The number of points on the uniform grid. 5000 is plenty for most images

        Returns
        -------
        igrid: np.ndarray
            The grid of x,y,z points to be interpolated to
        theta_interp: np.ndarray
            The theta spherical coordinates (physics) of the "npoints"
        phi_interp: np.ndarray
            The phi spherical coordinates (physics) of the "npoints"
        '''

        indices = np.arange(0, npoints) + 0.5

        # Based on equal amount of points in equal area on a sphere...
        theta = np.arccos(1. - 2.*indices/float(npoints))
        phi = np.pi * (1 + 5**0.5) * indices  - 2*np.pi*np.floor(np.pi * (1 + 5**0.5) * indices / (2 * np.pi))

        # create the interp_grid. It is written in this fashion to work with interpolation, z,y,x
        igrid = self._get_igrid(radius, theta, phi, npoints)

        return igrid, theta, phi

    def _get_cgrid(self):
        '''
        Constructs the PPMStar cartesian grid from either the internal rprofset or the assumed xc
        saved in whatever(0).

        Returns
        -------
        Boolean
            True on success.
            False on failure.
        '''

        # check, do we already have this?
        if not self._cgrid_exists:

            # Ok, we will always have a dump in memory so carry on!
            if isinstance(self._rprofset, RprofSet):

                # we can construct the xc_array
                rprof = self._rprofset.get_dump(self._rprofset.get_dump_list()[0])
                dx = rprof.get('deex')

                # 4 * dx * (ngridpoints/2.) gives me the right boundary but I want the central value so
                right_xcbound = 4. * dx * (self.moms_ngridpoints/2.) - 4. * (dx/2.)
                left_xcbound = -right_xcbound

                # So unfortunately float32 is terrible, only accurate to 1e-6. This isn't strictly uniform...
                grid_values = 4. * dx * np.arange(0,self.moms_ngridpoints) + left_xcbound

                xc_array = np.ones((self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)) * grid_values

            else:
                # We assume that x is always the zero varloc
                # We also make sure it is a copy and separated from self._momsdata.data
                temp_array = self._get(self._varloc['0'],self._many_momsdata_keys[0]).copy()

                # now, this is NOT uniform and we really need it to be for interpolation. I will force it to be
                unique_x = temp_array[0,0,:]
                right_xcbound = np.mean(np.diff(unique_x)) * (self.moms_ngridpoints/2.) - np.mean(np.diff(unique_x))/2.
                left_xcbound = -right_xcbound

                # So unfortunately float32 is terrible, only accurate to 1e-6. This isn't strictly uniform...
                grid_values = (np.mean(np.diff(unique_x)) * np.arange(0,self.moms_ngridpoints)) + left_xcbound

                xc_array = np.ones((self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)) * grid_values

            # x contains all the info for y and z, just different order. I figured this out
            # and lets use strides so that we don't allocate new wasteful arrays (~230Mb for 1536!)
            xc_strides = xc_array.strides
            xorder = [i for i in xc_strides]

            yc_array = np.lib.stride_tricks.as_strided(xc_array,shape=(self.moms_ngridpoints,
                                                                         self.moms_ngridpoints,
                                                                         self.moms_ngridpoints),
                                                        strides=(xorder[1],xorder[2],xorder[0]))

            zc_array = np.lib.stride_tricks.as_strided(xc_array,shape=(self.moms_ngridpoints,
                                                                         self.moms_ngridpoints,
                                                                         self.moms_ngridpoints),
                                                        strides=(xorder[2],xorder[0],xorder[1]))

            # unfortunately we have to flatten these. This creates copies as
            # they are not contiguous memory chunks... (data is a portion of ghostdata)
            self._xc = np.ravel(xc_array)
            self._yc = np.ravel(yc_array)
            self._zc = np.ravel(zc_array)

            # creating a new array, radius
            self._radius = np.sqrt(np.power(self._xc,2.0) + np.power(self._yc,2.0) +\
                                    np.power(self._zc,2.0))

            # from this, I will always setup vars for a rprof
            # we need a slight offset from the lowest value and highest value of grid for interpolation!
            delta_r = 2*np.min(self._xc[np.where(np.unique(self._xc)>0)])
            eps = 0.000001
            self._radial_boundary = np.linspace(delta_r + eps*delta_r, delta_r *
                                                (self.moms_ngridpoints/2.) - eps*delta_r *
                                                (self.moms_ngridpoints/2.),
                                                int(np.ceil(self.moms_ngridpoints/2.)))

            # these are the boundaries, now I need what is my "actual" r value
            self.radial_axis = self._radial_boundary - delta_r/2.

            # # construct the bins for computing averages ON radial_axis, these are "right edges"
            delta_r = (self.radial_axis[1] - self.radial_axis[0])/2.
            radialbins = self.radial_axis + delta_r
            self.radial_bins = np.insert(radialbins,0,0)

            # in some cases, it is more convenient to work with xc[z,y,x] so lets store views
            self._xc_view = self._xc.view()
            self._xc_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)
            self._yc_view = self._yc.view()
            self._yc_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)
            self._zc_view = self._zc.view()
            self._zc_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)

            self._radius_view = self._radius.view()
            self._radius_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)

            # grab unique values along x-axis
            self._unique_coord = self._xc_view[0,0,:]

            # all is good, set that we have made our grid
            self._cgrid_exists = True

            return True

        else:
            return True

    def _get_sgrid(self):
        '''
        Constructs the PPMStar spherical coordinates grid

        Returns
        -------
        Boolean
            True on success.
            False on failure.
        '''

        # check if we already have this in memory or not
        if not self._sgrid_exists:

            # ok we are good to go for the spherical coordinates

            # we have the radius already, need theta and phi
            self._theta = np.arctan2(np.sqrt(np.power(self._xc,2.0) + np.power(self._yc,2.0)), self._zc)

            # with phi we have a problem with the way np.arctan2 works, we get negative
            # angles in quadrants 3 and 4. we can fix this by adding 2pi to the negative values
            self._phi = np.arctan2(self._yc, self._xc)
            self._phi[self._phi < 0] += 2. * np.pi

            # in some cases, it is more convenient to work with xc[z,y,x] so lets store views
            self._theta_view = self._theta.view()
            self._theta_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)
            self._phi_view = self._phi.view()
            self._phi_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)

            # ok all is good, set our flag that everything is good
            self._sgrid_exists = True

            return True

        else:
            return True

    def _get_mgrid(self):
        '''
        Constructs the PPMStar mollweide spherical coordinates grid

        Returns
        -------
        Boolean
            True on success.
            False on failure.
        '''

        # check if we already have this in memory or not
        if not self._mollweide_exists:

            # ok we now check to see if spherical coordinates has been made or not
            if not self._sgrid_exists:
                self._get_sgrid()

            # we have a transform method, let's use it
            self._mollweide_theta, self._mollweide_phi = self._transform_mollweide(self._theta.copy(),
                                                                                   self._phi.copy())

            # DS: I will save this code, it may be used in the future
            # and is a nice way of calculating this directly
            # # we have the radius already, need theta and phi
            # self._mollweide_theta = np.arctan2(self._zc,np.sqrt(np.power(self._xc,2.0) + np.power(self._yc,2.0)))

            # # with phi we have a problem with the way np.arctan2 works, we get negative
            # # angles in quadrants 3 and 4. This is what we want
            # self._mollweide_phi = np.arctan2(self._yc,self._xc)

            # in some cases, it is more convenient to work with xc[z,y,x] so lets store views
            self._mollweide_theta_view = self._mollweide_theta.view()
            self._mollweide_theta_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)
            self._mollweide_phi_view = self._mollweide_phi.view()
            self._mollweide_phi_view.shape = (self.moms_ngridpoints, self.moms_ngridpoints, self.moms_ngridpoints)

            # ok all is good, set our flag that everything is good
            self._mollweide_exists = True

            return True

        else:
            return True

    def _get_interpolation(self, var, igrid, method):
        '''
        This function controls the which method of interpolation is done and how it is done.

        Parameters
        ----------
        var: np.ndarray
            The quantity on the grid
        igrid: np.ndarray
            The array that contains all of the points that are to be interpolated to
            igrid.shape = [ninterpolation_points,3]
            igrid[:,0] = z, igrid[:,1] = y, igrid[:,2] = x
        method: str
            'trilinear': Use a trilinear method to interpolate onto the points on igrid
            'moments': Use a moments averaging within a cell and using a quadratic function
                       as the form for the interpolation

        Returns
        -------
        var_interp: np.ndarray
            The var interpolated onto the 'igrid' points
        '''

        # what method?

        # trilinear
        if method == self._interpolation_methods[0]:

            # first we create interpolation object from scipy
            linear_interp = scipy.interpolate.RegularGridInterpolator(
                (self._unique_coord, self._unique_coord, self._unique_coord), var)

            # we have a "flattened" in radii igrid, just pass all arguments to the interpolator
            var_interp = linear_interp(igrid)

            # we will exit here
            return var_interp

        # moments
        else:

            # before I begin, are there any igrid values that are on the boundary or outside of it
            upper_bound = np.max(self._unique_coord) - 1/2. * np.mean(np.abs(np.diff(self._unique_coord)))
            lower_bound = np.min(self._unique_coord) + 1/2. * np.mean(np.abs(np.diff(self._unique_coord)))

            # I will count zeros
            out_of_bounds = np.logical_or((igrid > upper_bound),(igrid < lower_bound))

            if np.count_nonzero(out_of_bounds.flatten()) > 0:
                err = 'There are {:d} grid points that are at or outside of the boundary of the simulation'\
                      .format(np.count_nonzero(out_of_bounds.flatten()))

                self._messenger.error(err)
                raise ValueError

            # first find the indices that have the closest igrid to our unique coordinates
            # store the indexes
            x_idx = np.zeros((np.shape(igrid)[0]),dtype=np.intp)
            y_idx = np.zeros((np.shape(igrid)[0]),dtype=np.intp)
            z_idx = np.zeros((np.shape(igrid)[0]),dtype=np.intp)

            # find the index of unique coord that is closest to igrid values
            x_idx = np.searchsorted(self._unique_coord,igrid[:,2])
            y_idx = np.searchsorted(self._unique_coord,igrid[:,1])
            z_idx = np.searchsorted(self._unique_coord,igrid[:,0])

            # search sorted finds index to the "right" in value from the igrid points. However, we need to find the index
            # where igrid points are closest to for appropriate interpolation. This corrects for that
            x_idx[np.where((self._unique_coord[x_idx] - igrid[:,2]) > np.mean(np.abs(np.diff(self._unique_coord)))/2.)] -= 1
            y_idx[np.where((self._unique_coord[y_idx] - igrid[:,1]) > np.mean(np.abs(np.diff(self._unique_coord)))/2.)] -= 1
            z_idx[np.where((self._unique_coord[z_idx] - igrid[:,0]) > np.mean(np.abs(np.diff(self._unique_coord)))/2.)] -= 1

            # now we call the actual interpolation
            var_interp = self._interpolation_moments(var, igrid, x_idx, y_idx, z_idx)
            return var_interp

    def _get_jacobian(self, x, y, z, r):
        '''
        This function creates the Jacobian to convert quantities defined in cartesian
        coordinates to spherical coordinates. This is a very large array of
        9 x self.moms_gridresolution which will be stored in memory. It is defined as the "physics"
        spherical coordinates so the array has rhat, theta-hat, phi-hat -> xhat, yhat, zhat

        Parameters
        ----------
        x: np.ndarray
            The x coordinates of the grid
        y: np.ndarray
            The y coordinates of the grid
        z: np.ndarray
            The z coordinates of the grid
        r: np.ndarray
            The r coordinates of the grid

        Returns
        -------
        jacobian: np.ndarray
            The jacobian for the transformation between cartesian and spherical coordinates
        '''

        # are we working with a flattened, (x,y,z) or a matrix?
        if len(x.shape) > 1:

            # since we work in spherical coordinates, the phi-hat dot z-hat component is zero, so it is 8x(nxnxn)
            jacobian = np.zeros((8, x.shape[0], y.shape[0], z.shape[0]),dtype='float32')
        else:

            # since we work in spherical coordinates, the phi-hat dot z-hat component is zero, so it is 8x(n)
            jacobian = np.zeros((8, x.shape[0]))

        # need the cylindrical radius
        rcyl = np.sqrt(np.power(x,2.0) + np.power(y,2.0))

        # rhat -> xhat, yhat, zhat
        np.divide(x,r,out=jacobian[0])
        np.divide(y,r,out=jacobian[1])
        np.divide(z,r,out=jacobian[2])

        # theta-hat -> xhat, yhat, zhat
        # we use "placeholders" of jacobian slots to not make new memory
        np.divide(np.multiply(x, z, out=jacobian[3]),  np.multiply(r, rcyl, out=jacobian[4]),
                  out = jacobian[3])
        np.divide(np.multiply(y, z, out=jacobian[4]), np.multiply(r, rcyl, out=jacobian[5]),
                  out = jacobian[4])
        np.divide(-rcyl, r, out=jacobian[5])

        # phi-hat -> xhat, yhat, zhat
        np.divide(-y, rcyl, out=jacobian[6])
        np.divide(x, rcyl, out=jacobian[7])
        # phi-hat dot z-hat = 0

        # jacobian transformation matrix has been computed
        return jacobian

    def _get(self, varloc, fname=None):
        '''
        Returns the variable var which is referenced with varloc at a specific dump/time in the
        simulation. This is used internally for var claims that will be references that are garbage
        collected in a method. IMPORTANT: The arrays are NOT flattened but if they need
        to be a NEW array will be made

        Parameters
        ----------
        varloc: str, int
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
        fname: None,int
            None: default option, will grab current dump
            int: Dump number

        Returns
        -------
        var: np.ndarray
            Variable on the grid
        '''

        # if fname is None, use current dump
        if fname == None:
            fname = self.what_dump_am_i

        # quick check if we already have the momsdata in memory
        if str(fname) in self._many_momsdata:

            try:
                return self._many_momsdata[str(fname)].get(self._varloc[str(varloc)])

            except KeyError as e:
                err = 'Invalid key for varloc. A list of keys: \n'
                err += ', '.join(sorted(map(str,self._varloc.keys())))
                self._messenger.error(err)
                raise e

        else:

            # grab a new datacube. This updates self._momsdata.data
            self._get_dump(fname)

            try:
                return self._many_momsdata[str(fname)].get(self._varloc[str(varloc)])

            except KeyError as e:
                err = 'Invalid key for varloc. A list of keys: \n'
                err += ', '.join(sorted(map(str,self._varloc.keys())))
                self._messenger.error(err)
                raise e

    # def get_ray_interpolation(self, radius, theta, phi, nrays):
    #     """

    #     """

    def get_dump_list(self):
        '''
        Returns a list of dumps available.

        Returns
        -------
        dumps: list
            List of dumps that are available
        '''

        return list(self._dumps)

    def get_interpolation(self, varloc, igrid, fname=None, method='trilinear', logvar=False):
        '''
        Returns the interpolated array of values (with a particular method) of the var given by
        'varloc' at the [z,y,x] grid points of igrid

        Parameters
        ----------
        varloc: str, int, np.ndarray
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
            np.ndarray: quantity you want to have interpolated on the grid
        igrid: np.ndarray
            The array that contains all of the points that are to be interpolated to
            igrid.shape = [ninterpolation_points,3]
            igrid[:,0] = z, igrid[:,1] = y, igrid[:,2] = x
        fname: None, int
            None: default option, will grab current dump
            int: Dump number
        method: str
            'trilinear': Use a trilinear method to interpolate onto the points on igrid
            'moments': Use a moments averaging within a cell and using a quadratic function
                       as the form for the interpolation
        logvar: bool
            For better fitting should I do var = np.log10(var)? The returned var_interpolated
            will be scaled back to linear

        Returns
        -------
        var_interp: np.ndarray
            The var interpolated onto the 'igrid' points
        '''

        # first check if we have a np.ndarray or not
        if isinstance(varloc, np.ndarray):

            # for consistency of naming..
            var = varloc

            # check if it is the same shape as self._xc_view
            if var.shape != self._xc_view.shape:

                # we can try reshaping
                try:
                    var.reshape(self._xc_view.shape)
                except ValueError as e:
                    err = 'The varloc given cannot be reshaped into ' + str(self._xc_view.shape)
                    self._messenger.error(err)
                    raise e

        else:

            # varloc is a reference for a get method
            var = self._get(varloc, fname)

        # varloc is good, are we applying log10 to it?
        if logvar:
            var = np.log10(var.copy())

        # make sure that our method string is actually a real method
        if not list(filter(lambda x: method in x, self._interpolation_methods)):
            err = 'The inputted method, '+method+' is not any of the known methods, '
            err_add = ', '.join(self._interpolation_methods)
            err += err_add
            self._messenger.error(err)
            raise ValueError

        # make sure that igrid is the correct shape
        if len(igrid.shape) != 2 or igrid.shape[1] != 3:
            err = 'The igrid is not the correct shape. It must be [npoints,3] but it is {0}'\
                  .format(igrid.shape)
            self._messenger.error(err)
            raise ValueError

        # Now all of the hard work is done in other methods for the interpolation
        var_interp = self._get_interpolation(var, igrid, method)

        # did we log it?
        if logvar:
            return np.power(10., var_interp)
        else:
            return var_interp

    def get_rprof(self, varloc, radius=None, fname=None, method='trilinear', logvar=False):
        '''
        Compute a 1d radial profile of variable, var, given by 'varloc'

        Parameters
        ----------
        varloc: str, int, np.ndarray
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
            np.ndarray: quantity you want to have interpolated on the grid
        radius: None, np.ndarray
             None: default option, will use the internal radial_axis (every cell width)
             np.ndarray: array for the radial values to calculate the profile on
        fname: None, int
            None: default option, will grab current dump
            int: Dump number
        method: str
            'trilinear': Use a trilinear method to interpolate onto the points on igrid
            'moments': Use a moments averaging within a cell and using a quadratic function
                       as the form for the interpolation
        logvar: bool
            For better fitting should I do var = np.log10(var)? The returned rprof will be scaled
            back to linear

        Returns
        -------
        rad_prof: np.ndarray
            Spherical average profile of the var referenced by 'varloc'
        radial_axis: np.ndarray
            The radial coordinates in which var was averaged on
        '''

        # are we using self.radial_axis?
        if isinstance(radius, np.ndarray):

            # make sure nothing is too large
            if np.max(radius) > (np.max(self.radial_axis) + np.mean(abs(np.diff(self.radial_axis)))):
                err = 'The input radius has a radius, {0:0.2f}, which is outside of the simulation box {1:0.2f}'\
                      .format(np.max(radius),np.max(self.radial_axis))
                self._messenger.error(err)
                raise ValueError

            # we basically just call interpolation over radius, trilinear is default
            # now, if we do have a derivative, quantity is a list
            quantity = self.get_spherical_interpolation(varloc, radius, fname, method=method, logvar=logvar)

            # for an rprof we average all of those quantities at each radius
            quantity = np.mean(quantity, axis=1)

        else:
            # we basically just call interpolation over self.radius, trilinear is default
            # now, if we do have a derivative, quantity is a list
            quantity = self.get_spherical_interpolation(varloc, self.radial_axis, fname, method=method, logvar=logvar)

            # for an rprof we average all of those quantities at each radius
            quantity = np.mean(quantity,axis=1)

        if isinstance(radius, np.ndarray):
            return quantity, radius
        else:
            return quantity, self.radial_axis

        # DS: We will hold onto the old method, it uses binning which is not a bad way of doing it
        # # check if we have array or not
        # if type(varloc) == np.ndarray:
        #     quantity = np.ravel(varloc)
        # else:
        #     # get the grid from a momsdata cube
        #     # because we need a flattened array, ravel will create a new array
        #     quantity = np.ravel(self._get(varloc,fname))

        # # This will apply a "mean" to the quantity that is binned by radialbins
        # # using the self._radius values
        # average_quantity, bin_edge, binnumber = scipy.stats.binned_statistic(self._radius,quantity,'mean',self.radial_bins)

        # # return the radprof and radial_axis
        # return average_quantity, self.radial_axis

    def get_cgrid(self):
        '''
        Returns the central values of the grid for x, y and z of the moments data cube currently
        held in memory. This is the cartesian grid and it is formatted as xc[z,y,x]

        IMPORTANT NOTE: This is a copy of the actual data. This is to ensure that a dump's data
        will be deleted when new data is deleted i.e we are preserving that there will be no
        references!

        Returns
        -------
        xc: np.ndarray
            The x-coordinate at the center of all cells in the grid
        yc: np.ndarray
            The y-coordinate at the center of all cells in the grid
        zc: np.ndarray
            The z-coordinate at the center of all cells in the grid
        '''

        # we use these internally, so we give copies
        return self._xc_view.copy(), self._yc_view.copy(), self._zc_view.copy()

    def get_sgrid(self):
        '''
        Returns the central values of the grid for r, theta and phi of the moments data cube
        currently held in memory. The "physics" definition of the theta and phi angles are:

        - Theta runs from 0 -> pi while going down from the positive z-axis
        - Phi runs from 0 -> 2pi for the angle around the positive x-axis

        IMPORTANT NOTE: This is a copy of the actual data. This is to ensure that a dump's data
        will be deleted when new data is deleted i.e we are preserving that there will be no
        references!

        Returns
        -------
        r: np.ndarray
            The r-coordinate at the center of all cells in the grid
        theta: np.ndarray
            The theta-coordinate at the center of all cells in the grid
        phi: np.ndarray
            The phi-coordinate at the center of all cells in the grid
        '''

        # does this exist yet?
        if not self._sgrid_exists:
            self._get_sgrid()

        # these are not used internally and so we can give them the real grid (except for radius!)
        return self._radius_view.copy(), self._theta_view.copy(), self._phi_view.copy()

    def get_mgrid(self):
        '''
        Returns the central values of the grid for r, theta and phi of the moments data cube
        currently held in memory. The "mollweide" definition of the theta and phi angles are:

        - Theta runs from pi/2 -> -pi/2 going down from the positive z-axis
        - Phi goes from 0 -> pi from quadrants 1->2 and then 0 -> -pi from quadrants 4->3.

        IMPORTANT NOTE: This is a copy of the actual data. This is to ensure that a dump's data
        will be deleted when new data is deleted i.e we are preserving that there will be no
        references!

        Returns
        -------
        r: np.ndarray
            The r-coordinate at the center of all cells in the grid
        theta: np.ndarray
            The theta-coordinate at the center of all cells in the grid
        phi: np.ndarray
            The phi-coordinate at the center of all cells in the grid
        '''

        # DOES this exist yet?
        if not self._mollweide_exists:
            self._get_mgrid()

        # these are not used internally and so we can give them the real grid
        return self._radius_view.copy(), self._mollweide_theta_view.copy(), self._mollweide_phi_view.copy()

    def get_mollweide_coordinates(self, theta, phi):
        '''
        Transforms a "physics" spherical coordinates array into the "mollweide" spherical
        coordinates that matplotlib uses for projection plots.

        Parameters
        ----------
        theta: np.ndarray
            An array of theta with the "physics" defined angle to be converted
        phi: np.ndarray
            An array of phi with the "physics" defined angle to be converted

        Returns
        -------
        theta: np.ndarray
            theta transformed to the "mollweide" defined angle
        phi: np.ndarray
            phi transformed to the "mollweide" defined angle
        '''

        # am I converting?
        if isinstance(theta,np.ndarray) and isinstance(phi,np.ndarray):

            # I want to convert this array
            theta_copy, phi_copy = self._transform_mollweide(theta,phi)

            return theta_copy, phi_copy

        else:
            return None

    def get_spherical_coordinates(self, theta, phi):
        '''
        Transforms a "mollweide" spherical coordinates array into the "physics" spherical
        coordinates.

        Parameters
        ----------
        theta: np.ndarray
            An array of theta with the "mollweide" defined angle to be converted
        phi: np.ndarray
            An array of phi with the "mollweide" defined angle to be converted

        Returns
        -------
        theta: np.ndarray
            theta transformed to the "physics" defined angle
        phi: np.ndarray
            phi transformed to the "physics" defined angle
        '''

        # am I converting?
        if isinstance(theta,np.ndarray) and isinstance(phi,np.ndarray):

            # I want to convert this array
            theta_copy, phi_copy = self._transform_spherical(theta,phi)

            return theta_copy, phi_copy

        else:
            return None

    def get_spherical_interpolation(self, varloc, radius, fname=None, npoints=5000, method='trilinear', logvar=False,
                                    plot_mollweide=False, get_igrid=False):
        '''
        Returns the interpolated array of values of 'varloc' at a radius of 'radius' for a computed uniform
        distribution of points, 'npoints', on that sphere(s). It can return the 'theta,phi' (mollweide)
        coordinates of the 'varloc' values as well.

        Parameters
        ----------
        varloc: str, int, np.ndarray
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
            np.ndarray: quantity you want to have interpolated on the sphere
        radius: float or np.ndarray
            The radius of the sphere you want 'varloc' to be interpolated to
        fname: None, int
            None: default option, will grab current dump
            int: Dump number
        npoints: int
            The number of 'theta and phi' points you want for a projection plot
        method: str
            'trilinear': Use a trilinear method to interpolate onto the points on igrid
            'moments': Use a moments averaging within a cell and using a quadratic function
                       as the form for the interpolation
        logvar: bool
            For better fitting should I do var = np.log10(var)? The returned var_interpolated
            will be scaled back to linear
        plot_mollweide: bool
            If you want returned the theta, phi coordinates of the interpolated values so that it
            can be plotted with a projection method
        get_igrid: bool
            If you want returned the actual grid points (x,y,z) used for the interpolation

        Returns
        -------
        var_interpolated: np.ndarray
            The array containing var interpolated at 'radius'
        theta_grid: np.ndarray
            The array containing the "mollweide" theta points that were interpolated to
        phi_grid: np.ndarray
            The array containing the "mollweide" phi points that were interpolated to
        igrid: np.ndarray
            The array containing the x,y,z coordinates of the points that were interpolated to
        '''

        # I will construct an appropriate igrid and let get_interpolation do the rest

        # do we have many radii?
        try:
            first_r = radius[0]
        except (TypeError, IndexError) as e:
            # ok, we have an error, it is a single float or int
            radius = np.array([radius])

        # get the grid to be interpolated to
        igrid, theta_grid, phi_grid = self._constantArea_spherical_grid(radius, npoints)

        # for mollweide we have to transform these
        theta_grid, phi_grid = self._transform_mollweide(theta_grid, phi_grid)

        # More checks will be done with get_interpolation
        var_interp = self.get_interpolation(varloc, igrid, fname, method, logvar)

        # This var_interp and igrid COULD be a flattened array, let's reshape if so
        if len(radius) > 1:

            # to prevent copying array
            igrid.shape = (len(radius),npoints,3)
            var_interp.shape = (len(radius),npoints)

        # Are we plotting mollweide, and/or returning igrid?
        if get_igrid:
            if plot_mollweide:
                return var_interp, theta_grid, phi_grid, igrid
            else:
                return var_interp, igrid
        else:
            if plot_mollweide:
                return var_interp, theta_grid, phi_grid
            else:
                return var_interp

    def get_spherical_components(self, ux, uy, uz, fname=None, igrid=None):
        '''
        Vector quantities are output in the cartesian coordinates but we can transform them to
        spherical coordinates using unit vectors. This returns the spherical components of u

        Parameters
        ----------
        ux, uy, uz: int, str, np.ndarray
            int: integer referring to varloc
            str: string referring to quantity varloc
            np.ndarray: array with quantities
        fname: None, int
            None: default option, will grab current dump
            int: Dump number
        igrid: np.ndarray
            If the quantity is not defined on the entire grid we can still convert it if we know the
            cartesian points that it is on. Note:

            igrid.shape = (len(ux.flatten()),3)
            igrid[:,0] = z, igrid[:,1] = y, igrid[:,2] = x

        Returns
        -------
        u_spherical: list of np.ndarray
            The spherical components of u
        '''

        # first check if we are using the grid or not
        if not isinstance(igrid,np.ndarray):
            if not self._grid_jacobian_exists:

                # create the grid jacobian, keep this in memory
                self._grid_jacobian = self._get_jacobian(self._xc_view,self._yc_view,self._zc_view, self._radius_view)
                self._grid_jacobian_exists = True

            # local variable to reference internal jacobian
            jacobian = self._grid_jacobian
        else:
            radius = np.sqrt(np.power(igrid[:,2],2.0) + np.power(igrid[:,1],2.0) + np.power(igrid[:,0],2.0))
            jacobian = self._get_jacobian(igrid[:,2],igrid[:,1],igrid[:,0],radius)

        # first grab quantities if we need to
        if not isinstance(ux,np.ndarray):
            ux = self._get(ux,fname)
        if not isinstance(uy,np.ndarray):
            uy = self._get(uy,fname)
        if not isinstance(uz,np.ndarray):
            uz = self._get(uz,fname)

        ur = ux * jacobian[0] + uy * jacobian[1] + uz * jacobian[2]
        utheta = ux * jacobian[3] + uy * jacobian[4] + uz * jacobian[5]
        uphi = ux * jacobian[6] + uy * jacobian[7]

        return [ur, utheta, uphi]

    def get(self, varloc, fname=None):
        '''
        Returns variable var at a specific point in the simulation's time evolution. var is
        referenced from 'varloc' which can be a string (referring to var's name that you
        specified and instantiation) or an integer referring to whatever[varloc]

        IMPORTANT NOTE: This is a copy of the actual data. This is to ensure that a dump's data
        will be deleted when new data is deleted i.e we are preserving that there will be no
        references!

        Parameters
        ----------
        varloc: str, int
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
        fname: None,int
            None: default option, will grab current dump
            int: Dump number

        Returns
        -------
        var: np.ndarray
            Variable referenced with 'varloc' as given by MomsData.get() if the MomsData
            corresponding to 'fname' exists.
        '''

        # if fname is not specified use current dump
        if fname == None:
            fname = self.what_dump_am_i

        # quick check if we already have the momsdata in memory
        if str(fname) in self._many_momsdata:

            # This is public, we must give a copy
            # let's try this, if we get key error then obviously...
            try:
                return self._many_momsdata[str(fname)].get(self._varloc[str(varloc)]).copy()

            except KeyError as e:
                err = 'Invalid key for varloc. A list of keys: \n'
                err += ', '.join(sorted(map(str,self._varloc.keys())))
                self._messenger.error(err)
                raise e

        else:

            # grab a new datacube. If we don't have this in memory already (self._many_momsdata) we grab a new datacube
            self._get_dump(fname)

            # This is public, we must give a copy
            # let's try this, if we get key error then obviously...
            try:
                return self._many_momsdata[str(fname)].get(self._varloc[str(varloc)]).copy()

            except KeyError as e:
                err = 'Invalid key for varloc. A list of keys: \n'
                err += ', '.join(sorted(map(str,self._varloc.keys())))
                self._messenger.error(err)
                raise e

    def gradient(self, f, fname=None):
        '''
        Take the gradient of a scalar field in CARTESIAN coordinates. This uses central
        differences using points directly on the grid (no interpolation).

        Parameters
        ----------
        f: np.ndarray
            scalar field defined on the grid
        fname: None,int
            None: default option, will grab current dump
            int: Dump number

        Returns
        -------
        grad_f: list of np.ndarray
            list containing fx, fy and fz
        '''

        if not isinstance(f,np.ndarray):
            f = self._get(f,fname)
        else:
            # check len of shape of f
            if len(f.shape) != 3:
                err = 'The input f does not have its data formatted as f[z,y,x], make sure the shape is ({:0},{:0},{:0})'\
                .format(self.moms_ngridpoints)
                self._messenger.error(err)
                raise ValueError

        # we use the unique coordinates as the values on the grid (these should have had uniform spacing but don't...)
        gradf = np.gradient(f,self._unique_coord,self._unique_coord,self._unique_coord)

        # we get fz, fy and then fx, rearrange
        gradf.reverse()

        return gradf

    def sphericalHarmonics_format(self, varloc, radius, fname=None, lmax=None, method='trilinear',
                                  get_theta_phi_grids=False, get_igrid=False):
        '''
        To describe a function on a sphere it can be decomposed into its modes through spherical
        harmonics. To do this efficiently, a particular theta, phi grid is used which is different
        from what the self.get_spherical_interpolation uses. Keep in mind that for a given lmax:

        The number of theta subdivisions across its domain is N = 2*(l+1)
        The number of phi subdivisions across its domain is 2*N = 4*(l+1)
        The number of points being interpolated to is npoints = 8*(l+1)**2

        Parameters
        ----------
        varloc: str, int, np.ndarray
            String: for the variable you want if defined on instantiation
            Int: index location of the variable you want
            np.ndarray: quantity you want to have interpolated on the sphere
        radius: float
            The radius of the sphere you want 'varloc' to be interpolated to
        fname: None, int
            None: default option, will grab current dump
            int: Dump number
        lmax: None, int
            None: default option will use the maximum resolvable l for this 'radius' and moments
                  data grid size, i.e lmax = pi * radius / self.moms_gridresolution
            int: The maximum l value that you wish to use
        method: str
            'trilinear': Use a trilinear method to interpolate onto the points on igrid
            'moments': Use a moments averaging within a cell and using a quadratic function
                       as the form for the interpolation
        get_theta_phi_grids: bool
            If you want returned the theta, phi coordinates of the interpolated values ("physics")
        get_igrid: bool
            If you want returned the actual grid points (x,y,z) used for the interpolation

        Returns
        -------
        var_interpolated: np.ndarray
            The array containing var interpolated at 'radius'. Note that its shape is (N, 2*N)
            to conform with pyshtools
        theta_grid: np.ndarray
            The array containing the "physics" theta points that were interpolated to
        phi_grid: np.ndarray
            The array containing the "physics" phi points that were interpolated to
        igrid: np.ndarray
            The array containing the x,y,z coordinates of the points that were interpolated to
        '''

        # first get our lmax
        if lmax is None:

            # I will calculate it
            lmax, N, npoints = self.sphericalHarmonics_lmax(radius)

            # radius must be an array for creating an igrid
            radius = np.array([radius])

        else:

            # This is from a user, ensure it is an integer
            lmax = int(lmax)
            N = int(2 * (lmax + 1))
            npoints = int(N * 2*N)

            # radius must be an array for creating an igrid
            radius = np.array([radius])

        # Now I need to get my grid to interpolate
        igrid, theta_grid, phi_grid = self._sphericalHarmonics_grid(radius , N)

        # interpolate our quantity
        var_interp = self.get_interpolation(varloc, igrid, fname, method)

        # reshape var_interp to what is needed
        var_interp.shape = (N, 2*N)

        # return everything that is wanted
        if get_igrid:
            if get_theta_phi_grids:
                return var_interp, theta_grid, phi_grid, igrid
            else:
                return var_interp, igrid
        else:
            if get_theta_phi_grids:
                return var_interp, theta_grid, phi_grid
            else:
                return var_interp

    def sphericalHarmonics_lmax(self, radius):
        """
        Calculate the maximum l (minimum angular scale) that can be resolved with our moments data
        resolution. This is simply using the nyquist sampling theorem:

        lambda = 2 * pi * r / sqrt(l*(l+1))
        lambda_min = 2 * dx

        With large l, sqrt(l*(l+1)) ~ l
        lmax ~ pi * r / dx

        -------------------------------------------------------------------------------------------
        Keep in mind that for a given lmax:

        The number of theta subdivisions across its domain is N = 2*(l+1)
        The number of phi subdivisions across its domain is 2*N = 4*(l+1)
        The number of points being interpolated to is npoints = 8*(l+1)**2

        Parameters
        ----------
        radius: float
            The radius that you want to know the maximum l resolvable

        Returns
        -------
        lmax: int
            The maximum l resolvable
        N: int
            The number of subdivisions of theta between np.pi and 0
        npoints: int
            The number of points to be interpolated at "radius"
        """

        # using approximation above
        lmax = int(np.pi * radius / self.moms_gridresolution)
        N = int(2 * (lmax + 1))
        npoints = int(8 * (lmax + 1)**2)

        return lmax, N, npoints

    def norm(self, ux, uy, uz, fname=None):
        '''
        Norm of some vector quantity. It is written as ux, uy, uz which will give |u| through
        the definition |u| = sqrt(ux**2 + uy**2 + uz**2). The vector must be defined in an
        orthogonal basis. i.e we can also do |u| = sqrt(ur**2 + uphi**2 + utheta**2)

        Parameters
        ----------
        ux, uy, uz: int, str, np.ndarray
            int: integer referring to varloc
            str: string referring to quantity varloc
            np.ndarray: array with vector quantities
        fname: None,int
            None: default option, will grab current dump
            int: Dump number

        Returns
        -------
        |u|: np.ndarray
            The norm of vector u
        '''

        if not isinstance(ux,np.ndarray):
            ux = self._get(ux,fname)
        if not isinstance(uy,np.ndarray):
            uy = self._get(uy,fname)
        if not isinstance(uz,np.ndarray):
            uz = self._get(uz,fname)

        return np.sqrt(np.power(ux,2.0)+np.power(uy,2.0)+np.power(uz,2.0))

    def build_cmap(self, colours, ranges, num_colours, vmin, vmax):
        diff = vmax - vmin
        reduced_colours = list(colours[0:num_colours])
        reduced_ranges = list(ranges[0:num_colours])
        cmap_rgba = [matplotlib.colors.to_rgba(colour) for colour in reduced_colours]
        cmap = list(zip(reduced_ranges, cmap_rgba))
        sorted_cmap = sorted(cmap, key = lambda tup: tup[0])
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', sorted_cmap)
        pl.register_cmap('custom_cmap', colormap)

        return colormap

    def get_indexed_quantities(self):
        return [('x', 0), ('u_x', 1), ('u_y', 2), ('u_z', 3), ('|u_t|', 4), ('u_r', 5), ('|w|', 6), ('P', 7), ('rho' , 8), ('fv' , 9)]

    def get_colourpicker(self, active_colours):
        '''
            colorpicker gui for slice and mollweide_gui
        '''
        colour0 = widgets.ColorPicker(concise=False, value='blue', disabled=False, layout=widgets.Layout(width='150px'))
        colour1 = widgets.ColorPicker(concise=False, value='white', disabled=False, layout=widgets.Layout(width='150px'))
        colour2 = widgets.ColorPicker(concise=False, value='red', disabled=False, layout=widgets.Layout(width='150px'))
        colour3 = widgets.ColorPicker(concise=False, value='yellow', disabled=True, layout=widgets.Layout(width='150px'))
        colour4 = widgets.ColorPicker(concise=False, value='green', disabled=True, layout=widgets.Layout(width='150px'))
        colour5 = widgets.ColorPicker(concise=False, value='gray', disabled=True, layout=widgets.Layout(width='150px'))
        colour6 = widgets.ColorPicker(concise=False, value='lime', disabled=True, layout=widgets.Layout(width='150px'))
        colour7 = widgets.ColorPicker(concise=False, value='cyan', disabled=True, layout=widgets.Layout(width='150px'))
        colour8 = widgets.ColorPicker(concise=False, value='olive', disabled=True, layout=widgets.Layout(width='150px'))
        colour9 = widgets.ColorPicker(concise=False, value='teal', disabled=True, layout=widgets.Layout(width='150px'))
        colour10 = widgets.ColorPicker(concise=False, value='maroon', disabled=True, layout=widgets.Layout(width='150px'))
        colour11 = widgets.ColorPicker(concise=False, value='silver', disabled=True, layout=widgets.Layout(width='150px'))

        range0 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.00, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range1 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.50, disabled=False, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range2 = widgets.FloatSlider(min=0., max=1., step=0.01, value=1.00, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range3 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.60, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range4 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.80, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range5 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.20, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range6 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.10, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range7 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.30, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range8 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.40, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range9 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.75, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range10 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.90, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))
        range11 = widgets.FloatSlider(min=0., max=1., step=0.01, value=1.00, disabled=True, readout_format='.2f', continuous_update=False, \
            layout=widgets.Layout(width='200px'))

        def on_colour_change(change):
            if change['name'] == 'value' and (change['new'] != change['old']):
                ranges = [range0, range1, range2, range3, range4, range5, range6, range7, range8, range9, range10, range11]
                colours = [colour0, colour1, colour2, colour3, colour4, colour5, colour6, colour7, colour8, colour9, colour10, colour11]
                for i in range(0, len(ranges)):
                    if i < change['new']:
                        ranges[i].disabled = False
                        colours[i].disabled = False
                    else:
                        ranges[i].disabled = True
                        colours[i].disabled = True
                ranges[change['new']-1].value = 1.0
                ranges[change['new']-1].disabled = True
        colour_select = widgets.Dropdown(options=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], value=3, description='Colours', disabled=False, layout=widgets.Layout(width='150px'))
        colour_select.observe(on_colour_change)
        if active_colours != 3:
            range2.value = 0.7
        colour_select.value = active_colours

        combo0 = widgets.HBox([colour0, range0])
        combo1 = widgets.HBox([colour1, range1])
        combo2 = widgets.HBox([colour2, range2])
        combo3 = widgets.HBox([colour3, range3])
        combo4 = widgets.HBox([colour4, range4])
        combo5 = widgets.HBox([colour5, range5])
        combo6 = widgets.HBox([colour6, range6])
        combo7 = widgets.HBox([colour7, range7])
        combo8 = widgets.HBox([colour8, range8])
        combo9 = widgets.HBox([colour9, range9])
        combo10 = widgets.HBox([colour10, range10])
        combo11 = widgets.HBox([colour11, range11])

        colour_vbox1 = widgets.VBox([colour_select])
        colour_vbox2 = widgets.VBox([combo0, combo1, combo2, combo3])
        colour_vbox3 = widgets.VBox([combo4, combo5, combo6, combo7])
        colour_vbox4 = widgets.VBox([combo8, combo9, combo10, combo11])
        colourPicker = widgets.HBox([colour_vbox1, colour_vbox2, colour_vbox3, colour_vbox4])

        return [colour0, colour1, colour2, colour3, colour4, colour5, colour6, colour7, colour8, colour9, colour10, colour11], \
            [range0, range1, range2, range3, range4, range5, range6, range7, range8, range9, range10, range11], colour_select, colourPicker

    def slice_plot(self, dump, quantity, direction, vmin, vmax, log, slice_index, colours, ranges, num_colours, size, ifig, interpolation):
        cmap = self.build_cmap(colours, ranges, num_colours, vmin, vmax)
        values = self.get(quantity, dump)
        x, y, z = self.get_cgrid()
        indexed_quantities = self.get_indexed_quantities()

        if direction == 'x':
            trimmed_vals = values[slice_index, :, :]
            extent = [np.min(y),np.max(y),np.min(z),np.max(z)]
        elif direction == 'y':
            trimmed_vals = values[:, slice_index, :]
            extent = [np.min(x),np.max(x),np.min(z),np.max(z)]
        else:
            trimmed_vals = values[:, :, slice_index]
            extent = [np.min(x),np.max(x),np.min(y),np.max(y)]
        pl.close(ifig)
        fig = pl.figure(ifig)
        fig.canvas.layout.height = str(0.9*size)+'in'   # This is a hack to prevent ipympl
        fig.canvas.layout.width  = str(1.1*size)+'in'   # to adjust horizontal figure size

        if self.slice_gui_range_update == True and log == False:
            vmin = np.amin(trimmed_vals)
            vmax = np.amax(trimmed_vals)

        if log == True:
            log_min = np.abs(np.log10(vmin))
            for i in range(len(trimmed_vals[:][0])-1):
                row = trimmed_vals[i][:]
                rowBool = row > 0
                for index, val in enumerate(row):
                    if rowBool[index] == True:
                        if vmin <= val and val <= vmax:
                            row[index] = np.log10(val) + log_min
                        else:
                            row[index] = 0
                    else:
                        if val != 0:
                            if -vmax <= val and val <= -vmin:
                                row[index] = -np.log10(-val) - log_min
                            else:
                                row[index] = 0
                trimmed_vals[i][:] = row
            pl.imshow(trimmed_vals, extent=extent, vmin=np.amin(trimmed_vals), vmax=np.amax(trimmed_vals), cmap=cmap, interpolation=interpolation)
        else:
            pl.imshow(trimmed_vals, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)

        pl.colorbar()
        return [vmin, vmax]

    def slice_gui(self, size=8, ifig=123, interpolation='kaiser'):
        dump_min, dump_max = self.get_dump_list()[0], self.get_dump_list()[-1]
        dump_mean = int(2*(-dump_min + dump_max)/3.)
        self.slice_gui_range_update = True

        # left-most widgets in graph settings
        dump = widgets.IntSlider(min=dump_min, max=dump_max, step=1, value=dump_mean, description='Dump', disabled=False, layout=widgets.Layout(width='335px'))
        slice_index = widgets.IntSlider(min=0, max=191, step=1, value=45, description='Slice', disabled=False, layout=widgets.Layout(width='335px'))
        quantity = widgets.Dropdown(options=self.get_indexed_quantities(), value=4, description="Quantity", disabled=False)
        direction = widgets.RadioButtons(options=['x', 'y', 'z'], description='Slice plane', disabled=False, layout= widgets.Layout(display='flex', flex_flow='row'))

        left_widgets = widgets.VBox([dump, slice_index, quantity, direction], layout=widgets.Layout(width="30%"))

        # center widgets in graph settings
        value_label = widgets.Label(value="Value Range to Display:")
        value_min = widgets.BoundedFloatText(value=-85., min=-1000., max=84.9999999, step=1e-7, readout_format=".7f", description="Min Value", layout=widgets.Layout(width='335px'))
        value_max = widgets.BoundedFloatText(value=85, min=-84.9999999, max=1000., step=1e-7, readout_format=".7f", description="Max Value", layout=widgets.Layout(width='335px'))

        log_label = widgets.Label(value="Log Value Range to Display (Applied symmetrically):")
        log_min = widgets.BoundedFloatText(value=1e-5, min=1e-7, max=9.99999e-2, step=1e-7, readout_format=".7f", description="Log Min", layout=widgets.Layout(width='335px'))
        log_max = widgets.BoundedFloatText(value=1e-1, min=1.0001e-5, max=2e4, step=1e-7, readout_format=".7f", description="Log Max", layout=widgets.Layout(width='335px'))

        value_range = widgets.VBox([value_label, value_min, value_max], layout=widgets.Layout(width='100%'))
        value_range.layout.display = 'none' # hide these controls intially as log == True on init
        log_range = widgets.VBox([log_label, log_min, log_max], layout=widgets.Layout(width='100%'))

        log = widgets.Checkbox(value=True, description='Log Values', disabled=False, indent=False)

        center_widgets = widgets.VBox([value_range, log_range, log], layout=widgets.Layout(width='30%'));

        # right-most widgets in graphs settings
        save_load_label = widgets.Label(value="Save/Load Configuration (.pickle):")
        save_filename = widgets.Text(value='', placeholder='Filename (w/out extension)', disabled=False, layout=widgets.Layout(width='200px'))
        save_button = widgets.Button(description='Save Config', tooltip='Save Config', disabled=False)
        save_combo = widgets.HBox([save_button, save_filename], layout=widgets.Layout(margin="0px 0px 0px 20px"))

        load_filename = widgets.Text(value='', placeholder='Filename (w/out extension)', disabled=False, layout=widgets.Layout(width='200px'))
        load_button = widgets.Button(description='Load Config', tooltip='Load Config', disabled=False)
        load_combo = widgets.HBox([load_button, load_filename], layout=widgets.Layout(margin="0px 0px 0px 20px"))

        right_widgets = widgets.VBox([save_load_label, save_combo, load_combo], layout=widgets.Layout(width='30%'))

        colours, ranges, colour_select, colourpicker = self.get_colourpicker(6)
        graph_settings = widgets.HBox([left_widgets, center_widgets, right_widgets])

        ui = widgets.Tab(children=[graph_settings, colourpicker])
        ui.set_title(0, "Graph Settings")
        ui.set_title(1, "Colour Picker")

        def plot(dump, quantity, direction, log, slice_index, vmin, vmax, log_min, log_max, colour0, colour1, colour2, colour3, colour4, colour5, colour6, \
            colour7, colour8, colour9, colour10, colour11, range0, range1, range2, range3, range4, range5, range6, range7, range8, range9, range10, range11, colour_select):
            if log == True:
                self.slice_plot(dump, quantity, direction, log_min, log_max, log, slice_index, [colour0, colour1, colour2, colour3, colour4, colour5, colour6, colour7, \
                    colour8, colour9, colour10, colour11], [range0, range1, range2, range3, range4, range5, range6, range7, range8, range9, range10, range11], colour_select, \
                    size, ifig, interpolation)
            else:
                max_min = self.slice_plot(dump, quantity, direction, vmin, vmax, log, slice_index, [colour0, colour1, colour2, colour3, colour4, colour5, colour6, \
                    colour7, colour8, colour9, colour10, colour11], [range0, range1, range2, range3, range4, range5, range6, range7, range8, range9, range10, range11], \
                    colour_select, size, ifig, interpolation)
                if self.slice_gui_range_update == True:
                    value_min.value = max_min[0]
                    value_max.value = max_min[1]
                    self.slice_gui_range_update = False

        def on_click_save(b):
            pickle_info = {
                'dump': dump.value,
                'slice_index': slice_index.value,
                'value_min': value_min.value,
                'value_max': value_max.value,
                'log_min': log_min.value,
                'log_max': log_max.value,
                'quantity': quantity.value,
                'direction': direction.value,
                'log': log.value,
                'colour_select': colour_select.value
            }
            pickle_info['colours'] = []
            pickle_info['ranges'] = []
            for index in range(0, 12):
                pickle_info['colours'].append((colours[index].value, colours[index].disabled))
                pickle_info['ranges'].append((ranges[index].value, ranges[index].disabled))
            try:
                if save_filename.value != '':
                    f = open('%s.pickle' % save_filename.value, 'wb')
                else:
                    f = open('%s.pickle' % date.today(), 'wb')
                pickle.dump(pickle_info, f)
                f.close()
            except:
                print('Failed to save file')
        save_button.on_click(on_click_save)

        def on_click_load(b):
            if load_filename.value != '':
                try:
                    f = open('%s.pickle' % load_filename.value, 'rb')
                    pickle_info = pickle.load(f)
                    dump.value = pickle_info['dump']
                    slice_index.value = pickle_info['slice_index']
                    value_min.value = pickle_info['value_min']
                    value_max.value = pickle_info['value_max']
                    log_min.value = pickle_info['log_min']
                    log_max.value = pickle_info['log_max']
                    quantity.value = pickle_info['quantity']
                    direction.value = pickle_info['direction']
                    log.value = pickle_info['log']
                    colour_select.value = pickle_info['colour_select']
                    saved_colours = pickle_info['colours']
                    saved_ranges = pickle_info['ranges']
                    for i in range(0, 12):
                        colours[i].value, colours[i].disabled = saved_colours[i][0], saved_colours[i][1]
                        ranges[i].value, ranges[i].disabled = saved_ranges[i][0], saved_ranges[i][1]
                    f.close()
                except:
                    print('Failed to load file')
        load_button.on_click(on_click_load)

        def min_max_link(change):
            if change['owner'].description == "Min Value":
                value_max.min = value_min.value + 1e-7
            elif change['owner'].description == "Log Min":
                log_max.min = log_min.value + 1e-7
            elif change['owner'].description == "Max Value":
                value_min.max = value_max.value - 1e-7
            elif change['owner'].description == "Log Max":
                log_min.max = log_max.value - 1e-7
        value_min.observe(min_max_link)
        value_max.observe(min_max_link)
        log_min.observe(min_max_link)
        log_max.observe(min_max_link)

        def on_ddqs_change(change):
            if change['name'] == 'value' and (change['new'] != change['old']):
                self.slice_gui_range_update = True
        dump.observe(on_ddqs_change)
        direction.observe(on_ddqs_change)
        quantity.observe(on_ddqs_change)
        slice_index.observe(on_ddqs_change)

        def on_log_change(change):
            if change['name'] == 'value' and (change['new'] != change['old']):
                self.slice_gui_range_update = True
                if change['new'] == True:
                    log_range.layout.display = 'block'
                    value_range.layout.display = 'none'
                else:
                    value_range.layout.display = 'block'
                    log_range.layout.display = 'none'
        log.observe(on_log_change)

        output = widgets.interactive_output(plot, {'dump': dump, 'quantity': quantity, 'direction': direction, 'log': log, \
            'slice_index': slice_index, 'vmin': value_min, 'vmax': value_max, 'log_min': log_min, 'log_max': log_max, \
            'colour0': colours[0], 'colour1': colours[1], 'colour2': colours[2], 'colour3': colours[3], 'colour4': colours[4], \
            'colour5': colours[5], 'colour6': colours[6], 'colour7': colours[7], 'colour8': colours[8], 'colour9': colours[9], \
            'colour10': colours[10], 'colour11': colours[11], 'range0': ranges[0], 'range1': ranges[1], 'range2': ranges[2], \
            'range3': ranges[3], 'range4': ranges[4], 'range5': ranges[5], 'range6': ranges[6], 'range7': ranges[7], \
            'range8': ranges[8], 'range9': ranges[9], 'range10': ranges[10], 'range11': ranges[11], 'colour_select': colour_select})

        display(ui, output)

    def get_mollweide_data(self, dump, radius, quantity):
        constants = {
            'atomicnoH': 1.,
            'atomicnocld': self._rprofset.get('atomicnocld', fname=0),
            'fkcld': self._rprofset.get('fkcld', fname=0),
            'airmu': self._rprofset.get('airmu', fname=0),
            'cldmu': self._rprofset.get('cldmu', fname=0)
        }
        npoints = self.sphericalHarmonics_lmax(radius)[-1]
        ux = self.get(1, dump)
        uy = self.get(2, dump)
        uz = self.get(3, dump)
        ur, utheta, uphi = self.get_spherical_components(ux, uy, uz)
        ur_r, utheta_r, uphi_r = self.get_spherical_interpolation(ur, radius, npoints=npoints, plot_mollweide=True)
        plot_val = []

        if quantity == 0:
            plot_val = ur_r
        elif quantity == 1:
            u_t = self.get(4, dump)
            plot_val = self.get_spherical_interpolation(u_t, radius, npoints=npoints)
        elif quantity == 2 or quantity == 3:
            fv = self.get(9, dump)
            if quantity == 2:
                plot_val = self.get_spherical_interpolation(fv, radius, npoints=npoints)
            else:
                Xcld = fv/((1. - fv)*(constants['airmu']/constants['cldmu']) + fv)
                XH = constants['atomicnoH']*(constants['fkcld']/constants['atomicnocld'])*Xcld
                plot_val = self.get_spherical_interpolation(XH, radius, npoints=npoints)
        elif quantity == 4:
            rho = self.get(8, dump)
            rho_trilinear_r = self.get_spherical_interpolation(rho, radius, npoints=npoints, method='trilinear')
            avg_rho_trilinear = rho_trilinear_r.mean()
            plot_val = (rho_trilinear_r - avg_rho_trilinear) / avg_rho_trilinear
        elif quantity == 5:
            omega = self.get(6, dump)
            plot_val = self.get_spherical_interpolation(omega, radius, npoints=npoints)

        return {
            'utheta_r': utheta_r,
            'uphi_r': uphi_r,
            'npoints': npoints,
            'plot_val': plot_val
        }

    def mollweide_plot(self, quantity, log, vmin, vmax, colour_select, colours, ranges, ifig):
        plot_val = self.mollweide_data['plot_val']
        mollweide_plot = self.mollweide_fig.add_axes([0.1, 0.2, 0.88, 0.88], projection='mollweide')
        mollweide_plot.grid("True")
        cax = self.mollweide_fig.add_axes([0.12, 0.2, 0.84, 0.02])
        cmap = self.build_cmap(colours, ranges, colour_select, vmin, vmax)

        if log == True:
            log_min = np.abs(np.log10(vmin))
            plot_bool = plot_val > 0
            for index, val in enumerate(plot_val):
                if plot_bool[index] == True:
                    if vmin <= val and val <= vmax:
                        plot_val[index] = np.log10(val) + log_min
                    else:
                        plot_val[index] = 0
                else:
                    if val != 0:
                        if -vmax <= val and val <= -vmin:
                            plot_val[index] = -np.log10(-val) - log_min
                        else:
                            plot_val[index] = 0
            mollweide_plot.scatter(self.mollweide_data['uphi_r'], self.mollweide_data['utheta_r'], s=(72./self.mollweide_fig.dpi)**2, marker=',', c=plot_val, cmap=cmap, vmin=plot_val.min(), vmax=plot_val.max())
        else:
            mollweide_plot.scatter(self.mollweide_data['uphi_r'], self.mollweide_data['utheta_r'], s=(72./self.mollweide_fig.dpi)**2, marker=',', c=plot_val, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar1 = self.mollweide_fig.colorbar(mollweide_plot.collections[0], cax=cax, orientation='horizontal')
        pl.show(ifig)

    def mollweide_gui(self, rad_def=-1, rad_range=[0,-1], size=10, ifig=124):
        self.mollweide_data_update = False
        dump_min, dump_max = self.get_dump_list()[0], self.get_dump_list()[-1]
        dump_mean = int(2*(-dump_min + dump_max)/3.)

        # left-most widgets in graph settings
        dump = widgets.IntSlider(value=dump_mean, min=dump_min, max=dump_max, step=1, description="Dump", disabled=False, continuous_update=False, orientation="horizontal", layout=widgets.Layout(width='350px'))

        #Fix max radius bug:
        radii = self._rprofset.get('R',fname=dump.value)
        rad_max = max(radii)
        rad_range = [0,rad_max] if rad_range[-1] < 0 else rad_range
        #Set default to the median radius:
        rad_med = np.median(radii)
        rad_def = rad_med if rad_def < 0 else rad_def

        radius = widgets.FloatSlider(value=rad_def, min=rad_range[0], max=rad_range[1], step=0.1, description="Radius", disabled=False, continuous_update=False, layout=widgets.Layout(width='350px'))
        quantity = widgets.Dropdown(options=[('u_r', 0), ('u_t', 1), ('fv', 2), ('X_H', 3), ('rho', 4), ('|w|', 5)], value=0, description="Quantity", layout=widgets.Layout(width='200px'))
        log = widgets.Checkbox(value=True, description="Log Values", disabled=False, indent=True)
        quant_log = widgets.HBox([quantity, log], layout=widgets.Layout(margin='0px 0px 0px 10px'))
        plot_button = widgets.Button(description="Render Plot", disabled=False, layout=widgets.Layout(margin="10px 0px 0px 20px"))

        left_widgets = widgets.VBox([dump, radius, quant_log, plot_button], layout=widgets.Layout(width='30%'))

        # center widgets in graph settings
        value_label = widgets.Label(value="Value Range to Display:")
        value_min = widgets.BoundedFloatText(value=-85., min=-1000., max=84.9999999, step=1e-7, readout_format=".7f", description="Min Value", layout=widgets.Layout(width='335px'))
        value_max = widgets.BoundedFloatText(value=85, min=-84.9999999, max=1000., step=1e-7, readout_format=".7f", description="Max Value", layout=widgets.Layout(width='335px'))

        log_label = widgets.Label(value="Log Value Range to Display (Applied symmetrically):")
        log_min = widgets.BoundedFloatText(value=1e-3, min=1e-7, max=9.99999e-2, step=1e-7, readout_format=".7f", description="Log Min", layout=widgets.Layout(width='335px'))
        log_max = widgets.BoundedFloatText(value=1e-1, min=1.0001e-3, max=2e4, step=1e-7, readout_format=".7f", description="Log Max", layout=widgets.Layout(width='335px'))

        value_range = widgets.VBox([value_label, value_min, value_max], layout=widgets.Layout(width='100%'))
        value_range.layout.display = 'none' # hide these controls intially as log == True on init
        log_range = widgets.VBox([log_label, log_min, log_max], layout=widgets.Layout(width='100%'))

        center_widgets = widgets.VBox([value_range, log_range], layout=widgets.Layout(width='30%'));

        # right-most widgets in graph settings
        save_load_label = widgets.Label(value="Save/Load Configuration (.pickle):")
        save_button = widgets.Button(description="Save Config", disabled=False)
        save_filename = widgets.Text(placeholder="Enter name w/out file extension", disabled=False, layout=widgets.Layout(width='250px'))
        save_combo = widgets.HBox([save_button, save_filename], layout=widgets.Layout(margin="0px 0px 0px 20px"))

        load_button = widgets.Button(description="Load Config", disabled=False)
        load_filename = widgets.Text(placeholder="Enter filename w/out file extension", disabled=False, layout=widgets.Layout(width='250px'))
        load_combo = widgets.HBox([load_button, load_filename], layout=widgets.Layout(margin="0px 0px 0px 20px"))

        right_widgets = widgets.VBox([save_load_label, save_combo, load_combo], layout=widgets.Layout(margin="0px 0px 0px 20px", width='30%'))

        # Layout of all widgets and tabs
        graph_settings = widgets.HBox([left_widgets, center_widgets, right_widgets])
        colours, ranges, colour_select, colourpicker = self.get_colourpicker(3)
        gui = widgets.Tab(children=[graph_settings, colourpicker])
        gui.set_title(0, "Graph Settings")
        gui.set_title(1, "Colour Picker")

        # Setup observe and onclick events
        def on_click_plot_button(b):
            plot_button.disabled = True
            self.mollweide_data = self.get_mollweide_data(dump.value, radius.value, quantity.value)
            if self.mollweide_data_update == True:
                if log.value == False:
                    value_min.value = self.mollweide_data["plot_val"].min()
                    value_max.value = self.mollweide_data["plot_val"].max()
                self.mollweide_data_update = False
            colour_values = [colours[0].value, colours[1].value, colours[2].value, colours[3].value, colours[4].value, colours[5].value, colours[6].value, \
                colours[7].value, colours[8].value, colours[9].value, colours[10].value, colours[11].value]
            range_values = [ranges[0].value, ranges[1].value, ranges[2].value, ranges[3].value, ranges[4].value, ranges[5].value, ranges[6].value, \
                ranges[7].value, ranges[8].value, ranges[9].value, ranges[10].value, ranges[11].value]
            self.mollweide_fig.clear()
            if log.value == True:
                self.mollweide_plot(quantity.value, log.value, log_min.value, log_max.value, colour_select.value, colour_values, range_values, ifig)
            else:
                self.mollweide_plot(quantity.value, log.value, value_min.value, value_max.value, colour_select.value, colour_values, range_values, ifig)
            plot_button.disabled = False
        plot_button.on_click(on_click_plot_button)

        def on_dqr_change(change):
            if (change['new'] != change['old']):
                self.mollweide_data_update = True
        dump.observe(on_dqr_change)
        quantity.observe(on_dqr_change)
        radius.observe(on_dqr_change)

        def on_log_change(change):
            if change['name'] == 'value' and (change['new'] != change['old']):
                self.mollweide_data_update = True
                if change['new'] == True:
                    log_range.layout.display = 'block'
                    value_range.layout.display = 'none'
                else:
                    value_range.layout.display = 'block'
                    log_range.layout.display = 'none'
        log.observe(on_log_change)

        def min_max_link(change):
            if change['owner'].description == "Min Value":
                value_max.min = value_min.value + 1e-7
            elif change['owner'].description == "Log Min":
                log_max.min = log_min.value + 1e-7
            elif change['owner'].description == "Max Value":
                value_min.max = value_max.value - 1e-7
            elif change['owner'].description == "Log Max":
                log_min.max = log_max.value - 1e-7
        value_min.observe(min_max_link)
        value_max.observe(min_max_link)
        log_min.observe(min_max_link)
        log_max.observe(min_max_link)

        def on_click_save(b):
            pickle_info = {
                'dump': dump.value,
                'radius': radius.value,
                'quantity': quantity.value,
                'log': log.value,
                'value_min': value_min.value,
                'value_max': value_max.value,
                'log_min': log_min.value,
                'log_max': log_max.value,
                'colour_select': colour_select.value,
                'colours': [],
                'ranges': []
            }
            for index in range(0, 12):
                pickle_info['colours'].append((colours[index].value, colours[index].disabled))
                pickle_info['ranges'].append((ranges[index].value, ranges[index].disabled))
            try:
                if save_filename.value != '':
                    f = open('%s.pickle' % save_filename.value, 'wb')
                else:
                    f = open('%s.pickle' % date.today(), 'wb')
                pickle.dump(pickle_info, f)
                f.close()
            except:
                print('Failed to save file')
        save_button.on_click(on_click_save)

        def on_click_load(b):
            if load_filename.value != '':
                try:
                    f = open('%s.pickle' % load_filename.value, 'rb')
                    pickle_info = pickle.load(f)
                    dump.value = pickle_info['dump']
                    radius.value = pickle_info['radius']
                    quantity.value = pickle_info['quantity']
                    log.value = pickle_info['log']
                    colour_select.value = pickle_info['colour_select']
                    value_min.value = pickle_info['value_min']
                    value_max.value = pickle_info['value_max']
                    log_min.value = pickle_info['log_min'],
                    log_max.value = pickle_info['log_max'],
                    saved_colours = pickle_info['colours']
                    saved_ranges = pickle_info['ranges']
                    for i in range(0, 12):
                        colours[i].value, colours[i].disabled = saved_colours[i][0], saved_colours[i][1]
                        ranges[i].value, ranges[i].disabled = saved_ranges[i][0], saved_ranges[i][1]
                    f.close()
                except:
                    print('Failed to load file')
        load_button.on_click(on_click_load)

        display(gui)

        #setup needed initial plot info
        pl.rcParams.update({'font.size': 5})
        self.mollweide_fig = pl.figure(ifig, dpi=300)
        self.mollweide_fig.canvas.layout.height = str(0.9*size) + 'in' # This is a hack to prevent ipympl
        self.mollweide_fig.canvas.layout.width  = str(1.1*size) + 'in' # to adjust horizontal figure size
        self.mollweide_fig.clear()

        #setup intial data run and plot
        self.mollweide_data = self.get_mollweide_data(dump.value, radius.value, quantity.value)
        colour_values = [colours[0].value, colours[1].value, colours[2].value, colours[3].value, colours[4].value, colours[5].value, colours[6].value, \
            colours[7].value, colours[8].value, colours[9].value, colours[10].value, colours[11].value]
        range_values = [ranges[0].value, ranges[1].value, ranges[2].value, ranges[3].value, ranges[4].value, ranges[5].value, ranges[6].value, \
            ranges[7].value, ranges[8].value, ranges[9].value, ranges[10].value, ranges[11].value]
        self.mollweide_plot(quantity.value, log.value, log_min.value, log_max.value, colour_select.value, colour_values, range_values, ifig)


# now the 2X classes will override a couple of methods in the instantiation processes
class MomsData2X(MomsData):
    '''
    MomsData2x reads in the half briquette resolution datacube which will contain only one quantity
    (we need 8 "varlocs" for one quantity!) In the MomentsDataSet2X it is assumed that an rprof is
    supplied so that a suitable grid can be made.

    Parameters
    ----------
    file_path: string
        Path to the .aaa file.
    verbose: integer
        Verbosity level as defined in class Messenger.
    '''

    def __init__(self, file_path, verbose=3):

        # we will call MomsData to read in everything in the standard fashion
        super().__init__(file_path, verbose)

        # now I have a formatted self.var. I will assume the first 8 quantities
        # are in fact the "2X" quantity and so I will construct the double
        # resolution grid
        self.ngridpoints = int(2*self.ngridpoints)
        self.var2x = np.zeros((self.ngridpoints,self.ngridpoints,self.ngridpoints),
                               dtype=float32)

        # for an example, this is supposed to be fv2x then the following convention
        # for the order of varlocs is:
        # fvlbn, fvrbn, fvltn, fvrtn, fvlbf, fvrbf, fvltf, fvrtf
        # Where:
        # for x coordinate: l is "left", r is "right"
        # for y coordinate: b is "bottom", t is "top"
        # for z coordinate: n is "near", f is "far"

        fvlbn = self.var[0]
        fvrbn = self.var[1]

        fvltn = self.var[2]
        fvrtn = self.var[3]

        fvlbf = self.var[4]
        fvrbf = self.var[5]

        fvltf = self.var[6]
        fvrtf = self.var[7]

        # since we have [z,y,x] we can do...
        self.var2x[0:-1:2,0:-1:2,0:-1:2] = fvlbn
        self.var2x[0:-1:2,0:-1:2,1::2] = fvrbn

        self.var2x[0:-1:2,1::2,0:-1:2] = fvltn
        self.var2x[0:-1:2,1::2,1::2] = fvrtn

        self.var2x[1::2,0:-1:2,0:-1:2] = fvlbf
        self.var2x[1::2,0:-1:2,1::2] = fvrbf

        self.var2x[1::2,1::2,0:-1:2] = fvltf
        self.var2x[1::2,1::2,1::2] = fvrtf

        # delete var
        del self.var

    # override get
    def get(self, varloc):
        '''
        Returns a 3d array of the variable that is defined at whatever(varloc).

        Parameters
        ----------
        varloc: integer
            integer index of the quantity that is defined under whatever(varloc).
            This is completely ignored as there is only one variable

        Returns
        -------
        np.ndarray
            The 2X variable
        '''

        # we have no other variables, only return data2x
        return self.var2x


class MomsDataSet2X(MomsDataSet):
    '''
    MomsDataSet2X contains a set of dumps of MomsData2X from a single run of the
    Moments reader from PPMstar 2.0.

    Parameters
    ----------
    dir_name: string
        Name of the directory to be searched for .aaa uncompressed moms datacubes
    rprofset: RprofSet
        The grid MUST be constructed
    init_dump_read: integer
        The initial dump to read into memory when object is initialized
    dumps_in_mem: integer
        The number of dumps to be held into memory. These datacubes can be large (~2Gb for 384^3)
    var_list: list
        This is a list that can be filled with strings that will reference data. Since
        this is a 2X quantity, there is only one quantity!
    verbose: integer
        Verbosity level as defined in class Messenger.
    '''

    def __init__(self, dir_name, rprofset, init_dump_read=0, dumps_in_mem=2, var_list=[], verbose=3):

        # we can call super init but override a few methods
        super().__init__(dir_name, init_dump_read, dumps_in_mem, var_list,
                         rprofset=rprofset, verbose=3)


    def _get_dump(self, dump):
        '''
        Gets a new dump for MomsData2X or instantiates MomsData2X

        Parameters
        ----------
        dump: integer
            The dump which you want to read in
        '''

        if dump not in self._dumps:
            err = 'Dump {:d} is not available.'.format(dump)
            self._messenger.error(err)
            return None

        file_path = '{:s}{:04d}/{:s}-BQav{:04d}.aaa'.format(self._dir_name, \
                                                             dump, self._run_id, dump)

        # we first check if we can add a new moments data to memory
        # without removing another
        if len(self._many_momsdata) < self._dumps_in_mem:

            # add it to our dictionary!
            self._many_momsdata.update(zip([str(dump)],[MomsData2X(file_path)]))

            # append the key. This keeps track of order of read in
            self._many_momsdata_keys.append(str(dump))

        else:

            # we gotta remove one of them, this will be index 0 of a list
            del self._many_momsdata[str(self._many_momsdata_keys[0])]
            self._many_momsdata_keys.remove(self._many_momsdata_keys[0])

            # now add a new momsdata object to our dict
            self._many_momsdata.update(zip([str(dump)],[MomsData2X(file_path)]))

            # append the key. This keeps track of order of read in
            self._many_momsdata_keys.append(str(dump))

        # all is good. update what_dump_am_i
        self.what_dump_am_i = dump

    def _set_dictionaries(self, var_list):
        '''
        This function will setup the dictionaries that will house multiple moments data objects and
        a convenience dictionary to refer to the SINGLE FV2X variable by a string

        Parameters
        ---------
        var_list: list
            A list that may contain strings to associate the integer locations "varloc"

        Returns
        -------
        Boolean
            True if successful
            False if failure
        '''

        # because we only have one variable...
        self._number_of_whatevers = 1

        # check if the list is empty
        if not var_list:

            # ok it is empty, construct default dictionary
            var_keys = [str(i) for i in range(self._number_of_whatevers)]
            var_vals = [i for i in range(self._number_of_whatevers)]

        else:

            # first we check that var_list is the correct length
            if len(var_list) != self._number_of_whatevers:

                # we use the default
                var_keys = [str(i) for i in range(self._number_of_whatevers)]
                var_vals = [i for i in range(self._number_of_whatevers)]

            else:

                # ok we are in the clear
                var_keys = [str(i) for i in var_list]
                var_vals = [i for i in range(self._number_of_whatevers)]

                # I will also allow for known internal varloc to point to the same things
                # with this dictionary, i.e xc: varloc = 0 ALWAYS
                var_keys2 = [str(i) for i in range(self._number_of_whatevers)]

                self._varloc.update(zip(var_keys2,var_vals))

        # construct the variable dictionary
        self._varloc.update(zip(var_keys,var_vals))

        return True

    def _get_cgrid(self):
        '''
        Constructs the PPMStar cartesian grid from the internal rprofset

        Returns
        -------
        Boolean
            True on success.
            False on failure.
        '''

        # check, do we already have this?
        if not self._cgrid_exists:

            # we can construct the xc_array
            rprof = self._rprofset.get_dump(self._rprofset.get_dump_list()[0])
            dx = rprof.get('deex')

            # 4 * dx * (ngridpoints/2.) gives me the right boundary but I want the central value so
            right_xcbound = 2. * dx * (self.moms_ngridpoints/2.) - 2. * (dx/2.)
            left_xcbound = -right_xcbound

            # So unfortunately float32 is terrible, only accurate to 1e-6. This isn't strictly uniform...
            grid_values = 2. * dx * np.arange(0,self.moms_ngridpoints) + left_xcbound

            xc_array = np.ones((self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)) * grid_values

            # x contains all the info for y and z, just different order. I figured this out
            # and lets use strides so that we don't allocate new wasteful arrays (~230Mb for 1536!)
            xc_strides = xc_array.strides
            xorder = [i for i in xc_strides]

            yc_array = np.lib.stride_tricks.as_strided(xc_array,shape=(self.moms_ngridpoints,
                                                                         self.moms_ngridpoints,
                                                                         self.moms_ngridpoints),
                                                        strides=(xorder[1],xorder[2],xorder[0]))

            zc_array = np.lib.stride_tricks.as_strided(xc_array,shape=(self.moms_ngridpoints,
                                                                         self.moms_ngridpoints,
                                                                         self.moms_ngridpoints),
                                                        strides=(xorder[2],xorder[0],xorder[1]))

            # unfortunately we have to flatten these. This creates copies as
            # they are not contiguous memory chunks... (data is a portion of ghostdata)
            self._xc = np.ravel(xc_array)
            self._yc = np.ravel(yc_array)
            self._zc = np.ravel(zc_array)

            # creating a new array, radius
            self._radius = np.sqrt(np.power(self._xc,2.0) + np.power(self._yc,2.0) +\
                                    np.power(self._zc,2.0))

            # from this, I will always setup vars for a rprof
            # we need a slight offset from the lowest value and highest value of grid for interpolation!
            delta_r = 2*np.min(self._xc[np.where(np.unique(self._xc)>0)])
            eps = 0.000001
            self._radial_boundary = np.linspace(delta_r+eps*delta_r,delta_r*(self.moms_ngridpoints/2.)-eps*delta_r*(self.moms_ngridpoints/2.),int(np.ceil(self.moms_ngridpoints/2.)))

            # these are the boundaries, now I need what is my "actual" r value
            self.radial_axis = self._radial_boundary - delta_r/2.

            # construct the bins for computing averages ON radial_axis, these are "right edges"
            delta_r = (self.radial_axis[1] - self.radial_axis[0])/2.
            radialbins = self.radial_axis + delta_r
            self.radial_bins = np.insert(radialbins,0,0)

            # in some cases, it is more convenient to work with xc[z,y,x] so lets store views
            self._xc_view = self._xc.view()
            self._xc_view.shape = (self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)
            self._yc_view = self._yc.view()
            self._yc_view.shape = (self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)
            self._zc_view = self._zc.view()
            self._zc_view.shape = (self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)

            self._radius_view = self._radius.view()
            self._radius_view.shape = (self.moms_ngridpoints,self.moms_ngridpoints,self.moms_ngridpoints)

            # grab unique values along x-axis
            self._unique_coord = self._xc_view[0,0,:]

            # all is good, set that we have made our grid
            self._cgrid_exists = True

            return True

        else:
            return True

    # The following methods are overridden as they are able to grab variables with varloc which is not
    # used in MomsDataSet2X
    def get_spherical_components(self,ux ,uy, uz, igrid=None):
        '''
        Vector quantities are output in the cartesian coordinates but we can transform them to
        spherical coordinates using unit vectors. This returns the spherical components of u

        Parameters
        ----------
        ux, uy, uz: int, str, np.ndarray
            int: integer referring to varloc
            str: string referring to quantity varloc
            np.ndarray: array with quantities
        fname: None,int
            None: default option, will grab current dump
            int: Dump number
        igrid: np.ndarray
            If the quantity is not defined on the entire grid we can still convert it if we know the
            cartesian points that it is on. Note:

            igrid.shape = (len(ux.flatten()),3)
            igrid[:,0] = z, igrid[:,1] = y, igrid[:,2] = x

        Returns
        -------
        u_spherical: list of np.ndarray
            The spherical components of u
        '''

        # first check if we are using the grid or not
        if not isinstance(igrid,np.ndarray):
            if not self._grid_jacobian_exists:

                # create the grid jacobian, keep this in memory
                self._grid_jacobian = self._get_jacobian(self._xc_view,self._yc_view,self._zc_view, self._radius_view)
                self._grid_jacobian_exists = True

            # local variable to reference internal jacobian
            jacobian = self._grid_jacobian
        else:
            radius = np.sqrt(np.power(igrid[:,2],2.0) + np.power(igrid[:,1],2.0) + np.power(igrid[:,0],2.0))
            jacobian = self._get_jacobian(igrid[:,2],igrid[:,1],igrid[:,0],radius)

        # I cannot grab quantities!
        if not isinstance(ux,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError

        if not isinstance(uy,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError

        if not isinstance(uz,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError


        ur = ux * jacobian[0] + uy * jacobian[1] + uz * jacobian[2]
        utheta = ux * jacobian[3] + uy * jacobian[4] + uz * jacobian[5]
        uphi = ux * jacobian[6] + uy * jacobian[7]

        return [ur, utheta, uphi]

    def norm(self,ux,uy,uz):
        '''
        Norm of some vector quantity. It is written as ux, uy, uz which will give |u| through
        the definition |u| = sqrt(ux**2 + uy**2 + uz**2). The vector must be defined in an
        orthogonal basis. i.e we can also do |u| = sqrt(ur**2 + uphi**2 + utheta**2)

        Parameters
        ----------
        ux, uy, uz: int, str, np.ndarray
            array with vector quantities

        Returns
        -------
        |u|: np.ndarray
            The norm of vector u
        '''

        # I cannot grab quantities!
        if not isinstance(ux,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError

        if not isinstance(uy,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError

        if not isinstance(uz,np.ndarray):
            err = 'MomsData2X does not support multiple quantities. Arrays must be inputted for ux, uy and uz'
            self._messenger.error(err)
            raise ValueError

        return np.sqrt(np.power(ux,2.0)+np.power(uy,2.0)+np.power(uz,2.0))

# DS: my convenient plot figure handling functions
def add_plot(celln, ifig, ptrack):
    '''
    Add a plot to be tracked in ptrack. This ensures that it will be tracked and
    closed when needed with close_plot.

    Parameters
    ----------
    celln: integer
        This is an integer to track unique plots in a notebook
    ifig: integer
        This is the current ifig for the unique plot corresponding to celln
    ptrack: dict
        This dict that holds ifigs corresponding to key celln
    '''

    # can we update our stored ifig for celln?
    try:
        stored_ifig = ptrack[str(celln)]

        # if stored isnt equal to what we have, lets update!
        if stored_ifig != ifig:
            ptrack[str(celln)] = ifig

    # if this failed, we are not tracking unique figure celln
    except KeyError:
        ptrack[str(celln)] = ifig

def close_plot(celln, ifig, ptrack):
    '''
    Continuously close a unique figure plot celln or create a new figure

    Parameters
    ----------
    celln: integer
        This is an integer to track unique plots in a notebook
    ifig: integer
        This is the current ifig for the unique plot corresponding to celln
    ptrack: dict
        This dict that holds ifigs corresponding to key celln

    Returns
    -------
    tuple (int,int):
        first integer is whether to open or close figure
        second integer is the current ifig
    '''

    # check if we already have this plot added to ptrack?
    try:
        if ifig == ptrack[str(celln)]:
            # ifig matches, we are re-running a celln
            return (1,ifig)
        else:
            # do we already have this cells figure open? Itll be in ptrack
            for i in range(len(pl.get_fignums())):
                if i == ptrack[str(celln)]:
                    return (1,ifig)
            # how did this happen!?
            return (0,ifig)

    except KeyError:
        # plot is not added, make sure we dont close
        return (0,ifig)
