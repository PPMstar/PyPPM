"""ppmpy.derived
Utility registry for derived (computed) quantities.

The goal is to have a single, central place where formulas that combine or
post-process raw simulation variables live.  Client classes such as
`RprofSet`, `MomsDataSet`, etc. can simply ask for a variable name and if
that variable is not present in the raw data it will be computed on-the-fly
using the registered function.

Example
-------
>>> from ppmpy.derived import derived_quantity, list_derived
>>> @derived_quantity('ur')
... def radial_velocity(obj, fname, num_type='ndump', **kwargs):
...     ux = obj.get('ux', fname=fname, num_type=num_type)
...     ...
...

Functions
---------
derived_quantity(name)  - decorator to register a function
get_derived(name)       - fetch a registered callable by name
list_derived()          - list available derived quantity names

Classes
-------
DerivedMixin            - mix-in that provides `_get_derived()` helper

Notes
-----
*  The registered callable must accept the *host* object instance as first
   argument so that it can fetch raw variables via the usual `.get()` or
   other helper methods.
*  A very small cache is maintained on the host object to avoid recomputing
   the same quantity repeatedly within a single call chain.
"""

from __future__ import annotations

from typing import Callable, Dict, List

__all__ = [
    'derived_quantity',
    'get_derived',
    'list_derived',
    'DerivedMixin',
]

# -----------------------------------------------------------------------------
# Registry implementation
# -----------------------------------------------------------------------------

_DERIVED_REGISTRY: Dict[str, Callable] = {}


def derived_quantity(name: str) -> Callable[[Callable], Callable]:
    """Decorator that registers *func* in the derived-quantity registry.

    The decorated function must have the signature
    `func(obj, *args, **kwargs)` where *obj* is the hosting data object
    (`RprofSet`, `MomsDataSet`, etc.).
    """
    name = name.lower()

    def decorator(func: Callable) -> Callable:
        if name in _DERIVED_REGISTRY:
            raise RuntimeError(f"Derived quantity '{name}' already registered")
        _DERIVED_REGISTRY[name] = func
        return func

    return decorator


def get_derived(name: str) -> Callable:
    """Return the function that computes *name*.

    Raises
    ------
    KeyError
        If *name* is not a registered derived quantity.
    """
    return _DERIVED_REGISTRY[name.lower()]


def list_derived() -> List[str]:
    """Return a sorted list of all registered derived quantity names."""
    return sorted(_DERIVED_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Convenience mix-in
# -----------------------------------------------------------------------------

class DerivedMixin:
    """Mixin that enables objects to access derived quantities easily."""

    # Attribute name that will hold the per-instance cache.
    _CACHE_NAME = '_derived_cache'

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def _get_derived(self, name: str, *args, **kwargs):
        """Compute (or retrieve from cache) the derived quantity *name*.

        Parameters
        ----------
        name : str
            Name of the derived quantity (case-insensitive).
        *args, **kwargs
            Passed verbatim to the registered callable.
        """

        # Lazily create cache dictionary on the instance.
        cache = getattr(self, self._CACHE_NAME, None)
        if cache is None:
            cache = {}
            setattr(self, self._CACHE_NAME, cache)

        lname = name.lower()
        if lname in cache:
            return cache[lname]

        try:
            func = get_derived(lname)
        except KeyError:
            raise  # Let caller decide what to do.

        value = func(self, *args, **kwargs)
        cache[lname] = value
        return value 

# -----------------------------------------------------------------------------
# Derived quantity implementations
# -----------------------------------------------------------------------------

import numpy as np
from numpy import zeros_like


def _determine_object_type(obj):
    """
    Safely determine if an object is a MomsDataSet or RprofSet.
    
    Returns:
        str: 'moms' for MomsDataSet, 'rprof' for RprofSet, 'unknown' for others
    """
    # Try the explicit flags first
    if hasattr(obj, '__isMomsDataSet') and obj.__isMomsDataSet:
        return 'moms'
    
    if hasattr(obj, '__isRprofSet') and obj.__isRprofSet:
        return 'rprof'
    
    # Fallback: use method existence to distinguish
    if hasattr(obj, 'sphericalHarmonics_format'):
        # MomsDataSet has this method
        return 'moms'
    elif hasattr(obj, 'get') and hasattr(obj, 'get_dump'):
        # RprofSet has these methods
        return 'rprof'
    else:
        return 'unknown'


@derived_quantity('ur')
def compute_ur_from_components(obj, fname, num_type='ndump', radius=None, lmax=None, 
                              _for_rprof=False, method='trilinear', logvar=False, **kwargs):
    """Compute radial velocity component from ux, uy, uz components.
    
    For MomsDataSet:
    - If _for_rprof=False: returns full 3D array of ur for use with get()
    - If _for_rprof=True: returns radial profile using RMS over spherical surfaces
    
    For RprofSet, returns radial profile.
    
    Note: For the signed quantity ur, radial profiles use RMS (root mean square)
    to avoid cancellation of inward/outward flows.
    
    Parameters
    ----------
    obj : RprofSet or MomsDataSet
        Host object that provides access to raw data
    fname : int or str
        Dump number or filename
    num_type : str
        Type of fname ('ndump' or 't')
    radius : float or np.ndarray, optional
        Radius for spherical interpolation (only used when _for_rprof=True)
    lmax : int, optional
        Maximum spherical harmonic degree (MomsDataSet only) - ignored for 3D computation
    _for_rprof : bool, optional
        If True, compute full radial profile; if False, return 3D array
    method : str, optional
        Interpolation method for radial profile
    logvar : bool, optional
        Whether to use log scaling for radial profile
    **kwargs
        Additional arguments passed to underlying methods
    """
    # Check if this is a MomsDataSet (3D data) or RprofSet (radial profiles)
    obj_type = _determine_object_type(obj)
    
    if obj_type == 'moms':
        # MomsDataSet case - use existing get_spherical_components method
        
        # Get spherical components using existing method
        ur, utheta, uphi = obj.get_spherical_components('ux', 'uy', 'uz', fname=fname)
        
        if _for_rprof:
            # Do the full radial profile calculation
            if radius is None:
                # Ensure grid is set up before accessing radial_axis
                if not hasattr(obj, '_cgrid_exists') or not obj._cgrid_exists:
                    obj._get_cgrid()
                radius = obj.radial_axis
            
            # Use existing spherical interpolation method
            quantity = obj.get_spherical_interpolation(ur, radius, method=method, logvar=logvar)
            # For signed quantity ur, use RMS over spherical surface at each radius
            quantity = np.sqrt(np.mean(quantity**2, axis=1))
            return quantity
        else:
            # Return 3D array for regular get() calls
            return ur
        
    elif obj_type == 'rprof':
        # RprofSet case - compute from radial profile data
        try:
            # Try to get the velocity components directly
            Ut = obj.get('|Ut|', fname, num_type=num_type, resolution='l')
            U = obj.get('|U|', fname, num_type=num_type, resolution='l')
            ur = np.sqrt(U**2 - Ut**2)
            return ur
        except:
            # If that fails, we might need to compute from individual components
            # This would require more complex logic for RprofSet
            raise ValueError("Cannot compute 'ur' from components for RprofSet - missing required data")
    else:
        raise ValueError("Object type not supported for 'ur' computation")


@derived_quantity('utot')
def compute_utot_from_components(obj, fname, num_type='ndump', radius=None, lmax=None, 
                                _for_rprof=False, method='trilinear', logvar=False, **kwargs):
    """Compute total velocity magnitude from ux, uy, uz components.
    
    For MomsDataSet:
    - If _for_rprof=False: returns full 3D array of utot for use with get()
    - If _for_rprof=True: returns radial profile for use with get_rprof()
    
    For RprofSet, returns radial profile.
    
    Parameters
    ----------
    obj : RprofSet or MomsDataSet
        Host object that provides access to raw data
    fname : int or str
        Dump number or filename
    num_type : str
        Type of fname ('ndump' or 't')
    radius : float or np.ndarray, optional
        Radius for spherical interpolation (only used when _for_rprof=True)
    lmax : int, optional
        Maximum spherical harmonic degree (MomsDataSet only) - ignored for 3D computation
    _for_rprof : bool, optional
        If True, compute full radial profile; if False, return 3D array
    method : str, optional
        Interpolation method for radial profile
    logvar : bool, optional
        Whether to use log scaling for radial profile
    **kwargs
        Additional arguments passed to underlying methods
    """
    obj_type = _determine_object_type(obj)
    
    if obj_type == 'moms':
        # MomsDataSet case - compute full 3D array
        
        # Get 3D velocity components
        ux = obj._get('ux', fname)
        uy = obj._get('uy', fname)
        uz = obj._get('uz', fname)
        
        # Compute total velocity magnitude
        utot_3d = np.sqrt(ux**2 + uy**2 + uz**2)
        
        if _for_rprof:
            # Do the full radial profile calculation
            if radius is None:
                # Ensure grid is set up before accessing radial_axis
                if not hasattr(obj, '_cgrid_exists') or not obj._cgrid_exists:
                    obj._get_cgrid()
                radius = obj.radial_axis
            
            # Use existing spherical interpolation method
            quantity = obj.get_spherical_interpolation(utot_3d, radius, method=method, logvar=logvar)
            # Average over spherical surface at each radius
            quantity = np.mean(quantity, axis=1)
            return quantity
        else:
            # Return 3D array for regular get() calls
            return utot_3d
        
    elif obj_type == 'rprof':
        # RprofSet case
        try:
            U = obj.get('|U|', fname, num_type=num_type, resolution='l')
            return U
        except:
            raise ValueError("Cannot compute 'utot' for RprofSet - missing required data")
    else:
        raise ValueError("Object type not supported for 'utot' computation")


@derived_quantity('ut_phi')
def compute_ut_phi_from_components(obj, fname, num_type='ndump', radius=None, lmax=None, 
                                  _for_rprof=False, method='trilinear', logvar=False, **kwargs):
    """Compute tangential velocity phi component from ux, uy, uz components.
    
    For MomsDataSet:
    - If _for_rprof=False: returns full 3D array of ut_phi for use with get()
    - If _for_rprof=True: returns radial profile for use with get_rprof()
    
    Only works for MomsDataSet (3D data).
    """
    obj_type = _determine_object_type(obj)
    
    if obj_type == 'moms':
        # MomsDataSet case - use existing get_spherical_components method
        
        # Get spherical components using existing method
        ur, utheta, uphi = obj.get_spherical_components('ux', 'uy', 'uz', fname=fname)
        
        if _for_rprof:
            # Do the full radial profile calculation
            if radius is None:
                # Ensure grid is set up before accessing radial_axis
                if not hasattr(obj, '_cgrid_exists') or not obj._cgrid_exists:
                    obj._get_cgrid()
                radius = obj.radial_axis
            
            # Use existing spherical interpolation method
            quantity = obj.get_spherical_interpolation(uphi, radius, method=method, logvar=logvar)
            # Average over spherical surface at each radius
            quantity = np.mean(quantity, axis=1)
            return quantity
        else:
            # Return 3D array for regular get() calls
            return uphi
        
    else:
        raise ValueError("'ut_phi' computation only supported for MomsDataSet")


@derived_quantity('ut_theta')
def compute_ut_theta_from_components(obj, fname, num_type='ndump', radius=None, lmax=None, 
                                    _for_rprof=False, method='trilinear', logvar=False, **kwargs):
    """Compute tangential velocity theta component from ux, uy, uz components.
    
    For MomsDataSet:
    - If _for_rprof=False: returns full 3D array of ut_theta for use with get()
    - If _for_rprof=True: returns radial profile for use with get_rprof()
    
    Only works for MomsDataSet (3D data).
    """
    obj_type = _determine_object_type(obj)
    
    if obj_type == 'moms':
        # MomsDataSet case - use existing get_spherical_components method
        
        # Get spherical components using existing method
        ur, utheta, uphi = obj.get_spherical_components('ux', 'uy', 'uz', fname=fname)
        
        if _for_rprof:
            # Do the full radial profile calculation
            if radius is None:
                # Ensure grid is set up before accessing radial_axis
                if not hasattr(obj, '_cgrid_exists') or not obj._cgrid_exists:
                    obj._get_cgrid()
                radius = obj.radial_axis
            
            # Use existing spherical interpolation method
            quantity = obj.get_spherical_interpolation(utheta, radius, method=method, logvar=logvar)
            # Average over spherical surface at each radius
            quantity = np.mean(quantity, axis=1)
            return quantity
        else:
            # Return 3D array for regular get() calls
            return utheta
        
    else:
        raise ValueError("'ut_theta' computation only supported for MomsDataSet")


@derived_quantity('ut')
def compute_ut_from_components(obj, fname, num_type='ndump', radius=None, lmax=None, 
                              ut_direction=0, _for_rprof=False, method='trilinear', logvar=False, **kwargs):
    """Compute tangential velocity in specified direction from ux, uy, uz components.
    
    For MomsDataSet:
    - If _for_rprof=False: returns full 3D array of ut for use with get()
    - If _for_rprof=True: returns radial profile for use with get_rprof()
    
    Parameters
    ----------
    obj : MomsDataSet
        Host object that provides access to raw data
    fname : int or str
        Dump number or filename
    num_type : str
        Type of fname ('ndump' or 't')
    radius : float or np.ndarray, optional
        Radius for spherical interpolation (only used when _for_rprof=True)
    lmax : int, optional
        Maximum spherical harmonic degree (MomsDataSet only) - ignored for 3D computation
    ut_direction : float, optional
        Direction in degrees (0 = phi direction, 90 = theta direction)
    _for_rprof : bool, optional
        If True, compute full radial profile; if False, return 3D array
    method : str, optional
        Interpolation method for radial profile
    logvar : bool, optional
        Whether to use log scaling for radial profile
    **kwargs
        Additional arguments passed to underlying methods
    """
    obj_type = _determine_object_type(obj)
    
    if obj_type == 'moms':
        # MomsDataSet case - use existing get_spherical_components method
        
        # Get spherical components using existing method
        ur, utheta, uphi = obj.get_spherical_components('ux', 'uy', 'uz', fname=fname)
        
        # Combine theta and phi components based on direction
        alpha = np.deg2rad(ut_direction)
        ut = np.cos(alpha)*uphi + np.sin(alpha)*utheta
        
        if _for_rprof:
            # Do the full radial profile calculation
            if radius is None:
                # Ensure grid is set up before accessing radial_axis
                if not hasattr(obj, '_cgrid_exists') or not obj._cgrid_exists:
                    obj._get_cgrid()
                radius = obj.radial_axis
            
            # Use existing spherical interpolation method
            quantity = obj.get_spherical_interpolation(ut, radius, method=method, logvar=logvar)
            # Average over spherical surface at each radius
            quantity = np.mean(quantity, axis=1)
            return quantity
        else:
            # Return 3D array for regular get() calls
            return ut
        
    else:
        raise ValueError("'ut' computation only supported for MomsDataSet") 