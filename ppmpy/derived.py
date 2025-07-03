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
derived_quantity(name)  → decorator to register a function
get_derived(name)       → fetch a registered callable by name
list_derived()          → list available derived quantity names

Classes
-------
DerivedMixin            → mix-in that provides `_get_derived()` helper

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