"""
Utility classes for SkyMapUQ usage.

Created 2024-08-26 by Tom Loredo (partly refactored from skymapuq.py)
"""
import timeit

import numpy as np
import scipy
from numpy import *


class Timer:
    """
    Context manager tracking elapsed time.
    """

    def __init__(self, name=None, units='auto', log=True):
        """
        Setup a timer context manager.

        If units=None, the elapsed time is not printed; it is available
        via `secs` and `msecs` attributes.

        If units='s' or 'ms', the appropriate attribute is printed when
        the context exits; if `name` is not None, it is used to label
        the output.  If units='auto', seconds are used for times >=1 s,
        otherwise milliseconds are used.
        """
        self.name = name
        self.units = units
        self.log = log

    def __enter__(self):
        if self.units:
            if self.log and self.name:
                print('* {}...'.format(self.name))
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.log:
            if self.units == 'auto':
                if self.secs >= 1.:
                    units = 's'
                else:
                    units = 'ms'
            else:
                units = self.units
            if units == 's':
                print('-> Elapsed time = {:f} s'.format(self.secs))
            elif units == 'ms':
                print('-> Elapsed time = {:f} ms'.format(self.msecs))


class MomentAccumulator:

    def __init__(self, hist=False):
        """
        Initialize accumulator tracking the mean, standard deviation, and max
        of a sequence of scalars.  Keep a history of all values if `hist` is
        True.

        Call .done() when the sequence has ended to finish the computation.

        Results are accessible as attributes: .avg, .sig, .max, .imax.
        If requested via `hist=True`, the stored history is in the array .hist.
        """
        self.n = 0
        self.avg = 0.
        self._v = 0.
        self.sig = None
        self.max = None
        self.imax = None

        if hist:
            self.hist = []
        else:
            self.hist = None

    def update(self, x):
        self.n += 1
        if self.hist is not None:
            self.hist.append(x)
        avg_ = self.avg + (x - self.avg)/self.n  # recursive update of mean
        self._v += (x - self.avg)*(x - avg_)
        self.avg = avg_
        if self.n == 1:
            self.max = x
            self.imax = self.n
        elif x > self.max:
            self.max = x
            self.imax = self.n

    def done(self):
        self.sig = sqrt(self._v/self.n)
        if self.hist:
            self.hist = array(self.hist)
