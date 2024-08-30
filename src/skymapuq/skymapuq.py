"""
SkyMapUQ - HEALPix sky map uncertainty quantification.

Some parts of the SkyMapUQ class are based on an IGWN tutorial:

Working with Flat Resolution Sky Maps - IGWN | Public Alerts User Guide
https://emfollow.docs.ligo.org/userguide/tutorial/skymaps.html

Created 2024-08-21 by Tom Loredo
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *

import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap import postprocess

from .utils import Timer, MomentAccumulator


class SkyMapUQ:
    """
    HEALPix sky map container supporting location uncertainty quantification
    via a variety of measures.
    """

    def __init__(self, fits_path, name=None, ligo=False, timer=False):
        """
        Load HEALPix data from a FITS file at the specified path.

        If `timer` is True, report time spent on subtasks.
        """
        with Timer('HEALPix file loading') as timer:
            self.ligo = ligo
            if not ligo:
                # HEALPix data gets loaded as a 1D ndarray + FITS header.
                self.hpx, header = hp.read_map(fits_path, h=True)
                self.meta = dict(header)
            else:
                self.hpx, self.meta = read_sky_map(fits_path)

        self.name = name
        self.n_px = len(self.hpx)
        self.nside = hp.npix2nside(self.n_px)  # lateral resolution
        sky_area = 4*pi * (180/pi)**2  # sq deg over sky
        # self.sd_px = sky_area / npix  # square degrees per pixel
        self.sd_px = hp.nside2pixarea(self.nside, degrees=True)
        self.sam_px = 60**2 * self.sd_px  # sq arcmin per pixel
        self.sas_px = 3600**2 * self.sd_px  # sq arsec per pixel

        with Timer('HPD region calculations') as time:
            # Array of indices by decreasing probability density:
            # self.i_ddens = np.flipud(np.argsort(self.hpx))
            self.i_ddens = flip(self.hpx.argsort())
            self.p_ddens = self.hpx[self.i_ddens]
            # Restrict UQ calculations to pixels with nonzero density.
            self.n_nz = nonzero(self.p_ddens)[0][-1]+1
            # Ranked credible region probabilities (levels) for
            # pixel-based HPD regions:
            self.hpd_levels_r = cumsum(self.p_ddens[:self.n_nz])
            # Pre-fill the HPD level-by-pixel array with 1, so pixels with 0
            # probability are labeled as outside any HPD region with P<1.
            self.hpd_levels_px = ones_like(self.p_ddens)
        self.hpd_levels_px[self.i_ddens[:self.n_nz]] = self.hpd_levels_r

        # Handle search probabilities lazily; mark as not calculated.
        self.p_search = None

        # Information-theoretic summaries:
        with Timer('Info theory calculations') as timer:
            self.entropy = -nansum(self.p_ddens[:self.n_nz]*log2(self.p_ddens[:self.n_nz]))
            self.info_range = 2**self.entropy

    def _init_search(self):
        """Compute conditional search probabilities for each pixel,
        conditioned on nondiscovery in previous pixels.
        """
        with Timer('Computing search probabilities') as timer:
            # Restrict to pixels with HPD levels < 1; this may omit many pixels
            # with negligible nonzero probability due to finite precision.
            self.n_search = where(self.hpd_levels_r < 1.)[0][-1] + 1
            # Add a single pixel with HPD level of 1 if available.
            if self.n_search < self.n_px:
                self.n_search += 1
            # There are several ways to compute p_search, including using
            # HPD region quantities already available, but beware of
            # roundoff issues.  This particular algorithm is robust to
            # situations with huge dynamic range across pixels.
            a = flip(self.p_ddens[:self.n_search])
            self.denoms = flip(cumsum(a))
            self.p_search = self.p_ddens[:self.n_search]/self.denoms

        # These two approaches fail to correctly compute p_search for
        # pixels with very small probabilities, due to roundoff error.
        # The errors can be huge for the small-probability pixels, but
        # they are innocuous for results of interest, since problematic
        # pixels have probabilities so small that they never get
        # explored in simulated searches.
        if False:
            self.p_search1 = self.p_ddens[:self.n_search].copy()
            self.p_search1[1:] /= (1. - self.hpd_levels_r[0:self.n_search-1])
            self.denoms1 = 1. - self.hpd_levels_r[0:self.n_search-1]
            self.p_search2 = self.p_ddens[:self.n_search].copy()
            q = 1. - self.p_ddens[:self.n_search-1]
            self.denoms2 = cumsum(q) - arange(len(q))
            self.p_search2[1:] /= self.denoms2

    def mollview(self, title=None):
        if title is None:
            if self.name is None:
                hp.mollview(self.hpx)
            else:
                hp.mollview(self.hpx, title=self.name)
        else:
            hp.mollview(self.hpx, title=title)
        hp.graticule(**{'color':'gray'})

    def px2celestial(self, ipx):
        """
        Convert pixel number to celestial coordinates (ra, dec), in degrees.
        """
        theta, phi = hp.pix2ang(self.nside, ipx)
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec

    def celestial2px(self, ra, dec):
        """
        Return the index for the pixel containing the specified direction,
        given in celestial coordinates (degrees).
        """
        phi = deg2rad(ra)
        theta = 0.5*pi - deg2rad(dec)
        return hp.ang2pix(self.nside, theta, phi)

    def px2hpd_level(self, ipx):
        """
        Return the probability content (level) of an HPD credible region
        just containing pixel `ipx`.
        """
        return self.hpd_levels_px[ipx]

    def hpdr_area(self, level):
        # Indicators for pixels in the specified HPD region:
        ind = self.hpd_levels_px <= level
        return sum(ind) * self.sd_px

    def mode_px(self):
        ipx = self.hpx.argmax()
        return ipx, self.hpx[ipx] / self.sd_px

    def mode_celestial(self):
        ipx, pdf = self.mode_px()
        return self.px2celestial(ipx), pdf

    def p_circle(self, ra, dec, radius):
        phi = deg2rad(ra)
        theta = 0.5*pi - deg2rad(dec)
        radius = deg2rad(radius)
        # Cartesian coordinate tuple:
        xyz = hp.ang2vec(theta, phi)
        px_disc = hp.query_disc(self.nside, xyz, radius)
        return self.hpx[px_disc].sum()

    def ex_search_effort(self):
        """
        Compute the expected search effort to discover the object location
        in a perfect, optimized search over individual pixels.

        Return the effort as the expected number of pixels, the area
        of those pixels (sq deg), and the level of an HPD region
        comprising those pixels.
        """
        # Candidate number of searches:
        n = arange(1, self.n_px+1)
        xn = sum(n*self.p_ddens)
        return xn, xn*self.sd_px, self.p_ddens[0:int(xn)+1].sum()

    def effort_histo(self, edges):
        """
        Compute a PMF for the search effort (# of perfect searches to
        discovery), binned according to the bin edges in `edges`.

        The PMF values will be normalized over the full possible number
        of searches (which may extend beyond the provided bin edges).
        """
        n_bins = len(edges) - 1
        n_search = arange(1, self.n_px+1)
        return histogram(n_search, bins=edges, weights=self.p_ddens)[0]
        # pmf = zeros(n_bins)

    def sim_search(self):
        """
        Return the number of searches until object discovery in a simulated
        perfect optimal search of the pixels.
        """
        if self.p_search is None:
            self._init_search()
        i = 0
        while True:
            u = random.rand()
            if u <= self.p_search[i]:
                break
            else:
                i += 1
        return i+1

    def sim_searches(self, n_searches, tlog=False):
        """
        Simulate `n_searches` perfect optimal searches of the pixels,
        returning results in a `MomentAccumulator` instance (including
        the full history of search efforts).

        The `tlog` flag controls whether runtime is logged to the terminal.
        """
        with Timer('Simulating searches', log=tlog) as timer:
            n_searches = n_searches
            accum = MomentAccumulator(hist=True)
            for i in range(n_searches):
                n = self.sim_search()
                accum.update(n)
                if (i+1)%500 == 0:
                    print(i, n)
            accum.done()
        return accum
