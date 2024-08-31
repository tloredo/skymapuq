# SkyMapUQ - Sky Map Uncertainty Quantification

This Python package provides a class, `SkyMapUQ`, providing a variety of methods for quantifying uncertainty in a HEALPix sky map describing an object's or event's sky location uncertainty via a posterior probability density function (PDF) on the sky, stored as a probability mass function (PMF) across HEALPix pixels (that is, the sky map stores values of the PDF integrated over each pixel).

Uncertainty quantifications:

* Highest posterior density (HPD) credible regions
* Expected search effort: The area of the sky one would expect to search to discover the object in a perfect optimal search
* Information-equivalent area: Area of a region with a uniform PDF inside it (and zero PDF outside) for which the information gain (with respect to an all-sky uniform prior) is the same as the information gain in the posterior PDF.

The `SkyMapUQ` team includes:

* Tom Loredo, Cornell Center for Astrophysics and Planetary Science, and Department of Statistics and Data Science, Cornell University (lead developer)
* Tamás Budavári, Department of Applied Mathematics and Statistics, Johns Hopkins University
* Matthew Graham, Department of Astronomy, Caltech

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
