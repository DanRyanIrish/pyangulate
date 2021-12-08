import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst


def test_inscribe_max_area_ellipse_in_parallelogram():
    vertices = np.array([[2, 0], [10, -1], [14, 3], [6, 4]]).T
    ellipse = tie_pointing.inscribe_max_area_ellipse_in_parallelogram(vertices)
    return ellipse
    assert np.allclose(ellipse(0), np.array([12, 1]))
    assert np.allclose(ellipse(np.pi/2), np.array([10, 3.5]))
    assert np.allclose(ellipse(np.pi), np.array([4, 2]))
    assert np.allclose(ellipse(-np.pi/2), np.array([6, -0.5]))
