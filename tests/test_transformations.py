import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst

from tie_pointing import tie_pointing


def test_derive_projective_collineation_from_five_points():
    points = np.array([[[1, 1, 1, 1], [2, 10, 14, 6], [0, -1, 3, 4]],
                       [[1, 1, 1, 1], [2, 10, 14, 6], [0, -1, 3, 4]]])
    images = np.array([[[1, 1, 1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]],
                       [[1, 1, 1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]]])
    point5 = np.array([[[1], [8], [1.5]],
                       [[1], [8], [1.5]]])
    image5 = np.array([[[1], [0], [0]],
                       [[1], [0], [0]]])
    expected = np.array([[[18, 0, 0], [-26, 4, -4], [-20, 1, 8]],
                         [[18, 0, 0], [-26, 4, -4], [-20, 1, 8]]]) / 18
    output = tie_pointing.derive_projective_collineation_from_five_points(
        points, images, point5, image5)
    #return output, expected[0], points, images, point5, image5
    assert np.allclose(output, expected)
