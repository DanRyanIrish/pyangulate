import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst

from tie_pointing import transformations
from tie_pointing.elliptical import elliptical


def test_inscribe_ellipse_in_3d_para():
    vertices = np.stack([np.array([[2, 0, 0], [10, -1, 0], [14, 3, 0], [6, 4, 0]]).T]*2, axis=0)
    R = transformations.derive_plane_rotation_matrix(np.array([0, 0, 1]), np.array([0, 0.5, 1]))
    vertices = R @ vertices
    output_vertices = elliptical.inscribe_ellipse_in_3d(vertices)
    expected_center = np.array([[8, 8], [1.5, 1.5]])
    expected_major = np.array([[12.46525045, 12.46525045], [2.05815631,  2.05815631]])
    expected_minor = np.array([[8.24806947,  8.24806947], [-0.48455575, -0.48455575]])
    expected_vertices = np.stack((expected_center, expected_major, expected_minor), axis=-1)
    return output_vertices, expected_vertices
    assert np.allclose(output_center, expected_center)
    assert np.allclose(output_major, expected_major)
    assert np.allclose(output_minor, expected_minor)


def test_inscribe_ellipse_para():
    vertices = np.stack([np.array([[2, 0], [10, -1], [14, 3], [6, 4]]).T]*2, axis=0)
    output_vertices = elliptical.inscribe_ellipse(vertices)
    expected_center = np.array([[8, 8], [1.5, 1.5]])
    expected_major = np.array([[12.46525045, 12.46525045], [2.05815631,  2.05815631]])
    expected_minor = np.array([[8.24806947,  8.24806947], [-0.48455575, -0.48455575]])
    expected_vertices = np.stack((expected_center, expected_major, expected_minor), axis=-1)
    assert np.allclose(output_vertices, expected_vertices)


def test_inscribe_ellipse_quad():
    vertices = np.stack([np.array([[2, 0], [10, -1], [16, 7], [6, 4]]).T]*2, axis=0)
    output_center, output_major, output_minor = \
        elliptical.inscribe_ellipse(vertices)
    expected_center = np.array([[3.24945734, 2.40058265], [3.24945734, 2.40058265]])
    expected_major = np.array([[5.77585523, 5.98216733], [5.77585523, 5.98216733]])
    expected_minor = np.array([[1.20586074, 3.84210605], [1.20586074, 3.84210605]])
    assert np.allclose(output_center, expected_center)
    assert np.allclose(output_major, expected_major)
    assert np.allclose(output_minor, expected_minor)


def test_identify_vertices():
    vertices = np.array([[2, 0], [10, -1], [14, 3], [6, 4]]).T
    vertices = np.stack((vertices, vertices[:, ::-1]), axis=0)
    ll, lr, ur, ul = elliptical._identify_vertices(vertices)
    expected_ll = np.array([[2, 0], [2, 0]])
    expected_lr = np.array([[10, -1], [10, -1]])
    expected_ur = np.array([[14, 3], [14, 3]])
    expected_ul = np.array([[6, 4], [6, 4]])
    assert (ll == expected_ll).all()
    assert (lr == expected_lr).all()
    assert (ur == expected_ur).all()
    assert (ul == expected_ul).all()


def test_inscribe_max_area_ellipse_in_parallelogram():
    vertices = np.stack([np.array([[2, 0], [10, -1], [14, 3], [6, 4]]).T]*2, axis=0)
    output_center, output_major, output_minor = \
        elliptical.inscribe_max_area_ellipse_in_parallelogram(vertices)
    expected_center = np.array([[8, 8], [1.5, 1.5]])
    expected_major = np.array([[12.46525045, 12.46525045], [2.05815631,  2.05815631]])
    expected_minor = np.array([[8.24806947,  8.24806947], [-0.48455575, -0.48455575]])
    assert np.allclose(output_center, expected_center)
    assert np.allclose(output_major, expected_major)
    assert np.allclose(output_minor, expected_minor)


def test_inscribe_ellipse_in_quadrilateral():
    vertices = np.stack([np.array([[2, 0], [10, -1], [16, 7], [6, 4]]).T]*2, axis=0)
    output_center, output_major, output_minor = \
        elliptical.inscribe_ellipse_in_quadrilateral(vertices)
    expected_center = np.array([[3.24945734, 2.40058265], [3.24945734, 2.40058265]])
    expected_major = np.array([[5.77585523, 5.98216733], [5.77585523, 5.98216733]])
    expected_minor = np.array([[1.20586074, 3.84210605], [1.20586074, 3.84210605]])
    assert np.allclose(output_center, expected_center)
    assert np.allclose(output_major, expected_major)
    assert np.allclose(output_minor, expected_minor)
