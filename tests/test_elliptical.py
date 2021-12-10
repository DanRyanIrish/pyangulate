import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst

from tie_pointing import elliptical


def test_inscribe_ellipse_para():
    pass


def test_inscribe_ellipse_quad():
    pass


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


def test_get_equation_of_max_area_ellipse_in_parallelogram():
    vertices = np.stack([np.array([[2, 0], [10, -1], [14, 3], [6, 4]]).T]*2, axis=0)
    output = elliptical.get_equation_of_max_area_ellipse_in_parallelogram(vertices)
    h, k = np.array([8., 8.]), np.array([1.5, 1.5])
    a, b, c, d = np.array([4., 4.]), np.array([2., 2.]), np.array([-0.5, -0.5]), np.array([2., 2.])
    assert np.allclose(output[0], h)
    assert np.allclose(output[1], k)
    assert np.allclose(output[2], a)
    assert np.allclose(output[3], b)
    assert np.allclose(output[4], c)
    assert np.allclose(output[5], d)


def test_get_ellipse_semi_axes_coords():
    h, k, = np.array([8., 7.]), np.array([1.5, 0.5])
    a, b, c, d = np.array([4., 4.]), np.array([2., 2.]), np.array([-0.5, -0.5]), np.array([2., 2.])
    output_major, output_minor = elliptical.get_ellipse_semi_axes_coords(h, k, a, b, c, d)
    expected_major = np.array([[12.46525045,  2.05815631], [11.46525045,  1.05815631]]).T
    expected_minor = np.array([[ 8.24806947, -0.48455575], [ 7.24806947, -1.48455575]]).T
    assert np.allclose(output_major, expected_major)
    assert np.allclose(output_minor, expected_minor)


def test_compute_isometry_matrix():
    new_origin = np.array([[2, 1], [-2, 1], [-2, -1], [2, -1]])
    new_y = np.array([[6, 4], [-6, 4], [6, 4], [-6, -4]])
    output, output_inv = elliptical.compute_isometry_matrices(new_origin, new_y)
    expected = np.array([[[1, 0, 0],
                          [-0.4, 0.6, -0.8],
                          [-2.2, 0.8, 0.6]],
                         [[1, 0, 0],
                          [0.4, 0.6, 0.8],
                          [-2.2, -0.8, 0.6]],
                         [[1, 0, 0],
                          [0.21199958, 0.52999894, -0.8479983],
                          [2.22599555, 0.8479983, 0.52999894]],
                         [[1, 0, 0],
                          [1.63857606, -0.35112344, 0.93632918],
                          [1.52153491, -0.93632918, -0.35112344]]])
    expected_inv = np.array([[[1, 0,  0],
                              [2, 0.6, 0.8],
                              [1, -0.8, 0.6]],
                             [[1, 0, 0],
                              [-2, 0.6, -0.8],
                              [1, 0.8, 0.6]],
                             [[1, 0, 0],
                              [-2, 0.52999894, 0.8479983],
                              [-1, -0.8479983, 0.52999894]],
                             [[1, 0, 0],
                              [2, -0.35112344, -0.93632918],
                              [-1, 0.93632918, -0.35112344]]])
    assert np.allclose(output, expected)
    assert np.allclose(output_inv, expected_inv)


def test_inscribe_ellipse_in_quadrilateral():
    pass


def test_compute_isometry():
    pass
