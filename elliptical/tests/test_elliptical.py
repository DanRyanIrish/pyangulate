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
    input_vertices = np.array([[2, 10, 14, 6],
                               [0, -0.89442719, 2.68328157, 3.57770876],
                               [0, 0.4472136, -1.34164079, -1.78885438]])
    input_vertices = np.stack([input_vertices]*2, axis=0)
    expected_vertices = np.array([[8, 12.46525045, 8.24806947],
                                  [1.34164078, 1.84087096, -0.43339984],
                                  [-0.6708204, -0.92043549, 0.21669992]])
    expected_vertices = np.stack([expected_vertices]*2, axis=0)
    # Test when input vertices > 2D
    output_vertices = elliptical.inscribe_ellipse_in_3d(input_vertices)
    assert np.allclose(output_vertices, expected_vertices)
    # Test when input vertices are 2-D
    output_vertices0 = elliptical.inscribe_ellipse_in_3d(input_vertices[0])
    assert np.allclose(output_vertices0, expected_vertices[0])


def test_inscribe_ellipse_in_3d_quad():
    input_vertices = np.array([[2, 10, 14, 6],
                               [0, -0.89442719, 2.68328157, 3.57770876],
                               [0, 0.4472136, -1.34164079, -1.78885438]])
    input_vertices = np.stack([input_vertices]*2, axis=0)
    expected_vertices = np.array([[8, 12.46525045, 8.24806947],
                                  [1.34164078, 1.84087096, -0.43339984],
                                  [-0.6708204, -0.92043549, 0.21669992]])
    expected_vertices = np.stack([expected_vertices]*2, axis=0)
    # Test when input vertices > 2D
    output_vertices = elliptical.inscribe_ellipse_in_3d(input_vertices)
    assert np.allclose(output_vertices, expected_vertices)
    # Test when input vertices are 2-D
    output_vertices0 = elliptical.inscribe_ellipse_in_3d(input_vertices[0])
    assert np.allclose(output_vertices0, expected_vertices[0])


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
    output_center, output_major, output_minor = elliptical.inscribe_ellipse(vertices)
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
