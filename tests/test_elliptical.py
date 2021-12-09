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
    # Define inputs.
    reps = 2
    lower_left = np.array([[0.5, 0.4, 0.1]] * reps)
    upper_left = np.array([[0.6, 0.3, 0.05]] * reps)
    C = np.linalg.norm(upper_left - lower_left, axis=-1)
    third_point = np.array([0, 0, 0.2])
    plane_normal = tie_pointing.derive_3d_plane_coefficients(lower_left[0], upper_left[0],
                                                             third_point)
    plane_normal = np.array([plane_normal[:3]] * reps)
    # Execute test.
    # Calculate isometry and inverse isometry.
    isometry, inverse_isometry = tie_pointing.compute_isometry_matrices(lower_left, upper_left,
                                                                        plane_normal)
    # Apply isometry to upper_left point.
    hom_upper_left = np.expand_dims(np.ones((reps, 4)), -1)
    hom_upper_left[:, :3, 0] = upper_left
    result = isometry @ hom_upper_left
    # Define expected result of transformation of upper_left point.
    expected = np.zeros((reps, 4))
    expected[:, -1] = 1
    expected[:, 1] = C
    expected = np.expand_dims(expected, -1)

    # Apply inverse transformation to result of forward transformation.
    # It should reproduce the input to the forward transform.
    inverse_result = inverse_isometry @ result

    # Assert results are as expected.
    assert np.allclose(result, expected)
    assert np.allclose(inverse_result, hom_upper_left)
