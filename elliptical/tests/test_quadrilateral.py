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
