import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst

from tie_pointing import utils


def test_convert_to_homogenous_coords_locations():
    coords = np.array([[2, 2]]).T
    component_axis = 0
    coord_axis = 1
    homogenous_idx = 0
    hom = utils.convert_to_homogeneous_coords(
        coords, vector=False, component_axis=component_axis,
        coord_axis=coord_axis, homogeneous_idx=homogenous_idx)
    expected = np.array([[1, 2, 2]]).T
    assert np.allclose(hom, expected)


def test_convert_to_homogenous_coords_vectors():
    coords = np.array([[2, 2]])
    component_axis = 1
    coord_axis = 0
    homogenous_idx = 0
    output = utils.convert_to_homogeneous_coords(
        coords, vector=True, component_axis=component_axis,
        coord_axis=coord_axis, homogeneous_idx=homogenous_idx)
    expected = np.array([[0, 2, 2]])
    assert np.allclose(output, expected)


def test_hee_from_hee_xyz():
    expected_lat, expected_lon, expected_r = [-0.12072942]*u.rad, [-1.69241535]*u.rad, np.array([0.9133455])
    input_x, input_y, input_z = 0.11, 0.9, -0.11
    output_lat, output_lon, output_r = utils.hee_from_hee_xyz(input_x, input_y, input_z)
    return output_lat, expected_lat, output_lon, expected_lon, output_r, expected_r
    assert output_lat == expected_lat
    assert output_lon == expected_lon
    assert output_r == expected_r
