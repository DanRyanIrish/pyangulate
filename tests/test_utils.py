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


def test_get_quadrilateral_slopes():
    vertices = np.array([[[3, 2, 1, 4],
                          [0, 3, 1, 2]],
                         [
                          [3, 2, 1, 4],
                          [0, 3, 1, 10]]])
    ll = vertices[..., 2]
    lr = vertices[..., 0]
    ur = vertices[..., 3]
    ul = vertices[..., 1]
    # Run test
    out_m_ll_lr, out_m_lr_ur, out_m_ur_ul, out_m_ul_ll = utils.get_quadrilateral_slopes(
        ll, lr, ur, ul)
    expected_m_ll_lr = np.array([-0.5, -0.5])
    expected_m_lr_ur = np.array([2, 10])
    expected_m_ur_ul = np.array([-0.5, 3.5])
    expected_m_ul_ll = np.array([2, 2])
    assert (out_m_ll_lr == expected_m_ll_lr).all()
    assert (out_m_lr_ur == expected_m_lr_ur).all()
    assert (out_m_ur_ul == expected_m_ur_ul).all()
    assert (out_m_ul_ll == expected_m_ul_ll).all()


def test_is_parallelogram():
    vertices = np.array([[[3, 2, 1, 4],
                          [0, 3, 1, 2]],
                         [
                          [3, 2, 1, 4],
                          [0, 3, 1, 10]]])
    expected = np.array([True, False])
    output = utils.is_parallelogram(vertices, keepdims=False)
    assert (output == expected).all()
    # Also test keepdims=True case
    expected = np.ones(vertices.shape, dtype=bool)
    expected[1][:] = False
    output = utils.is_parallelogram(vertices, keepdims=True)
    assert (output == expected).all()


def test_repeat_over_new_axes():
    data = np.arange(12).reshape(3, 4)
    output = utils.repeat_over_new_axes(data, np.array([0, 0]), np.array([2, 3]))
    expected = np.stack([data]*3, axis=0)
    expected = np.stack([expected]*2, axis=0)
    assert np.allclose(output, expected)


def test_add_z_to_xy():
    cols = 4
    xy = np.ones((2, cols))
    output = utils.add_z_to_xy(xy, 0)
    expected = np.ones((3, cols))
    expected[2] = np.zeros(cols)
    assert np.allclose(output, expected)


def test_convert_to_homogeneous_coords_locations():
    coords = np.array([[2, 2]]).T
    component_axis = 0
    coord_axis = 1
    trailing_convention = False
    hom = utils.convert_to_homogeneous_coords(
        coords, vector=False, component_axis=component_axis,
        trailing_convention=trailing_convention)
    expected = np.array([[1, 2, 2]]).T
    assert np.allclose(hom, expected)


def test_convert_to_homogeneous_coords_vectors():
    coords = np.array([[2, 2]])
    component_axis = 1
    coord_axis = 0
    trailing_convention = False
    output = utils.convert_to_homogeneous_coords(
        coords, vector=True, component_axis=component_axis,
        trailing_convention=trailing_convention)
    expected = np.array([[0, 2, 2]])
    assert np.allclose(output, expected)


def test_hee_from_hee_xyz():
    expected_lon = [1.4491773]*u.rad
    expected_lat = [-0.12072942]*u.rad
    expected_r = np.array([0.9133455])
    input_x, input_y, input_z = 0.11, 0.9, -0.11
    output_lon, output_lat, output_r = utils.hee_from_hee_xyz(input_x, input_y, input_z)
    assert np.allclose(output_lon.value, expected_lon.to_value(output_lon.unit))
    assert np.allclose(output_lat.value, expected_lat.to_value(output_lat.unit))
    assert np.allclose(output_r, expected_r)
