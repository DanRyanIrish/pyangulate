import numpy as np

from tie_pointing import utils
from tie_pointing.elliptical import quadrilateral


def test_compute_isometry_transform():
    n = 2
    ll = np.array([2, 0])
    ll = np.stack([ll]*n, axis=0)
    lr = np.array([10, -1])
    lr = np.stack([lr]*2, axis=0)
    ul = np.array([6, 4])
    ul = np.stack([ul]*2, axis=0)
    hom_vertices = utils.convert_to_homogeneous_coords(np.stack((ll, lr, ul), axis=-1), component_axis=-2)
    output, output_inv = quadrilateral.compute_isometry_transform(ll, lr, ul)
    expected = np.array([[1, 0, 0],
                         [-0.22222222, 0.11111111, -0.11111111],
                         [-0.05555556, 0.02777778, 0.22222222]])
    expected = np.stack([expected]*2, axis=0)
    expected_inv = np.linalg.inv(expected)
    assert np.allclose(output, expected)
    assert np.allclose(output_inv, expected_inv)
