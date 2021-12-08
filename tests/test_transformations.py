import numpy as np

from tie_pointing import transformations


def test_rotate_plane_to_xy():
    points = np.array([[[1, 1, 1], [0.5, 0, 0], [4, -1, -1]],
                       [[1, 1, -1], [-7, 0, 0], [3, -1, 1]]])
    points = np.swapaxes(points, -1, -2)
    output_points, output_matrix = transformations.rotate_plane_to_xy(points)
    expected_points = np.array([[[1, -2**0.5, 0], [0.5, 0, 0], [4, 2**0.5, 0]],
                                [[1, 2**0.5, 0], [-7, 0, 0], [3, -2**0.5, 0]]])
    expected_points = np.swapaxes(expected_points, -1, -2)
    a = 2**(-0.5)
    expected_matrix = np.array([[[1, 0, 0],
                                 [0, -a, -a],
                                 [0, a, -a]],
                                [[1, 0, 0],
                                 [0, a, -a],
                                 [0, a, a]]])
    assert np.allclose(output_points, expected_points)
    assert np.allclose(output_matrix, expected_matrix)


def test_derive_plane_rotation_matrix():
    plane = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    new_plane = np.array([0, 1, 0])
    expected = np.stack([np.eye(3)]*3, axis=0)
    expected[0, 1:, 1:] = np.array([[0, 1], [-1, 0]])
    expected[1, :2, :2] = 2**(-0.5)
    expected[1, 0, 1] *= -1
    output = transformations.derive_plane_rotation_matrix(plane, new_plane)
    assert np.allclose(output, expected)


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
    output = transformations.derive_projective_collineation_from_five_points(
        points, images, point5, image5)
    assert np.allclose(output, expected)
