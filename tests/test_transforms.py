import numpy as np

from tie_pointing import transforms, utils


def test_transform_to_xy_plane():
    points = np.array([[[1, 1, 5], [0.5, 0, 4], [4, -1, 3]],
                       [[1, 1, 2], [-7, 0, 3], [3, -1, 4]]])

    points = np.swapaxes(points, -1, -2)
    output_points, output_matrix = transforms.transform_to_xy_plane(points)
    expected_points = np.array([[
                                 [1, 1, 1],
                                 [1, 0.5, 4],
                                 [4.24264069, 2.82842712, 1.41421356],
                                 [0, 0, 0]],
                                [
                                 [1, 1, 1],
                                 [1, -7, 3],
                                 [-0.70710678, -2.12132034, -3.53553391],
                                 [0, 0, 0]]])
    theta0 = -np.pi/4
    theta1 = np.pi/4
    expected_matrix = np.eye(4)
    expected_matrix = utils.repeat_over_new_axes(expected_matrix, 0, 2)
    expected_matrix[0, 2:, 2:] = transforms.rotation_matrix_2d(theta0)
    expected_matrix[1, 2:, 2:] = transforms.rotation_matrix_2d(theta1)
    expected_matrix[0, -1, 0] = -2.82842712
    expected_matrix[1, -1, 0] = -2.12132034
    #return output_points, expected_points, output_matrix, expected_matrix
    assert np.allclose(output_points, expected_points)
    assert np.allclose(output_matrix, expected_matrix)


def test_derive_plane_rotation_matrix():
    plane = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    new_plane = np.array([0, 1, 0])
    expected = np.stack([np.eye(3)]*3, axis=0)
    expected[0, 1:, 1:] = np.array([[0, 1], [-1, 0]])
    expected[1, :2, :2] = 2**(-0.5)
    expected[1, 0, 1] *= -1
    output = transforms.derive_plane_rotation_matrix(plane, new_plane)
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
    output = transforms.derive_projective_collineation_from_five_points(
        points, images, point5, image5)
    assert np.allclose(output, expected)
