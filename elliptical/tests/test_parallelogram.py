

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
