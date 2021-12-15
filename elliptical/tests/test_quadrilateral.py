


def test_compute_isometry_transform():
    new_origin = np.array([[2, 1], [-2, 1], [-2, -1], [2, -1]])
    new_y = np.array([[6, 4], [-6, 4], [6, 4], [-6, -4]])
    output, output_inv = elliptical.compute_isometry_transform(new_origin, new_y)
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


def test_inscribe_ellipses_in_quadrilaterals():
    l = 2
    input_shape = (l, 1)
    A = np.array([1]*l).reshape(input_shape)
    B = np.array([-2]*l).reshape(input_shape)
    C = np.array([1]*l).reshape(input_shape)
    s = np.array([4]*l).reshape(input_shape)
    t = np.array([9]*l).reshape(input_shape)
    h_fit, k_fit, a_fit, b_fit, theta_fit = quadrilateral.inscribe_ellipses_in_quadrilaterals(
        A, B, C, s, t)
    expected_h = np.array([1.00930847]).reshape(input_shape)
    expected_k = np.array([1.19769491]).reshape(input_shape)
    expected_a = np.array([1.7670193]).reshape(input_shape)
    expected_b = np.array([0.80730511]).reshape(input_shape)
    expected_theta = np.array([1.17515864]).reshape(input_shape)
    assert np.allclose(h_fit, expected_h)
    assert np.allclose(k_fit, expected_k)
    assert np.allclose(a_fit, expected_a)
    assert np.allclose(b_fit, expected_b)
    assert np.allclose(theta_fit, expected_theta)

