import copy
from functools import partial

import numpy as np
import scipy.optimize

from tie_pointing import utils


def compute_isometry_transform(new_origin, new_y, trailing_convention=False):
    """
    Calculate the affine transformation matrix that does the following:
    1. translates by -new_origin (i.e. the origin is moved to new_origin);
    2. rotates the line from the new_origin to new_y so it is aligned with the y-axis.

    Also calculate the inverse affine transform.

    Parameters
    ----------
    new_origin: `numpy.ndarray`
        The point which serves as the new origin, i.e. the matrix translates
        by subtracting this point.
        The last dimension must be length 2 and represent the coordinate's x-y components.
        Other dimensions are assumed to represent other transformations and are
        broadcast through the calculation.
    new_y: `numpy.ndarray`
        The point which, along with new_origin, defines the new y-axis.
        Must be same shape as new_origin and dimensions are interpretted in the same way.
    trailing_convention: `bool`
        Determines whether the non-physical dimension in the affine transformation will be
        the first or last dimension.

    Returns
    -------
    isometry: `numpy.ndarray`
        The affine transformation matrix.
        Will have N+1 dimensions where N is the number of dimensions in the input.
        Dimensions 0,..,N-1 will have the same shape has the corresponding dimensions
        in the inputs.
        The final two dimensions will represent 3x3 matrices, i.e. affine transformations.
        See Notes for more.

    inverse_isometry: `numpy.ndarray`
        The inverse affine transformation.
        Same shape and interpretation of dimensions as the forward transform.

    Notes
    -----
    Affine transformations combine linear transformations such as
    rotation, reflection, shearing etc., with translations which cannot be achieved
    via NxN (2x2) matrix multiplication. They do this by adding a N+1th (3rd) row and
    column where the extra column represents the translation, and the extra row is all 0
    except its value corresponding to the extra column, which is 1.
    There are two conventions: either the extra row and column can be placed as the 1st
    row/column or the last row/column. N-D points are then made compatible with these
    transformation by adding a 1 at the start/end of the coordinate depending on
    the leading or trailing convention is used for the affine tranaformation.
    Coordinates with this extra theoretical dimension are known as
    "homogeneous coordinates". This way of representing translation is the same as
    shearing (a linear transformation) in a N+1th dimensions and then projecting the new
    position onto the N-D plane, i.e. the value to the N+1th coordinate is 0.

    References
    ----------
    Computer Graphics 2012, Lect. 5 - Linear and Affine Transformations
    Part 1: https://www.youtube.com/watch?v=heVndBwDffI
    Part 2: https://www.youtube.com/watch?v=7z1rs9Di77s&list=PLDFA8FCF0017504DE&index=11
    """
    # Sanitize inputs
    input_shape = new_origin.shape
    if ((new_y.shape != input_shape) or (new_origin.shape[-1] != new_y.shape[-1] != 2)):
        raise ValueError("All inputs must have same shape and final axes must be length 2. "
                         f"new_origin: {new_origin.shape}. "
                         f"new_y: {new_y.shape}. ")
    if trailing_convention:
        hom_idx = -1
        mat_slice = slice(0, 2)
    else:
        hom_idx = 0
        mat_slice = slice(1, 3)
    tiled_identity = np.eye(3)
    if len(input_shape) > 1:
        tiled_identity = utils.repeat_over_new_axes(
            tiled_identity, [0]*(len(input_shape)-1), input_shape[:-1])
    # Calculate matrix that shifts origin to new_origin.
    T = copy.deepcopy(tiled_identity)
    T[..., hom_idx] = utils.convert_to_homogeneous_coords(
        -new_origin, trailing_convention=trailing_convention)
    T_inv = copy.deepcopy(tiled_identity)
    T_inv[..., hom_idx] = utils.convert_to_homogeneous_coords(
        new_origin, trailing_convention=trailing_convention)

    # Calculate rotation matrix that brings line from new_origin to new_y in line with y_axis.
    shifted_y = new_y - new_origin
    theta = -np.arctan(shifted_y[..., 1] / shifted_y[..., 0]) + np.pi/2
    if not np.isscalar(theta):
        theta[shifted_y[..., 0] < 0] += np.pi
    elif shifted_y[0] < 0:
        theta += np.pi
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    r_inv = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
    if r.ndim > 2:
        # If R has more that 2 dimensions, move its 2 matrix dimensions to last 2 axes.
        r = np.moveaxis(r, 0, -1)
        r = np.moveaxis(r, 0, -1)
        r_inv = np.moveaxis(r_inv, 0, -1)
        r_inv = np.moveaxis(r_inv, 0, -1)
    R = copy.deepcopy(tiled_identity)
    R[..., mat_slice, mat_slice] = r
    R_inv = copy.deepcopy(tiled_identity)
    R_inv[..., mat_slice, mat_slice] = r_inv

    # Combine translation and rotation into affine transform.
    A = R @ T
    A_inv = T_inv @ R_inv
    return A, A_inv


def parametric_ellipse_angled(h, k, a, b, theta, phi):
    """
    Return x-y coordinate on the point on an ellipse at an angle phi from the semi-major axis.

    Parameters
    ----------
    h, k: `numpy.ndarray`
        The center of the ellipse(s)
    a, b: `numpy.ndarray`
        The semi-major and semi_minor axes of the ellipse.
    theta: `numpy.ndarray`
        The rotation of the major axis from the x-axis.
    phi: `numpy.ndarray`
        The angle formed by the semi-major axis and the line segment defined by the ellipse
        center and desired point.

    Returns
    -------
    x, y: `numpy.ndarray`
        The x and y coordinates of the point on the ellipse.
    """
    x = h + a * np.cos(phi) * np.cos(theta) - b * np.sin(phi) * np.sin(theta)
    y = k + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
    return x, y


def inscribe_ellipses_in_quadrilaterals(A, B, C, s, t):
    """
    Calculate parameters of ellipse(s) tangent to all sides of a quadrilateral(s).

    The quadrilateral must not be a parallelogram and the left and right sides must not be parallel.
    The lower left vertex must be at the origin, while the upper left must be on the y-axis.
    Algorithm is based on [1], [2] and [3] (see references section).
    Kwargs are passed to scipy.optimize.minimize.

    Parameters
    ----------
    A, B: `float` or `numpy.ndarray`
        The x and y coordinates of the lower right vertex
    C: `float` or `numpy.ndarray`
        The y coordinate of the upper left vertex. The x coord must be 0.
        If array, must be broadcastable other inputs.
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.

    Returns
    -------
    h, k: `float` or `numpy.ndarray`
        The x and y coordinate of the ellipse center.
    a, b: `float` or `numpy.ndarray`
        The semi-major and semi-minor axes, respectively.
    theta:
        The rotation angle of the ellipse from the x-axis.

    References
    ----------
    [1] Byrne, J.P et al.
    Propagation of an Earth-directed coronal mass ejection in three dimensions
    Nat. Comms, DOI: 10.1038/ncomms1077, (2010)
    [2] Horwitz, A.
    Finding ellipses and hyperbolas tangent to two, three, or four given lines.
    Southwest J. Pure Appl. Math. 1, 6–32 (2002)
    [3] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    # Sanitize input.
    shape_error_msg = "inputs must all be same shape"
    scalar_input = np.isscalar(A)
    if scalar_input:
        if not (np.isscalar(B) and np.isscalar(C) and np.isscalar(s) and np.isscalar(t)):
            raise ValueError(shape_error_msg)
        A = np.array([A])
        B = np.array([B])
        C = np.array([C])
        s = np.array([s])
        t = np.array([t])
    input_shape = A.shape
    if input_shape != B.shape != C.shape != s.shape != t.shape:
        raise ValueError(shape_error_msg)
    # Define arrays to hold output.
    input_size = A.size
    bad_value = np.nan
    h = np.empty(input_size, dtype=float)
    h[:] = bad_value
    k = np.empty(input_size, dtype=float)
    k[:] = bad_value
    a = np.empty(input_size, dtype=float)
    a[:] = bad_value
    b = np.empty(input_size, dtype=float)
    b[:] = bad_value
    theta = np.empty(input_size, dtype=float)
    theta[:] = bad_value
    # For each set of vertices, derive ellipse parameters.
    for i, (A_i, B_i, C_i, s_i, t_i) in enumerate(zip(A.flatten(), B.flatten(), C.flatten(),
                                                      s.flatten(), t.flatten())):
        h_i, k_i, a_i, b_i, theta_i, fit_successful = inscribe_ellipse_in_single_quadrilateral(
            A_i, B_i, C_i, s_i, t_i)
        if fit_successful:
            h[i] = h_i
            k[i] = k_i
            a[i] = a_i
            b[i] = b_i
            theta[i] = theta_i
    if scalar_input:
        h = h[0]
        k = k[0]
        a = a[0]
        b = b[0]
    else:
        h = h.reshape(input_shape)
        k = k.reshape(input_shape)
        a = a.reshape(input_shape)
        b = b.reshape(input_shape)
        theta = theta.reshape(input_shape)
    return h, k, a, b, theta


def inscribe_ellipse_in_single_quadrilateral(A, B, C, s, t, **kwargs):
    """
    Calculate parameters of ellipse tangent to all sides of a quadrilateral.

    The quadrilateral must not be a parallelogram and the left and right sides must not be parallel.
    The lower left vertex must be at the origin, while the upper left must be on the y-axis.
    Algorithm is based on [1], [2] and [3] (see references section).
    Kwargs are passed to scipy.optimize.minimize.

    Parameters
    ----------
    A, B: `float` or `numpy.ndarray`
        The x and y coordinates of the lower right vertex
    C: `float` or `numpy.ndarray`
        The y coordinate of the upper left vertex. The x coord must be 0.
        If array, must be broadcastable other inputs.
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.

    Returns
    -------
    h, k: `float`
        The x and y coordinate of the ellipse center.
    a, b: `float`
        The semi-major and semi-minor axes, respectively.
    theta: `float`
        The rotation angle of the ellipse from the x-axis.
    fit_successful: `bool`
        True if the residuals have been minimized to 0.  False otherwise.

    References
    ----------
    [1] Byrne, J.P et al.
    Propagation of an Earth-directed coronal mass ejection in three dimensions
    Nat. Comms, DOI: 10.1038/ncomms1077, (2010)
    [2] Horwitz, A.
    Finding ellipses and hyperbolas tangent to two, three, or four given lines.
    Southwest J. Pure Appl. Math. 1, 6–32 (2002)
    [3] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    # Derive bounds
    bounds = kwargs.pop("bounds", None)
    if bounds is None:
        mu1_x = (C + A) / 2
        mu2_x = s / 2
        mu_x = [mu1_x, mu2_x]
        h_bounds = (min(mu_x), max(mu_x))
        theta_bounds = (0, np.pi)
        bounds = (h_bounds, theta_bounds)
    # Derive initial guesses.
    x0 = kwargs.pop("x0", None)
    if x0 is None:
        h_init = np.linspace(h_bounds[0], h_bounds[1], 100)
        h_init = h_init[np.logical_and(h_init != A/2, h_init != s/2)]
        h_init = h_init.reshape(len(h_init), 1)
        theta_init = np.linspace(0, np.pi, 100)
        theta_init = theta_init.reshape(1, len(theta_init))
        resid = _ellipse_tangent_residuals(A, B, C, s, t, [h_init, theta_init])
        idx = np.where(resid == resid.min())
        x0 = (h_init[:, 0][idx[0]], theta_init[0][idx[1]])
    # Find optimal h and theta.
    func = partial(_ellipse_tangent_residuals, A, B, C, s, t)
    optimize_result = scipy.optimize.minimize(func, x0, bounds=bounds, **kwargs)
    h, theta = optimize_result.x
    fit_successful = np.isclose(optimize_result.fun, 0)
    # Derive corresponding ellipse parameters
    k, a, b = _derive_ellipse_axes(A, B, C, s, t, h)
    return h, k, a, b, theta, fit_successful


def _ellipse_tangent_residuals(A, B, C, s, t, x):
    h, theta = x[0], x[1]
    m0 = B / A
    c0 = B - m0 * A
    m1 = (t - B) / (s - A)
    c1 = t - m1 * s
    m2 = (C - t) / (-s)
    c2 = C
    delta0 = _ellipse_tangent_discriminant(A, B, C, s, t, m0, c0, h, theta)
    delta1 = _ellipse_tangent_discriminant(A, B, C, s, t, m1, c1, h, theta)
    delta2 = _ellipse_tangent_discriminant(A, B, C, s, t, m2, c2, h, theta)
    resid = delta0**2 + delta1**2 + delta2**2
    return resid


def _ellipse_tangent_discriminant(A, B, C, s, t, m, c, h, theta):
    """
    Calculate the discriminant, which when zero, means the ellipse is tangent to the line.

    Parameters
    ----------
    A, B: `float` or `numpy.ndarray`
        The x and y coordinates of the lower right vertex
    C: `float` or `numpy.ndarray`
        The y coordinate of the upper left vertex. The x coord must be 0.
        If array, must be broadcastable other inputs.
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.
    h:
        The x coordinate of the ellipse center.
        The y coordinate, k, is derived from h as it must lie on the line segment
        joining the midpoints of the quadrilateral diagonals.
        If array, must be broadcastable other inputs.

    Returns
    -------
    k: `float` or `numpy.ndarray`
        The y coordinate of the ellipse center.
    a, b: `float` or `numpy.ndarray`
        The semi-major and semi-minor axes, respectively.
    """
    k, a, b = _derive_ellipse_axes(A, B, C, s, t, h)
    sin = np.sin(theta)
    cos = np.cos(theta)
    kappa = c - k
    alpha = ((cos**2 + 2 * m * cos * sin + m**2 * sin**2) / a**2
             + (sin**2 - 2 * m * cos * sin + m**2 * cos**2) / b**2)
    beta = ((2 * m * kappa * sin**2 + 2 * (kappa - m * h) * cos * sin - 2 * h * cos**2) / a**2
            + (2 * kappa * m * cos**2 - 2 * (kappa - m * h) * cos * sin - 2 * h * sin**2) / b**2)
    gamma = ((h**2 * cos**2 - 2 * h * kappa * cos * sin + kappa**2 * sin**2) / a**2
             + (h**2 * sin**2 + 2 * h * kappa * cos * sin + kappa**2 * cos**2) / b**2 - 1)
    return beta**2 - 4 * alpha * gamma


def _derive_ellipse_axes(A, B, C, s, t, h):
    if s == A:
        raise ValueError("Invalid vertices. Right side must not be vertical.")
    h_bounds = [s/2, (C + A) / 2]
    if np.isscalar(h):
        h_outofbounds = h < min(h_bounds) or h > max(h_bounds)
    else:
        h_outofbounds = np.logical_or(h < min(h_bounds), h > max(h_bounds)).any()
    if h_outofbounds:
        raise ValueError(f"h must lie between the x coords of the midpoints of diagonals: h={h}; range={(min(h_bounds), max(h_bounds))}")
    s_A = s - A
    k = (h - s/2) * (t - B - C) / s_A + t/2
    h_A_2 = (h - A) / 2
    r1 = (
            4 * (s_A**2 - (t - B - C)**2) * h_A_2**2
            + 4 * s_A * (A*s_A + B*(B - t) + C*(C - t)) * h_A_2
            + s_A**2 * (A**2 - (C - B)**2)
            )
    r2 = (
            8 * (t - B - C) * s_A * h_A_2**2
            + 4 * s_A * (A*t + C*s + B*s - 2*A*B) * h_A_2
            + 2 * A * s_A**2 * (B - C)
            )
    u_h = r1**2 + r2**2
    R = np.sqrt(u_h / (16 * s_A**4))
    W = 0.25 * (C / s_A**2) * (2 * (B*s - A*(t - C))*h - A*C*s) * (2*h - A) * (2*h - s)
    a = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) + R))
    b = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) - R))
    return k, a, b


