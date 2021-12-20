import copy
from functools import partial

import numpy as np
import scipy.optimize

from tie_pointing import utils


def compute_isometry_transform(ll, lr, ul, trailing_convention=False):
    """
    Calculate the affine transformation matrix that does the following:
    1. translates by -new_origin (i.e. the origin is moved to new_origin);
    2. rotates the line from the new_origin to new_y so it is aligned with the y-axis;
    3. Shears in the y-direction so the bottom side lies along the x-axis;
    4. Scales in x and y so the lower right vertes sits at (1, 0) and the
    upper left sits at (0, 1).

    Also calculate the inverse affine transform.

    Parameters
    ----------
    ll, lr, ul:`numpy.ndarray`
        The lower left, lower right and upper left vertices. Must all be same shape.
        Can have any number of dimensions but last one must be length-2 and represent
        the x-y coordinates of the vertices.  Other dimensions are assumed to represent
        other sets of vertices reuqiring their own transform.
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
    input_shape = ll.shape
    if ((lr.shape != ul.shape != input_shape) or (ll.shape[-1] != lr.shape[-1] != ul.shape != 2)):
        raise ValueError("All inputs must have same shape and final axes must be length 2. "
                         f"ll: {ll.shape}. "
                         f"lr: {lr.shape}. "
                         f"ul: {ul.shape}.")
    # Convert vertices to homogeneous coords.
    if trailing_convention:
        hom_idx = -1
        mat_slice = slice(0, 2)
    else:
        hom_idx = 0
        mat_slice = slice(1, 3)
    vertices = np.stack((ll, lr, ul), axis=-1)
    component_axis = -2
    hom_vertices = utils.convert_to_homogeneous_coords(vertices, component_axis=component_axis,
                                                       trailing_convention=trailing_convention)
    tiled_identity = np.eye(3)
    if len(input_shape) > 1:
        tiled_identity = utils.repeat_over_new_axes(
            tiled_identity, [0]*(len(input_shape)-1), input_shape[:-1])
    # Calculate matrix that shifts origin to new_origin.
    T = copy.deepcopy(tiled_identity)
    T[..., hom_idx] = utils.convert_to_homogeneous_coords(
        -ll, trailing_convention=trailing_convention)
    new_vertices = (T @ hom_vertices)[..., mat_slice, :]

    # Calculate rotation matrix that brings line from lower left to upper left vertices
    # in line with y_axis.
    shifted_y = new_vertices[..., 2] - new_vertices[..., 0]
    theta = -np.arctan(shifted_y[..., 1] / shifted_y[..., 0]) + np.pi/2
    if not np.isscalar(theta):
        theta[shifted_y[..., 0] < 0] += np.pi
    elif shifted_y[0] < 0:
        theta += np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    if R.ndim > 2:
        # If R has more than 2 dimensions, move its 2 matrix dimensions to last 2 axes.
        R = np.moveaxis(R, 0, -1)
        R = np.moveaxis(R, 0, -1)  # Repetition is intentional.
    new_vertices = R @ new_vertices

    # Calculate shear in y direction needed so bottom side lies along x-axis.
    m = new_vertices[..., 1, 1] / new_vertices[..., 0, 1]
    S1 = copy.deepcopy(tiled_identity[..., :2, :2])
    S1[..., 1, 0] = -m
    new_vertices = S1 @ new_vertices

    # Calculate shift to bring lower right and upper left vertices to (1, 0) and (0, 1).
    S2 = copy.deepcopy(tiled_identity[..., :2, :2])
    S2[..., 0, 0] = 1 / new_vertices[..., 0, 1]
    S2[..., 1, 1] = 1 / new_vertices[..., 1, 2]

    # Combine linear transform and extend result for homogenous coordinates
    # so it can be combined with shift.
    L = S2 @ S1 @ R
    M = copy.deepcopy(tiled_identity)
    M[..., mat_slice, mat_slice] = L
    # Combine translation and rotation into affine transform.
    A = M @ T
    A_inv = np.linalg.inv(A)
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


def get_parameters_of_max_area_ellipse_in_quadrilateral(s, t):
    """
    Calculate parameters of mas area ellipse(s) tangent to all sides of a quadrilateral(s).

    The quadrilateral must not be a parallelogram and the left and right sides must not be parallel.
    The lower left, lower right and upper left vertices must be at (0, 0), (1, 0) and (0, 1),
    respectively, where coordinates are given as (x, y).
    The upper right vertex is given by (s, t) where s > 0 and t > 0.
    Since the quadrilateral is not a parallelogram => s != 0.
    Algorithm is based on Theorem 3.3. in [1] (see references section).

    Parameters
    ----------
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
    [1] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    # Sanitize inputs. Quadrilateral vertices must be (0, 0), (1, 0), (s, t), (0, 1)
    # where s > 0 and t > 0.  Furthermore, quadrilateral cannot be a parallelogram, i.e. s != 1.
    if s <= 0 or s == 1:
        raise ValueError("s must be >0 and not equal to 1")
    if t <= 0:
        raise ValueError("t must be >0")
    # Derive ellipse center.
    if s == 1:
        h, k = _derive_ellipse_center_trapezoid(t)
    else:
        h, k = _derive_ellipse_center_quadrilateral(s, t)
    # Derive ellipse semi-major and minor axes.
    a, b = _derive_ellipse_axes(s, t)
    # Derive ellipse tilt angle (anti-clockwise angle from x-axis to semi-major axis).
    theta = _derive_ellipse_tilt(h, k, a, b, s, t)
    return h, k, a, b, theta


def _derive_ellipse_center_quadrilateral(s, t):
    """
    Derive coordinate of center of ellipse inscribed in a quadrilateral.

    See Theorem 3.3 of [1] for algorithm.

    Parameters
    ----------
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.

    Returns
    -------
    h, k: `float` or `numpy.ndarray`
        The x and y coordinate of the ellipse center.

    References
    ----------
    [1] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    A, B, C = 1, 0, 1
    # Derive ellipse center, h and k
    mu_x = (0.5, 0.5 * s)
    h_bounds = (min(mu_x), max(mu_x))
    alpha = -24 * (t - 1)
    beta = 8 * ((s+1) * (t-1) - s)
    gamma = c = 2*s*(s - t + 2)
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    h_roots = np.array([-beta - delta, -beta + delta]) / (2*alpha)
    h = h_roots[np.logical_and(h_roots > h_bounds[0], h_roots < h_bounds[1])]
    if len(h) != 1:
        raise ValueError("Unique solution to h not found within valid range.")
    h = h[0]
    k = (h - s/2) * (t - B - C) / (s - A) + t/2
    return h, k


def  _derive_ellipse_center_trapezoid(s, t):
    """
    Derive coordinate of center of ellipse inscribed in a trapezoid.

    See Theorem 3.3 of [1] for algorithm.

    Parameters
    ----------
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.

    Returns
    -------
    h, k: `float` or `numpy.ndarray`
        The x and y coordinate of the ellipse center.

    References
    ----------
    [1] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    h = 0.5
    k = 0.25 * t + 0.25
    return h, k


def _derive_ellipse_axes(h, k, s, t):
    """
    Derive semi-major and minor axes of an ellipse inscribed in a quadrilateral.

    That is, the ellipse is tangent to all four sides.
    See [1] for algorithm.

    Parameters
    ----------
    s, t:
        The x and y coordinates of the upper right vertex.
        If array, must be broadcastable other inputs.
    h, k: `float` or `numpy.ndarray`
        The x and y coordinate of the ellipse center.

    Returns
    -------
    a, b: `float` or `numpy.ndarray`
        The semi-major and semi-minor axes, respectively.

    References
    ----------
    [1] Horwitz, A.
    Ellipses of maximal area and of minimal eccentricity inscribed in a convex quadrilateral.
    Austral. J. Math. Anal. Appl. 2, 1–12 (2005)
    """
    s_A = s - A
    h_A_2 = h - A/2
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
    return a, b


def _derive_ellipse_tilt(h, k, a, b, s, t):
    # Define coefficients of the lines making up the quadrilateral except the line along the y-axis.
    # It's slope is inf and so cannot be used. Fortunately, we don't need it.
    A, B, C = 1, 0, 1
    m0 = B / A
    c0 = B - m0 * A
    m1 = (t - B) / (s - A)
    c1 = t - m1 * s
    m2 = (C - t) / (-s)
    c2 = C
    raise NotImplementedError(
        "Algorithm to find ellipse tilt not implemented. Implement it here if you need it.")
