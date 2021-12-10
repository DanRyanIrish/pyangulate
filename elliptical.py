import copy
from functools import partial

import numpy as np

from tie_pointing import transformations, utils


def inscribe_ellipse_in_3d(col_vertices):
    # Rotate vertices to xy-plane
    xy_vertices, R1 = transformations.rotate_plane_to_xy(col_vertices)
    R1_inv = np.linalg.inv(R1)
    ellipse_vertices = inscribe_ellipse(xy_vertices)
    # Convert 2-D ellipse vertices to 3-D coords
    ellipse_vertices = R1_inv @ ellipse_vertices
    return ellipse_vertices


def inscribe_ellipse(xy_vertices):
    component_axis = -2
    vertex_axis = -1
    scalar_set = False
    if xy_vertices.ndim < 2:
        raise ValueError("vertices must have more than 1 dimensions.")
    elif xy_vertices.ndim == 2:
        xy_vertices = np.expand_dims(xy_vertices, 0)
        scalar_set = True
    # Determine which sets of vertices correspond to a parallelogram and
    # which to other quadrilaterals.
    para_idx = utils.is_parallelogram(xy_vertices, keepdims=True)
    quad_idx = np.logical_not(para_idx)
    para_vertices = xy_vertices[para_idx]
    quad_vertices = xy_vertices[quad_idx]
    # Define array to hold vertices of the ellipse, i.e. the center and
    # a point at the end of the semi-major and semi-minor axes.
    ellipse_vertices = np.zeros(tuple(list(xy_vertices.shape[:-2]) + [3, 3]), dtype=float)
    # Calculate ellipse vertices for parallelograms.
    if para_idx.any():
        ellipse_vertices[para_idx][..., :2, :] = \
            inscribe_max_area_ellipse_in_parallelogram(para_vertices)
    # Calculate ellipse vertices for quadrilaterals.
    if quad_idx.any():
        ellipse_vertices[quad_idx][..., :2, :] = inscribe_ellipse_in_quadrilateral(quad_vertices)
    # Remove extra dimension if one was added.
    if scalar_set:
        ellipse_vertices = ellipse_vertices[0]
    return ellipse_vertices


def _identify_vertices(xy_vertices):
    """
    Given 4 x-y locations, assign them to vertices of a quadrilateral.

    This selection determined which locations will be the lower left, lower right,
    upper left and upper right vertices after an affine transform is applied.

    Parameters
    ----------
    xy_vertices: `numpy.ndarray`
        The x-y locations representing the vertices. Can have any number of dimensions
        but the penultimate one must be length 2 and represent the x-y coordinates,
        while the last one must be length 4 and represent the four vertices of a specific
        quadrilateral. Other dimensions will represent other quadrilaterls.

    Returns
    -------
    ll, lr, ur, ul: `numpy.ndarray`
        The lower left, lower right, upper right, and upper left vertices
        if they were to be transformed to the positive quadrant of the coordinate system.
        These represent columns in the input array and so the vertices dimension
        is removed compared to the input.
    """
    # Put some fixed values into variables.
    component_axis = -2
    vertex_axis = -1
    input_shape = xy_vertices.shape
    n_xy = input_shape[component_axis]
    no_vertex_shape = list(input_shape)
    del no_vertex_shape[vertex_axis]
    no_vertex_shape = tuple(no_vertex_shape)
    vertex1_shape = list(input_shape)
    vertex1_shape[vertex_axis] = 1
    vertex1_shape = tuple(vertex1_shape)
    # Select lower left vertex (relative to final position after transformation)
    # as the one closest to the origin.
    norms = np.linalg.norm(xy_vertices, axis=component_axis)
    tmp_vertex_axis = vertex_axis + 1 if component_axis > vertex_axis else vertex_axis
    ll_idx = np.isclose(norms - norms.min(axis=tmp_vertex_axis, keepdims=True), 0)
    ll_idx = utils.repeat_over_new_axes(ll_idx, component_axis, n_xy)
    ll = xy_vertices[ll_idx].reshape(no_vertex_shape)
    ll_1vertex = ll.reshape(vertex1_shape)
    # Find vertex diagonal to lower left one.
    diagonal_norm = np.linalg.norm(xy_vertices - ll_1vertex, axis=component_axis)
    ur_idx = np.isclose(diagonal_norm - diagonal_norm.max(axis=tmp_vertex_axis, keepdims=True), 0)
    ur_idx = utils.repeat_over_new_axes(ur_idx, component_axis, n_xy)
    ur = xy_vertices[ur_idx].reshape(no_vertex_shape)
    # Get axes of corner vertices relative to lower left.
    # To do this in an array-based way, define v1 as the vertex closer
    # to lower left and v2 as the further from lower left.
    diagonal_norm_sorted = np.sort(diagonal_norm, axis=tmp_vertex_axis)
    v1_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[..., 1:2], 0)
    v1_idx = utils.repeat_over_new_axes(v1_idx, component_axis, n_xy)
    v2_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[..., 2:3], 0)
    v2_idx = utils.repeat_over_new_axes(v2_idx, component_axis, n_xy)
    v1 = xy_vertices[v1_idx].reshape(no_vertex_shape)
    v2 = xy_vertices[v2_idx].reshape(no_vertex_shape)
    # Then set the lower right vertex as the one whose line with the lower left
    # forms a positive angle with diagonal. The remaining vertex is the upper left one.
    v1_shifted = v1 - ll
    v2_shifted = v2 - ll
    diagonal_shifted = ur - ll
    tmp_component_axis = component_axis + 1 if component_axis < vertex_axis else component_axis
    x_item = [slice(None)] * v1.ndim
    x_item[tmp_component_axis] = 0
    x_item = tuple(x_item)
    y_item = [slice(None)] * v1.ndim
    y_item[tmp_component_axis] = 1
    y_item = tuple(y_item)
    theta = (np.arctan2(v1_shifted[y_item], v1_shifted[x_item])
             - np.arctan2(diagonal_shifted[y_item], diagonal_shifted[x_item]))
    theta = utils.repeat_over_new_axes(theta, tmp_component_axis, n_xy)
    pos_theta_idx = theta < 0
    neg_theta_idx = theta > 0
    lr = np.zeros(no_vertex_shape)
    ul = np.zeros(no_vertex_shape)
    lr[pos_theta_idx] = v1[pos_theta_idx]
    lr[neg_theta_idx] = v2[neg_theta_idx]
    ul[neg_theta_idx] = v1[neg_theta_idx]
    ul[pos_theta_idx] = v2[pos_theta_idx]
    return ll, lr, ur, ul


    # Compute ellipse points for parallograms and other quadrilaterals separately.
    parallelograms_idx = n_parallel_pairs == 2
    quadrilaterals_idx = np.logical_not(parallelograms)

    return ll_idx, lr_idx, ur_idx, ul_idx, parallelograms_idx, quadrilaterals_idx


def inscribe_max_area_ellipse_in_parallelogram(vertices):
    """Derive the maximum-area ellipse inscribed in a parallelogram.

    The ellipse is represented in the returned values as the coordinates of
    the ellipse center, a point at one end of the major-axis and minor-axis.

    Parameters
    ----------
    vertices: `numpy.ndarray`
        ...,2x4 array giving the 2-D x-y coordinates of the 4 vertices of the parallelogram.
        The penultimate axis gives the coordinates of a single vertex while
        the last axis iterates from vertex to vertex.
        Other axes and take any shape and can be used to represent other ellipses.

    Returns
    -------
    center: `numpy.ndarray`
        The x-y coordinates of the center of the ellipse.
        Has same shape as vertices with axes having the same meaning.
    major_coord: `numpy.ndarray` 
        The x-y coordinates of the point at one end of the major-axis.
        Has same shape as vertices with axes having the same meaning.
    minor_coord: `numpy.ndarray` 
        The x-y coordinates of the point at one end of the minor-axis.
        Has same shape as vertices with axes having the same meaning.
    """
    component_axis = -2
    h, k, a, b, c, d = get_equation_of_max_area_ellipse_in_parallelogram(vertices)
    major_coord, minor_coord = get_ellipse_semi_axes_coords(h, k, a, b, c, d)
    center = np.stack((h, k), axis=component_axis)
    return center, major_coord, minor_coord


def get_equation_of_max_area_ellipse_in_parallelogram(vertices):
    """Derive parametic equation of the maximum-area ellipse inscribed in a parallelogram.

    Parameters
    ----------
    vertices: `numpy.ndarray`
        ...,2x4 array giving the 2-D x-y coordinates of the 4 vertices of the parallelogram.
        The penultimate axis gives the coordinates of a single vertex while
        the last axis iterates from vertex to vertex.
        The vertices must be in the following order:
        lower left, lower right, upper right, upper left.
        Other axes and take any shape and can be used to represent other ellipses.

    Returns
    -------
    h, k:
        The x-y coordinates of the center of the ellipse.
    a, b, c, d:
        The coefficients of the parametric ellipse equation of the form:
        x = h + a*cos(phi) + b*cos(phi);  y = k + c(cos(phi) + d*sin(phi)

    Notes
    -----
    Outline
    To derive the equation of the maximum-area ellipse inside a parallelogram,
    we use the affine transformation (projective collineation) that transforms the
    parallelogram to the 2-unit square (sides of length 2 units).
    Because the maximum-area ellipse inscribed in the 2-unit square is the unit circle,
    the inverse of the above transform transformis the parametric equation of the
    unit circle to that of the maximum-area ellipse in the original parallelogram.

    5th point
    An ellipse can only be uniquely defined from 5 points. While the vertices of
    a quadrilateral only give us 4, we can use the fact that the center of an
    ellipse inscribed in a parallelgram must be the intersection of the parallelograms
    diagonals. This is a special case of the more general case of inscribing an ellipse
    in a quadrilateral. In the general case, the center must lie on the line segment
    between the midpoints of the diagonals. However, since for a parallelogram, the midpoints
    for the diagonals are equal, we have a unique solution. This means that the method used
    here is specific to the parallogram case. Although the center point is not on the ellipse,
    it can still be used to uniquely define the transformation from the parallelogram to the
    2-unit square.

    Projective Collineation
    The projective collineations is an affine transformation that converts the
    parallelogram vertices to those of the 2-unit swaure. See the docstring of
    the function used to calculate the projective collineation for more info.

    rojective Collineation
    The projective collineations is an affine transformation that converts the
    parallelogram vertices to those of the 2-unit swaure. See the docstring of
    the function used to calculate the projective collineation for more info.

    Parametric Equation of the Ellipse
    Once the transformation, T, from the parallelogram to the 2-unit square is found,
    the parametic equation of the inscribed ellipse, e, is found by transforming the
    parametric equation of the unit circle, c,  via the inverse of T. i.e.
    e = T^-1 c
    where c = [1 cos(theta) sin(theta)].
    Substituting theta into into e will give the x-y coordinate on the ellipse that
    correspond to the point on the unit circle at the angle theta.

    Homogenous Coordinates Convention
    Above we have used the convention of placing the homogeneous coordinate at the
    front of the point vectors. However, the above workflow works equally well if the
    homogeneous coordinates are placed at the end of the vectors. The only result is that
    the x-y values in the parametric ellipse equation will correspond to the 1st and 2nd
    entries instead of the 2nd and 3rd entries.  (Or if the points are 3-D, the x, y, z
    values will correspond to the 1st, 2nd and 3rd entries rather than the 2nd, 3rd, and
    4th entries.

    References
    ----------
    [1] Hayes 2016, MAXIMUM AREA ELLIPSES INSCRIBING SPECIFIC QUADRILATERALS,
        Proceedings of The Canadian Society for Mechanical Engineering International Congress 2016
        http://faculty.mae.carleton.ca/John_Hayes/Papers/CCToMM_MJDH_MaxEllipse_RevF.pdf
    [2] Zhang 1993, Estimating Projective Transformation Matrix (Collineation, Homography),
        Microsoft Research Techical Report MSR-TR-2010-63,
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-63.pdf
    """
    # Calculate center of parallelogram, given by the midpoint of the diagonals.
    # Since the midpoints of both diagonals are equal by definition,
    # we only need to use 1 diagonal.
    center = (vertices[..., 0:1] + vertices[..., 2:3]) / 2

    # Define coordinates of the 2-unit square.
    square_vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T
    square_center = np.array([[0, 0]]).T
    # Convert to homogeneous coords.
    trailing_convention = False
    homogeneous_idx = -1 if trailing_convention else 0
    component_axis = -2
    vertex_axis = -1
    vertices = utils.convert_to_homogeneous_coords(
        vertices, component_axis=component_axis,
        trailing_convention=trailing_convention, vector=False)
    square_vertices = utils.convert_to_homogeneous_coords(
        square_vertices, component_axis=component_axis,
        trailing_convention=trailing_convention, vector=False)
    center = utils.convert_to_homogeneous_coords(
        center, component_axis=component_axis,
        trailing_convention=trailing_convention, vector=False)
    square_center = utils.convert_to_homogeneous_coords(
        square_center, component_axis=component_axis,
        trailing_convention=trailing_convention, vector=False)

    # Calculate projective collineation
    T = transformations.derive_projective_collineation_from_five_points(
            vertices, square_vertices, center, square_center)
    T_inv = np.linalg.inv(T)
    # Extract equation coefficients.
    blank_item = [slice(None)] * T_inv.ndim
    h = T_inv[..., 1, homogeneous_idx]
    k = T_inv[..., 2, homogeneous_idx]
    idx0 = 1 + homogeneous_idx
    a = T_inv[..., idx0, idx0]
    b = T_inv[..., idx0, idx0+1]
    c = T_inv[..., idx0+1, idx0]
    d = T_inv[..., idx0+1, idx0+1]

    return h, k, a, b, c, d


def get_ellipse_semi_axes_coords(h, k, a, b, c, d):
    """
    Calculate coords of one end of the semi-major and -minor axes of an ellipse.

    To do this, the parametric equation of the ellipse is used:
    x = h + a*cos(phi) + bsin(phi); y = k + c*cos(phi) + d*sin(phi)
    where (h, k) and the (x, y) coordinates of the ellipse center and
    phi is the angle at which the ellipse coordinates are desired.

    Parameters
    ----------
    h, k: `numpy.ndarray` or `float`
        The x-y coordinates of the ellipse center
        Must be same shape.
    a, b, c, d: `numpy.ndarray` or `float`
        Coefficients of the parametric ellipse equation as a function of angle.

    Returns
    -------
    semi_major_point:
        The semi_major axis.
    semi_minor_point:
        The semi-minor axis.

    Notes
    -----
    The algorithm used in this function is based on the follow derivation.
    Given the parametric equation of an ellipse as a function of angle, phi,
        x = h + a*cos(phi) + bsin(phi); y = k + c*cos(phi) + d*sin(phi)
    where (h, k) are the (x, y) coordinates of the ellipse center,
    we can determine the radius of the ellipse at phi using the distance formula:
        r^2 = (x - h)^2 + (y - k)^2
            = (a*cos(phi) + bsin(phi))^2 + (c*cos(phi) + d*sin(phi))^2
    Rearranging this equation gives
        r^2 = (a^2 + c^2)cos^2(phi) + 2(ab + cd)cos(phi)sin(phi) + (b^2 + d^2)sin^2(phi)
    The semi-major and -minor axes are the maximum and minimum of this equation.
    To find these, we set its derivative to zero and solve for phi.
    This will result in two valid solutions, corresponding to the semi-major and -minor axes
    Taking the derivative and rearranging gives:
        d(r^2)/d(phi) = 2Pcos^2(phi) + 2Qcos(phi)sin(phi) - 2Psin^2(phi)
    where
        P = ab + cd;  Q = b^2 + d^2 - a^2 - c^2
    Setting the derivative to 0 and dividing both sides by 2P gives
        cos^2(phi) + (P/Q)cos(phi)sin(phi) - sin^2(phi) = 0
    This can be factorized with initially unknown coefficients, u, v, s, t as:
        (u*cos(phi) + v*sin(phi)(s*cos(phi) + t*sin(phi)) = 0
    Expanding this factorization and equating the coefficients to the form above gives:
        us = 1 => s = 1/u;  vt = -1 => t = -1/v;  us + vs = P/Q
    The above equation can only be 0 when one of the factors is zero:
        u*cos(phi) + v*sin(phi) = 0;  OR  s*cos(phi) + t*sin(phi) = 0
         sin(phi) / cos(phi) = -u/v;       sin(phi) / cos(phi) = -s/t
                    tan(phi) = -u/v;                  tan(phi) = -s/t
    Recalling the ut + vs = P/Q and substituting for s and t gives:
                            ut + vs = P/Q
        =>               -u/v + v/u = P/Q
        =>   tan(phi) -  1/tan(phi) = P/Q
        => tan^2(phi) - (P/Q)tan(phi) - 1 = 0
    This is a quadratic in tan(phi). So to get the two possible solutions
    (i.e. semi-major and semi-minor axes) we use the standard quadratic root equation
    where (letting R = -P/Q) A = 1, B = R, C = -1
        tan(phi) = (-B +/- (B^2 - 4AC)^0.5) / (2A)
                 = (-R +/- (R^2 + 4)^0.5) / 2
    where R, expressed in terms of the original coefficients is:
        R = (a^2 + c^2 - b^2 - d^2) / (ab + cd)
    Taking the arctan of both sides gives the two angles at which the semi-major and -minor
    axes are located. Entering these angles into the parametric ellipse equation gives
    the points at one end the axes. If points at both ends of the major and minor axes are
    wanted, the negative of the above angles and also be plugged into the ellipse equation
    If the values of the semi-major and semi-minor axes are wanted, it is only required
    to compute these points' distances from the ellipse center using the
    standard distance formula.

    Note that tan has a periodicity of pi. This guarantees that our two above solutions
    correspond the semi-major and semi-minor axes, and not two symmetric solutions
    to just the semi-major (or semi-minor) axis.
    """
    if not (np.isscalar(h) and np.isscalar(k)) and h.shape != k.shape:
        raise ValueError(f"h and k must be same shape. h: {h}; k: {k}")
    # Derive the angles at which the semi-major and -minor axes are located.
    # See Notes section of docstring.
    R = (a**2 + c**2 - b**2 - d**2) / (a*b + c*d)
    phi1 = np.arctan((-R + np.sqrt(R**2 + 4)) / 2)
    phi2 = np.arctan((-R - np.sqrt(R**2 + 4)) / 2)
    # Find the points on the ellipse corresponding to the semi-major and -minor axes.
    ellipse = partial(_parametric_ellipse, h, k, a, b, c, d)
    xy_axis = -1
    xy1 = np.stack(ellipse(phi1), axis=xy_axis)
    xy2 = np.stack(ellipse(phi2), axis=xy_axis)
    # Caculate the semi- axes from the distance between the above points and
    # the ellipse center.
    center = np.stack((h, k), axis=xy_axis)
    semi_axis1 = np.linalg.norm(xy1 - center, axis=xy_axis)
    semi_axis2 = np.linalg.norm(xy2 - center, axis=xy_axis)
    # Find major and minor axes.
    semi_major_coord = copy.deepcopy(center)
    semi_minor_coord = copy.deepcopy(center)
    major_idx1 = semi_axis1 > semi_axis2
    minor_idx1 = np.logical_not(major_idx1)
    semi_major_coord[major_idx1] = xy1[major_idx1]
    semi_minor_coord[major_idx1] = xy2[major_idx1]
    semi_major_coord[minor_idx1] = xy2[minor_idx1]
    semi_minor_coord[minor_idx1] = xy1[minor_idx1]
    # Convert to row vectors.
    if semi_major_coord.ndim == 1:
        semi_major_coord = semi_major_coord.reshape(len(semi_major_coord), 1)
    else:
        semi_major_coord = np.swapaxes(semi_major_coord, -1, -2)
    if semi_minor_coord.ndim == 1:
        semi_minor_coord = semi_minor_coord.reshape(len(semi_minor_coord), 1)
    else:
        semi_minor_coord = np.swapaxes(semi_minor_coord, -1, -2)
    return semi_major_coord, semi_minor_coord


def _parametric_ellipse(h, k, a, b, c, d, phi):
    x = h + a * np.cos(phi) + b * np.sin(phi)
    y = k + c * np.cos(phi) + d * np.sin(phi)
    return x, y


def inscribe_ellipse_in_quadrilateral(xy_vertices):
    # Select which vertices will be lower left, lower right, upper left, and upper right
    # for the purposes of the transformations.
    ll, lr, ur, lr = _identify_vertices(xy_vertices)
    # In the case of quadrilaterals that have 2 parallel sides, the lower left vertex
    # must not be at the base of one of those parallel sides.
    # If this is the case, shift vertices one step around the quadrilateral.
    # Let m0 be the slope of the line from lower left to lower right,
    # m1 the slope from lower right to upper right,
    # m2 the slope from upper right to upper left
    # and m3 the sloper from upper left to lower left.
    # ind cases where slopes m1 and m3 are parallel.
    m0, m1, m2, m3 = utils.get_quadrilateral_slopes(ll, lr, ur, ul)
    wrong_ll_idx = np.isclose(m1, m3)
    # Before proceding, confirm the other lines are not also parallel.
    # This would mean that the vertices represent a parallelogram and this
    # algorithm is not valid.
    is_parallelogram = np.isclose(m0, m2) + wrong_ll_idx
    if is_parallelogram.any():
        raise ValueError(
            "Some input vertices represent a parallelogram, for which this function is not valid.")
    if wrong_ll_idx.any():
        ll_idx[wrong_ll_idx], lr_idx[wrong_ll_idx], ur_idx[wrong_ll_idx], ll_idx[wrong_ll_idx] = \
            lr_idx[wrong_ll_idx], ur_idx[wrong_ll_idx], ul_idx[wrong_ll_idx], ll_idx[wrong_ll_idx]


def inscribe_ellipse_in_quadrilateral_old(vertices):
    # Apply isometry to vertices, converting them to a 2-D coordinate system whereby:
    # lower left: (0, 0)
    # lower right: (A, B)
    # upper left: (0, C)
    # upper right: (s,t)
    # Start by calculating the affine transformation (and its inverse) that represent
    # the isometry.
    # To do this we need a vector normal to the plane in which the vertices lie.
    # This can be found be taking the cross product of two vectors in the plane.
    # Do not use the origin for this calculation.
    i = 0
    cross_vertices = []
    while i < vertices.shape[-2] and len(cross_vertices) < 2:
        vertex = vertices[..., i, :]
        if not np.any(np.all(vertex == 0, axis=-1)):
            cross_vertices.append(vertex)
        i += 1
    if len(cross_vertices) < 2:
        raise ValueError("Could not find 2 sets of vertices not including the origin.")
    plane_normal = np.cross(*cross_vertices)
    isometry, inverse_isometry = compute_isometry_matrices(vertices[..., 0, :],
                                                           vertices[..., 2, :], plane_normal)
    # Convert vertices to homogeneous coords, (i.e. add a 4th coord equalling 1)
    # as required for affine transformations, and represent last axis as a 4x1 matrix
    # for purposes of matrix multiplication rather than a length-4 row vector.
    hom_vertices = np.expand_dims(
            convert_to_homogeneous_coords(vertices, component_axis=-1, coord_axis=-2), -1)
    # Apply isometry and extract A, B, C, s, t.
    iso_vertices = (isometry @ hom_vertices)[..., 0]  # Remove dummy axis retained by mat. mul.
    A = iso_vertices[..., 1, 0]
    B = iso_vertices[..., 1, 1]
    C = iso_vertices[..., 2, 1]
    s = iso_vertices[..., 3, 0]
    t = iso_vertices[..., 3, 1]

    # Fix ellipse center, (h, k), by setting h as somewhere along the
    # open line segment connecting the midpoints of the diagonals of
    # the quadrilateral. Determine k from the equation of a line.
    h = 0.5 * (s/2 + A/2)
    k = (h - s/2) * ((t - B - C) / (s - A)) + t/2
    # To solve for the ellipse tangent to the four sides of the quadrilateral,
    # we can solve for the ellipse tangent to the three sides of a triangle,
    # the vertices of which are the complex points:
    # z1 = 0; z2 = A + Bi; z3 = - ((At - Bs) / (s - A))i
    # and the two ellipse foci are then the zeroes of the equation:
    # ph(z) = (s - A)z^2 - 2(a-A)(h - ik)z - (B - iA)(s - 2h)C
    # the discriminant of which can be denoted by
    # r(h) = r1(h) + ir2(h) where:
    s_A = s - A
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
    # Thus, we need to determine the quartic polynomial u(h) - |r(h)|^2 = r1^2 + r2^2
    # and we can then solve for the ellipse semi-major axis, a, and semi-minor axis, b,
    # from the equations:
    R = np.sqrt((r1**2 + r2**2) / (16 * s_A**4))
    W = 0.25 * (C / s_A**2) * (2 * (B*s - A*(t - C))*h - A*C*s) * (2*h - A) * (2*h - s)
    a = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) + R))
    b = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) - R))
    # Knowing the axes, we can generate the ellipse and float its tilt angle d until it
    # sits tangent to each side of the quadrilateral, using the inclined ellipse equation
    # and and the equation of the tangent line.
    return h, k, a, b, iso_vertices, A, B, C, s, t, r1, r2, R, W


def compute_isometry_matrices(new_origin, new_y, plane_normal):
    """
    Calculate the affine transformation matrix that does the following:
    1. translates by -new_origin (i.e. the origin is moved to new_origin);
    2. rotates such that the plane on which the point lies becomes parallel to the z-axis;
    3. rotates the line from the new_origin to new_y around the new z-axis so
    to is aligned with the y-axis.

    Also calculate the inverse affine transform.

    Parameters
    ----------
    new_origin: `numpy.ndarray`
        The point which serves as the new origin, i.e. the matrix translates
        by subtracting this point.
        The last dimension must be length 3 and represent the coordinate components.
        Other dimensions are assumed to represent other transformations and are
        broadcast through the calculation.
    new_y: `numpy.ndarray`
        The point which, along with new_origin, defines the new y-axis.
        Must be same shape as new_origin and dimensions are interpretted in the same way.
    plane_normal: `numpy.ndarray`
        A vector normal to the plane in which new_origin and new_y lie.
        Must be same shape as new_origin and dimensions are interpretted in the same way.

    Returns
    -------
    isometry: `numpy.ndarray`
        The affine transformation matrix.
        Will have N+1 dimensions where N is the number of dimensions in the input.
        Dimensions 0,..,N-1 will have the same shape has the corresponding dimensions
        in the inputs.
        The final two dimensions will represent 4x4 matrices, i.e. affine transformations.
        See Notes for more.

    inverse_isometry: `numpy.ndarray`
        The inverse affine transformation.
        Same shape and interpretation of dimensions as the forward transform.

    Notes
    -----
    Affine transformations combine linear transformations such as
    rotation, reflection, shearing etc., with translations which cannot be achieved
    via NxN (3x3) matrix multiplication. They do this by adding a N+1th (4th) row and
    column where the final column represents the translation, the final row is all 0,
    and the last element in the final row/column is 1.
    N-D points are then made compatible with these transformation by appending a 1
    for their N+1th (4th) dimensions, while vectors have a 0 appended.
    This is known as "homogeneous coordinates" where the real coordinates are simply
    the first N elements in the homogeneous coordinate.
    This way of representing translation is the same as shearing (a linear transformation)
    in a N+1th dimensions and then projecting the new position onto the N-D plane,
    i.e. the value to the N+1th coordinate is 0.

    References
    ----------
    Computer Graphics 2012, Lect. 5 - Linear and Affine Transformations
    Part 1: https://www.youtube.com/watch?v=heVndBwDffI
    Part 2: https://www.youtube.com/watch?v=7z1rs9Di77s&list=PLDFA8FCF0017504DE&index=11
    """
    # Sanitize inputs
    input_shape = new_origin.shape
    if ((new_y.shape != plane_normal.shape != input_shape) or
            (new_origin.shape[-1] != new_y.shape[-1] != plane_normal.shape[-1] != 3)):
        raise ValueError("All inputs must have same shape and final axes must be length 3. "
                         f"new_origin: {new_origin.shape}. "
                         f"new_y: {new_y.shape}. "
                         f"plane_normal: {plane_normal.shape}.")
    # Compute the unit vectors, u, v, w, of the orthogonal coordinate system where
    # the origin is at new_origin,
    # the w-axis is normal to the plane,
    # the v-axis is along the open new_origin-new_y line with the positive direction
    # going from new_origin to new_y,
    # and the u-axis is the cross-product of the v- and w-axes.
    plane_normal_norm = np.linalg.norm(plane_normal, axis=-1)
    w = plane_normal / np.expand_dims(np.linalg.norm(plane_normal, axis=-1), -1)  # Normalize length
    v = new_y - new_origin
    v /= np.expand_dims(np.linalg.norm(v, axis=-1), -1)
    u = np.cross(v, w) # j x k = i; k x j = -i

    # Translation matrix translates so that lower left become the origin.
    # Create a 4x4 matrix as required for 3-D affine transformation matrices and broadcast
    # to the other dimensions of the input.
    matrix_shape = list(input_shape[:-1]) + [4, 4]
    tiled_affine_identity_matrix = np.broadcast_to(np.identity(4), matrix_shape)
    T = copy.deepcopy(tiled_affine_identity_matrix)
    # Set translation (4th) column to subtract new origin, leaving 4th element as 1.
    T[...,:3, 3] = -new_origin

    # First rotation is to rotate the plane so its normal is parallel to the z-axis.
    # The rotation matrix is defined by the row vectors of the unit vectors, u, v, w.
    # Enter these into athe top left 3x3 sub-matrix of a 4x4 affine transformation matrix.
    # As rotation is a purely linear transformation, the last row and column are the same
    # as the identity matrix.
    R1 = copy.deepcopy(tiled_affine_identity_matrix)
    R1[..., :-1, :-1] = np.stack((u, v, w), axis=-2)

    # Second rotation is to rotate around the rotated plane normal (now the z-axis)
    # so the new_origin -> new_y line aligns with the y-axis.
    # To calculate the angle between this line and the y-axis,
    # apply the translation and R1 rotation, and then use arctan on the resulting
    # intermediate (x,y)/(u,v) coordinates.
    # Start by converting the new_y coord to homogeneous coords, i.e. an append
    # a 4th point to final axis whose value is 1.  See Notes in docstring.
    hom_new_y = convert_to_homogeneous_coords(new_y)
    # Re-represent last axis as a 4x1 matrix rather than a length-4 homogeneous location coord.
    hom_new_y = np.expand_dims(hom_new_y, -1)
    partial_isometry = R1 @ T
    tmp_y = (partial_isometry @ hom_new_y)[...,0]  # Remove trailing dummy axis.
    theta = np.arctan(tmp_y[..., 0] / tmp_y[..., 1])  # Angle to y-axis
    # Create affine transformation of a rotation by theta in the xy (uv) plane
    # so the new_origin -> new_y line aligns with the y-axis.
    rot_shape = np.array(matrix_shape)
    rot_shape[-2:] = 2
    rot = np.zeros(tuple(rot_shape))
    rot[..., 0, 0] = np.cos(theta)
    rot[..., 0, 1] = -np.sin(theta)
    rot[..., 1, 0] = np.sin(theta)
    rot[..., 1, 1] = np.cos(theta)
    R2 = copy.deepcopy(tiled_affine_identity_matrix)
    R2[..., :2, :2] = rot

    # Final isometery is given by the matrix product, R2.R1.T
    isometry = R2 @ partial_isometry

    # Also compute inverse isometry.
    # R2^-1 is simply a rotation around the z-axis by -theta.
    rot_inv = np.zeros(tuple(rot_shape))
    rot_inv[..., 0, 0] = np.cos(-theta)
    rot_inv[..., 0, 1] = -np.sin(-theta)
    rot_inv[..., 1, 0] = np.sin(-theta)
    rot_inv[..., 1, 1] = np.cos(-theta)
    R2_inv = copy.deepcopy(tiled_affine_identity_matrix)
    R2_inv[..., :2, :2] = rot_inv

    # Whereas R1 was given by the row vectors of the u, v, w unit vectors.
    # inverse of R1 is given by the column vectors of u, v, w.
    R1_inv = copy.deepcopy(tiled_affine_identity_matrix)
    R1_inv[..., :-1, :-1] = np.stack((u, v, w), axis=-1)

    # The inverse translation is simply a positive translation by new_origin
    # rather that a negative translation.
    T_inv = copy.deepcopy(tiled_affine_identity_matrix)
    T_inv[...,:3, 3] = new_origin

    # Final inverse isometry in the matrix produc of the components in reverse
    # order to the forward transformation: T^-1 . R1^-1 . R2^-1
    inverse_isometry = T_inv @ R1_inv @ R2_inv

    return isometry, inverse_isometry
