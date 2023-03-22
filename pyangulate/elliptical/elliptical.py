import copy
import warnings
from functools import partial

import numpy as np

from pyangulate import transforms, utils
from pyangulate.elliptical import parallelogram, quadrilateral


def inscribe_ellipse_in_3d(vertices):
    """
    Calculate parameters of an ellipse tangent to all sides of a quadrilateral.

    Parameters
    ----------
    vertices: `numpy.ndarray`
        The x-y-z locations representing the vertices. Can have any number of dimensions
        but the penultimate one must be length 3 and represent the x-y-z coordinates,
        while the last one must be length 4 and represent the four vertices of a specific
        quadrilateral. Other dimensions will represent other quadrilaterls.

    Returns
    -------
    ellipse_vertices: `numpy.ndarray`
        The x-y-z coordinates of the ellipse center, one of ends of the major axis,
        and one of the ends of the minor axis.  Dimensions correspond to those of
        the vertices input, except that the last axis has length 3 and and its
        elements correspond to the center, major coord, and minor coord, repectively.
    """
    # If vertices is only 2-D, add leading dummy axis to make array arithmetic work.
    if vertices.ndim == 2:
        leading_dummy = True
        vertices = vertices[np.newaxis]
    else:
        leading_dummy = False
    # Rotate vertices to xy-plane
    component_axis = -2
    xy_vertices, A = transforms.transform_to_xy_plane(vertices)
    A_inv = np.linalg.inv(A)
    # Strip z-axis (which should all now be zeros) and homogeneous axis.
    z_item = utils._item(xy_vertices.ndim, component_axis, -1)
    z_vertices = xy_vertices[z_item]
    if not np.allclose(z_vertices, np.zeros(z_vertices.shape)):
        raise ValueError(
            "Transformation to x-y plane failed. All vertices may not be on same plane.")
    xy_item = utils._item(xy_vertices.ndim, component_axis, slice(1, 3))
    xy_vertices = xy_vertices[xy_item]
    # Get ellipse vertices in 2D.
    ellipse_xy_vertices = inscribe_ellipse(xy_vertices)
    # Convert 2-D ellipse vertices to 3-D coords and transform back to original plane.
    ellipse_z0_vertices = utils.add_z_to_xy(ellipse_xy_vertices, component_axis)
    ellipse_hom_vertices = utils.convert_to_homogeneous_coords(
        ellipse_z0_vertices, component_axis=component_axis)
    ellipse_vertices = A_inv @ ellipse_hom_vertices
    # Strip homogeneous axis.
    nat_item = utils._item(ellipse_vertices.ndim, component_axis, slice(1, None))
    ellipse_vertices = ellipse_vertices[nat_item]
    if leading_dummy:
        ellipse_vertices = ellipse_vertices[0]
    return ellipse_vertices


def inscribe_ellipse(xy_vertices):
    """
    Calculate parameters of an ellipse tangent to all sides of a quadrilateral.

    Parameters
    ----------
    xy_vertices: `numpy.ndarray`
        The x-y locations representing the vertices. Can have any number of dimensions
        but the penultimate one must be length 2 and represent the x-y coordinates,
        while the last one must be length 4 and represent the four vertices of a specific
        quadrilateral. Other dimensions will represent other quadrilaterls.

    Returns
    -------
    ellipse_vertices: `numpy.ndarray`
        The x-y coordinates of the ellipse center, one of ends of the major axis,
        and one of the ends of the minor axis.  Dimensions correspond to those of
        the xy_vertices input, except that the last axis has length 3 and and its
        elements correspond to the center, major coord, and minor coord, repectively.
    """
    component_axis = -2
    vertex_axis = -1
    scalar_set = False
    if xy_vertices.ndim < 2:
        raise ValueError("vertices must have more than 1 dimensions.")
    elif xy_vertices.ndim == 2:
        # If input vertices are 2-D, add a dummy leading axis so it works better
        # with other code elements. Remove this dummy axis before returning result.
        xy_vertices = np.expand_dims(xy_vertices, 0)
        scalar_set = True
    # Define array to hold vertices of the ellipse, i.e. the center and
    # a point at the end of the semi-major and semi-minor axes.
    ellipse_vertices = np.zeros(tuple(list(xy_vertices.shape[:-2]) + [2, 3]), dtype=float)
    # Determine which sets of vertices correspond to a parallelogram and
    # which to other quadrilaterals.
    para_idx = utils.is_parallelogram(xy_vertices, keepdims=True)
    # Calculate ellipse coordinates for parallelograms.
    if para_idx.any():
        ellipse_vertices[para_idx[...,:3]]  = _calculate_and_reshape_ellipse_vertices(
            para_idx, xy_vertices, inscribe_max_area_ellipse_in_parallelogram)
    # Calculate ellipse coordinates for quadrilaterals.
    quad_idx = np.logical_not(para_idx)
    if quad_idx.any():
        ellipse_vertices[quad_idx[...,:3]]  = _calculate_and_reshape_ellipse_vertices(
            quad_idx, xy_vertices, inscribe_ellipse_in_quadrilateral)
    # Remove extra dimension if one was added.
    if scalar_set:
        ellipse_vertices = ellipse_vertices[0]
    return ellipse_vertices


def _calculate_and_reshape_ellipse_vertices(idx, xy_vertices, func):
    shape = (int(idx.sum() / 2 / 4), 2, 4)
    vertices = xy_vertices[idx].reshape(shape)
    ellipse = func(vertices)
    ellipse = np.stack(ellipse, axis=-1)
    return ellipse.flatten()


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
    # If two or more points are equidistant from origin, shift vertices slightly
    # and recalculate so there is a unique closest vertex.
    ll_idx_sum = ll_idx.sum(axis=-1)
    if ll_idx_sum.any():
        tmp_xy_vertices = copy.copy(xy_vertices)
        tmp_xy_vertices[ll_idx_sum > 1] += 0.001
        norms = np.linalg.norm(tmp_xy_vertices, axis=component_axis)
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


def inscribe_max_area_ellipse_in_parallelogram(xy_vertices):
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
    coord_axis = -1
    xy_vertices = np.stack(_identify_vertices(xy_vertices), axis=coord_axis)
    h, k, a, b, c, d = parallelogram.get_equation_of_max_area_ellipse_in_parallelogram(xy_vertices)
    major_coord, minor_coord = parallelogram.get_ellipse_semi_axes_coords(h, k, a, b, c, d)
    tmp_component_axis = component_axis + 1 if component_axis < coord_axis else component_axis
    center = np.stack((h, k), axis=tmp_component_axis)
    return center, major_coord, minor_coord


def inscribe_ellipse_in_quadrilateral(xy_vertices):
    """
    Calculate parameters of ellipse(s) tangent to all sides of a quadrilateral(s).

    Algorithm is based on [1], [2] and [3] (see references section).
    The quadrilateral must not be a parallelogram and the left and right sides must not be parallel.
    The lower left vertex must be at the origin, while the upper left must be on the y-axis.

    Parameters
    ----------
    xy_vertices: `numpy.ndarray`
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
    warnings.warn("Fitting ellipses to irregular quadrilaterals is unstable. "
                  "Check results manually.")
    # Select which vertices will be lower left, lower right, upper left, and upper right
    # for the purposes of the transformations.
    ll, lr, ur, ul = _identify_vertices(xy_vertices)
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
        ll[wrong_ll_idx], lr[wrong_ll_idx], ur[wrong_ll_idx], ll[wrong_ll_idx] = \
            lr[wrong_ll_idx], ur[wrong_ll_idx], ul[wrong_ll_idx], ll[wrong_ll_idx]
        m0, m1, m2, m3 = m1, m2, m3, m0
    # Tranform vertices so lower left is at (0, 0) and upper left is at (0, C).
    A, A_inv = quadrilateral.compute_isometry_transform(ll, ul)
    hom_vertices = utils.convert_to_homogeneous_coords(np.stack((ll, lr, ur, ul), axis=-1),
                                                       component_axis=-2)
    iso_vertices = A @ hom_vertices
    # Extract variables for deriving ellipse.
    s = iso_vertices[..., 1, 2]
    t = iso_vertices[..., 2, 2]
    # Fit ellipse to quadrilateral
    h, k, a, b, theta = quadrilateral.get_parameters_of_max_area_ellipse_in_quadrilateral(s, t)
    # Finally, find the coordinates of the points at one end of the major axes.
    ellipse = partial(quadrilateral.parametric_ellipse_angled, h, k, a, b, theta)
    xy_axis = -1
    major_coord = np.stack(ellipse(0), axis=xy_axis)
    minor_coord = np.stack(ellipse(np.pi/2), axis=xy_axis)
    center = np.stack((h, k), axis=xy_axis)

    return center, major_coord, minor_coord


def parametric_ellipse_3d(center, major_coord, minor_coord, phi):
    component_axis = -2
    coord_axis = -1
    # Rotate ellipse vertices to xy-plane
    vertices = np.stack((center, major_coord, minor_coord), axis=coord_axis)
    xy_vertices, R = transforms.transform_to_xy_plane(vertices)
    # Calculate angle between semi-major axis and x-axis.
    center_idx = utils._item(xy_vertices.ndim, np.array([component_axis, coord_axis]),
                             np.array([slice(1, 3), 0]))
    major_idx = utils._item(xy_vertices.ndim, np.array([component_axis, coord_axis]),
                             np.array([slice(1, 3), 1]))
    minor_idx = utils._item(xy_vertices.ndim, np.array([component_axis, coord_axis]),
                             np.array([slice(1, 3), 2]))
    xy_center = xy_vertices[center_idx]
    xy_major = xy_vertices[major_idx]
    xy_minor = xy_vertices[minor_idx]
    tmp_component_axis = component_axis + 1 if component_axis < coord_axis else component_axis
    tmp_x_idx = utils._item(xy_center.ndim, tmp_component_axis, 0)
    tmp_y_idx = utils._item(xy_center.ndim, tmp_component_axis, 1)
    shifted_xy_major = xy_major - xy_center
    shifted_x_major = shifted_xy_major[tmp_x_idx]
    shifted_y_major = shifted_xy_major[tmp_y_idx]
    m1 = 0
    m2 = shifted_y_major / shifted_x_major
    # Shift angle depending on quadrant.
    theta = np.arctan(np.abs((m2 - m1) / (1 + m1*m2)))
    theta_is_scalar = False
    if np.isscalar(theta):
        theta = np.array([theta])
        theta_is_scalar = True
    idx = np.logical_and(shifted_x_major < 0, shifted_y_major >= 0)
    theta[idx] = np.pi - theta[idx]
    idx = np.logical_and(shifted_x_major < 0, shifted_y_major < 0)
    theta[idx] += np.pi
    idx = np.logical_and(shifted_x_major >= 0, shifted_y_major < 0)
    theta[idx] = 2*np.pi - theta[idx]
    if theta_is_scalar:
        theta = theta[0]
    # Calculate semi-axes.
    a = np.linalg.norm(xy_major - xy_center, axis=tmp_component_axis)
    b = np.linalg.norm(xy_minor - xy_center, axis=tmp_component_axis)
    h = xy_center[tmp_x_idx]
    k = xy_center[tmp_y_idx]
    # Derive xy points of ellipse from phi
    h = np.expand_dims(h, coord_axis)
    k = np.expand_dims(k, coord_axis)
    a = np.expand_dims(a, coord_axis)
    b = np.expand_dims(b, coord_axis)
    theta = np.expand_dims(theta, coord_axis)
    shape = [1] * h.ndim
    shape[coord_axis] = len(phi)
    phi = phi.reshape(tuple(shape))
    xy_ellipse = np.stack(quadrilateral.parametric_ellipse_angled(h, k, a, b, theta, phi),
                          axis=component_axis)
    # Convert ellipse points to 3-D homogeneous points
    ellipse = utils.add_z_to_xy(xy_ellipse, component_axis)
    ellipse = utils.convert_to_homogeneous_coords(ellipse, component_axis=component_axis)
    # Rotate back to original plane.
    R_inv = np.linalg.inv(R)
    ellipse = R_inv @ ellipse
    # Strip homogeneous axis.
    item = utils._item(ellipse.ndim, component_axis, slice(1, None))
    ellipse = ellipse[item]
    return ellipse
