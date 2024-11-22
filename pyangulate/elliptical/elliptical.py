import copy
import warnings
from functools import partial

import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from sunpy.coordinates import HeliocentricEarthEcliptic

from pyangulate import transforms, utils
from pyangulate.elliptical import parallelogram, quadrilateral


def inscribe_ellipse(vertices):
    """
    Calculate parameters of an ellipse tangent to all sides of a quadrilateral.

    Coordinates are given in astropy SkyCoords.

    Parameters
    ----------
    vertices: `~astropy.coordinates.SkyCoord`
        The coordinates of the locations representing the vertices. Can have any
        number of dimensions but the last one must be length 4 and represent the
        four vertices of a specific quadrilateral. Other dimensions will represent
        other quadrilaterls.

    Returns
    -------
    ellipse_vertices: `~astropy.coordinates.SkyCoord`
        The coordinates of the ellipse center, one of ends of the major axis,
        and one of the ends of the minor axis.  Dimensions correspond to those of
        the vertices input, except that the last axis has length 3 and and its
        elements correspond to the center, major coord, and minor coord, repectively.
    """
    # Sanitize inputs
    base_frame = HeliocentricEarthEcliptic
    if not isinstance(vertices, SkyCoord) or not vertices.is_transformable_to(base_frame):
        raise TypeError(f"vertices must be a SkyCoord transformable to {base_frame.name}")
    if vertices.isscalar or vertices.shape[-1] != 4:
        raise ValueError("Last axis of vertices must be length 4.")
    input_frame = vertices.frame

    # Convert vertices to HeliocentricEarthEcliptic and extract their xyz coordinates.
    # Ensure the penultimate axis represents the xyz components.
    vertices = vertices.transform_to(base_frame)
    xyz_vertices = np.moveaxis(vertices.cartesian.xyz, 0, -2)
    base_unit = xyz_vertices.unit
    xyz_vertices = xyz_vertices.value

    # Fit ellipse in quadrilateral and return centers and ends of semi-axes.
    ellipse_xyz_vertices = inscribe_ellipse_xyz(xyz_vertices)

    # Reform vertices from xyz to input coordinate frame.
    ellipse_vertices = SkyCoord(CartesianRepresentation(ellipse_xyz_vertices[..., 0, :],
                                                        ellipse_xyz_vertices[..., 1, :],
                                                        ellipse_xyz_vertices[..., 2, :],
                                                        unit=base_unit),
                                frame=base_frame)
    return ellipse_vertices


def inscribe_ellipse_xyz(xyz_vertices):
    """
    Calculate parameters of an ellipse tangent to all sides of a quadrilateral.

    Coordinates are given in arbitrary 3D Cartesian coordinates.

    Parameters
    ----------
    xyz_vertices: `numpy.ndarray`
        The x-y-z locations representing the vertices. Can have any number of dimensions
        but the penultimate one must be length 3 and represent the x-y-z coordinates,
        while the last one must be length 4 and represent the four vertices of a specific
        quadrilateral. Other dimensions will represent other quadrilaterls.

    Returns
    -------
    ellipse_xyz_vertices: `numpy.ndarray`
        The x-y-z coordinates of the ellipse center, one of ends of the major axis,
        and one of the ends of the minor axis.  Dimensions correspond to those of
        the vertices input, except that the last axis has length 3 and and its
        elements correspond to the center, major coord, and minor coord, repectively.
    """
    # If vertices is only 2-D, add leading dummy axis to make array arithmetic work.
    if xyz_vertices.ndim == 2:
        leading_dummy = True
        xyz_vertices = xyz_vertices[np.newaxis]
    else:
        leading_dummy = False
    # Rotate xyz_vertices to xy-plane
    component_axis = -2
    xy_vertices, A = transforms.transform_to_xy_plane(xyz_vertices)
    A_inv = np.linalg.inv(A)
    # Strip z-axis (which should all now be zeros) and homogeneous axis.
    z_item = utils._item(xy_vertices.ndim, component_axis, -1)
    z_vertices = xy_vertices[z_item]
    if not np.allclose(z_vertices, np.zeros(z_vertices.shape)):
        warnings.warn("Transformation to x-y plane resulted in non-zero z-values. "
                      "All vertices may not be on same plane. "
                      f"Max transformed z-value: {z_vertices.max()}; "
                      f"Max pre-transformed z-value: {xyz_vertices[..., 2, :].max()}")
    xy_item = utils._item(xy_vertices.ndim, component_axis, slice(1, 3))
    xy_vertices = xy_vertices[xy_item]
    # Get ellipse vertices in 2D.
    ellipse_xy_vertices = inscribe_ellipse_xy(xy_vertices)
    # Convert 2-D ellipse vertices to 3-D coords and transform back to original plane.
    ellipse_z0_vertices = utils.add_z_to_xy(ellipse_xy_vertices, component_axis)
    ellipse_hom_vertices = utils.convert_to_homogeneous_coords(
        ellipse_z0_vertices, component_axis=component_axis)
    ellipse_xyz_vertices = A_inv @ ellipse_hom_vertices
    # Strip homogeneous axis.
    nat_item = utils._item(ellipse_xyz_vertices.ndim, component_axis, slice(1, None))
    ellipse_xyz_vertices = ellipse_xyz_vertices[nat_item]
    if leading_dummy:
        ellipse_xyz_vertices = ellipse_xyz_vertices[0]
    return ellipse_xyz_vertices


def inscribe_ellipse_xy(xy_vertices):
    """
    Calculate parameters of an ellipse tangent to all sides of a quadrilateral.

    Coordinates are given in arbitrary 2D Cartesian coordinates.

    Parameters
    ----------
    xy_vertices: `numpy.ndarray`
        The x-y locations representing the vertices. Can have any number of dimensions
        but the penultimate one must be length 2 and represent the x-y coordinates,
        while the last one must be length 4 and represent the four vertices of a specific
        quadrilateral. Other dimensions will represent other quadrilaterls.

    Returns
    -------
    ellipse_xy_vertices: `numpy.ndarray`
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
    ellipse_xy_vertices = np.zeros(tuple(list(xy_vertices.shape[:-2]) + [2, 3]), dtype=float)
    # Determine which sets of vertices correspond to a parallelogram and
    # which to other quadrilaterals.
    para_idx = utils.is_parallelogram(xy_vertices, keepdims=True)
    # Calculate ellipse coordinates for parallelograms.
    if para_idx.any():
        ellipse_xy_vertices[para_idx[...,:3]]  = _calculate_and_reshape_ellipse_vertices(
            para_idx, xy_vertices, inscribe_max_area_ellipse_in_parallelogram)
    # Calculate ellipse coordinates for quadrilaterals.
    quad_idx = np.logical_not(para_idx)
    if quad_idx.any():
        ellipse_xy_vertices[quad_idx[...,:3]]  = _calculate_and_reshape_ellipse_vertices(
            quad_idx, xy_vertices, inscribe_ellipse_in_quadrilateral)
    # Remove extra dimension if one was added.
    if scalar_set:
        ellipse_xy_vertices = ellipse_xy_vertices[0]
    return ellipse_xy_vertices


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


def get_canonical_ellipse_coeffs_from_xyvertices(xy_center, xy_major, xy_minor):
    """
    Get the coefficients of the canonical ellipse equation.

    Do this given center and point at end of semi-major and semi-minor axes.
    This function is only valid in 2-D space.
    The canonical equation of an ellipse is given by:
    ((x−h)cos(phi)+(y−k)sin(phi))**2 / a**2 + ((x−h)sin(phi)−(y−k)cos(phi))**2 / b**2 = 1
    where (h, k) are the (x, y) coordinates of the center, phi is the angle by
    which the ellipse is tilted relative to the x-axis, and a and b are the
    length of the semi-major and semi-minor axes, respectively.

    Parameters
    ----------
    center: `numpy.ndarray`
        The x-y coordinates of the center.  (x, y) must be represented by the last axis
        of the array.  x coord must be center[..., 0], while y must be at center[..., 1].
    major: `numpy.ndarray`
        The x-y coordinates of one end of the major axis.  (x, y) must be represented
        by the last axis of the array.  x coord must be major[..., 0], while y must
        be at major[..., 1].
    minor: `numpy.ndarray`
        The x-y coordinates of one end of the minor axis.  (x, y) must be represented
        by the last axis of the array.  x coord must be minor[..., 0], while y must
        be at minor[..., 1].

    Returns
    -------
    h : `numpy.ndarray`
        x coordinate of the ellipse center
    k : `numpy.ndarray`
        y coordinate of the ellipse center
    a : `numpy.ndarray`
        Semi-major axis.
    b : `numpy.ndarray`
        Semi-minor axis
    theta : `numpy.ndarray`
        Tilt angle of major axis from x-axis in units of rad.
    """
    component_axis = -1
    x_idx = utils._item(xy_center.ndim, component_axis, 0)
    y_idx = utils._item(xy_center.ndim, component_axis, 1)
    shifted_xy_major = xy_major - xy_center
    shifted_x_major = shifted_xy_major[x_idx]
    shifted_y_major = shifted_xy_major[y_idx]
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
    a = np.linalg.norm(xy_major - xy_center, axis=component_axis)
    b = np.linalg.norm(xy_minor - xy_center, axis=component_axis)
    h = xy_center[x_idx]
    k = xy_center[y_idx]
    return h, k, a, b, theta


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
    h, k, a, b, theta = get_canonical_ellipse_coeffs_from_xyvertices(xy_center, xy_major_coord, xy_minor_coord)
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


def get_canonical_from_general_ellipse_coeffs(A, B, C, D, E, F):
    """
    Derive center, semi-axes and rotation angle of ellipse from general coefficients.

    General Equation of Ellipse: Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0

    Parameters
    ----------
    A: `float`

    B: `float`

    C: `float`

    D: `float`

    E: `float`

    F: `float`

    Returns
    -------
    h: `float` or array-like
        The x-coordinate of the ellipse center.
    k: `float` or array-like
        The y-coordinate of the ellipse center.
    a: `float` or array-like
        The length of the semi-major axis of the ellipse.
    b: `float` or array-like
        The length of the semi-minor axis of the ellipse.
    phi: `float` or array-like
        The angle between the semi major axis and the x-axis in radians.
    """
    # Derive semi-axes
    denom = B**2 - 4*A*C
    base_axis = 2 * (A*E**2 + C*D**2 - B*D*E + denom*F)
    diff_axis = np.sqrt((A - C)**2 + B**2)
    axis1 = - np.sqrt(base_axis * ((A + C) + diff_axis)) / denom
    axis2 = - np.sqrt(base_axis * ((A + C) - diff_axis)) / denom
    axes = np.stack([axis1, axis2], axis=-1)
    a = axes.max(axis=-1)
    b = axes.min(axis=-1)

    # Derive center
    h = (2*C*D - B*E) / denom
    k = (2*A*E - B*D) / denom

    # Derive rotation angle
    if np.isclose(B, 0):
        if A > C:
            phi = 90
        else:
            phi = 0
        phi = (phi * u.deg).to(u.rad)
    else:
        phi = -np.arctan((C - A - diff_axis) / B) * u.rad

    return h, k, a, b, phi


def find_intersections_ellipse_and_line(h, k, a, b, phi, m, c):
    """
    Find the point(s) of intersection between an ellipse and a line.

    Inputs can be float or array as long as they are the same shape or broadcastable.

    Parameters
    ----------
    h: `float` or array-like
        The x-coordinate of the ellipse center.
    k: `float` or array-like
        The y-coordinate of the ellipse center.
    a: `float` or array-like
        The length of the semi-major axis of the ellipse.
    b: `float` or array-like
        The length of the semi-minor axis of the ellipse.
    phi: `float` or array-like
        The angle between the semi major axis and the x-axis in radians.
    m: `float` or array-like
        The slope of the line.
    c: `float` or array-like
        The intercept of the line.

    Returns
    -------
    x, y:
        The x and y coordinates of the intersection point(s)
    """
    ksi = c - k
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    lam = cos_phi + m * sin_phi
    kappa = sin_phi - m * cos_phi
    new = h * cos_phi - ksi * sin_phi
    tau = h * sin_phi + ksi * cos_phi
    alpha = b**2 * lam**2 + a**2 * kappa**2
    beta = -2 * (b**2 * lam * new + a**2 * kappa * tau)
    gamma = b**2 * new**2 + a**2 * tau**2 - a**2 * b**2
    sqrt = np.sqrt(beta**2 - 4*alpha*gamma)
    if hasattr(sqrt, "ndim"):
        sqrt = sqrt * np.array([-1, 1]).reshape([2] + [1]*sqrt.ndim)
    else:
        sqrt *= np.array([-1, 1])
    x = (-beta + sqrt) / (2 * alpha)
    y = m * x + c
    return x, y


def fit_ellipse(x, y):
    """Fit Ellipse to a set of x,y pointsvia least squares minimization

    ||alpha*x - beta ||^2
    where alpha is the x,y terms (without coefficients) of the general equation of an ellipse:
    Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0 where F = -1

    Parameters
    ----------
    x: array-like
        The x-coordinates of the points to be fitted.
    y: array-like
        The y-coordinates of the points to be fitted.

    Parameters
    ----------
    h: `float` or array-like
        The x-coordinate of the ellipse center.
    k: `float` or array-like
        The y-coordinate of the ellipse center.
    a: `float` or array-like
        The length of the semi-major axis of the ellipse.
    b: `float` or array-like
        The length of the semi-minor axis of the ellipse.
    phi: `float` or array-like
        The angle between the semi major axis and the x-axis in radians.
    """
    # Calculate minimizer
    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    # Derive/fit coefficients
    A, B, C, D, E = np.linalg.lstsq(A, b, rcond=-1)[0].squeeze()
    F = -1
    # Convert canonical ellipse coefficients to general coeffs and return.
    return get_canonical_from_general_ellipse_coeffs(A, B, C, D, E, F)
