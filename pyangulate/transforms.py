import copy

import numpy as np

from pyangulate import utils


def transform_to_xy_plane(points):
    """Rotate points on a 2-D plane in 3-D space to the x-y plane.

    Parameters
    ----------
    points: `numpy.ndarray`
        Points on the same 2-D plane in 3-D space. Can have any shape but penultimate axis
        must give the 3-D coordinates of the points and the final axis must represent
        points in a single plane.  The order of the coordinate components must be (x, y, z).

    Returns
    -------
    xy_points: `numpy.ndarray`
        Points rotated to x-y plane. As all z-coords are now 0, last axis is 2-D
        and only gives x-y values.
    rotation: `numpy.ndarray`
        Rotation matrix.
    """
    # Sanitize inputs.
    component_axis = -2
    coord_axis = -1
    nd = 3
    if points.ndim == 1:
        if len(points) == nd:
            points = points.reshape((nd, 1))
    elif points.ndim < 1 or points.shape[component_axis] != nd:
        raise ValueError("Points must be at least 2-D with penultimate axis of length 3. "
                         f"Input shape: {points.shape}")
    # Derive unit vector normal to the plane in which the first two points lie.
    plane_normal = _derive_unit_normal(points, component_axis, coord_axis)
    # Derive rotation to make plane parallel to x-y plane
    R = derive_plane_rotation_matrix(plane_normal, np.array([0, 0, 1]))
    rotated_points = R @ points
    # Derive affine transformation including shift to x-y axis.
    shift_item = utils._item(points.ndim, np.array([component_axis, coord_axis]),
                             np.array([2, 0]))
    shift = rotated_points[shift_item]
    A = np.eye(4)
    ext_dim = rotated_points.ndim - 2
    if ext_dim > 0:
        ext_shape = points.shape[:-2]
        A = utils.repeat_over_new_axes(A, np.zeros(ext_dim), np.array(ext_shape))
    A[..., -1, 0] = -shift
    # Convert points to homogenous coords and translate to x-y plane.
    hom_rot_points = utils.convert_to_homogeneous_coords(
        rotated_points, component_axis=component_axis, trailing_convention=False)
    xy_points = A @ hom_rot_points
    # Combine rotation and shift into affine matrix and return transformed points and matrix.
    A[..., 1:, 1:] = R
    return xy_points, A


def _derive_unit_normal(points, component_axis, coord_axis):
    # Derive unit vector normal to the plane in which the first three points lie.
    ndim = points.ndim
    item0 = utils._item(ndim, coord_axis, 0)
    item1 = utils._item(ndim, coord_axis, 1)
    item2 = utils._item(ndim, coord_axis, 2)
    vector1 = points[item1] - points[item0]
    vector2 = points[item2] - points[item0]
    tmp_component_axis = component_axis + 1 if component_axis < coord_axis else component_axis
    plane_normal = np.cross(vector1, vector2, axis=tmp_component_axis)
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=tmp_component_axis,
                                                 keepdims=True)
    return plane_normal


def derive_plane_rotation_matrix(plane_normal, new_plane_normal):
    """Derive matrix that rotates one plane to another.

    Parameters
    ----------
    plane: `numpy.ndarray`
        A vector normal to the original plane. Vector axis must be length 3,
        i.e. vector must be 3-D. The order of the coordinates must be (x, y, z).
        Other dimensions represent different planes that need rotating.
    new_plane: `numpy.ndarray`
        A vector normal to the plane to rotate to. Can have the following dimensionalities
        which are interpretted in the following ways:
        Same as plane arg:
            Last axis corresponds to (x, y, z) coordinates while other axes give
            the planes which the corresponding original planes are to be rotated to.
            Thus, multiple rotations are derived simultaneously.
        1-D:
            All original planes are rotated to this single new plane.
        N-D if plane arg is 1-D:
            Last axis corresponds to (x, y, z) coordinates while other axes give
            different planes for which rotations from original plane are desired.  Again,
            this means multiple rotations are calculated at once, but this time all
            from the same original plane.

    Returns
    -------
    R: `numpy.ndarray`
        The 3x3 rotation matrix. If input arrays are >1D, the last two axis will represent
        the matrix while preceding axes will correspond to the non-vector axes of the
        input arrays, i.e. the axes not corresponding to the axis kwarg.

    Notes
    -----
    Formula for rotation matrix is outline at reference [1].

    References
    ----------
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    component_axis = -1
    old_norm = np.linalg.norm(plane_normal, axis=component_axis)
    new_norm = np.linalg.norm(new_plane_normal, axis=component_axis)
    cos = (utils.dot_product_single_axis(plane_normal, new_plane_normal, axis=component_axis)
           / (old_norm * new_norm))
    # If cos = 1, then old plane and new plane are parallel. Return identity matrix.
    if (np.isscalar(cos) and cos == 1) or (cos == 1).all():
        R = np.eye(3)
        if cos.ndim > 2:
            R = utils.repeat_over_new_axes(R, np.zeros(cos.ndim-2), cos.shape[:-2])
        return R
    sin = np.sqrt(1 - cos**2)
    C = 1 - cos
    rot_axis = np.cross(plane_normal, new_plane_normal, axis=component_axis)
    rot_axis  = rot_axis / np.linalg.norm(rot_axis, axis=component_axis, keepdims=True)
    ndim = rot_axis.ndim
    x_item = utils._item(ndim, component_axis, 0)
    y_item = utils._item(ndim, component_axis, 1)
    z_item = utils._item(ndim, component_axis, 2)
    x = rot_axis[x_item]
    y = rot_axis[y_item]
    z = rot_axis[z_item]
    R = np.empty(tuple(list(cos.shape) + [3, 3]))
    R[..., 0, 0] = x**2 * C + cos
    R[..., 0, 1] = x * y * C - z * sin
    R[..., 0, 2] = x * z * C + y * sin
    R[..., 1, 0] = y * x * C + z * sin
    R[..., 1, 1] = y**2 * C + cos
    R[..., 1, 2] = y * z * C - x * sin
    R[..., 2, 0] = z * x * C - y * sin
    R[..., 2, 1] = z * y * C + x * sin
    R[..., 2, 2] = z**2 * C + cos
    # If some planes are parallel, replace with their matrices with identity.
    if not np.isscalar(cos) and (cos == 1).any():
        R[cos == 1] = np.eye(3)
    return R


def derive_projective_collineation_from_five_points(points, images, point5, image5):
    """Derive projection collineation mapping points to their images.

    Requires 5 points and their images.

    Parameters
    ----------
    points: `numpy.ndarray`
        n x 4 array of 4 points.
    images: `numpy.ndarray`
        n x 4 array of 4 images of the above points after the transformation.
    point5: `numpy.ndarray`
        n x 1 array of a 5th point used in the calculation.
    image5: `numpy.ndarray`
        n x 1 array giving the image of the 5th point after transformation.

    Returns
    -------
    T: `numpy.ndarray`
        n x n matrix defining the transformation from the points to their images.

    Notes
    -----
    Given 5 points and their images after transformation, we can define a projective
    collineation (affine transformation) that maps the 5 points to their images.
    Let the homogeneous coordinates of the 5 points be:
    m1 = [1 ]  m2 = [1 ]  m3 = [1 ]  m4 = [1 ]  m5 = [1 ]
         [w1]       [x1]       [y1]       [z1]       [v1]
         [w2]       [x2]       [y2]       [z2]       [v2]
    and their images after transformation be:
    M1 = [1 ]  M2 = [1 ]  M3 = [1 ]  M4 = [1 ]  M5 = [1 ]
         [W1]       [X1]       [Y1]       [Z1]       [V1]
         [W2]       [X2]       [Y2]       [Z2]       [V2]
    In our algorithm, let m1,...,m4 be the vertices of the parallelogram and m5 its center.
    M1,...,M4 are then the vertices of the 2-unit square and M5 is the origin.
    The projective collineation (affine transform), T, transforming the points to their
    images (parallelogram to 2-unit square), is given by
        T = A_2 A_1^-1
    where A_1^-1 is the inverse of A_1, the matrix that transforms a set of standard
    reference points to the original points.
    Similarly, A_2 is the matrix that transforms the same set of standard reference
    points to the image points. Hence it follows that T will transform the original
    points to the image points.
    A_1 is given by:
    A_1 = [c1*m1 c2*m2 c3*m3 c4*m4]
    where c1,...,c4 are given by the matrix product
    c = m^-1 m5
    where m^-1 is the inverse of the matrix m = [m1 m2 m3 m4].
    A_2 is found is exactly the same way using the matrix M = [M1 M2 M3 M4] and M5.

    In the above description we have assumed that the points given are 2-D and that the
    homogeneous element is given in the first row. However, the same workflow works
    equally well in 3-D by simply adding a 3rd dimension to the vectors describing the
    original and image points.  It also works if the homogeneous element is in the last row.
    The resulting matrix will differ in that it will be suited to transformations
    where the homogeneous element of the locations are in the last row, but should give
    the same transformations.

    References
    ----------
    [1] Zhang 1993, Estimating Projective Transformation Matrix (Collineation, Homography),
        Microsoft Research Techical Report MSR-TR-2010-63,
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-63.pdf
    """
    # Calculate matrix A1. See Notes in docstring.
    points_inv = np.linalg.pinv(points)  # inverse matrix of vertices.
    lam = points_inv @ point5
    A1 = points * np.swapaxes(lam, -2, -1)
    # Use swapaxes above instead of transpose as we only want to transpose last 2 axes.
    A1_inv = np.linalg.pinv(A1)
    # Repeat for calculation of matrix A2.  See Notes in docstring.
    images_inv = np.linalg.pinv(images)
    lam_prime = images_inv @ image5
    A2 = images * np.swapaxes(lam_prime, -2, -1)
    # Calculate the projective collineation (affine transformation) from
    # parallelogram vertices to 2-unit square vertices.  See Notes in docstring.
    return A2 @ A1_inv


def rotation_matrix_2d(theta):
    """Return 2-D rotation matrix given angle in radians."""
    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    if not np.isscalar(theta):
        matrix = np.moveaxis(matrix, 0, -1)
        matrix = np.moveaxis(matrix, 0, -1)  # Repeat is deliberate.
    return matrix
