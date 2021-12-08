


def inscribe_ellipse_in_quadrilateral(vertices):
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
