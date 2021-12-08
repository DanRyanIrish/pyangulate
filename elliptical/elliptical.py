import numpy as np

def inscribe_ellipse(col_vertices):
    # Rotate vertices to x-y plane
    coord_axis = -2
    vertex_axis = -1
    # Rotate vertices to xy-plane
    xy_vertices, R1 = rotate_plane_to_xy(col_vertices)
    R1_inv = np.linalg.inv(R1)
    # Get indices of vertices that will be in the
    # lower left, lower right, upper right, upper left positions after transformation.
    ll_idx, lr_idx, ur_idx, ul_idx, parallelograms_idx, quadrilaterals_idx = identify_vertices(
        xy_vertices)

    # Define array to hold vertices of the ellipse, i.e. the center and
    # a point at the end of the semi-major and semi-minor axes.
    planes_shape = vertices.shape[:-2]
    ellipse_vertices = np.zeros(tuple(list(planes_shape) + [3, 3]), dtype=float)

    # Calculate ellipse vertices for parallelograms.
    if parallelograms_idx.any():
        ellipse_vertices[..., :2, :][parallelograms_idx] = inscribe_ellipse_in_parallelogram(
            col_vertices[parallelograms_idx], ll_idx, lr_idx, ur_idx, ul_idx)

    # Calculate ellipse vertices for quadrilaterals.
    if quadrilaterals_idx.any():
        ellipse_vertices[..., :2, :][quadrilaterals_idx] = inscribe_ellipse_in_quadrilateral(
            vertices[quadrilaterals_idx], ll_idx, lr_idx, ur_idx, ul_idx)

    # Convert 2-D ellipse vertices to 3-D coords
    ellipse_vertices = R1_inv @ ellipse_vertices

    return ellipse_vertices


def identify_vertices(xy_vertices):
    # Determine which vertices should be lower left, lower right, upper right and upper left
    # after the the isometry is applied.
    # vertices must have shape (..., 4, 3), i.e. (..., vertices, coords)
    coord_axis = -2
    vertex_axis = -1

    # Select lower left vertex (relative to final position after transformation)
    # as the one closest to the origin.
    norms = np.linalg.norm(xy_vertices, axis=coord_axis)
    ll_idx = np.isclose(norms - norms.min(axis=vertex_axis, keepdims=True), 0)  #TODO: expand bool idx to coord axis.
    # Find vertex diagonal to lower left one.
    diagonal_norm = np.linalg.norm(xy_vertices - xy_vertices[ll_idx], axis=coord_axis)
    tmp_vertex_axis = vertex_axis + 1 if coord_axis > vertex_axis else vertex_axis
    ur_idx = np.isclose(diagonal_norm - diagonal_norm.max(axis=tmp_vertex_axis, keepdims=True), 0) #TODO: expand bool idx to coord axis.
    # Get axes of corner vertices relative to lower left.
    # To do this in an array-based way, define v1 as the vertex closer
    # to lower left and v2 as the further from lower left.
    diagonal_norm_sorted = diagonal_norm.sort(axis=tmp_vertex_axis)
    v1_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[..., 1], 0) #TODO: expand bool idx to coord axis.
    v2_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[..., 2], 0) #TODO: expand bool idx to coord axis.
    # Then set the lower right vertex as the one whose line with the lower left
    # forms a negative with diagonal.
    diagonal = xy_vertices[ur_idx] - xy_vertices[ll_idx]
    v1 = xy_vertices[v1_idx] - xy_vertices[ll_idx]
    v2 = xy_vertices[v2_idx] - xy_vertices[ll_idx]
    tmp_coord_axis = coord_axis + 1 if coord_axis < vertex_axis else coord_axis
    x_item = [slice(None)] * ndim(v1)
    x_item[tmp_coord_axis] = 0
    y_item = [slice(None)] * ndim(v1)
    y_item[tmp_coord_axis] = 1
    v1_theta = np.arctan2(v1[y_item], v1[x_item]) - np.arctan2(diagonal[y_item], diagonal[x_item])
    lr_idx = np.zeros(xy_vertices.shape, dtype=bool)
    ul_idx = np.zeros(xy_vertices.shape, dtype=bool)
    lr_idx[v1_theta < 0] = v1_theta[v1_theta < 0]
    lr_idx[v1_theta > 0] = v2_theta[v1_theta > 0]
    ul_idx[v1_theta > 0] = v1_theta[v1_theta > 0]
    ul_idx[v1_theta < 0] = v2_theta[v1_theta < 0]
    # Now determine which pairs of lines are parallel, if any.
    # Let m0 be the slope of the line from lower left to lower right,
    # m1 the slope from lower right to upper right,
    # m2 the slope from upper right to upper left
    # and m3 the sloper from upper left to lower left.
    ll = xy_vertices[ll_idx]
    lr = xy_vertices[lr_idx]
    ur = xy_vertices[ur_idx]
    ul = xy_vertices[ul_idx]
    m0 = (lr[y_item] - ll[y_item]) / (lr[x_item] - ll[x_item])
    m1 = (ur[y_item] - lr[y_item]) / (ur[x_item] - lr[x_item])
    m2 = (ul[y_item] - ur[y_item]) / (ul[x_item] - ur[x_item])
    m3 = (ll[y_item] - ul[y_item]) / (ll[x_item] - ul[x_item])
    # Find cases where quadrilateral has two parallel sides and slopes m1 and m3 are parallel.
    # In these cases, the lower left point must be changed to one adjacent to it.
    # Otherwise algorithm for such quadrilaterals will not work.
    vertical_parallels = np.isclose(m1, m3)
    n_parallel_pairs = vertical_parallels + np.isclose(m0, m2)
    wrong_ll_idx = np.logical_and(n_parallel_pairs == 1, vertical_parallels)
    if wrong_ll_idx.any():
        ll_idx[wrong_ll_idx], lr_idx[wrong_ll_idx], ur_idx[wrong_ll_idx], ll_idx[wrong_ll_idx] = \
            lr_idx[wrong_ll_idx], ur_idx[wrong_ll_idx], ul_idx[wrong_ll_idx], ll_idx[wrong_ll_idx]

    # Compute ellipse points for parallograms and other quadrilaterals separately.
    parallelograms_idx = n_parallel_pairs == 2
    quadrilaterals_idx = np.logical_not(parallelograms)

    return ll_idx, lr_idx, ur_idx, ul_idx, parallelograms_idx, quadrilaterals_idx
