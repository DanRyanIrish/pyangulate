


def inscribe_max_area_ellipse_in_parallelogram(vertices):
    component_axis = -2
    h, k, a, b, c, d = get_equation_of_max_area_ellipse_in_parallelogram(vertices)
    major_point, minor_point = get_ellipse_semi_axes_coords(h, k, a, b, c, d)
    center = np.stack((h, k), axis=component_axis)
    return center, major_point, minor_point

def get_equation_of_max_area_ellipse_in_parallelogram(vertices):
    """Derive parametic equation of the maximum-area ellipse inscribed in a parallelogram.

    Parameters
    ----------
    vertices: `numpy.ndarray`
        2x4 array giving the 2-D x-y coordinates of the 4 vertices of the parallelogram.
        The 1st axis gives the coordinates of a single vertex while
        the 2nd axis iterates from vertex to vertex.
        The vertices must be in the following order:
        lower left, lower right, upper right, upper left.

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
    homogeneous_idx = 0
    component_axis = -2
    vertex_axis = -1
    vertices = convert_to_homogeneous_coords(
        vertices, vector=False, component_axis=component_axis, coord_axis=vertex_axis,
        homogeneous_idx=homogeneous_idx)
    square_vertices = convert_to_homogeneous_coords(
        square_vertices, vector=False, component_axis=component_axis, coord_axis=vertex_axis,
        homogeneous_idx=homogeneous_idx)
    center = convert_to_homogeneous_coords(
        center, vector=False, component_axis=component_axis, coord_axis=vertex_axis,
        homogeneous_idx=homogeneous_idx)
    square_center = convert_to_homogeneous_coords(
        square_center, vector=False, component_axis=component_axis, coord_axis=vertex_axis,
        homogeneous_idx=homogeneous_idx)

    # Calculate projective collineation
    T = derive_projective_collineation_from_five_points(vertices, square_vertices,
                                                        center, square_center)
    T_inv = np.linalg.inv(T)
    # Extract equation coefficients.
    blank_item = [slice(None)] * T_inv
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
    if not (np.isscalar(h) and np.isscalar(k)) or h.shape != k.shape:
        raise ValueError("h and k must be same shape.")
    # Derive the angles at which the semi-major and -minor axes are located.
    # See Notes section of docstring.
    R = (a**2 + c**2 - b**2 - d**2) / (a*b + c*d)
    phi1 = np.arctan((-R + np.sqrt(R**2 + 4)) / 2)
    phi2 = np.arctan((-R - np.sqrt(R**2 + 4)) / 2)
    # Find the points on the ellipse corresponding to the semi-major and -minor axes.
    ellipse = partial(_parametric_ellipse, a, b, c, d)
    xy_axis = -2
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
    semi_major_coord[major_idx1] = semi_axis1[major_idx1]
    semi_minor_coord[major_idx1] = semi_axis2[major_idx1]
    semi_major_coord[minor_idx1] = semi_axis2[minor_idx1]
    semi_minor_coord[minor_idx1] = semi_axis1[minor_idx1]

    return semi_major_coord, semi_minor_coord


def _parametric_ellipse(h, k, a, b, c, d, phi):
    x = h + a * np.cos(phi) + b * np.sin(phi)
    y = k + c * np.cos(phi) + d * np.sin(phi)
    return x, y

