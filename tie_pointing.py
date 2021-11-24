import numbers

import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst


def triangulate(observer1, observer2, epipolar_origin,
                x_feature_obs1, y_feature_obs1, x_feature_obs2, y_feature_obs2,
                wcs_obs1, wcs_obs2):
    """Reconstruct the 3-D locations from position as seen by two observers.
    
    Use Tie-pointing.
    
    Inputs can be scalar, 1-D or 3-D.
    If 3-D, the first axis must correspond to the number of distinct observer-pair times/locations.
    The second axis must correspond to the number of epipolar planes per observer-pair.
    The third axis must correspond to the number positions being reconstructed per epipolar plane.
    Because of this, the output is 3-D with each axis having the same meaning as above.
    
    Parameters
    ----------
    observer1: `astropy.coordinates.SkyCoord`
        The position of observer 1. Must be scalar, 1-D or 3-D.
    observer2: `astropy.coordinates.SkyCoord`
        The position of observer 2. Must be same shape as observer1.
    epipolar_origin: `astropy.coordinates.SkyCoord`
        The 3rd point defining the epipolar plane on which the feature lies.
        Also acts os the origin of the epipolar plane's coordinate system.
        Must be scalar, 1-, or 3-D. If 3-D, must 0th axis must have length 1 or
        have the same length as axis 0 of observer1/oberver2.
        The third axis must be length 1 or be the same length as the number of 
        positions being reconstructed.
    x_feature_obs1: `float` or `numpy.ndarray`
        The x-pixel coordinate of the feature in the image plane of observer1.
        Must be scalar, 1-D or 3-D. If 3-D 1st axis must be length-1 or equal
        the number of observer-pairs. The 2nd axis must be length-1 or equal
        to the number of epiolar planes per observer-pair.
    y_feature_obs1: `float` or `numpy.ndarray`
        The y-pixel coordinate of the feature in the image plane of observer1.
        Must be scalar, 1-D or 3-D. If 3-D 1st axis must be length-1 or equal
        the number of observer-pairs. The 2nd axis must be length-1 or equal
        to the number of epiolar planes per observer-pair.
    x_feature_obs2: `float` or `numpy.ndarray`
        The x-pixel coordinate of the feature in the image plane of observer2.
        Must be scalar, 1-D or 3-D. If 3-D 1st axis must be length-1 or equal
        the number of observer-pairs. The 2nd axis must be length-1 or equal
        to the number of epiolar planes per observer-pair.
    y_feature_obs2: `float` or `numpy.ndarray`
        The y-pixel coordinate of the feature in the image plane of observer2.
        Must be scalar, 1-D or 3-D. If 3-D 1st axis must be length-1 or equal
        the number of observer-pairs. The 2nd axis must be length-1 or equal
        to the number of epiolar planes per observer-pair.
    wcs_obs1: `astropy.wcs.WCS`
        The WCS describing the pixel <-> world coordinate transformations for observer1.
    wcs_obs2: `astropy.wcs.WCS`
        The WCS describing the pixel <-> world coordinate transformations for observer2.
        
    Returns
    -------
    feature: `astropy.coordinates.SkyCoord`
        The 3-D position(s) of the input feature(s).
        Returned shape had 3 dimensions corresponding to the number of observer-pairs,
        the number of epipolar planes per observer-pair, and the number of features.
    """
    if observer1.isscalar:
        observer1 = observer1.reshape((1,))
    if observer2.isscalar:
        observer2 = observer2.reshape((1,))
    observer_shape = observer1.shape
    if observer2.shape != observer_shape:
        raise ValueError("observer inputs must be same shape")
    output_ndim = 3
    observer1 = np.expand_dims(observer1, tuple(range(-1, -(1 + output_ndim - observer1.ndim), -1)))
    observer2 = np.expand_dims(observer2, tuple(range(-1, -(1 + output_ndim - observer2.ndim), -1)))
    if epipolar_origin.isscalar:
        epipolar_origin = epipolar_origin.reshape((1,))
    if epipolar_origin.ndim < 2:
        epipolar_origin = np.expand_dims(epipolar_origin, 0)
    if epipolar_origin.ndim < output_ndim:
        epipolar_origin = np.expand_dims(epipolar_origin, -1)
    if isinstance(x_feature_obs1, numbers.Real):
        x_feature_obs1 = np.array([x_feature_obs1])
    if isinstance(y_feature_obs1, numbers.Real):
        y_feature_obs1 = np.array([y_feature_obs1])
    if isinstance(x_feature_obs2, numbers.Real):
        x_feature_obs2 = np.array([x_feature_obs2])
    if isinstance(y_feature_obs1, numbers.Real):
        y_feature_obs2 = np.array([y_feature_obs2])
    feature_shape = x_feature_obs1.shape
    if (y_feature_obs1.shape != feature_shape or x_feature_obs2.shape != feature_shape
        or y_feature_obs2.shape != feature_shape):
        raise ValueError("feature inputs must all be same shape.")
    if x_feature_obs1.ndim == 0:
        x_feature_obs1 = x_feature_obs1.reshape((1,))
        y_feature_obs1 = y_feature_obs1.reshape((1,))
        x_feature_obs2 = x_feature_obs2.reshape((1,))
        y_feature_obs2 = y_feature_obs2.reshape((1,))
    if x_feature_obs1.ndim < output_ndim:
        extra_dims = tuple(range(output_ndim - x_feature_obs1.ndim))
        x_feature_obs1 = np.expand_dims(x_feature_obs1, extra_dims)
        y_feature_obs1 = np.expand_dims(y_feature_obs1, extra_dims)
        x_feature_obs2 = np.expand_dims(x_feature_obs2, extra_dims)
        y_feature_obs2 = np.expand_dims(y_feature_obs2, extra_dims)
    feature_pix_obs1 = np.stack((x_feature_obs1, y_feature_obs1), axis=-1)
    feature_pix_obs2 = np.stack((x_feature_obs2, y_feature_obs2), axis=-1)
        
    # Convert observer positions to Heliocentric Earth Ecliptic frame and
    # calculate distances between observers and origin
    observer1 = observer1.transform_to(HeliocentricEarthEcliptic)
    observer2 = observer2.transform_to(HeliocentricEarthEcliptic)
    epipolar_origin = epipolar_origin.transform_to(HeliocentricEarthEcliptic)
    observer1_xyz = observer1.cartesian.xyz
    dist_unit = observer1_xyz.unit
    observer1_xyz = observer1_xyz.value
    observer2_xyz = observer2.cartesian.xyz.to_value(dist_unit)
    epipolar_origin_xyz = epipolar_origin.cartesian.xyz.to_value(dist_unit)
    # Move the cartesian axis to the last position
    observer1_xyz = np.moveaxis(observer1_xyz, 0, -1)
    observer2_xyz = np.moveaxis(observer2_xyz, 0, -1)
    epipolar_origin_xyz = np.moveaxis(epipolar_origin_xyz, 0, -1)
    # Calculate distance between observers and epipolar origin.
    d_1 = get_distance(observer1_xyz, epipolar_origin_xyz)
    d_2 = get_distance(observer2_xyz, epipolar_origin_xyz)
    d_12 = get_distance(observer2_xyz, observer1_xyz)

    # Calculate the angles from the origin to the feature in the image plane of the two observers.
    x_origin_obs1, y_origin_obs1 = wcs_obs1.world_to_pixel(epipolar_origin)
    origin_pix_obs1 = np.stack((x_origin_obs1, y_origin_obs1), axis=-1)
    s_1 = u.Quantity(get_distance(feature_pix_obs1, origin_pix_obs1) * wcs_obs1.wcs.cdelt[0],
                     unit=wcs_obs1.wcs.cunit[0])
    x_origin_obs2, y_origin_obs2 = wcs_obs2.world_to_pixel(epipolar_origin)
    origin_pix_obs2 = np.stack((x_origin_obs2, y_origin_obs2), axis=-1)
    s_2 = u.Quantity(get_distance(feature_pix_obs2, origin_pix_obs2)  * wcs_obs2.wcs.cdelt[0],
                     unit=wcs_obs2.wcs.cunit[0])
    # Determine sign of s_1 and s_2. This depends on whether they to
    # the right (+) of left (-) the line from the image centre to the projection
    # of the epipolar origin.
    # This is best done in pixel units.
    cen_pix_obs1 = np.array(wcs_obs1.pixel_shape) / 2
    cen_pix_obs2 = np.array(wcs_obs2.pixel_shape) / 2
    idx = np.logical_or(np.logical_and(origin_pix_obs1[...,1] >= cen_pix_obs1[...,1],
                                       feature_pix_obs1[...,0] < origin_pix_obs1[...,0]),
                        np.logical_and(origin_pix_obs1[...,1] < cen_pix_obs1[...,1],
                                       feature_pix_obs1[...,0] > origin_pix_obs1[...,0]))
    s_1[idx] *= -1
    idx = np.logical_or(np.logical_and(origin_pix_obs2[...,1] >= cen_pix_obs2[...,1],
                                       feature_pix_obs2[...,0] < origin_pix_obs2[...,0]),
                        np.logical_and(origin_pix_obs2[...,1] < cen_pix_obs2[...,1],
                                       feature_pix_obs2[...,0] > origin_pix_obs2[...,0]))
    s_2[idx] *= -1

    # Let a, b be the 2-D coordinate system on the epipolar plane.
    # Give a, b in same units as x, y with the epipolar_origin input as the origin.
    # Let the a-axis be the line that passes through the origin and
    # is perpendicular to the line joining the observers.
    # Let the positive direction be towards the observers line.
    # Let the b-axis be the line rotated anti-clockwise to the a-axis that
    # passes through the origin.
    # By definition this line is parallel to the Earth-SolO line.
    # The equation of the a-axis is defined by the origin and the
    # origin's projection onto the observers' line, p.
    # This is given by:  p = observer1 + t.d
    # where d is the direction vector from observer1 to observer2,
    # and t is the distance between observer1 and p.
    d = (observer2_xyz - observer1_xyz) / np.expand_dims(d_12, -1)
    t = ((epipolar_origin_xyz - observer1_xyz) * d).sum(axis=-1) # dot product of last axis of both vectors.
    p = observer1_xyz + np.expand_dims(t, -1) * d
    # The direction vector of the 3-D line definining the a-axis this therefore:
    dv_a = calculate_3d_line_direction_vector(epipolar_origin_xyz, p)
    # The direction vector of the b-axis is:
    if observer2.lon < 0 * u.deg:
        dv_b = calculate_3d_line_direction_vector(observer2_xyz, observer1_xyz)
    else:
        dv_b = calculate_3d_line_direction_vector(observer1_xyz, observer2_xyz)
    # Thus observers (a, b) coordinates are (|origin p|, |p Observer|)
    # The sign needs to be worked out depending on the positions of the spacecraft.
    a = get_distance(epipolar_origin_xyz, p)
    p_lon = get_lon_from_xy(p[...,0], p[...,1])
    b_1 = get_distance(observer1_xyz, p)
    b_1[observer1.lon < p_lon] *= -1
    b_2 = get_distance(observer2_xyz, p)
    b_2[observer2.lon < p_lon] *= -1
    observer1_ab = np.stack([a, b_1], axis=-1)
    observer2_ab = np.stack([a, b_2], axis=-1)

    # Find the point of intersection of the lines of view of the feature from the two observers
    # in the a-b coordinate system.
    # Define the vectors, v_e and v_s as the unit vectors pointing from the observers to the origin of the plane.
    v_1 = -1 * observer1_ab / np.expand_dims(d_1, -1)
    v_2 = -1 * observer2_ab / np.expand_dims(d_2, -1)
    # Derive the a-b coordinates of intersection of the lines of view.
    dr_a, dr_b = derive_epipolar_coords_of_a_point(d_2, v_2, s_2, d_1, v_1, s_1)
    # Convert feature's a-b coords to xyz coords
    feature_xyz = (calculate_point_along_3d_line(dv_a,
                                                 epipolar_origin_xyz,
                                                 np.expand_dims(dr_a.value, -1)) +
                   calculate_point_along_3d_line(dv_b,
                                                 epipolar_origin_xyz,
                                                 np.expand_dims(dr_b.value, -1)))
    feature = SkyCoord(feature_xyz[...,0], feature_xyz[...,1], feature_xyz[...,2],
                       unit=[dist_unit] * 3,
                       representation_type='cartesian',
                       frame='heliocentricearthecliptic',
                       obstime=observer1.obstime
                      ).transform_to(HeliocentricEarthEcliptic)
    return feature


def get_distance(p1, p2):
    return np.sqrt(((p2 - p1)**2).sum(axis=-1))


def get_lon_from_xy(x, y):
    lon = np.arctan(y/x)
    pi = np.pi
    lon[x < 0] += pi
    lon[np.logical_and(x >= 0, y < 0)] += 2 * pi
    idx = lon > pi
    lon[idx] = -1 * (2 * pi - lon[idx])
    return lon * u.rad

def test_get_lon_from_xy():
    x = np.arange(-1, 1.1, 0.5)[:, np.newaxis]
    y = np.arange(-1, 1.1, 0.5)[np.newaxis, :]


def calculate_3d_line_direction_vector(p, q):
    """Return direction vector (l, m n) for defining a line in 3-D

    Equation of the line: (x - x0) / l = (y - y0) / m = (z - z0) / n
    where (x0, y0, z0) is any point on the line, e.g. the input point, p
    """
    lmn = q - p
    return lmn


def derive_epipolar_coords_of_a_point(d_1, v_1, s_1,
                                      d_2, v_2, s_2):
    """
    Derive the intersection of lines of view of a feature from two observers.

    The intersection section is defined in a 2-D coordinate system on the epipolar
    plane that includes the two observers and an origin.

    Parameters
    ----------
    d_1, d_2: `float`
        Distance from observer 1 (2) to the origin of the epipolar plane.
    v_1, v_2: `numpy.ndarray`
        The unit vector pointing from observer 1 (2) to the origin.
        Defined in the 2-D coordinate system of the epipolar plane
        and so must be length-2 array.
    s_1, s_2: `float` or `numpy.ndarray`
        The angle from the origin of the epipolar plane to the feature(s)
        in the image plane of observer 1 (2).

    Returns
    -------
    dr_a, dr_b: `float` or `numpy.ndarray`
        The coordinates of the intersection
        in the 2-D coordinate system of the epipolar plane.
    """
    # Derive the unit vector rotated 90 degrees in the epipolar plane
    # from the unit vector pointing from the observer to the origin.
    e_1 = v_1[...,::-1]
    e_1[...,1] *= -1
    e_2 = v_2[...,::-1]
    e_2[...,1] *= -1
    # Derive the terms of the required translation equations.
    b_term1 = d_2 * np.tan(s_2) / e_2[...,0]
    b_term2 = d_1 * np.tan(s_1) / e_1[...,0]
    b_term3 = e_2[...,1] / e_2[...,0]
    b_term4 = e_1[...,1] / e_1[...,0]
    dr_b = (b_term1 - b_term2) / (b_term3 - b_term4)
    dr_a = (d_2 * np.tan(s_2) - e_2[...,1] * dr_b) / e_2[...,0]
    return dr_a, dr_b


def calculate_point_along_3d_line(lmn, p, distance):
    """Return the point along a 3-D a given distance from another point, p.

    Parameters
    ----------
    lmn: length-3 1-d `numpy.ndarray`
        The direction vector of the line.
    p: length-3 1-d `numpy.ndarray`
        The point on the line from which the output point will be calculated.
    distance: `float`
        The distance along the line from p that the output point lies.
    """
    denom = np.sqrt((lmn**2).sum(axis=-1))
    if not isinstance(denom, numbers.Real):
        denom = np.expand_dims(denom, -1)  # Make sure axis is not dropped by summing.
    return p + lmn * distance / denom


def derive_3d_plane_coefficients(p0, p1, p2):
    """
    Derive the coefficients of the equation of a plane given 3 points in that plane.

    The equation of a plane is given by ax + by + cz = d.

    Parameters
    ----------
    p0: `numpy.ndarray`
        Length-3 array giving a point in the 3D plane.

    p1: same as p0

    p2: same as p0

    Returns
    -------
    a, b, c, d:
        The coefficients of the equation of the plane.
    """
    v01 = p1 - p0
    v02 = p2 - p0
    norm = np.cross(v01, v02)
    d = (norm * p0).sum()
    a, b, c = norm
    return a, b, c, d


def hee_skycoord_from_xyplane(x, y, a, b, c, d, obstime=None, dist_unit=None):
    if dist_unit is None:
        if isinstance(x, u.Quantity):
            dist_unit = x.unit
        elif isinstance(y, u.Quantity):
            dist_unit = y.unit
    if isinstance(x, u.Quantity):
        x = x.to_value(dist_unit)
    if isinstance(y, u.Quantity):
        y = y.to_value(dist_unit)
    if np.isscalar(x):
        x = np.array([x])
    if np.isscalar(y):
        y = np.array([y])
    z = (d - (a * x + b * y)) / c
    lon, lat, dist = hee_from_hee_xyz(x, y, z)
    return SkyCoord(lon=lon.flatten().to(u.deg),
                    lat=lat.flatten().to(u.deg),
                    distance=u.Quantity(dist.flatten(), unit=dist_unit),
                    frame=HeliocentricEarthEcliptic(obstime=obstime))


def hee_from_hee_xyz(x, y, z):
    """
    Compute the lat, lon, and distance in the heliocentric Earth ecliptic frame.
    
    Inputs are cartesian with x and y lying in the plane of ecliptic and z is height
    above the ecliptic.  The Earth is assumed to lie at (x, y, z) = (-1, 0, 0)
    """
    # Sanitize inputs
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
    n = max([len(x), len(y), len(z)])
        
    # Calculate lon
    # Combine x and y coords into single array where each row is a position.
    xy = np.concatenate((x.reshape(n, 1), y.reshape(n, 1)), axis=1)
    earth = np.zeros(xy.shape)
    earth[:, 0] = 1
    # Use formula for cosine between two vectors to get longitude.
    r_xy = np.linalg.norm(xy, axis=1)
    cos_lon = np.prod((earth, xy), axis=0).sum(axis=1) / (1 * r_xy)
    lon = np.arccos(cos_lon)
    # If angle opens to to the east, reverse sign.
    neg_idx = y < 0
    lon[neg_idx] = -1 * lon[neg_idx]
    
    # Calculate latitude using tan formula
    lat = np.arctan(z / r_xy)
    
    # Calculate distance from sun center
    r = z / np.sin(lat)
    
    return lon * u.rad, lat * u.rad, r


def isometerize_vertices(vertices):
    """
    Apply an isometry so the lower left is at the origin and upper left is on the y-axis.

    The vertices are given in 3-D space and are assumed to all be in the same plane.
    Therefore vertices is a 2-D array where the last axis is length 3 and 
    represent the 3 coordinates. The penultimate axis indexes between vertices where
    vertices[...,0, :] is the lower left, vertices[...,1, :] is the lower right,
    vertices[...,2, :] is the upper left, and vertices[...,3, :] is the upper right.
    All other axes index through different sets of vertices.

    This function converts points to a 2-D planar geometry, with axes, u,v.
    The origin is the first vertex.
    The positive direction of the v-axis is the line from the first to the 3rd vertex.
    The positive direction of the u-axis is the v-axis rotated clockwise by 90 degrees.
    Project points so they have the following 2-D coords (u, v):
    vertices[...,0, :]: (0, 0)
    vertices[...,1, :]: (A, B)
    vertices[...,2, :]: (0, C)
    vertices[...,3, :]: (s, t)
    """
    if vertices.ndim < 2 or vertices.shape[-2] != 4:
        raise ValueError(
            "'vertices' must have >1 dimension and penultimate dimension must have length of 4.")
        vertices = np.expand_dims(vertices, 0)
    lower_left, upper_left, right_vertices = \
        vertices[..., 0:1, :], vertices[..., 2:3, :], vertices[..., 1::2, :]
    # First find C, i.e. the distance between the 1st and 3rd vertices.
    C = get_distance(lower_left, upper_left)[0]
    # Next, project the lower and top right vertices onto the v-axis.
    v_projections = project_point_onto_line(lower_left, upper_left, right_vertices)
    # Calculate the v-coordinates of the lower and upper right vertices by getting the
    # ditance between their v-projections and the origin, and then comparing them
    # with C to get the sign.
    v_coords = get_distance(v_projections, lower_left)
    vc_dist = get_distance(v_projections, upper_left)
    tol = np.stack((vc_dist, v_coords), axis=-1).min(axis=-1) * 1e-5
    idx == np.isclose(v_coords + C, rtol=0, atol=tol)
    v_coords[idx] *= -1
    B = v_coords[..., 0]
    t = v_coords[..., 1]
    # Find the distance from the right vertices to their projections to get their u-coordinates.
    # By definition the u-coordinates of these vertices are positive.
    u_coords = get_distance(v_projections, right_vertices)
    A = u_coords[..., 0]
    s = c_coords[..., 1]
    return A, B, C, s, t

def compute_isometry_matrix(lower_left, upper_left, plane_normal):
    # Calculate the affine transformation matrix that transforms any 3-D point
    # (x, y, z) by translating it by -lower_left (i.e. the origin is moved to lower_left),
    # rotates it so the plane on which the point lies is paralle to the z-axis,
    # and the line from the lower_left to upper_left points is along the y-axis.

    # Start by computing unit vectors, u, v, w, where
    # the origin is the lower left vertex,
    # the w-axis is normal to the plane,
    # the v-axis is along the lower left-upper left line with the positive direction
    # going from lower left to upper left,
    # and the u-axis is the cross-product of the w- and v-axes.
    w = plane_normal / np.expand_dims(get_distance(plane_normal, np.zeros(plane_normal.shape)), -1)
    c_line_norm = get_distance(lower_left, upper_left)
    v = (upper_left - lower_left) / np.expand_dims(c_line_norm, -1)
    u = np.cross(v, w, axis=-1) # j x k = i; k x j = -i
    # Translation matrix translates so that lower left become the origin.
    T = np.identity(4)
    T[:3, 3] = -lower_left
    # First rotation is to rotate the point so the normal of the plane is parallel to the z-axis.
    # This is defined by the row vectors of the unit vectors, u, v, w.
    R1 = np.identity(4)
    R1[:-1, :-1] = np.stack((u, v, w), axis=-2)
    # Second rotation is to rotate around the plane normal (now the z-axis) so the 
    # lower left -> upper left line aligns with the y-axis.
    # To calculate the angle between this line and the y-axis,
    # apply the translation and R1 rotation, and then use the tan^-1.
    # Start by converting the upper left coord to homogenous coords, i.e. an append
    # a 4th point to final axis whose value is one.
    hom_shape = np.array(upper_left.shape)
    hom_shape[..., -1] = 1
    hom_upper_left = np.concatenate((upper_left, np.ones(hom_shape)), -1)
    if hom_upper_left.ndim == 1:
        hom_upper_left = np.expand_dims(hom_upper_left, -1)
    else:
        hom_upper_left, np.swapaxes(hom_upper_left, -1, -2)
    partial_isometry = R1 @ T
    tmp_upper_left = np.matmul(partial_isometry, hom_upper_left)
    theta = np.arctan(tmp_upper_left[..., 0, :] / tmp_upper_left[..., 1, :])
    theta = theta.squeeze()
    R2 = np.identity(4)
    R2[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    isometry = R2 @ partial_isometry
    return isometry


def deisometerize(points, origin, v_axis_coeffs, plane_coeffs):
    pass    


def fit_ellipse_within_quadrilateral(A, B, C, s, t):
    # Fix ellipse center, (h, k), by setting h as somewhere along the
    # open line segment connecting the midpoints of the diagonals of
    # the quadrilateral. Determine k from the equation of a line.
    h = 0.5 * (s/2 + A/2)
    k = (h - s/2) * ((t - B - C) / (s - A)) + t/2
    # To solve for the ellipse tangent to the four sides of the quadrilateral,
    # we can solve for the ellipse tangent to the three sides of a triangle
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
    # and we can then solve for the ellipse semimajor axis a and semiminor axis b
    # from the equations:
    R = np.sqrt((r1**2 + r2**2) / (16 * s_A**4))
    W = 0.25 * (C / s_A**2) * (2 * (B*t - A*(t - c))*h - A*C*s) * (2*h - A) * (2*h - s)
    a = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) + R))
    b = np.sqrt(0.5 * (np.sqrt(R**2 + 4*W) - R))
    # Knowing the axes, we can generate the ellipse and float its tilt angle d until it
    # sits tangent to each side of the quadrilateral, using the inclined ellipse equation:
    

def project_point_onto_line(a, b, p):
    """
    Project point, p, onto line defined by points, a, b.

    Works in any number of dimensions.
    Input arrays can have any number of dimensions. But is is assumed that
    the final dimension represents the axes defining the points while other axes
    represent different points.
    """
    ap = p-a
    ab = b-a
    dot_ratio = dot_product_single_axis(ap, ab) / dot_product_single_axis(ab, ab)
    return a + np.expand_dims(dot_ratio, -1) * ab


def dot_product_single_axis(a, b, axis=-1):
    return (a * b).sum(axis=axis)
