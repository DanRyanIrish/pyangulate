import copy
import numbers
from functools import partial

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


def inscribe_ellipse(vertices):
    # Rotate vertices to x-y plane
    coord_axis = -1
    vertex_axis = -2
    # Rotate vertices to xy-plane
    col_vertices = np.swapaxes(vertices, coord_axis, vertex_axis)
    xy_col_vertices, R1 = rotate_plane_to_xy_plane(col_vertices)
    R1_inv = np.linalg.inv(R1)
    xy_vertices = np.swapaxes(col_vertices, coord_axis, vertex_axis)
    # Get indices of vertices that will be in the
    # lower left, lower right, upper right, upper left positions after transformation.
    ll_idx, lr_idx, ur_idx, ul_idx, parallelograms_idx, quadrilaterals_idx = identify_vertices(xy_vertices)

    # Define array to hold vertices of the ellipse, i.e. the center and
    # a point at the end of the semi-major and semi-minor axes.
    planes_shape = vertices.shape[:-2]
    ellipse_vertices = np.zeros(tuple(list(planes_shape) + [3, 3]), dtype=float)

    # Calculate ellipse vertices for parallelograms.
    if parallelograms_idx.any():
        ellipse_vertices[..., :2][parallelograms_idx] = inscribe_ellipse_in_parallelogram(
            np.swapaxes(vertices[parallelograms_idx], coord_axis, vertex_axis),
            ll_idx, lr_idx, ur_idx, ul_idx)

    # Calculate ellipse vertices for quadrilaterals.
    if quadrilaterals_idx.any():
        ellipse_vertices[..., :2][quadrilaterals_idx] = inscribe_ellipse_in_quadrilateral(
            vertices[quadrilaterals_idx], ll_idx, lr_idx, ur_idx, ul_idx)

    # Convert 2-D ellipse vertices to 3-D coords
    ellipse_vertices = R1_inv @ ellipse_vertices

    return ellipse_vertices


def identify_vertices(xy_vertices):
    # Determine which vertices should be lower left, lower right, upper right and upper left
    # after the the isometry is applied.
    # vertices must have shape (..., 4, 3), i.e. (..., vertices, coords)
    coord_axis = -1
    vertex_axis = -2

    # Select lower left vertex (relative to final position after transformation)
    # as the one closest to the origin.
    norms = np.linalg.norm(xy_vertices, axis=coord_axis)
    tmp = norms - norms.min(axis=vertex_axis)
    ll_idx = np.isclose(norms - norms.min(axis=vertex_axis, keepdims=True), 0)  #TODO: expand bool idx to coord axis.
    # Find vertex diagonal to lower left one.
    diagonal_norm = np.linalg.norm(xy_vertices - xy_vertices[ll_idx], axis=coord_axis)
    tmp_vertex_axis = vertex_axis + 1 if coord_axis > vertex_axis else vertex_axis
    ur_idx = np.isclose(diagonal_norm - diagonal_norm.max(axis=tmp_vertex_axis, keepdims=True), 0) #TODO: expand bool idx to coord axis.
    # Get axes of corner vertices relative to lower left.
    # To do this in an array-based way, define v1 as the vertex closer
    # to lower left and v2 as the further from lower left.
    diagonal_norm_sorted = diagonal_norm.sort(axis=tmp_vertex_axis)
    v1_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[...,1], 0) #TODO: expand bool idx to coord axis.
    v2_idx = np.isclose(diagonal_norm - diagonal_norm_sorted[...,2], 0) #TODO: expand bool idx to coord axis.
    # Then set the lower right vertex as the one whose line with the lower left
    # forms a negative with diagonal.
    diagonal = xy_vertices[ur_idx] - xy_vertices[ll_idx]
    v1 = xy_vertices[v1_idx] - xy_vertices[ll_idx]
    v2 = xy_vertices[v2_idx] - xy_vertices[ll_idx]
    v1_theta = np.arctan2(v1[..., 1], v1[..., 0]) - np.arctan2(diagonal[..., 1], diagonal[..., 0])
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
    m0 = (lr[..., 1] - ll[..., 1]) / (lr[..., 0] - ll[..., 0])
    m1 = (ur[..., 1] - lr[..., 1]) / (ur[..., 0] - lr[..., 0])
    m2 = (ul[..., 1] - ur[..., 1]) / (ul[..., 0] - ur[..., 0])
    m3 = (ll[..., 1] - ul[..., 1]) / (ll[..., 0] - ul[..., 0])
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


def rotate_plane_to_xy_plane(points):
    """Rotate locations on the same 2-D plane to the x-y plane.

    Parameters
    ----------
    points: `numpy.ndarray`
        Points on the same 2-D in 3-D space. Can have nay shape as long as penultimate
        axis gives the 3-D coordinates of the points.
        The order of the coordinates must be (x, y, z)
    return_matrix: `bool`
        If True, the rotation matrix used is also returned.

    Returns
    -------
    xy_points: `numpy.ndarray`
        Points rotated to x-y plane. As all z-coords are now 0, last axis is 2-D
        and only gives x-y values.
    rotation: `numpy.ndarray`
        Rotation matrix.
    """
    # Sanitize inputs.
    coord_axis = -2
    vertex_axis = -1
    nd = 3
    if points.ndim == 1:
        if len(points) == nd:
            points = points.reshape((nd, 1))
    elif points.ndim < 1 or points.shape[coord_axis] != nd:
        raise ValueError("Points must be at least 2-D with penultimate axis of length 3. "
                         f"Input shape: {points.shape}")
    # Derive unit vector normal to the plane.
    i = 0
    cross_points = []
    while i < points.shape[vertex_axis] and len(cross_points) < 2:
        point = points[..., i:i+1]
        if not np.any(np.all(point == 0, axis=coord_axis)):
            cross_points.append(point)
        i += 1
    if len(cross_points) < 2:
        raise ValueError("Could not find 2 sets of vertices not including the origin.")
    plane_normal = np.cross(*cross_points)
    plane_normal /= np.linalg.norm(plane_normal, axis=coord_axis, keepdims=True)
    rotation = derive_plane_rotation_matrix(plane_normal, np.array([0, 0, 1]), axis=coord_axis)
    xy_vertices = rotation @ points
    return xy_vertices[..., :2, :], rotation


def derive_plane_rotation_matrix(plane, new_plane):
    """Derive matrix that rotates one plane to another.

    Parameters
    ----------
    plane: `numpy.ndarray`
        A vector normal to the original plane. Vector axis must be length 3,
        i.e. vector must be 3-D. If array is >1D, the coordinates of the vector
        must be given by last axis. The order of the coordinates must be (x, y, z).
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
    coord_axis = -1
    old_norm = np.linalg.norm(old_normal, axis=coord_axis)
    new_norm = np.linalg.norm(new_normal, axis=coord_axis)
    cos = dot_product_single_axis(old_normal, new_normal, axis=coord_axis) / (old_norm * new_norm)
    sin = np.sqrt(1 - cos**2)
    C = 1 - cos
    rot_axis = np.cross(old_normal, new_normal, axis=coord_axis)
    rot_axis /= np.linalg.norm(rot_axis, axis=coord_axis, keepdims=True)
    x_idx, y_idx, z_idx = 0, 1, 2
    blank_item = [slice(None)] * rot_axis.ndim
    item = copy.deepcopy(blank_item)
    item[axis] = x_idx
    x = rot_axis[tuple(item)]
    item = copy.deepcopy(blank_item)
    item[axis] = y_idx
    y = rot_axis[tuple(item)]
    item = copy.deepcopy(blank_item)
    item[axis] = z_idx
    z = rot_axis[tuple(item)]
    R = np.empty(tuple(list(cos.shape) + [2]))
    R[..., 0, 0] = x**2 * C + cos
    R[..., 0, 1] = x * y * C - z * sin
    R[..., 0, 2] = x * z * C + y * sin
    R[..., 1, 0] = y * x * C + z * sin
    R[..., 1, 1] = y**2 * C + cos
    R[..., 1, 2] = y * z * C - x * sin
    R[..., 2, 0] = z * x * C - y * sin
    R[..., 2, 1] = z * y * C + x * sin
    R[..., 2, 2] = z**2 * C + cos
    return R


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


def convert_to_homogeneous_coords(coords, vector=False,
                                  component_axis=-1, coord_axis=-2, homogeneous_idx=-1):
    hom_shape = np.array(coords.shape)
    hom_shape[component_axis] = 1
    if vector:
        hom_component = np.zeros(hom_shape)
    else:
        hom_component = np.ones(hom_shape)
    c = (coords, hom_component)
    if np.isclose(homogeneous_idx, 0):
        c = c[::-1]
    elif not np.isclose(homogeneous_idx, -1):
        raise ValueError(f"homogeneous_idx must be 0 or -1. Given value = {homogeneous_idx}")
    return np.concatenate(c, axis=component_axis)


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
    parametric_ellipse = partial(_parametric_ellipse, a, b, c, d)
    x1, y1 = _parametric_ellipse(phi1)
    x2, y2 = _parametric_ellipse(phi2)
    # Caculate the semi- axes from the distance between the above points and
    # the ellipse center.
    semi_axis1 = np.sqrt((x1 - h)**2 + (y1 - k)**2)
    semi_axis2 = np.sqrt((x2 - h)**2 + (y2 - k)**2)
    # Find major and minor axes.
    xy_axis = -1
    semi_axis = np.stack((semi_axis1, semi_axis2), axis=xy_axis)
    semi_major = semi_axes.max(axis=xy_axis)
    semi_minor = semi_axes.min(axis=xy_axis)

    return semi_major, semi_minor


def _parametric_ellipse(h, k, a, b, c, d, phi):
    x = h + a * np.cos(phi) + b * np.sin(phi)
    y = k + c * np.cos(phi) + d * np.sin(phi)
    return x, y


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
