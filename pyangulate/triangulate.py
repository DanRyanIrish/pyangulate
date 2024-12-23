import numbers

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliocentricEarthEcliptic

from pyangulate import utils


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
    if isinstance(y_feature_obs2, numbers.Real):
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
    observer1_xyz = observer1_xyz.to_value(dist_unit)
    observer2_xyz = observer2.cartesian.xyz.to_value(dist_unit)
    epipolar_origin_xyz = epipolar_origin.cartesian.xyz.to_value(dist_unit)
    # Move the cartesian axis to the last position
    observer1_xyz = np.moveaxis(observer1_xyz, 0, -1)
    observer2_xyz = np.moveaxis(observer2_xyz, 0, -1)
    epipolar_origin_xyz = np.moveaxis(epipolar_origin_xyz, 0, -1)
    # Calculate distance between observers and epipolar origin.
    d_1 = np.linalg.norm(observer1_xyz - epipolar_origin_xyz, axis=-1)
    d_2 = np.linalg.norm(observer2_xyz - epipolar_origin_xyz, axis=-1)
    d_12 = np.linalg.norm(observer2_xyz - observer1_xyz, axis=-1)

    # Calculate the angles from the origin to the feature in the image plane of the two observers.
    x_origin_obs1, y_origin_obs1 = wcs_obs1.world_to_pixel(epipolar_origin)
    origin_pix_obs1 = np.stack((x_origin_obs1, y_origin_obs1), axis=-1)
    s_1 = u.Quantity(
        np.linalg.norm(feature_pix_obs1 - origin_pix_obs1, axis=-1) * wcs_obs1.wcs.cdelt[0],
        unit=wcs_obs1.wcs.cunit[0])
    x_origin_obs2, y_origin_obs2 = wcs_obs2.world_to_pixel(epipolar_origin)
    origin_pix_obs2 = np.stack((x_origin_obs2, y_origin_obs2), axis=-1)
    s_2 = u.Quantity(
        np.linalg.norm(feature_pix_obs2 - origin_pix_obs2, axis=-1)  * wcs_obs2.wcs.cdelt[0],
        unit=wcs_obs2.wcs.cunit[0])
    # Determine sign of s_1 and s_2. This depends on whether they to
    # the right (+) of left (-) the line from the image centre to the projection
    # of the epipolar origin.
    # This is best done in pixel units.
    cen_pix_obs1 = np.array(wcs_obs1.pixel_shape) / 2
    idx = _get_s_sign_idx(origin_pix_obs1, feature_pix_obs1)
    s_1[idx] *= -1
    cen_pix_obs2 = np.array(wcs_obs2.pixel_shape) / 2
    idx = _get_s_sign_idx(origin_pix_obs2, feature_pix_obs2)
    s_2[idx] *= -1

    # Let a, b be the 2-D coordinate system on the epipolar plane.
    # Given a, b in same units as x, y with the epipolar_origin input as the origin.
    # Let the a-axis be the line that passes through the origin and
    # is perpendicular to the line joining the observers.
    # Let the positive direction be towards the observers line.
    # Let the b-axis be the line rotated anti-clockwise to the a-axis that
    # passes through the origin.
    # By definition this line is parallel to the line joining the observers (epipole).
    # The equation of the a-axis is defined by the origin and the
    # origin's projection onto the observers' line, p.
    # This is given by:  p = observer1 + t.d
    # where d is the direction vector from observer1 to observer2,
    # and t is the distance between observer1 and p.
    d = (observer2_xyz - observer1_xyz) / np.expand_dims(d_12, -1)
    t = ((epipolar_origin_xyz - observer1_xyz) * d).sum(axis=-1) # dot product of last axis of both vectors.
    p = observer1_xyz + np.expand_dims(t, -1) * d
    # The direction vector of the 3-D line definining the a-axis this therefore:
    dv_a = utils.calculate_3d_line_direction_vector(epipolar_origin_xyz, p)
    # The direction vector of the b-axis is:
    if observer2.lon < observer1.lon:
        dv_b = utils.calculate_3d_line_direction_vector(observer2_xyz, observer1_xyz)
    else:
        dv_b = utils.calculate_3d_line_direction_vector(observer1_xyz, observer2_xyz)
    # Thus observers (a, b) coordinates are (|origin p|, |p Observer|)
    # The sign needs to be worked out depending on the positions of the spacecraft.
    a = np.linalg.norm(p - epipolar_origin_xyz, axis=-1)
    p_lon = _get_lon_from_xy(p[...,0], p[...,1])
    b_1 = np.linalg.norm(p - observer1_xyz, axis=-1)
    b_1[observer1.lon < p_lon] *= -1
    b_2 = np.linalg.norm(p - observer2_xyz, axis=-1)
    b_2[observer2.lon < p_lon] *= -1
    observer1_ab = np.stack([a, b_1], axis=-1)
    observer2_ab = np.stack([a, b_2], axis=-1)

    # Find the point of intersection of the lines of view of the feature from the two observers
    # in the a-b coordinate system.
    # Define the vectors, v_e and v_s as the unit vectors pointing from the observers to the origin of the plane.
    v_1 = -1 * observer1_ab / np.expand_dims(d_1, -1)
    v_2 = -1 * observer2_ab / np.expand_dims(d_2, -1)
    # Derive the a-b coordinates of intersection of the lines of view.
    dr_a, dr_b = derive_epipolar_coords_of_a_point_affine(d_2, v_2, s_2, d_1, v_1, s_1)
    # Convert feature's a-b coords to xyz coords
    feature_xyz = (
            (utils.calculate_point_along_3d_line(dv_a,
                                                 epipolar_origin_xyz,
                                                 np.expand_dims(dr_a.value, -1))
             + utils.calculate_point_along_3d_line(dv_b,
                                                   epipolar_origin_xyz,
                                                   np.expand_dims(dr_b.value, -1))
                  ) - epipolar_origin_xyz)

    feature = SkyCoord(feature_xyz[...,0], feature_xyz[...,1], feature_xyz[...,2],
                       unit=[dist_unit] * 3,
                       representation_type='cartesian',
                       frame='heliocentricearthecliptic',
                       obstime=observer1.obstime
                      ).transform_to(HeliocentricEarthEcliptic)
    return feature


def _get_s_sign_idx(origin_pix, feature_pix):
    """Calculate sign of angle, s, from epipolar origin to feature.

    This depends on whether they to the right (+) of left (-) the line from the
    image centre to the projection of the epipolar origin.  The indices returned
    indicate which values of the s array that should be negative.

    Parameters
    ----------
    origin_pix: `numpy.ndarray`
        The x and y pixel values of the epipolar origin in the image plane.
        Final dimension must be length 2 with 0th element giving the x coord and
        the other giving the y coord.  Preceding dimensions must be same shape as
        the array holding the s angle values.
    feature_pix: `numpy.ndarray`
        The x and y pixel values of the feature, which together with the epipolar origin,
        defines the angle s.  Must be same shape and format as origin_pix

    Returns
    -------
    idx: `numpy.ndarray`
        The indices of the array holding the angle s which are negative.
    """
    return feature_pix[...,0] < origin_pix[...,0]


def _get_lon_from_xy(x, y):
    lon = np.arctan(y/x)
    pi = np.pi
    lon[x < 0] += pi
    lon[np.logical_and(x >= 0, y < 0)] += 2 * pi
    idx = lon > pi
    lon[idx] = -1 * (2 * pi - lon[idx])
    return lon * u.rad


def derive_epipolar_coords_of_a_point_affine(d_1, v_1, s_1, d_2, v_2, s_2):
    """
    Derive intersection of lines of view to a feature from 2 observers assuming affine geometry.

    The intersection section is defined in a 2-D coordinate system on the epipolar plane that
    includes the two observers and an origin.
    The assumption of affine relates to the assumption that the lines of sight from the
    observer to the epipolar origin and from the observer to the feature are parallel, i.e.
    The angle between the lines of sight is small.  This assumption may lead to inaccuracies
    even when that angle is small.

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

    Notes
    -----
    Equations below are derived from Equation 5 in [1] which assumed an affine geometry.

    References
    ----------
    [1]: Inhester, Stereoscopy basics for the STEREO Mission, ISSI, 2006
    """
    # Derive the unit vector rotated 90 degrees in the epipolar plane
    # from the unit vector pointing from the observer to the origin.
    e_1 = v_1[...,::-1].copy()
    e_1[...,1] *= -1
    e_2 = v_2[...,::-1].copy()
    e_2[...,1] *= -1
    # Derive the terms of the required translation equations.
    b_term1 = d_2 * np.tan(s_2) / e_2[...,0]
    b_term2 = d_1 * np.tan(s_1) / e_1[...,0]
    b_term3 = e_2[...,1] / e_2[...,0]
    b_term4 = e_1[...,1] / e_1[...,0]
    dr_b = (b_term1 - b_term2) / (b_term3 - b_term4)
    dr_a = (d_2 * np.tan(s_2) - e_2[...,1] * dr_b) / e_2[...,0]
    return dr_a, dr_b


def derive_epipolar_coords_of_a_point(d_1, v_1, s_1, d_2, v_2, s_2):
    """
    Derive intersection of lines of view to a feature from 2 observers.

    The intersection section is defined in a 2-D coordinate system on the epipolar plane that
    includes the two observers and an origin. The translations used in this function assume
    a projective geometry. This does not suffer from the sam inaccuracies as teh assumption
    of affine geometry.

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

    Notes
    -----
    Equation below are derived from Equation 7 in [1]

    References
    ----------
    [1]: Inhester, Stereoscopy basics for the STEREO Mission, ISSI, 2006
    """
    # Derive the unit vector rotated 90 degrees in the epipolar plane
    # from the unit vector pointing from the observer to the origin.
    e_1 = v_1[...,::-1].copy()
    e_1[...,1] *= -1
    e_2 = v_2[...,::-1].copy()
    e_2[...,1] *= -1
    # Derive the terms of the required translation equations.
    alpha = e_1 - v_1 * np.tan(s_1)[...,np.newaxis]
    gamma = e_2 - v_2 * np.tan(s_2)[...,np.newaxis]
    b_term1 = d_2 * np.tan(s_2) / gamma[...,0]
    b_term2 = d_1 * np.tan(s_1) / alpha[...,0]
    b_term3 = gamma[...,1] / gamma[...,0]
    b_term4 = alpha[...,1] / alpha[...,0]
    dr_b = (b_term1 - b_term2) / (b_term3 - b_term4)
    dr_a = (d_2 * np.tan(s_2)) / gamma[...,1] - gamma[...,0] / gamma[...,1] * dr_b
    return dr_a, dr_b


def define_epipolar_planes(epipolar_origins, loc1, loc2, wcs1, wcs2):
    """
    Define a set of epipolar planes in the image frames of the two observers.

    Parameters
    ----------
    epipolar_origins: `astropy.coordinates.SkyCoord`
        The points in 3-D space that define each epipolar plane along with
        positions of th two observers. Must be iterable, i.e. not scalar.
        Also, frame must be transformable to HeliocentricEarthEcliptic.
    loc1: `astropy.coordinates.SkyCoord`
        The location of observer 1.  Must be scalar.
    loc2: `astropy.coordinates.SkyCoord`
        The location of observer 2.  Must be scalar.
    wcs1:
        The WCS describing observer 1's image plane. Must be APE-14-API-compliant.
    wcs2:
        The WCS describing observer 2's image plane. Must be APE-14-API-compliant.

    Returns
    -------
    epipolar_lines1: `numpy.ndarray`
        The coefficients of the lines in observer 1's image plane in pixel coordinates
        corresponding to the 
    epipolar_lines2: `numpy.ndarray`
        The coefficients of the lines in observer 2's image plane in pixel coordinates
        corresponding to the epipolar planes.
    """
    if epipolar_origins.isscalar:
        epipolar_origins.reshape((1,))
    epipolar_planes = []
    epipolar_lines1 = []
    epipolar_lines2 = []
    epipolar_points1 = []
    epipolar_points2 = []
    dist_unit = u.R_sun
    base_frame = HeliocentricEarthEcliptic
    epipolar_origins = epipolar_origins.transform_to(base_frame)
    loc1 = loc1.transform_to(base_frame)
    loc2 = loc2.transform_to(base_frame)
    for i, epipolar_origin in enumerate(epipolar_origins):
        epipolar_planes.append(utils.derive_3d_plane_coefficients(
            epipolar_origin.cartesian.xyz.to_value(dist_unit),
            loc1.cartesian.xyz.to_value(dist_unit),
            loc2.cartesian.xyz.to_value(dist_unit)))
        anchor1_x, anchor1_y = 1 * dist_unit, -1 * dist_unit
        anchor2_x, anchor2_y = -1 * dist_unit, 1 * dist_unit
        # Calculate anchor points in 3-D space to enable the evaluation and plotting of epipolar lines.
        anchor1 = utils.hee_skycoord_from_xyplane(
            anchor1_x, anchor1_y, *epipolar_planes[-1], obstime=loc1.obstime)
        anchor2 = utils.hee_skycoord_from_xyplane(
            anchor2_x, anchor2_y, *epipolar_planes[-1], obstime=loc1.obstime)

        # Calculate the pixel coords of the anchor points in the image planes of both observers.
        anchor1_x1, anchor1_y1 = wcs1.world_to_pixel(anchor1)
        anchor1_x1, anchor1_y1 = float(anchor1_x1), float(anchor1_y1)
        anchor2_x1, anchor2_y1 = wcs1.world_to_pixel(anchor2)
        anchor2_x1, anchor2_y1 = float(anchor2_x1), float(anchor2_y1)
        epipolar_points1.append((anchor1_x1, anchor1_y1, anchor2_x1, anchor2_y1))
        epipolar_lines1.append(get_line_equation_coeffs_2d(*epipolar_points1[-1]))

        anchor1_x2, anchor1_y2 = wcs2.world_to_pixel(anchor1)
        anchor1_x2, anchor1_y2 = float(anchor1_x2), float(anchor1_y2)
        anchor2_x2, anchor2_y2 = wcs2.world_to_pixel(anchor2)
        anchor2_x2, anchor2_y2 = float(anchor2_x2), float(anchor2_y2)
        epipolar_points2.append((anchor1_x2, anchor1_y2, anchor2_x2, anchor2_y2))
        epipolar_lines2.append(get_line_equation_coeffs_2d(*epipolar_points2[-1]))
    return np.array(epipolar_lines1), np.array(epipolar_lines2)


def define_epipolar_planes_from_solar_rotation_axis(lower, upper, loc1, loc2, wcs1, wcs2, n_planes):
    """
    Define a set of epipolar planes in the image frames of the two observers.

    The 3rd points (in addition to the locations of the two observers) that
    define the epipolar planes are themselves defined by their distance from
    Sun-center along the Sun's rotational axis.  The returned planes are
    spaced evenly between the input lower and upper limits.

    Parameters
    ----------
    lower: `astropy.units.Quantity`
        The distance from Sun-center along the Sun's rotational axis of the
        epipolar plane that bounds the lower edge of the source.
        Must be scalar.
    upper: `astropy.units.Quantity`
        The distance from Sun-center along the Sun's rotational axis of the
        epipolar plane that bounds the upper edge of the source.
        Must be scalar.
    loc1: `astropy.coordinates.SkyCoord`
        The location of observer 1.  Must be scalar.
    loc2: `astropy.coordinates.SkyCoord`
        The location of observer 2.  Must be scalar.
    wcs1:
        The WCS describing observer 1's image plane. Must be APE-14-API-compliant.
    wcs2:
        The WCS describing observer 2's image plane. Must be APE-14-API-compliant.
    n_planes: `int`
        The number of evenly spaced epipolar planes desired between the lower
        and upper bounding planes.

    Returns
    -------
    epipolar_lines1: `numpy.ndarray`
        The coefficients of the lines in observer 1's image plane in pixel coordinates
        corresponding to the epipolar planes. 
    epipolar_lines2: `numpy.ndarray`
        The coefficients of the lines in observer 2's image plane in pixel coordinates
        corresponding to the epipolar planes.
    epipolar_origins: `astropy.coordinates.SkyCoord`
        The 3-D locations that define the epipolar planes along with the two observers'
        positions.
    """
    distances = np.linspace(lower.value, upper.to_value(lower.unit), n_planes) * lower.unit
    epipolar_origins = SkyCoord(lon=np.zeros(n_planes)*u.deg,
                                lat=(np.zeros(n_planes) + 90)*u.deg,
                                distance=distances,
                                frame=HeliocentricEarthEcliptic,
                                obstime=loc1.obstime)
    epipolar_lines1, epipolar_lines2 = define_epipolar_planes(epipolar_origins, loc1, loc2,
                                                              wcs1, wcs2)
    return epipolar_lines1, epipolar_lines2, epipolar_origins


def project_point_to_line_2D(point, line):
    """
    Find the point on a line that is closest to a given point.
    
    That is the point of intersection between the line and the
    line perpdendicular to it that passes through the point.
    
    Parameters
    ----------
    point: `numpy.ndarray`
       The x-y coords of the point. Can be any number of dimensions
       but last one must correspond to x, y in that order.
    line: `numpy.ndarray`
        The line coefficients of the line.  Must be same shape as point
        input with last dimension corresponding to the intercept and
        slope, in that order.
        
    Returns
    -------
    point_intersect: `numpy.ndarray`
        Projection of point onto line. Same shape as input point.
    """
    x, y = point[...,0], point[...,1]
    intercept, slope = line[...,0], line[...,1]
    perp_slope = -1 / slope
    perp_intercept = y - perp_slope * x
    x_intersect = (perp_intercept - intercept) / (slope - perp_slope)
    y_intersect = slope * x_intersect + intercept
    point_intersect = np.stack((x_intersect, y_intersect), axis=-1)
    return point_intersect
