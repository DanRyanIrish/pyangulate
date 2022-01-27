import numbers

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliocentricEarthEcliptic

from tie_pointing import utils


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
    observer1_xyz = observer1_xyz.value
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
    # Given a, b in same units as x, y with the epipolar_origin input as the origin.
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
    dv_a = utils.calculate_3d_line_direction_vector(epipolar_origin_xyz, p)
    # The direction vector of the b-axis is:
    if observer2.lon < 0 * u.deg:
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
    dr_a, dr_b = derive_epipolar_coords_of_a_point(d_2, v_2, s_2, d_1, v_1, s_1)
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


def _get_lon_from_xy(x, y):
    lon = np.arctan(y/x)
    pi = np.pi
    lon[x < 0] += pi
    lon[np.logical_and(x >= 0, y < 0)] += 2 * pi
    idx = lon > pi
    lon[idx] = -1 * (2 * pi - lon[idx])
    return lon * u.rad


def derive_epipolar_coords_of_a_point(d_1, v_1, s_1, d_2, v_2, s_2):
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
