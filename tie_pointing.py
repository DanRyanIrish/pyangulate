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


def test_triangulate():
    # Define positions of observers, epipolar origin and feature for test.
    obstime = Time('2021-05-07T18:53:12.000', scale="utc", format="isot")
    solo_loc = SkyCoord(lon=-97.44924725974596*u.deg, lat=-5.557398340410003*u.deg, distance=0.9173762300274045*u.AU,
                        frame=HeliocentricEarthEcliptic, obstime=obstime)
    earth_loc = astropy.coordinates.get_body("Earth", obstime).transform_to(HeliocentricEarthEcliptic)
    sun_loc = SkyCoord(lon=0*u.deg, lat=0*u.deg, distance=0*u.AU,
                       frame=HeliocentricEarthEcliptic(obstime=obstime))
    reference_plane = derive_3d_plane_coefficients(solo_loc.cartesian.xyz.to_value(u.AU),
                                                   earth_loc.cartesian.xyz.to_value(u.AU),
                                                   sun_loc.cartesian.xyz.to_value(u.AU))
    feature_loc = hee_skycoord_from_xyplane(([1, 1, -1, -1 ]*u.R_sun).to(u.AU),
                                            ([1, -1, 1, -1]*u.R_sun).to(u.AU),
                                            *reference_plane, obstime=obstime)
    # Define sample maps representing view from Earth and SolO
    map_earth = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    sample_header = map_earth.wcs.to_header()
    sample_header["DATE-OBS"] = earth_loc.obstime.isot
    sample_header["MJD-OBS"] = earth_loc.obstime.mjd
    sample_header["DSUN_OBS"] = earth_loc.distance.to_value(u.m)
    earth_loc_hg = earth_loc.transform_to(HeliographicStonyhurst(obstime=earth_loc.obstime))
    sample_header["HGLN_OBS"] = earth_loc_hg.lon.to_value(u.deg)
    sample_header["HGLT_OBS"] = earth_loc_hg.lat.to_value(u.deg)
    wcs_earth = WCS(sample_header)
    wcs_earth.pixel_shape = (384, 384)

    map_solo = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    sample_header = map_solo.wcs.to_header()
    sample_header["DATE-OBS"] = solo_loc.obstime.isot
    sample_header["MJD-OBS"] = solo_loc.obstime.mjd
    sample_header["DSUN_OBS"] = solo_loc.distance.to_value(u.m)
    solo_loc_hg = solo_loc.transform_to(HeliographicStonyhurst(obstime=solo_loc.obstime))
    sample_header["HGLN_OBS"] = solo_loc_hg.lon.to_value(u.deg)
    sample_header["HGLT_OBS"] = solo_loc_hg.lat.to_value(u.deg)
    wcs_solo = WCS(sample_header)
    wcs_solo.pixel_shape = (256, 256)
    
    # Derive pixel coords of feature in both image planes.
    x_pix_feature_earth, y_pix_feature_earth = wcs_earth.world_to_pixel(feature_loc)
    x_pix_feature_solo, y_pix_feature_solo = wcs_solo.world_to_pixel(feature_loc)
    
    output = triangulate(earth_loc, solo_loc, sun_loc,
                         x_pix_feature_earth, y_pix_feature_earth, x_pix_feature_solo, y_pix_feature_solo,
                         wcs_earth, wcs_solo)
    assert u.allclose(output.cartesian.xyz.squeeze(), feature_loc.cartesian.xyz, rtol=0.005)


def test_hee_from_hee_xyz():
    expected_lat, expected_lon, expected_r = [-0.12072942]*u.rad, [-1.69241535]*u.rad, np.array([0.9133455])
    input_x, input_y, input_z = 0.11, 0.9, -0.11
    output_lat, output_lon, output_r = hee_from_hee_xyz()
    assert output_lat == expected_lat
    assert output_lon == expected_lon
    assert output_r == expected_r
