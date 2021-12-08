import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliocentricEarthEcliptic


def convert_to_homogeneous_coords(coords, component_axis=-1,
                                  trailing_convention=False, vector=False):
    """Convert N-D coordinate(s) to homogeneous coordinates.

    Parameters
    ----------
    coords: `numpy.ndarray`
        Array of coordinates to be converted.
    component_axis: `int`
        The axis of coords array corresponding to the coordinate components, e.g. x, y, z.
    trailing_convention: `bool`
        States which homogeneous convention is to be used.
        True means homogenous component is placed at end of coordinate.
        False means homogenous component is placed at start of coordinate.
    vector: `bool` or boolean array.
        States whether the coordinates are vectors or locations.
        If the coords are vectors (vector=True), the value of the homogenous component is 0.
        If the coords are locations (vector=False), the value of the homogenous component is 1.
        If a boolean array, then must be same shape as coords except 
        that the component axis must be length 1.
        In this case, coordinates corresponding to True will be deemed vectors.
    """
    hom_shape = np.array(coords.shape)
    hom_shape[component_axis] = 1
    if isinstance(vector, bool):
        if vector:
            hom_component = np.zeros(hom_shape)
        else:
            hom_component = np.ones(hom_shape)
    else:
        if vector.shape != hom_shape:
            raise ValueError(f"If vector is an array, it must have a shape of {hom_shape}.")
        hom_component = np.ones(hom_shape)
        hom_component[vector] = 0
    c = (hom_component, coords)
    if trailing_convention:
        c = c[::-1]
    return np.concatenate(c, axis=component_axis)


def dot_product_single_axis(a, b, axis=-1):
    return (a * b).sum(axis=axis)


def calculate_3d_line_direction_vector(p, q):
    """Return direction vector (l, m n) for defining a line in 3-D

    Equation of the line: (x - x0) / l = (y - y0) / m = (z - z0) / n
    where (x0, y0, z0) is any point on the line, e.g. the input point, p
    """
    lmn = q - p
    return lmn


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


def project_point_onto_line(a, b, p, axis=-1):
    """
    Project point, p, onto line defined by points, a, b.

    Works in any number of dimensions.
    Input arrays can have any number of dimensions. But is is assumed that
    the final dimension represents the axes defining the points while other axes
    represent different points.
    """
    ap = p-a
    ab = b-a
    dot_ratio = (dot_product_single_axis(ap, ab, axis=axis)
                 / dot_product_single_axis(ab, ab, axis=axis))
    return a + np.expand_dims(dot_ratio, axis) * ab