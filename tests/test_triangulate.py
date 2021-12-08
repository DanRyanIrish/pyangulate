import astropy.coordinates
import astropy.units as u
import numpy as np
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import HeliocentricEarthEcliptic, HeliographicStonyhurst

from tie_pointing import tie_pointing

__all__ = ["test_triangulate", "test_hee_from_hee_xyz", "test_compute_isometry_matrix"]


def test_triangulate():
    # Define positions of observers, epipolar origin and feature for test.
    obstime = Time('2021-05-07T18:53:12.000', scale="utc", format="isot")
    solo_loc = SkyCoord(lon=-97.44924725974596*u.deg, lat=-5.557398340410003*u.deg, distance=0.9173762300274045*u.AU,
                        frame=HeliocentricEarthEcliptic, obstime=obstime)
    earth_loc = astropy.coordinates.get_body("Earth", obstime).transform_to(HeliocentricEarthEcliptic)
    sun_loc = SkyCoord(lon=0*u.deg, lat=0*u.deg, distance=0*u.AU,
                       frame=HeliocentricEarthEcliptic(obstime=obstime))
    reference_plane = tie_pointing.derive_3d_plane_coefficients(
        solo_loc.cartesian.xyz.to_value(u.AU),
        earth_loc.cartesian.xyz.to_value(u.AU),
        sun_loc.cartesian.xyz.to_value(u.AU))
    feature_loc = tie_pointing.hee_skycoord_from_xyplane(([1, 1, -1, -1 ]*u.R_sun).to(u.AU),
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

    output = tie_pointing.triangulate(earth_loc, solo_loc, sun_loc,
                                      x_pix_feature_earth, y_pix_feature_earth,
                                      x_pix_feature_solo, y_pix_feature_solo,
                                      wcs_earth, wcs_solo)
    assert u.allclose(output.cartesian.xyz.squeeze(), feature_loc.cartesian.xyz, rtol=0.005)
