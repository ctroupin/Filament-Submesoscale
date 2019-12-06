import requests
import datetime
import re
import os
import netCDF4
import numpy as np
from lxml import html
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from osgeo import gdal, osr
import logging

def get_filelist_url(year, dayofyear):
    """
    Generate a list of file URLs (OPEnDAP) for the netCDF corresponding to `year` and `dayofyear`
    """

    urllist = []

    for mission in ["metop_a", "metop_b"]:
        baseurl = "https://opendap.jpl.nasa.gov/opendap/OceanWinds/ascat/preview/L2/{}/coastal_opt/{}/{}/contents.html".format(mission, year, str(dayofyear).zfill(3))
        opendapurl = "https://opendap.jpl.nasa.gov:443/opendap/OceanWinds/ascat/preview/L2/{}/coastal_opt/".format(mission)

        r = requests.get(baseurl)
        content = r.content
        soup = BeautifulSoup(content, "html.parser")

        for link in soup.find_all('a'):
            datalink = link.get('href')
            if datalink.startswith("ascat_") & datalink.endswith(".gz"):
                dataurl = os.path.join(opendapurl, str(year), str(dayofyear).zfill(3), datalink)
                urllist.append(dataurl)

    logger.info("Found {} files".format(len(urllist)))

    return urllist

def read_data_domain(dataurl, domain):
    """
    Read the wind data from the file is measurements are available in the domain
    Otherwise return None
    """

    lon2plot, lat2plot, uwind, vwind, speed = None, None, None, None, None

    try:
        # Read the coordinates
        with netCDF4.Dataset(dataurl) as nc:
            lon = nc.variables["lon"][:] - 360.
            lat = nc.variables["lat"][:]

        # Check if we have data in the domain of interest
        goodlon = (lon <= domain[1]) & (lon >= domain[0])
        goodlat = (lat <= domain[3]) & (lat >= domain[2])
        goodcoord = (lon <= domain[1]) & (lon >= domain[0]) & (lat <= domain[3]) & (lat >= domain[2])
        ngood = goodcoord.sum().sum()

        if ngood > 0:
            logger.info("Subsetting data to region of interest")
            lon2plot = lon[goodcoord]
            lat2plot = lat[goodcoord]
            with netCDF4.Dataset(dataurl) as nc:
                windspeed = nc.variables["wind_speed"][:]
                winddir = nc.variables["wind_dir"][:]
                winddir2plot = winddir[goodcoord]
                speed = windspeed[goodcoord]

            uwind = speed * np.sin(np.deg2rad(winddir2plot))
            vwind = speed * np.cos(np.deg2rad(winddir2plot))
        else:
            logger.info("No data in the region of interest")
    except OSError:
        logger.warning("Problem to access the file")

    return lon2plot, lat2plot, uwind, vwind, speed

def plot_wind_sat(lon, lat, u, v, speed, arr, extent, figname):

    myproj = ccrs.PlateCarree()
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection=myproj)
    ax.quiver(lon, lat, u, v, speed, scale=400, width=.001, cmap=plt.cm.hot_r, clim=[5, 20])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.add_wms(wms='http://ows.emodnet-bathymetry.eu/wms',
                   layers=['coastlines'])
    ax.imshow(arr, origin='upper', extent=extent, transform=myproj)
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close()
