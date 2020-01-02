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
import cartopy.mpl.ticker as cartopyticker
lon_formatter = cartopyticker.LongitudeFormatter()
lat_formatter = cartopyticker.LatitudeFormatter()
from osgeo import gdal, osr
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.info("Starting")


def decorate_map(ax, domain, xt, yt):
    """
    ```python
    decorate_map(ax, domain, xt, yt)
    ```
    Add labels to the map axes and limit the extent according the selected
    `domain`.

    Inputs:
    ------
    ax: a cartopy ax instance
    domain: a 4-element array storing lonmin, lonmax, latmin, latmax
    xt: array storing the xticks locations
    yt: array storing the yticks locations
    """

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[2], domain[3])

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

class Wind(object):

    def __init__(self, lon=None, lat=None, u=None, v=None, speed=None, angle=None):
        self.lon = lon
        self.lat = lat
        self.u = u
        self.v = v
        self.speed = speed
        self.angle = angle

    def read_from_ccmp(self, datafile, domain=None):
        """
        ```python
        read_from_ccmp(datafile, coordinates)
        ```
        Read the wind data from the CCMP product in `datafile` and subset in the region
        specified by `domain`
        """

        with netCDF4.Dataset(datafile, "r") as nc:
            self.lon = nc.get_variables_by_attributes(standard_name="longitude")[0][:]
            self.lat = nc.get_variables_by_attributes(standard_name="latitude")[0][:]
            self.lon[self.lon > 180.] -= 360.
            time = nc.get_variables_by_attributes(standard_name="time")[0][:]
            ntime = len(time)

            if domain is not None:
                goodlon = np.where( (self.lon<= domain[1]) & (self.lon>= domain[0]))[0]
                goodlat = np.where( (self.lat<= domain[3]) & (self.lat>= domain[2]))[0]
                self.lon = self.lon[goodlon]
                self.lat = self.lat[goodlat]

                if ntime == 12:
                    # Climatology product
                    self.uwind = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][:, goodlat, goodlon]
                    self.vwind = nc.get_variables_by_attributes(standard_name="northward_wind")[0][:, goodlat, goodlon]
                    self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:, goodlat, goodlon]
                else:
                    # Monthly product
                    self.uwind = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][goodlat, goodlon]
                    self.vwind = nc.get_variables_by_attributes(standard_name="northward_wind")[0][goodlat, goodlon]
                    self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][goodlat, goodlon]

            else:
                self.uwind = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][:]
                self.vwind = nc.get_variables_by_attributes(standard_name="northward_wind")[0][:]
                self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:]


    def read_from_scow(self, ufile, vfile, domain=None):
        """
        ```python
        read_from_scow(ufile, vfile, domain)
        ```
        Read the wind field from the SCOW product stored in the files
        `ufile` and `vfile`, and subset in the region specified by `domain`
        """
        if os.path.exists(ufile):
            with netCDF4.Dataset(ufile, "r") as nc:
                self.lat = nc.variables["latitude"][:]
                self.lon = nc.variables["longitude"][:]
                self.lon[self.lon > 180.] -= 360.

                if domain is not None:
                    goodlon = np.where( (self.lon<= domain[1]) & (self.lon>= domain[0]))[0]
                    goodlat = np.where( (self.lat<= domain[3]) & (self.lat>= domain[2]))[0]
                    self.lon = self.lon[goodlon]
                    self.lat = self.lat[goodlat]

                self.uwind = np.empty((len(self.lat), len(self.lon), 12))
                windstress_vars = nc.get_variables_by_attributes(units="N/m^2")
                for i in range(0, 12):
                    if coordinates is not None:
                        self.uwind[:,:,i] = windstress_vars[i][goodlat, goodlon]
                    else:
                        self.uwind[:,:,i] = windstress_vars[i][:, :]
            self.uwind = np.ma.masked_where(self.uwind==-9999.0, self.uwind)


        if os.path.exists(vfile):
            with netCDF4.Dataset(vfile, "r") as nc:

                self.vwind = np.empty((len(self.lat), len(self.lon), 12))
                windstress_vars = nc.get_variables_by_attributes(units="N/m^2")
                for i in range(0, 12):
                    if domain is not None:
                        self.vwind[:,:,i] = windstress_vars[i][goodlat, goodlon]
                    else:
                        self.vwind[:,:,i] = windstress_vars[i][:, :]
            self.vwind = np.ma.masked_where(self.vwind==-9999.0, self.vwind)


    def read_ascat(self, dataurl, domain=[-180., 180., -90., 90.]):
        """
        ```python
        read_ascat(dataurl, domain)
        ```
        Read the coordinates, the wind speed and components from the file stored
        in `dataurl` (OPEnDAP) and subset it on the `domain`.

        Inputs:
        dataurl: file path OPEnDAP URL
        domain: 4-element array storing the domain extension

        Outputs:
        res: boolean, True if data are found in the domain
        """

        res = False

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
                self.lon = lon[goodcoord]
                self.lat = lat[goodcoord]
                with netCDF4.Dataset(dataurl) as nc:
                    self.speed = nc.variables["wind_speed"][:][goodcoord]
                    self.angle = nc.variables["wind_dir"][:][goodcoord]

                self.u = self.speed * np.sin(np.deg2rad(self.angle))
                self.v = self.speed * np.cos(np.deg2rad(self.angle))
                res = True
            else:
                logger.info("No data in the region of interest")
                res = False
        except OSError:
            logger.warning("Problem to access the file")

        return res

    def add_to_plot(self, fig, ax, domain=None, cmap=plt.cm.hot_r,
                    clim=[0, 15], date=None, vis=False):
        """
        ```python
        wind.add_to_plot(fig, ax, domain, cmap, clim=[0, 15], date)
        ```
        Display the wind field defined by (u, v) on coordinates (lon, lat) as a
        quiver plot.

        Inputs:
        fig: matplotlib.figure.Figure instance
        ax: a 'cartopy.mpl.geoaxes.GeoAxesSubplot'
        domain: a 4-element tuple storing (lonmin, lonmax, latmin, latmax)
        cmap: the colormap
        clim: limits of the colorbar
        date: the date to be added to the plot
        """

        if date is not None:
            plt.text(0.15, 0.95, date, size=18, rotation=0.,
                     ha="center", va="center",
                     transform=ax.transAxes,
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 1., 1.),
                               alpha=.7
                               )
                     )
        qv = ax.quiver(self.lon, self.lat, self.u, self.v, self.speed,
                       scale=400, width=.002, cmap=cmap, clim=clim)

        if domain is not None:
            ax.set_xlim(domain[0], domain[1])
            ax.set_ylim(domain[2], domain[3])

        # Add high-resolution coastline
        ax.add_wms(wms='http://ows.emodnet-bathymetry.eu/wms',
                       layers=['coastlines'])

        cbar_ax = fig.add_axes([0.18, 0.75, 0.2, 0.015])
        if clim[0] == 0.:
            ext = "max"
        else:
            ext = "both"

        if vis is True:
            textcolor = "w"
        else:
            textcolor = "k"
        cb = plt.colorbar(qv, orientation="horizontal", cax=cbar_ax, extend=ext)
        cb.set_label("m/s (max. = {:.2f})".format(self.speed.max()), fontsize=12,
                     color=textcolor)
        cb.ax.xaxis.set_tick_params(color=textcolor)
        cb.outline.set_edgecolor(textcolor)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=textcolor)

class Visible(object):

    def __init__(self, lon=None, lat=None, proj=None, extent=None, image=None):
        self.lon = lon
        self.lat = lat
        self.proj = proj
        self.image = image
        self.extent = extent

    def read_geotiff(self, imagefile):
        """
        ```python
        read_geotiff(imagefile)
        ```
        Read an image and compute the coordinates from a geoTIFF file.

        Input: imagefile
        -----
        """
        ds = gdal.Open(imagefile, gdal.GA_ReadOnly)

        # Read the array and the transformation
        arr = ds.ReadAsArray()
        # Read the geo transform
        trans = ds.GetGeoTransform()
        # Compute the spatial extent
        self.extent = [trans[0], trans[0] + ds.RasterXSize*trans[1],
                      trans[3] + ds.RasterYSize*trans[5], trans[3]]

        # Get the info on the projection
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)
        self.proj = inproj

        # Compute the coordinates
        x = np.arange(0, ds.RasterXSize)
        y = np.arange(0, ds.RasterYSize)

        xx, yy = np.meshgrid(x, y)
        self.lon = trans[1] * xx + trans[2] * yy + trans[0]
        self.lat = trans[4] * xx + trans[5] * yy + trans[3]

        # Transpose
        self.image = np.transpose(arr, (1, 2, 0))

    def add_to_plot(self, ax, myproj):
        """
        ```python
        add_to_plot(ax, myproj)
        ```
        Add the geoTIFF image to the plot.

        Inputs:
        ------
        ax: a figure ax object
        myproj: a cartopy projection
        """
        ax.imshow(self.image, origin='upper', extent=self.extent, transform=myproj)
