import os
import netCDF4
import logging
import datetime
import numpy as np
#import seawater
import calendar
import rasterio
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from geopy.distance import geodesic
import cmocean
import scipy.io as sio
import warnings
import matplotlib.cbook
from matplotlib import colors

from matplotlib.font_manager import FontProperties
fa_dir = r"/home/ctroupin/.fonts/"
fp1 = FontProperties(fname=os.path.join(fa_dir, "Font Awesome 6 Free-Solid-900.otf"))

# from lxml import html
from bs4 import BeautifulSoup
import requests
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cartopyticker
myproj = ccrs.PlateCarree()
coast = cfeature.GSHHSFeature(scale="full")
lon_formatter = cartopyticker.LongitudeFormatter()
lat_formatter = cartopyticker.LatitudeFormatter()

logger = logging.getLogger("Filament")

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


class Bathymetry(object):
    """
    Bathymetry/tropography
    """

    def __init__(self, lon=None, lat=None, depth=None):
        self.lon = lon
        self.lat = lat
        self.depth = depth

    def read_from_EMODnet_dtm(self, fname, domain=None):
        """
        Extract the bathymetry from the `.dtm` (tile) file downloaded from
        [EMODnet Bathymetry](https://www.emodnet-bathymetry.eu/)
        If `domain` is defined, the bathymetry is extracted in the domain
        defined the bounding box (lonmin, lonmax, latmin, latmax)
        """
        with netCDF4.Dataset(fname) as nc:
            lon = nc.get_variables_by_attributes(standard_name="projection_x_coordinate")[0][:]
            lat = nc.get_variables_by_attributes(standard_name="projection_y_coordinate")[0][:]
            if domain is not None:
                if len(domain) == 4:
                    goodlon = np.where((lon <= domain[1]) & (lon >= domain[0]))[0]
                    goodlat = np.where((lat <= domain[3]) & (lat >= domain[2]))[0]
                    self.lon = lon[goodlon]
                    self.lat = lat[goodlat]
                    self.depth = nc.variables["DEPTH"][goodlat, goodlon]
                else:
                    logger.error("domain, if defined, should be a 4-element tuple")
            else:
                self.lon = lon
                self.lat = lat
                self.depth = nc.variables["DEPTH"][:]


class SST(object):
    """
    Sea surface temperature (SST) field
    """

    def __init__(self, lon=None, lat=None, field=None, qflag=None,
                 date=None, fname=None):
        self.lon = lon
        self.lat = lat
        self.field = field
        self.qflag = qflag
        self.date = date
        self.fname = fname

    def read_from_oceancolorL3(self, filename, domain=(-180., 180., -90., 90.)):
        """
        Read the coordinates and the SST from an Ocean Color level-3 file
        and subset the domain defined by `coordinates`
        If coordinates are not specified, then the whole domain is considered
        """
        with netCDF4.Dataset(filename) as nc:
            self.fname = filename
            self.date = datetime.datetime.strptime(nc.time_coverage_end, "%Y-%m-%dT%H:%M:%S.000Z")
            lon = nc.get_variables_by_attributes(long_name="Longitude")[0][:]
            lat = nc.get_variables_by_attributes(long_name="Latitude")[0][:]
            goodlon = np.where( (lon >= domain[0] ) & (lon <= domain[1]))[0]
            goodlat = np.where( (lat >= domain[2] ) & (lat <= domain[3]))[0]
            self.field = nc.get_variables_by_attributes(standard_name="sea_surface_temperature")[0][goodlat, goodlon]
            try:
                self.qflag = nc.get_variables_by_attributes(long_name="Quality Levels, Sea Surface Temperature")[0][goodlat, goodlon]
            except IndexError:
                self.qflag = nc.variables["qual_sst4"][:]
        self.lon = lon[goodlon]
        self.lat = lat[goodlat]

    def read_from_oceancolorL2(self, filename):
        """
        Load the SST from netCDF L2 file obtained from
        https://oceancolor.gsfc.nasa.gov
        :param filename: name of the netCDF file
        :return: lon, lat, field, qflag, year, dayofyear
        """

        if os.path.exists(filename):
            self.fname = filename
            with netCDF4.Dataset(filename) as nc:
                # Read platform
                sat = nc.platform
                # Read time information
                # Assume all the measurements made the same day (and same year)
                year = int(nc.groups['scan_line_attributes'].variables['year'][0])
                dayofyear = int(nc.groups['scan_line_attributes'].variables['day'][0])
                # Convert to date
                #self.date = datetime.datetime(year, 1, 1) + datetime.timedelta(dayofyear - 1)
                tstring = nc.time_coverage_start
                self.date = datetime.datetime.strptime(tstring, '%Y-%m-%dT%H:%M:%S.%fZ')

                # Read coordinates
                self.lon = nc.groups['navigation_data'].variables['longitude'][:]
                self.lat = nc.groups['navigation_data'].variables['latitude'][:]

                # Read geophysical variables
                try:
                    self.field = nc.groups['geophysical_data'].variables['sst'][:]
                    self.qflag = nc.groups['geophysical_data'].variables['qual_sst'][:]
                except KeyError:
                    try:
                        self.field = nc.groups['geophysical_data'].variables['sst4'][:]
                        self.qflag = nc.groups['geophysical_data'].variables['qual_sst4'][:]
                    except KeyError:
                        self.field = nc.groups['geophysical_data'].variables['sst_triple'][:]
                        try:
                            self.qflag = nc.groups['geophysical_data'].variables['qual_sst'][:]
                        except KeyError:
                            self.qflag = nc.groups['geophysical_data'].variables['qual_sst_triple'][:]

                # Remove bad coordinates (for plotting purposes)
                if np.ma.is_masked(self.lon):
                    lon0 = self.lon[:,0]
                    goodlon = ~lon0.mask
                    self.lon = self.lon[goodlon, :]
                    self.lat = self.lat[goodlon, :]
                    self.field = self.field[goodlon, :]
                    self.qflag = self.qflag[goodlon, :]


    def read_from_ghrsst(self, filename):
        """
        Load the SST from netCDF GHRSST file obtained from
        [PODAAC FTP](ftp://podaac-ftp.jpl.nasa.gov)
        :param filename: name of the netCDF file
        :return: lon, lat, field, qflag, year, dayofyear
        """
        if os.path.exists(filename):
            self.fname = filename
            with netCDF4.Dataset(filename) as nc:
                time = nc.variables["time"][0]
                timeunits = nc.variables["time"].units
                self.date = netCDF4.num2date(time, timeunits)
                self.lon = nc.variables["lon"][:]
                self.lat = nc.variables["lat"][:]
                self.field = nc.variables["sea_surface_temperature"][0,:,:] - 273.15
                self.qflag = nc.variables["quality_level"][0,:,:]

    def read_from_cmems(self, datafile, tindex=0, depthindex=0, domain=None):
        """Extract the surface temperature from CMEMS-IBI regional model
        at the time index `tindex`
        """
        self.fname = datafile
        with netCDF4.Dataset(datafile, "r") as nc:

            lon = nc.get_variables_by_attributes(standard_name="longitude")[0][:]
            lat = nc.get_variables_by_attributes(standard_name="latitude")[0][:]
            timevar = nc.get_variables_by_attributes(standard_name="time")[0]

            if domain is not None:
                goodlon = np.where((lon <= domain[1]) & (lon >= domain[0]))[0]
                goodlat = np.where((lat <= domain[3]) & (lat >= domain[2]))[0]
                self.lon = lon[goodlon]
                self.lat = lat[goodlat]
                self.field = nc.get_variables_by_attributes(standard_name="sea_water_potential_temperature")[0][tindex, depthindex,goodlat, goodlon]

            else:
                self.lon = lon
                self.lat = lat
                self.field = nc.get_variables_by_attributes(standard_name="sea_water_potential_temperature")[0][tindex,depthindex,:,:]


            depth = nc.get_variables_by_attributes(standard_name="depth")[0][depthindex]


            time = timevar[tindex]
            timeunits = timevar.units
            self.date = netCDF4.num2date(time, timeunits)

    def read_from_sentinel3(self, datafile):
        """Read the sea surface temperature from Sentinel-3 file
        downloaded using the
        """
        with netCDF4.Dataset(datafile, "r") as nc:
            self.lon = nc.variables["lon"][:]
            self.lat = nc.variables["lat"][:]
            self.qflag = nc.variables["quality_level"][0,:,:]
            self.field = nc.get_variables_by_attributes(standard_name="sea_surface_skin_temperature")[0][0,:,:] - 273.15
            time = nc.variables["time"][0]
            timeunits = nc.variables["time"].units
            self.date = netCDF4.num2date(time, timeunits)

    def apply_qc(self, qf=1):
        """
        Mask the sst values which don't match the mentioned quality flag
        """
        self.field = np.ma.masked_where(self.qflag != qf, self.field)


    def get_title(self):
        """
        Construct the title string based on the sensor, platform and date
        """
        with netCDF4.Dataset(self.fname, "r") as nc:
            try:
                titletext = "{} ({}) {}".format(nc.instrument, nc.platform, self.date.strftime("%Y-%m-%d"))
            except AttributeError:
                titletext = "{} ({}) {}".format(nc.sensor, nc.platform, self.date.strftime("%Y-%m-%d"))
        return titletext

    def get_figname(self):
        """
        Construct the figure name based on the sensor and the date
        """
        # with netCDF4.Dataset(filename, "r") as nc:
        #    figname = "-".join((nc.instrument, self.date.strftime("%Y_%m_%d")))
        figname = os.path.basename(self.fname)
        figname = os.path.splitext(figname)[0]
        figname = figname.replace(".", "-")
        return figname

    def add_date(self, ax):
        """
        Add a box with the date inside
        """
        plt.text(0.05, 0.95, self.date.strftime('%Y-%m-%d %H:%M:%S'), size=18, rotation=0.,
             ha="left", va="center",
             transform=ax.transAxes,
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 1., 1.),
                       alpha=.7
                       )
             )

    def add_to_plot(self, fig, ax, domain=None, cmap=cmocean.cm.thermal,
                    clim=[15., 30.], vis=False, shrink=.8,
                    cbarloc=[0.18, 0.75, 0.2, 0.015], alpha=1):
        """
        ```python
        sst.add_to_plot(fig, ax, domain, cmap, clim, date)
        ```
        Display the SST field

        Inputs:
        fig: matplotlib.figure.Figure instance
        ax: a 'cartopy.mpl.geoaxes.GeoAxesSubplot'
        domain: a 4-element tuple storing (lonmin, lonmax, latmin, latmax)
        cmap: the colormap
        clim: limits of the colorbar
        date: the date to be added to the plot
        proj: the Cartopy projection
        """

        pcm = ax.pcolormesh(self.lon.data, self.lat.data, self.field, cmap=cmap,
                    vmin=clim[0], vmax=clim[1], alpha=alpha, transform=ccrs.PlateCarree())

        if domain is not None:
            ax.set_extent(domain)
            #ax.set_xlim(domain[0], domain[1])
            #ax.set_ylim(domain[2], domain[3])

        if vis is True:
            textcolor = "w"
        else:
            textcolor = "k"

            if cbarloc is not None:
                cbar_ax = fig.add_axes(cbarloc)
                cb = plt.colorbar(pcm, orientation="horizontal",
                                  cax=cbar_ax, extend="both")
            else:
                cb = plt.colorbar(pcm, orientation="vertical",
                                  extend="both", shrink=shrink)

        cb.set_label("$^{\circ}$C", fontsize=12, color=textcolor,
                     rotation=0, ha="left")
        cb.ax.xaxis.set_tick_params(color=textcolor)
        cb.outline.set_edgecolor(textcolor)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=textcolor)
        return pcm, cb

    def make_plot(self, m, figname=None, visibleim=None, swot=None, titletext=None,
                  vmin=16., vmax=24., shrink=1.):
        """
        Make the plot of the SST on a map
        Input:
        m: projection from Basemap
        figname: name of the figure (None by default, means to figure saved)
        visfile: name of the file storing the visible scene (None by default)
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        if titletext is None:
            titletext = self.get_title()
            plt.title(titletext, fontsize=18)

        if swot is not None:
            m.plot(swot.lon, swot.lat, "k--", ms=0.5, latlon=True,
                   linewidth=0.5, alpha=0.7, zorder=7)

        if visibleim is not None:
            m.imshow(np.flipud(visibleim.array), zorder=4)
                #m.pcolormesh(visibleim.lon, visibleim.lat, visibleim.array[:,:,0], latlon=True,
                #             cmap=plt.cm.gray, zorder=1, alpha=0.7)
        else:
            m.fillcontinents(zorder=5)


        pcm = m.pcolormesh(self.lon, self.lat, self.field, latlon=True,
                          cmap=cmocean.cm.thermal, vmin=vmin, vmax=vmax, zorder=3)
        cb = plt.colorbar(pcm, extend="both", shrink=shrink)
        cb.set_label("$^{\circ}$C", rotation=0, ha="left")
        m.drawcoastlines(linewidth=0.25)
        #m.drawmapscale(-10., 27.25, -10., 27.25, 100, barstyle='simple',
        #               units='km', fontsize=10, fontcolor='k', zorder=5)
        m.drawmeridians(np.arange(m.lonmin, m.lonmax, 3.), labels=(0,0,0,1),
                        linewidth=.5, fontsize=12, zorder=4)
        m.drawparallels(np.arange(m.latmin, m.latmax, 2.), labels=(1,0,0,0),
                        linewidth=.5, fontsize=12, zorder=4)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    def make_plot_qf(self, m, figname=None, titletext=None, shrink=1.):
        """
        Create a plot with the Quality Flag values
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)

        if titletext is None:
            titletext = self.get_title()
            plt.title(titletext, fontsize=18)

        m.fillcontinents(zorder=5)

        cmap = plt.cm.YlOrRd
        norm = colors.BoundaryNorm(np.arange(-0.5, 4.5001, 1), cmap.N)

        pcm = m.pcolormesh(self.lon, self.lat, self.qflag, cmap=cmap,
                           vmin=-0.5, vmax=4.5, norm=norm, zorder=3)
        cb = plt.colorbar(pcm, ticks=[0, 1, 2, 3, 4], shrink=shrink)
        cb.set_ticklabels(["Best", "Good", "Questionable", "Bad", "Not processed"])

        m.drawcoastlines(linewidth=0.25)
        #m.drawmapscale(-10., 27.25, -10., 27.25, 100, barstyle='simple',
        #               units='km', fontsize=10, fontcolor='k', zorder=5)
        m.drawmeridians(np.arange(m.lonmin, m.lonmax, 3.), labels=(0,0,0,1),
                        linewidth=.5, fontsize=12, zorder=4)
        m.drawparallels(np.arange(m.latmin, m.latmax, 2.), labels=(1,0,0,0),
                        linewidth=.5, fontsize=12, zorder=4)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()


    def make_plot2(self, m, figname=None, visibleim=None, swot=None,
                   titletext=None, vmin=16., vmax=24.):
        """
        Make the plot of the SST on a map
        Input:
        m: projection from Basemap
        figname: name of the figure (None by default, means to figure saved)
        visfile: name of the file storing the visible scene (None by default)
        """
        fig = plt.figure(figsize=(8, 7))
        ax = plt.subplot(111)

        if swot is not None:
            m.plot(swot.lon, swot.lat, "k--", ms=0.5, latlon=True,
                   linewidth=0.5, alpha=0.7, zorder=7)

        if visibleim is not None:
            m.imshow(np.flipud(visibleim.array), zorder=4)
                #m.pcolormesh(visibleim.lon, visibleim.lat, visibleim.array[:,:,0], latlon=True,
                #             cmap=plt.cm.gray, zorder=1, alpha=0.7)
        else:
            m.fillcontinents(zorder=4)


        pcm = m.pcolormesh(self.lon, self.lat, self.field, latlon=True,
                          cmap=cmocean.cm.thermal, vmin=vmin, vmax=vmax, zorder=6)
        cb = plt.colorbar(pcm, orientation="horizontal", pad=0.05, extend="both", shrink=0.9)
        cb.set_label("$^{\circ}$C")
        m.drawcoastlines(linewidth=0.25)
        #m.drawmapscale(-10., 27.25, -10., 27.25, 100, barstyle='simple',
        #               units='km', fontsize=10, fontcolor='k', zorder=5)
        m.drawmeridians(np.arange(m.lonmin, m.lonmax, 3.), labels=(0,0,1,0),
                        linewidth=.5, fontsize=12, zorder=3)
        m.drawparallels(np.arange(m.latmin, m.latmax, 2.), labels=(1,0,0,0),
                        linewidth=.5, fontsize=12, zorder=3)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()


    def get_domain(self):
        """
        Create a rectangle defined by (lonrect, latrect) based on the SST
        coordinates (mostly for plotting purposes)
        """
        lonrect = np.concatenate((self.lon[:,0].compressed(),
                              self.lon[-1,:].compressed(),
                              np.flipud(self.lon[:,-1].compressed()),
                              np.flipud(self.lon[0,:].compressed())))
        latrect = np.concatenate((self.lat[:,0].compressed(),
                              self.lat[-1,:].compressed(),
                              np.flipud(self.lat[:,-1].compressed()),
                              np.flipud(self.lat[0,:].compressed())))
        return lonrect, latrect

class Current(object):
    """
    Ocean current, caracterised by its two components
    (mmight be merged with Wind object)
    """

    def __init__(self, lon=None, lat=None, u=None, v=None,
                 date=None, fname=None):
        self.lon = lon
        self.lat = lat
        self.u = u
        self.v = v
        self.date = date
        self.fname = fname

    def read_from_cmems(self, datafile, tindex=0, depthindex=0, domain=None):
        with netCDF4.Dataset(datafile) as nc:

            lon = nc.get_variables_by_attributes(standard_name="longitude")[0][:]
            lat = nc.get_variables_by_attributes(standard_name="latitude")[0][:]
            timevar = nc.get_variables_by_attributes(standard_name="time")[0]

            if domain is not None:
                goodlon = np.where((lon <= domain[1]) & (lon >= domain[0]))[0]
                goodlat = np.where((lat <= domain[3]) & (lat >= domain[2]))[0]
                self.lon = lon[goodlon]
                self.lat = lat[goodlat]
                self.u = nc.get_variables_by_attributes(standard_name="eastward_sea_water_velocity")[0][tindex, depthindex,goodlat, goodlon]
                self.v = nc.get_variables_by_attributes(standard_name="northward_sea_water_velocity")[0][tindex, depthindex,goodlat, goodlon]

            else:
                self.lon = lon
                self.lat = lat
                self.u = nc.get_variables_by_attributes(standard_name="eastward_sea_water_velocity")[0][tindex,depthindex,:,:]
                self.v = nc.get_variables_by_attributes(standard_name="northward_sea_water_velocity")[0][tindex,depthindex,:,:]


            depth = nc.get_variables_by_attributes(standard_name="depth")[0][depthindex]
            time = timevar[tindex]
            timeunits = timevar.units
            self.date = netCDF4.num2date(time, timeunits)


class Chloro(object):
    """
    Chlorophyll concentration
    """

    def __init__(self, lon=None, lat=None, field=None, qflag=None,
                 year=None, dayofyear=None, date=None, fname=None):
        self.lon = lon
        self.lat = lat
        self.field = field
        self.qflag = qflag
        self.timeunits = year
        self.year = year
        self.dayofyear = dayofyear
        self.date = date
        self.fname = fname

    def read_from_copernicus(self, filename, bbox=[-180., 180., -90., 90.]):
        """
        Load the chlorophyll concentration from netCDF level-3 file
        obtained from [Copernicus Marine Service](https://oceancolor.gsfc.nasa.gov) FTP
        :param filename: name of the netCDF file
        :return: lon, lat, field, qflag, year, dayofyear
        """
        if os.path.exists(filename):
            self.fname = filename
            with netCDF4.Dataset(filename) as nc:
                lon = nc.variables["lon"][:]
                goodlon = np.where((lon <= bbox[1]) & (lon >= bbox[0]))[0]
                self.lon = lon[goodlon]
                lat = nc.variables["lat"][:]
                goodlat = np.where((lat <= bbox[3]) & (lat >= bbox[2]))[0]
                self.lat = lat[goodlat]
                self.field = nc.variables["CHL"][0,goodlat[0]:goodlat[-1], goodlon[0]:goodlon[-1]]
                timevar = nc.variables["time"]
                self.date = netCDF4.num2date(timevar[0], timevar.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)

    def read_from_oceancolorL2(self, filename):
        """
        Load the chlorophyll concentration from netCDF level-2 file
        obtained from [Ocean Color](https://oceancolor.gsfc.nasa.gov)
        :param filename: name of the netCDF file
        :return: lon, lat, field, qflag, year, dayofyear
        """

        if os.path.exists(filename):
            self.fname = filename
            with netCDF4.Dataset(filename) as nc:
                # Read platform
                sat = nc.platform
                # Read time information
                # Assume all the measurements made the same day (and same year)
                self.year = int(nc.groups['scan_line_attributes'].variables['year'][0])
                self.dayofyear = int(nc.groups['scan_line_attributes'].variables['day'][0])
                # Convert to date
                self.date = datetime.datetime(self.year, 1, 1) + datetime.timedelta(self.dayofyear - 1)
                # Read coordinates
                self.lon = nc.groups['navigation_data'].variables['longitude'][:]
                self.lat = nc.groups['navigation_data'].variables['latitude'][:]
                # Read geophysical variables
                self.field = nc.groups['geophysical_data'].variables['chlor_a'][:]

    def add_to_plot(self, fig, ax, domain=None, cmap=cmocean.cm.haline_r,
                    clim=[0., 3.], vis=False,
                    cbarloc=None, alpha=1, extend="both"):
        """
        ```python
        chloro.add_to_plot(fig, ax, domain, cmap, clim, date)
        ```
        Display the chlorophyll concentrationß field

        Inputs:
        fig: matplotlib.figure.Figure instance
        ax: a 'cartopy.mpl.geoaxes.GeoAxesSubplot'
        domain: a 4-element tuple storing (lonmin, lonmax, latmin, latmax)
        cmap: the colormap
        clim: limits of the colorbar
        date: the date to be added to the plot
        """

        pcm = ax.pcolormesh(self.lon.data, self.lat.data, self.field, cmap=cmap,
                            vmin=clim[0], vmax=clim[1], alpha=alpha)

        if domain is not None:
            ax.set_xlim(domain[0], domain[1])
            ax.set_ylim(domain[2], domain[3])

        if vis is True:
            textcolor = "w"
        else:
            textcolor = "k"

        if cbarloc is not None:
            cbar_ax = fig.add_axes(cbarloc)
            cb = plt.colorbar(pcm, orientation="horizontal", cax=cbar_ax, extend=extend)
        else:
            cb = plt.colorbar(pcm, orientation="vertical", extend=extend)

        cb.set_label("mg/m$^{3}$", fontsize=12, color=textcolor, rotation=0, ha="left")
        cb.ax.xaxis.set_tick_params(color=textcolor)
        cb.outline.set_edgecolor(textcolor)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=textcolor)
        return pcm, cb

    def get_figname(self):
        """
        Construct the figure name based on the sensor and the date
        """
        figname = os.path.basename(self.fname)
        figname = os.path.splitext(figname)[0]
        figname = figname.replace(".", "-")
        return figname

class Swot(object):
    def __init__(self, lon=None, lat=None, rad=None):
        self.lon = lon
        self.lat = lat
        self.rad = rad

    def read_from(self, orbitfile):
        """
        Read the orbit from a text file
        """
        if os.path.exists(orbitfile):
            self.lon, self.lat, self.rad = np.loadtxt(orbitfile, comments="#", unpack=True)
            self.lon[self.lon>180.] = self.lon[self.lon>180.] - 360.

    def select_domain(self, coordinates):
        """
        Subset the coordinates in a region of interest defined by the
        tuple `coordinates`
        """
        goodlon = np.where(np.logical_and(self.lon>= coordinates[0], self.lon<= coordinates[1]))[0]
        goodlat = np.where(np.logical_and(self.lat>= coordinates[2], self.lat<= coordinates[3]))[0]
        goodcoords = np.intersect1d(goodlon, goodlat)
        self.lon = self.lon[goodcoords]
        self.lat = self.lat[goodcoords]

    def add_to_plot(self, m):
        m.plot(swot.lon, swot.lat, "ko--", ms=0.5, latlon=True)


class Wind(object):

    def __init__(self, lon=None, lat=None, u=None, v=None, time=None, speed=None, angle=None):
        self.lon = lon
        self.lat = lat
        self.u = u
        self.v = v
        self.time = time
        self.speed = speed
        self.angle = angle

    def compute_speed(self):
        """Compute the speed using the 2 velocity components
        """
        self.speed = np.sqrt(self.u * self.u + self.v * self.v)

    def read_from_quikscat(self, datafile, domain=[-180., 180., -90., 90.]):
        """
        ```python
        read_from_quikscat(datafile, domain)
        ```
        Extract the wind data from `datafile` and subset them on `domain`
        """
        with netCDF4.Dataset(datafile, "r") as nc:
            t = nc.variables["time"][0]
            timeunits = nc.variables["time"].units
            self.time = netCDF4.num2date(t, timeunits)
            lon = nc.get_variables_by_attributes(standard_name="longitude")[0][:] - 360.
            lat = nc.get_variables_by_attributes(standard_name="latitude")[0][:]

            logger.info(lon.shape)

            # Check if we have data in the domain of interest
            goodlon = (lon <= domain[1]) & (lon >= domain[0])
            goodlat = (lat <= domain[3]) & (lat >= domain[2])
            goodcoord = (lon <= domain[1]) & (lon >= domain[0]) & (lat <= domain[3]) & (lat >= domain[2])
            ngood = goodcoord.sum().sum()

            if ngood > 0:
                logger.info("Subsetting data to region of interest")
                self.lon = lon[goodcoord]
                self.lat = lat[goodcoord]
                self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:][goodcoord]
                self.angle = nc.get_variables_by_attributes(standard_name="wind_to_direction")[0][:][goodcoord]

                self.u = self.speed * np.sin(np.deg2rad(self.angle))
                self.v = self.speed * np.cos(np.deg2rad(self.angle))
                res = True
            else:
                logger.info("No data in the region of interest")
                res = False

        return res

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
            # self.lon[self.lon > 180.] -= 360.
            time = nc.get_variables_by_attributes(standard_name="time")[0][:]
            timeunits = nc.get_variables_by_attributes(standard_name="time")[0].units
            dates = netCDF4.num2date(time, timeunits)
            self.time = dates
            ntime = len(time)

            if domain is not None:
                goodlon = np.where( (self.lon<= domain[1]) & (self.lon>= domain[0]))[0]
                goodlat = np.where( (self.lat<= domain[3]) & (self.lat>= domain[2]))[0]
                self.lon = self.lon[goodlon]
                self.lat = self.lat[goodlat]

                if ntime == 12:
                    # Climatology product
                    self.u = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][:, goodlat, goodlon]
                    self.v = nc.get_variables_by_attributes(standard_name="northward_wind")[0][:, goodlat, goodlon]
                    try:
                        self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:, goodlat, goodlon]
                    except IndexError:
                        logger.warning("No variable `speed`  in the netCDF file")
                elif ntime == 4:
                    # Daily product
                    self.u = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][:, goodlat, goodlon]
                    self.v = nc.get_variables_by_attributes(standard_name="northward_wind")[0][:, goodlat, goodlon]
                    try:
                        self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:, goodlat, goodlon]
                    except IndexError:
                        logger.warning("No variable `speed`  in the netCDF file")
                else:
                    # Monthly product
                    self.u = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][0, goodlat, goodlon]
                    self.v = nc.get_variables_by_attributes(standard_name="northward_wind")[0][0, goodlat, goodlon]
                    try:
                        self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][0, goodlat, goodlon]
                    except IndexError:
                        logger.warning("No variable `speed`  in the netCDF file")

            else:
                self.u = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][:]
                self.v = nc.get_variables_by_attributes(standard_name="northward_wind")[0][:]
                try:
                    self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][:]
                except IndexError:
                    logger.warning("No variable `speed` in the netCDF file")


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

                self.u = np.empty((len(self.lat), len(self.lon), 12))
                windstress_vars = nc.get_variables_by_attributes(units="N/m^2")
                for i in range(0, 12):
                    if domain is not None:
                        self.u[:,:,i] = windstress_vars[i][goodlat, goodlon]
                    else:
                        self.u[:,:,i] = windstress_vars[i][:, :]
            self.u = np.ma.masked_where(self.u==-9999.0, self.u)


        if os.path.exists(vfile):
            with netCDF4.Dataset(vfile, "r") as nc:

                self.v = np.empty((len(self.lat), len(self.lon), 12))
                windstress_vars = nc.get_variables_by_attributes(units="N/m^2")
                for i in range(0, 12):
                    if domain is not None:
                        self.v[:,:,i] = windstress_vars[i][goodlat, goodlon]
                    else:
                        self.v[:,:,i] = windstress_vars[i][:, :]
            self.v = np.ma.masked_where(self.v==-9999.0, self.v)

    def read_knmi(self, datafile, domain=[-180., 180., -90., 90.]):
        """
        ```python
        read_knmi(dataurl, domain)
        ```
        Read the coordinates, the wind speed and components from the file stored
        in `datafile` (OPEnDAP) and subset it on the `domain`.

        Inputs:
        datafile: file path OPEnDAP URL
        domain: 4-element array storing the domain extension

        """
        with netCDF4.Dataset(datafile) as nc:
            lon = nc.get_variables_by_attributes(standard_name="longitude")[0][:].data
            lat = nc.get_variables_by_attributes(standard_name="latitude")[0][:].data
            goodlon = np.where((lon <= domain[1]) & (lon >= domain[0]))[0]
            goodlat = np.where((lat <= domain[3]) & (lat >= domain[2]))[0]

            self.lon = lon[goodlon]
            self.lat = lat[goodlat]

            timevar = nc.get_variables_by_attributes(standard_name="time")[0]
            self.dates = netCDF4.num2date(timevar[:], timevar.units, only_use_python_datetimes=True)
            self.u = nc.get_variables_by_attributes(standard_name="northward_wind")[0][0,goodlat, goodlon]
            self.v = nc.get_variables_by_attributes(standard_name="eastward_wind")[0][0,goodlat, goodlon]
            self.speed = nc.get_variables_by_attributes(standard_name="wind_speed")[0][0,goodlat, goodlon]
            self.angle = nc.get_variables_by_attributes(standard_name="wind_to_direction")[0][0,goodlat, goodlon]

    def read_from_cmems(self, datafile, domain=[-180., 180., -90., 90.]):
        with netCDF4.Dataset(datafile) as ds:
            lon = ds.get_variables_by_attributes(standard_name = "longitude")[0][:]
            lat = ds.get_variables_by_attributes(standard_name = "latitude")[0][:]
            goodlon = np.where((lon >= domain[0]) & (lon <= domain[1]))[0]
            goodlat = np.where((lat >= domain[2]) & (lat <= domain[3]))[0]

            self.lon = lon[goodlon]
            self.lat = lat[goodlat]

            timevar = ds.get_variables_by_attributes(standard_name="time")[0]
            self.dates = netCDF4.num2date(timevar[:], timevar.units, only_use_python_datetimes=True)

            self.u = ds.get_variables_by_attributes(standard_name="eastward_wind")[0][0,goodlat, goodlon]
            self.v = ds.get_variables_by_attributes(standard_name="northward_wind")[0][0,goodlat, goodlon]

    def get_uv(self):
        anglerad = np.deg2rad(self.angle)
        self.u = -self.speed * np.sin(anglerad)
        self.v = -self.speed * np.cos(anglerad)

    def read_cyms(self, datafile):
        """Read the data from the netCDF file
        """
        with netCDF4.Dataset(datafile) as nc:
            self.lon = nc.get_variables_by_attributes(standard_name="longitude")[0][0,:,:]
            self.lat = nc.get_variables_by_attributes(standard_name="latitude")[0][0,:,:]
            self.speed = nc.variables["wind_speed"][0,:,:]
            self.angle = nc.variables["wind_from_direction"][0,:,:]
        self.get_uv()

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
                time = nc.variables["time"][0][0]
                timeunits = nc.variables["time"].units
                self.time = netCDF4.num2date(time, timeunits)

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
                    visname=None, clim=[0., 15.],
                    quivscale=200, quivwidth=0.2,
                    cbarloc='lower right', cbarplot=True, zorder=5):
        """
        ```python
        wind.add_to_plot(fig, ax, domain, cmap, clim=[0, 15], date)
        ```
        Display the wind field defined by (u, v) on coordinatess (lon, lat) as as
        quiver plot.

        Inputs:
        fig: matplotlib.figure.Figure instance
        ax: a 'cartopy.mpl.geoaxes.GeoAxesSubplot'
        domain: a 4-element tuple storing (lonmin, lonmax, latmin, latmax)
        visname: a string setting the satellite used for the true color
        (VIIRS, AQUA, TERRA or NOAA)
        cmap: the colormap
        clim: limits of the colorbar
        """

        qv = ax.quiver(self.lon, self.lat, self.u, self.v, self.speed,
                       scale=quivscale, width=quivwidth, cmap=cmap, clim=clim,
                       transform=ccrs.PlateCarree())

        if domain is not None:
            ax.set_extent(domain)

        # Add high-resolution coastline
        #ax.add_wms(wms='http://ows.emodnet-bathymetry.eu/wms',
        #                layers=['coastlines'])

        if clim[0] == 0.:
            ext = "max"
        else:
            ext = "both"

        if visname is not None:
            textcolor = "w"
            backcolor = "k"
        else:
            textcolor = "k"
            backcolor = "w"

        if cbarplot is True:
            axins1 = inset_axes(ax, width="35%", height="3.5%", loc=cbarloc, borderpad=4)
            axins1.xaxis.set_ticks_position("bottom")
            cb = plt.colorbar(qv, cax=axins1, extend=ext, orientation="horizontal")
            cb.set_label("m/s (max. = {:.1f})".format(self.speed.max()), fontsize=10,
                     color=textcolor, path_effects=[PathEffects.withStroke(linewidth=1,
                                                        foreground=backcolor)])
            cb.ax.tick_params(labelsize=8)
            cb.ax.xaxis.set_tick_params(color=textcolor)
            cb.outline.set_edgecolor(textcolor)
            plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=textcolor, path_effects=[PathEffects.withStroke(linewidth=1, foreground=backcolor)])

        return qv

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

        with rasterio.open(imagefile) as r:
            truecolor = r.read()
            self.image = np.transpose(truecolor, [1,2,0])
            bbox = r.bounds
            self.extent = [bbox.left, bbox.right, bbox.bottom, bbox.top]
            self.proj = r.crs
            trans = r.transform

            height = truecolor.shape[0]
            width = truecolor.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(trans, rows, cols)
            self.lon = np.array(xs)
            self.lat = np.array(ys)

    def list_files(self, datadir, imdate):
        """
        List the files containing visible images either from
        [WorldView](https://worldview.earthdata.nasa.gov) or from
        [Sentinel-Hub](https://apps.sentinel-hub.com)
        """
        filelist = []
        dtime = imdate.strftime("%Y-%m-%d")

        if os.path.exists(datadir):
            for files in os.listdir(datadir):
                if (dtime in files) & (files.endswith(".tiff")):
                    filelist.append(files)
        else:
            logger.warning("Directory {} does not exist".format(datadir))
        return filelist

    def extract_area(self, domain):
        """
        ```python
        extract_area(domain)
        ```
        Extract the coordinates and the field in the region of interest
        :param domain: a 4-element iterable specifying the region of interest
        """
        lonvis = self.lon[0,:]
        latvis = self.lat[:,0]
        goodlon = np.where(np.logical_and(lonvis <= domain[1], lonvis >= domain[0]))[0]
        goodlat = np.where(np.logical_and(latvis <= domain[3], latvis >= domain[2]))[0]
        self.array = self.array[goodlat, :, :]
        self.array = self.array[:, goodlon, :]
        self.lon = self.lon[:,goodlon]
        self.lat = self.lat[goodlat,:]


    def add_to_plot(self, ax, myproj, **kwargs):
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
        ax.imshow(self.image, origin='upper', extent=self.extent, transform=myproj, **kwargs)


class Altimetry(object):

    """
    SLA field from altimetry
    """

    def __init__(self, lon=None, lat=None, sla=None, u=None, v=None,
                 time=None, date=None, speed=None):
        self.lon = lon
        self.lat = lat
        self.sla = sla
        self.u = u
        self.v = v
        self.time = time
        self.speed = speed

    def read_from_aviso(self, filename):
        """
        ```python
        read_from_aviso(filename)
        ```
        :param filename: name of the netCDF file
        :return: lon, lat, SLA, u, v, time
        """

        if os.path.exists(filename):
            with netCDF4.Dataset(filename) as nc:
                self.lon = nc.get_variables_by_attributes(standard_name='longitude')[0][:]
                self.lat = nc.get_variables_by_attributes(standard_name='latitude')[0][:]
                self.time = nc.get_variables_by_attributes(standard_name='time')[0][:]
                timeunits = nc.get_variables_by_attributes(standard_name='time')[0].units
                self.date = netCDF4.num2date(self.time, timeunits)

                self.sla = nc.get_variables_by_attributes(standard_name='sea_surface_height_above_sea_level')[0][0,:]
                self.u = nc.get_variables_by_attributes(standard_name='surface_geostrophic_eastward_sea_water_velocity')[0][0,:]
                self.v = nc.get_variables_by_attributes(standard_name='surface_geostrophic_northward_sea_water_velocity')[0][0,:]

    def get_speed(self):
        """
        Compute current speed
        """

        self.speed = np.sqrt(self.u * self.u + self.v * self.v )
        self.speed = np.ma.masked_greater(self.speed, 1.5)

    def get_vort(self):
        llon, llat = np.meshgrid(self.lon, self.lat)
        dx = llon[:, 1:] - llon[:, :-1]
        dy = llat[1:, :] - llat[:-1, :]
        dux, duy = np.gradient(self.u)
        dvx, dvy = np.gradient(self.v)
        self.vort = dvx/dx.mean() - duy/dy.mean()

    def plot_streamline(self, m=None, cmap=plt.cm.RdBu_r, vmax=0.15, density=3):

        if m is not None:
            llon, llat = np.meshgrid(self.lon, self.lat)
            self.sla[self.sla >= vmax] = vmax
            self.sla[self.sla <= -vmax] = -vmax
            m.streamplot(llon, llat, self.u, self.v, color=self.sla,
                       arrowstyle="fancy", density=density, linewidth=.5, cmap=cmap, latlon=True)
        else:
            plt.streamplot(self.lon, self.lat, self.u, self.v, color=self.sla,
                       arrowsize=2, density=density, linewidth=.5, cmap=cmap)

    def plot_sla(self, m=None, cmap=plt.cm.RdBu_r, slalevels=np.arange(-0.3, 0.3, 0.025)):

        if m is not None:
            llon, llat = np.meshgrid(self.lon, self.lat)
            xx, yy = m(llon, llat)
            plt.contour(xx, yy, self.sla, slalevels, cmap=cmap)

        else:
            plt.contour(self.lon, self.lat, self.sla, slalevels, cmap=cmap)

    def select_domain(self, coordinates):
        """
        Subset based on geographical positions
        """
        goodlon = np.where(np.logical_and(self.lon >= coordinates[0], self.lon <= coordinates[1]))[0]
        goodlat = np.where(np.logical_and(self.lat >= coordinates[2], self.lat <= coordinates[3]))[0]
        self.lon = self.lon[goodlon]
        self.lat = self.lat[goodlat]
        self.u = self.u[goodlat, :]
        self.u = self.u[:, goodlon]
        self.v = self.v[goodlat, :]
        self.v = self.v[:, goodlon]
        self.sla = self.sla[goodlat, :]
        self.sla = self.sla[:, goodlon]

class NAO(object):

    def __init__(self, values=None, times=None):

        if (values is not None) & (times is not None):
            if len(values) == len(times):
                self.values = values
                self.times = times
            else:
                logger.error("Values and times have not the same dimension")


    def read_from_esrl(self, filename, valex=-999.99):
        """
        ```python
        read_nao_esrl(filename, valex):
        ```
        Read the NAO monthly time series from the file obtained from
        https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/nao.long.data
        :param filename: path to the file containing the data
        """
        with open(filename, "r") as f1:
            ymin, ymax = f1.readline().rstrip().split()
            yearmin = int(ymin)
            yearmax = int(ymax)

            # Create a date vector
            datevec = []
            for iyear, yy in enumerate(range(yearmin, yearmax+1)):
                for mm in range(0, 12):
                    datevec.append(datetime.datetime(yy, mm+1, 1))

            # Read the NAO values
            naovalues = []
            for lines in f1:
                lsplit = lines.rstrip().split()
                if len(lsplit) == 13:
                    year = lsplit[0]
                    naoyear = [float(nao) for nao in lsplit[1:]]
                    naovalues.extend(naoyear)

        # Replace fill value by NaN
        self.times = np.array(datevec)
        naovalues = np.array(naovalues)
        naovalues[naovalues == valex] = np.nan
        self.values = naovalues

    def read_from_ucar(self, filename, valex=-999.99):
        """
        ```python
        read_nao_ucar(filename, valex):
        ```
        Read the NAO monthly time series from the file obtained from

        https://climatedataguide.ucar.edu/sites/default/files/nao_station_monthly.txt
        """
        with open(filename, "r") as f1:

            # Read the NAO values
            naovalues = []
            yearvec = []
            for lines in f1:
                lsplit = lines.rstrip().split()
                if len(lsplit) == 13:
                    yearvec.append(int(lsplit[0]))
                    naoyear = [float(nao) for nao in lsplit[1:]]
                    naovalues.extend(naoyear)

        yearmin, yearmax = yearvec[0], yearvec[-1]

        # Create a date vector
        datevec = []
        for iyear, yy in enumerate(range(yearmin, yearmax+1)):
            for mm in range(0, 12):
                datevec.append(datetime.datetime(yy, mm+1, 1))

        # Replace fill value by NaN
        self.times = np.array(datevec)
        self.values = np.array(naovalues)
        self.values[self.values == valex] = np.nan

    def read_from_noaa(self, filename, valex=-999.99):
        """
        ```python
        dates, noavalues = read_nao_ucar(filename):
        ```
        Read the NAO monthly time series from the file obtained from

        https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii
        """
        with open(filename, "r") as f1:

            # Read the NAO values
            naovalues = []
            datevec = []
            for lines in f1:
                lsplit = lines.rstrip().split()
                year = int(lsplit[0])
                month = int(lsplit[1])
                nao = float(lsplit[2])

                datevec.append(datetime.datetime(year, month, 1))
                naovalues.append(nao)

        # Replace fill value by NaN
        self.times = np.array(datevec)
        self.values = np.array(naovalues)
        self.values[self.values == valex] = np.nan

    def plot_nao_bars(datevec, naovalues, figname=None,
                      xmin=datetime.datetime(2000, 1, 1),
                      xmax=datetime.datetime(2018, 12, 31),
                      **kwargs):
        """
        Create a bar chart of a NAO time series
        """
        plt.bar(datevec, naovalues, **kwargs)
        plt.vlines(datetime.datetime(2010,1,1), -6., 6.,
                   linestyles='--')
        plt.vlines(datetime.datetime(2011,1,1), -6., 6.,
                   linestyles='--')
        plt.xlabel("Time")
        plt.ylabel("NAO index", rotation=0, ha="right")
        plt.xlim(xmin, xmax)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


def prepare_3D_scat():

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_aspect('equal')

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xticks(np.arange(-1., 0, 0.2))
    ax1.set_yticks(np.arange(36.8, 37.2, 0.2))
    ax1.set_title("Coastal glider")

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xticks(np.arange(-1., 0, 0.2))
    ax2.set_yticks(np.arange(36.8, 37.2, 0.2))
    ax2.set_title("Deep glider")
    fig.subplots_adjust(right=0.6)
    cbar_ax = fig.add_axes([0.65, 0.25, 0.015, 0.5])
    return fig, ax1, ax2, cbar_ax

def prepare_3D_scat4():
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_aspect('equal')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_aspect('equal')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_aspect('equal')

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xticks(np.arange(-1., 0, 0.2))
    ax1.set_yticks(np.arange(36.8, 37.2, 0.2))
    ax1.set_title("Coastal glider", fontsize=18)

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xticks(np.arange(-1., 0, 0.2))
    ax2.set_yticks(np.arange(36.8, 37.2, 0.2))
    ax2.set_title("Deep glider", fontsize=18)

    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_xticks(np.arange(-1., 0, 0.2))
    ax3.set_yticks(np.arange(36.8, 37.2, 0.2))

    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.set_xticks(np.arange(-1., 0, 0.2))
    ax4.set_yticks(np.arange(36.8, 37.2, 0.2))


    fig.subplots_adjust(right=0.8)
    cbar_ax1 = fig.add_axes([0.85, 0.525, 0.02, 0.35])
    cbar_ax2 = fig.add_axes([0.85, 0.125, 0.02, 0.35])
    return fig, ax1, ax2, ax3, ax4, cbar_ax1, cbar_ax2


def change_wall_prop(ax, coordinates, depths, angles):
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_xaxis.gridlines.set_linestyles(':')
    ax.w_yaxis.gridlines.set_linestyles(':')
    ax.w_zaxis.gridlines.set_linestyles(':')
    ax.view_init(angles[0], angles[1])
    ax.set_xlim(coordinates[0],coordinates[1])
    ax.set_ylim(coordinates[2],coordinates[3])
    ax.set_zlim(depths[0],depths[1])
    ax.set_zlabel('Depth (m)')

    ax.set_zticks(np.arange(depths[0],depths[1]+10,depths[2]))
    ax.set_zticklabels(range(int(-depths[0]),-int(depths[1])-10,-int(depths[2])))

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
    ax.set_extent(domain)


def add_vis_wind_caption(ax, visname=None, satname=None, date=None):
    """Add a text in the corner, based on the satellite names
    (true color and wind)
    """

    # Dictionaries with the satellite "long names"
    windsat_names = {'metopa': 'MetOp-A', 'metopb': 'MetOp-B',
                     'metopc': 'MetOp-C', 'CCMP': 'CCMP'}
    truecolor_names = {'VIIRS': 'Suomi NPP | VIIRS', 'TERRA': 'Terra | MODIS',
                       'AQUA': 'Aqua | MODIS', 'NOAA': 'NOAA-20 | VIIRS',
                       'Sentinel-2': 'Sentinel-2'}

    # Text properties (box etc)
    textdict = {'fontsize':10, 'ha': "left",
        'transform':ax.transAxes,
        'bbox': dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.), alpha=.7)}

    if (satname is not None) & (date is not None) & (visname is not None):
        logotext = "\uf7a2\n\uf72e"
        sattext = f" {truecolor_names[visname]}\n {windsat_names[satname]} | ASCAT ({date})"
    elif (satname is None) & (visname is not None) & (date is not None):
        logotext = "\uf7a2"
        sattext = f" {truecolor_names[visname]}"
    elif (satname is not None) & (visname is None) & (date is not None):
        logotext = "\uf72e"
        sattext = f" {windsat_names[satname]} | ASCAT ({date})"

    ax.text(0.01, 0.97, logotext, fontproperties=fp1, **textdict,
            va="top")
    ax.text(0.05, 0.97, sattext, **textdict, va="top")


def get_filelist_url(year, dayofyear):
    """
    Generate a list of file URLs (OPEnDAP) for the netCDF corresponding to `year` and `dayofyear`
    """

    urllist = []

    for mission in ["metop_a", "metop_b", "metop_c"]:
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

def get_filelist_url_quikscat(year, dayofyear):
    """
    Generate a list of file URLs (OPEnDAP) for the netCDF corresponding to `year` and `dayofyear`
    """
    urllist = []

    baseurl = "https://opendap.jpl.nasa.gov/opendap/OceanWinds/quikscat/L2B12/v4.0/{}/{}/contents.html".format(year, str(dayofyear).zfill(3))
    opendapurl = "https://opendap.jpl.nasa.gov/opendap/OceanWinds/quikscat/L2B12/v4.0/"
    r = requests.get(baseurl)
    content = r.content
    soup = BeautifulSoup(content, "html.parser")

    for link in soup.find_all('a'):
        datalink = link.get('href')
        if datalink.startswith("qs") & datalink.endswith(".nc"):
            dataurl = os.path.join(opendapurl, str(year), str(dayofyear).zfill(3), datalink)
            urllist.append(dataurl)

    logger.info("Found {} files".format(len(urllist)))

    return urllist

def extract_emodnet_bath(bathfile, domain):
    """
    Extract the bathymetry from a tile (in netCDF) downloaded from EMODnet Bathymetry format
    """
    with netCDF4.Dataset(bathfile) as nc:
        lon = nc.get_variables_by_attributes(standard_name="projection_x_coordinate")[0][:]
        lat = nc.get_variables_by_attributes(standard_name="projection_y_coordinate")[0][:]
        goodlon = np.where((lon <= domain[1]) & (lon >= domain[0]))[0]
        goodlat = np.where((lat <= domain[3]) & (lat >= domain[2]))[0]
        lon = lon[goodlon]
        lat = lat[goodlat]
        depth = nc.variables["DEPTH"][goodlat, goodlon]
    return lon, lat, depth

def get_monthly_filename(sat, sensor, year, month, res="9km"):
    """
    Return the file name according to the satellite, sensor and the period of
    interest, specified by `year` and `month`.

    ```python
    sstfilename = get_monthly_filename("TERRA", "MODIS", 2020, 6)
    > "TERRA_MODIS.20200601_20200630.L3m.MO.SST4.sst4.9km.nc"
    ```

    """
    mm = str(month).zfill(2)
    numdays = calendar.monthrange(year, month)[1]
    nd = str(numdays).zfill(2)
    fname = f"{sat}_{sensor}.{year}{mm}01_{year}{mm}{nd}.L3m.MO.SST4.sst4.{res}.nc"
    return fname

def get_monthly_clim_filename(sat, sensor, yearstart, yearend, month, res="9km"):
    """
    Return the file name according to the satellite, mission and the date

    ```python

    ```

    """
    mm = str(month).zfill(2)
    numdays = calendar.monthrange(yearend, month)[1]
    ddend = str(numdays).zfill(2)
    res = "9km"
    fname = f"{sat}_{sensor}.{yearstart}{mm}01_{yearend}{mm}{ddend}.L3m.MC.SST4.sst4.{res}.nc"

    return fname

def get_filelist_url_oceancolor(sat="MODIS-Terra", res="9km", variable="sst4"):
    """
    Generate a list of URLs for the netCDF files from Ocean Color.
    """

    urllist = []

    baseurl = f"https://oceandata.sci.gsfc.nasa.gov/{sat}/Mapped/Monthly/{res}/{variable}/"
    logger.info(baseurl)
    r = requests.get(baseurl)
    content = r.content
    soup = BeautifulSoup(content, "html.parser")

    for link in soup.find_all('a'):
        datalink = link.get('href')
        if isinstance(datalink, str):
            if datalink.endswith(".nc"):
            #if datalink.startswith("ascat_") & datalink.endswith(".gz"):
            #    dataurl = os.path.join(opendapurl, str(year), str(dayofyear).zfill(3), datalink)
                urllist.append(datalink)

    #logger.info("Found {} files".format(len(urllist)))

    return urllist
