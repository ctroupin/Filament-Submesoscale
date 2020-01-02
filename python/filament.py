import os
import netCDF4
import logging
import datetime
import numpy as np
import seawater
from osgeo import gdal
from scipy import interpolate
import matplotlib.pyplot as plt
import cartopy
import matplotlib.patches as patches
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from geopy.distance import vincenty
import cmocean
import scipy.io as sio
import warnings
import matplotlib.cbook
from matplotlib import colors

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class SST(object):
    """
    Sea surface temperature field
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
        Read the coordinates and the SST from a L3 file
        and subset the domain defined by `coordinates`
        If coordinates are not specified, then the whole domain
        is considered
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
                self.date = datetime.datetime(year, 1, 1) + datetime.timedelta(dayofyear - 1)
                # Read coordinates
                self.lon = nc.groups['navigation_data'].variables['longitude'][:]
                self.lat = nc.groups['navigation_data'].variables['latitude'][:]
                # Read geophysical variables
                try:
                    self.field = nc.groups['geophysical_data'].variables['sst'][:]
                    self.qflag = nc.groups['geophysical_data'].variables['qual_sst'][:]
                except KeyError:
                    self.field = nc.groups['geophysical_data'].variables['sst4'][:]
                    self.qflag = nc.groups['geophysical_data'].variables['qual_sst4'][:]

    def read_from_ghrsst(self, filename):
        """
        Load the SST from netCDF GHRSST file obtained from
        ftp://podaac-ftp.jpl.nasa.gov
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

    def apply_qc(self, qf=1):
        """
        Mask the sst values which don't match the mentioned quality flag
        """
        self.field = np.ma.masked_where(self.qflag > 1, self.field)


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


    def make_plot2(self, m, figname=None, visibleim=None, swot=None, titletext=None, vmin=16., vmax=24.):
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
        goodlon = np.where(np.logical_and(self.lon>= coordinates[0], self.lon<= coordinates[1]))[0]
        goodlat = np.where(np.logical_and(self.lat>= coordinates[2], self.lat<= coordinates[3]))[0]
        goodcoords = np.intersect1d(goodlon, goodlat)
        self.lon = self.lon[goodcoords]
        self.lat = self.lat[goodcoords]

    def add_to_plot(self, m):
        m.plot(swot.lon, swot.lat, "ko--", ms=0.5, latlon=True)


class Visible(object):

    def __init__(self, lon=None, lat=None, array=None):
        self.lon = lon
        self.lat = lat
        self.array = array

    def list_from_dir(self, datadir, dd, satnames=["AQUA", "TERRA", "VIIRS"]):
        """
        Generate a list of geoTIFF files located in the directory `visdir`
        """
        fnames = []
        for sat in satnames:
            fname = "".join((sat, "_", dd.strftime("%Y-%m-%dT%H_%M_%SZ"), ".tiff"))
            visname = os.path.join(datadir, fname)
            if os.path.exists(visname):
                fnames.append(visname)
        return fnames

    def read_from(self, filename):

        gtif = gdal.Open(filename)

        # info about the projection
        arr = gtif.ReadAsArray()
        trans = gtif.GetGeoTransform()
        self.extent = (trans[0], trans[0] + gtif.RasterXSize*trans[1],
                  trans[3] + gtif.RasterYSize*trans[5], trans[3])
        # Permute dimensions
        self.array = np.transpose(arr, (1, 2, 0))

        x = np.arange(0, gtif.RasterXSize)
        y = np.arange(0, gtif.RasterYSize)
        xx, yy = np.meshgrid(x, y)

        self.lon = trans[1] * xx + trans[2] * yy + trans[0]
        self.lat = trans[4] * xx + trans[5] * yy + trans[3]

    def extract_area(self, domain):
        """
        ```python
        extract_area(coordinates)
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

    def read_nao_ucar(self, filename, valex=-999.99):
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

    def read_nao_noaa(self, filename, valex=-999.99):
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
