import os
import glob
import numpy as np
import filament
import datetime
import cmocean
import logging
import netCDF4
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from importlib import reload
reload(filament)

logger = logging.getLogger("SSTanom")
logger.setLevel(logging.INFO)
logging.info("Starting")

year = 2010
figdir = "/data/SST/Global/figures/"
monthlydir = "/data/SST/Global/monthly/"
climdir = "/data/SST/Global/monthly_clim/"
monthfilelist = sorted(glob.glob(os.path.join(monthlydir, "T{0}*.L3m_MO_SST4_sst4_9km.nc").format(year)))
climfilelist = sorted(glob.glob(os.path.join(climdir, "T*.L3m_MC_SST4_sst4_9km.nc")))
logger.info("Found {} monthly files".format(len(monthfilelist)))


# ## Create dictionary for the monthly climatology files
# Each entry contains the month number (from 1 to 12) and the corresponding climatological file.
climDict = {}
for climfiles in climfilelist:
    logger.debug("Working on file {}".format(climfiles))
    with netCDF4.Dataset(climfiles, "r") as nc:
        time_coverage_start = nc.time_coverage_start[:]
        time_coverage_end = nc.time_coverage_end[:]
    date_start = datetime.datetime.strptime(time_coverage_start, "%Y-%m-%dT%H:%M:%S.000Z")
    date_end = datetime.datetime.strptime(time_coverage_end, "%Y-%m-%dT%H:%M:%S.000Z")
    m = date_end.month
    climDict[m] = climfiles


# ## Loop on the monthly files
# The goal is to find the correct monthly climatology files, knowing that
# * the first and last years are not always the same,
# * the days of the years may also change.

# ## Plot preparation
# ### Projection

# Read coordinates from the 1st file
SSTmonth = filament.SST()
SSTmonth.read_from_oceancolorL3(monthfilelist[0])

# Apply projection
m = Basemap(projection='ortho',lon_0=-52.,lat_0=10,resolution='c')
llon, llat = np.meshgrid(SSTmonth.lon, SSTmonth.lat)
lonp, latp = m(llon, llat)
lonp[lonp==lonp.max()] = np.nan
latp[latp==latp.max()] = np.nan


# ### Plotting function

def make_monthly_subplot(lon, lat, SSTanom, month, NN=1):
    plt.title(calendar.month_name[month], fontsize=18)
    m.drawcoastlines(linewidth=.2)
    pcm = m.pcolormesh(lon[::NN], lat[::NN], SSTanom[::NN],
                 zorder=3, cmap=plt.cm.RdBu_r, vmin=-3., vmax=3.)
    # m.fillcontinents(color='grey')
    #plt.colorbar(extend="both", shrink=.7)
    m.warpimage("world.topo.bathy.200403.3x5400x2700.jpg", zorder=2)
    #mpol2.drawcoastlines()
    #plt.savefig(os.path.join(figdir, "SST_anom_test"), dpi=300, bbox_inches="tight")
    # plt.show()
    #plt.close()
    return pcm


plt.close("all")
fig = plt.figure(figsize=(11.69,8.27))

for imonth, monthlyfile in enumerate(monthfilelist):
    logger.debug("Working on file {}".format(monthlyfile))

    with netCDF4.Dataset(monthlyfile, "r") as nc:
        time_coverage_start = nc.time_coverage_start[:]
        time_coverage_end = nc.time_coverage_end[:]
    date_start = datetime.datetime.strptime(time_coverage_start, "%Y-%m-%dT%H:%M:%S.000Z")
    date_end = datetime.datetime.strptime(time_coverage_end, "%Y-%m-%dT%H:%M:%S.000Z")
    date_mid = date_start + 0.5 * (date_end - date_start)
    month = date_mid.month
    logger.info("Working on month {}/12".format(month))

    # Read data (lon, lat and SST)
    SSTmonth = filament.SST()
    SSTmonth.read_from_oceancolorL3(monthlyfile)
    SSTclim = filament.SST()
    SSTclim.read_from_oceancolorL3(climDict[month])

    # Compute anomalies
    SSTanom = SSTmonth.field - SSTclim.field

    ax = plt.subplot(3, 4, imonth+1)
    pcm = make_monthly_subplot(lonp, latp, SSTanom, imonth+1, NN=1)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
normanom = mpl.colors.Normalize(vmin=-3., vmax=3.)
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.cm.RdBu_r,
                            norm=normanom, orientation='vertical', extend="both")

cb1.set_label("$^{\circ}$C", rotation=0, ha="left", fontsize=14)
fig.suptitle('Sea surface temperature anomalies ({})'.format(year), fontsize=24)
plt.savefig(os.path.join(figdir, "SSTanomalies{}06".format(year)), dpi=300, bbox_inches="tight")
# plt.show()
plt.close()
