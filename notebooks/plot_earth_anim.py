import os
import glob
import calendar
import datetime
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.info("Let's start")

plt.style.use('dark_background')

figdir = "../images/rotatingEarth/V2/"
if not os.path.exists(figdir):
    os.makedirs(figdir)
    logger.info("Create figure directory {}".format(figdir))

lons = np.arange(0, 360., 1.)
lat0 = 10.
nlons = len(lons)
for i, lon0 in enumerate(lons):
    m = Basemap(projection='ortho',lon_0=lon0, lat_0=lat0, resolution='c')
    logger.info("Working on figure {} / {}".format(i, nlons))
    ii = str(i).zfill(4)

    fig = plt.figure(1, figsize=(10, 10))
    #ax = plt.subplot(111)
    #ax.set_facecolor(".2")
    #m.fillcontinents()
    #m.drawcoastlines(linewidth=.5)
    m.warpimage("./world.topo.bathy.200403.3x5400x2700.jpg", zorder=2)
    plt.savefig(os.path.join(figdir, "earth_visible_{}.jpg".format(ii)), dpi=300, bbox_inches="tight")
    fig.clf()
    plt.close(fig)

    plt.close("all")
