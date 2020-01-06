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
        ax.imshow(self.image, origin='upper', extent=self.extent, transform=myproj)
