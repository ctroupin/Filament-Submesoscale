import os
import glob
import datetime
import numpy as np
from filament import filament
import unittest
import logging


class TestSST(unittest.TestCase):

    def setUp(self):
        self.fname = "./data/TERRA_MODIS.20191201_20191231.L3m.MO.SST4.sst4.9km.nc"
        self.assertTrue(os.path.isfile(self.fname))

    def test_read_L3(self):
        sstmonth = filament.SST()
        sstmonth.read_from_oceancolorL3(self.fname)

        self.assertEqual(len(sstmonth.lat), 2160)
        self.assertEqual(len(sstmonth.lon), 4320)
        self.assertAlmostEqual(sstmonth.lon[0], -179.95833, places=5)
        self.assertEqual(sstmonth.lat[10], 89.125)
        self.assertTrue(np.ma.is_masked(sstmonth.field[200, 300]))
        self.assertAlmostEqual(sstmonth.field[1200, 300], 28.994999, places=6)


    def test_get_filename(self):
        sstfilename = filament.get_monthly_filename("TERRA", "MODIS", 2020, 6)
        self.assertTrue(sstfilename == "TERRA_MODIS.20200601_20200630.L3m.MO.SST4.sst4.9km.nc")

        sstfilename = filament.get_monthly_filename("TERRA", "MODIS", 2020, 6, "4km")
        self.assertTrue(sstfilename == "TERRA_MODIS.20200601_20200630.L3m.MO.SST4.sst4.4km.nc")

    def test_get_monthly_filename(self):
        sstfilename = filament.get_monthly_clim_filename("TERRA", "MODIS", 2000, 2019, 2, res="9km")
        self.assertTrue(sstfilename == "TERRA_MODIS.20000201_20190228.L3m.MC.SST4.sst4.9km.nc")

        sstfilename = filament.get_monthly_clim_filename("TERRA", "MODIS", 2000, 2019, 2, "4km")
        self.assertTrue(sstfilename == "TERRA_MODIS.20000201_20190228.L3m.MC.SST4.sst4.9km.nc")


if __name__ == '__main__':
    unittest.main()
