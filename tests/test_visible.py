import os
import glob
import datetime
from filament import filament
import unittest
import logging


class TestVisible(unittest.TestCase):

    def setUp(self):
        self.datadir = "./data/"
        self.fname = "./data/snapshot-2019-12-28T00_00_00Z.tiff"
        self.assertTrue(os.path.isfile(self.fname))
        self.assertTrue(os.path.isdir(self.datadir))

    def test_list(self):
        vis = filament.Visible()
        date = datetime.datetime(2020, 1, 1)
        flist = vis.list_files(self.datadir, date)

        self.assertTrue(len(flist) == 4)
        #flist[0] == 'Sentinel-2 L1C from 2020-01-01.tiff'
        # flist[-1] == 'MODIS_Aqua-2020-01-01T00_00_00Z.tiff'

    def test_read(self):
        vis = filament.Visible()
        vis.read_geotiff(self.fname)
        self.assertTrue(vis.lon.shape == (353, 433))
        self.assertTrue(vis.lat.shape == (353, 433))
        self.assertTrue(vis.image.shape == (353, 433, 3))
        self.assertTrue(vis.lon[0][0] == -18.3262939453125)
        self.assertTrue(vis.lon[10][10] == -18.3043212890625)
        self.assertTrue(vis.lat[0][-1] == 29.0313720703125)
        self.assertTrue(vis.image[0][3][0] == 29)
        self.assertTrue(vis.image[0][3][1] == 43)
        self.assertTrue(vis.image[0][3][2] == 56)
        self.assertTrue(vis.extent == [-18.3262939453125, -17.3748779296875, 28.2557373046875, 29.0313720703125])

if __name__ == '__main__':
    unittest.main()
