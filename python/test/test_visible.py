import os
import glob
import datetime
import filament
import unittest
import logging


class TestVisible(unittest.TestCase):

    def setUp(self):
        self.datadir = "./test/data/"
        self.fname = "./test/data/snapshot-2019-12-28T00_00_00Z.tiff"
        self.assertTrue(os.path.isfile(self.fname))
        self.assertTrue(os.path.isdir(self.datadir))

    def test_list(self):
        vis = filament.Visible()
        date = datetime.datetime(2020, 1, 1)
        flist = vis.list_files(self.datadir, date)

        self.assertTrue(len(flist) == 4)
        #flist[0] == 'Sentinel-2 L1C from 2020-01-01.tiff'
        # flist[-1] == 'MODIS_Aqua-2020-01-01T00_00_00Z.tiff'

if __name__ == '__main__':
    unittest.main()
