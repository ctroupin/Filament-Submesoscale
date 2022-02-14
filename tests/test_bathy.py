import sys
sys.path.insert(0, '..')
import os
import numpy as np
import glob
import datetime
from filament import filament
import unittest
import logging


class TestBathymetry(unittest.TestCase):

    def setUp(self):
        self.datafile = "./data/E4_2018.dtm"
        self.assertTrue(os.path.isfile(self.datafile))

    def test_read(self):
        bath = filament.Bathymetry()
        bath.read_from_EMODnet_dtm(self.datafile)
        self.assertEqual(len(bath.lon), 9504)
        len(bath.lat) == 9024
        bath.depth.shape == (len(bath.lat), len(bath.lon))
        bath.lon[10] == -6.376562523333333
        bath.lat[22] == 43.13593763666667
        np.ma.is_masked(bath.depth[100, 200])
        bath.depth[1000, 1200] == -1852.0

if __name__ == '__main__':
    unittest.main()
