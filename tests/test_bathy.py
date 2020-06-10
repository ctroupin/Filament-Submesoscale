import os
import glob
import datetime
import filament
import unittest
import logging


class TestBathymetry(unittest.TestCase):

    def setUp(self):
        self.datafile = "./python/test/data/E4_2018.dtm"
        self.assertTrue(os.path.isdir(self.datafile))

    def test_read(self):
        bath = filament.Bathymetry()

        self.assertEqual(len(bath.lon), 9504)
        len(bath.lat) == 9024
        bath.depth.shape == (len(bath.lat), len(bath.lon))
        bath.lon[10] == -6.376562523333333
        bath.lat[22] == 43.13593763666667
        np.ma.is_masked(bath.depth[100, 200])
        bath.depth[1000, 1200] == -1852.0

if __name__ == '__main__':
    unittest.main()
