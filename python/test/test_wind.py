import os
import filament
import numpy as np
import unittest

class TestWindAscat(unittest.TestCase):

    def setUp(self):
        self.windfile = "./test/data/ascat_20180313_003000_metopa_59134_eps_o_coa_2401_ovw.l2.nc.gz.nc4"
        self.domain = (-50., -40., 0., 20.)
        self.domain2 = (100., 140., 0., 20.)
        self.assertTrue(os.path.isfile(self.windfile))

    def test_list(self):
        wind = filament.Wind()

    def test_read_ascat(self):
        wind = filament.Wind()
        r = wind.read_ascat(self.windfile)
        self.assertTrue(r)

        self.assertEqual(len(wind.lon), 136513)
        self.assertEqual(len(wind.lat), 136513)
        self.assertAlmostEqual(wind.lon[20], -52.24672999)
        self.assertEqual(wind.lat[200], 1.31254)
        self.assertEqual(wind.speed.mean(), 8.85660589984728)
        self.assertEqual(len(wind.speed.compressed()), 62205)
        self.assertTrue(np.ma.is_masked(wind.u[999]))
        self.assertTrue(np.ma.is_masked(wind.v[1999]))
        self.assertEqual(wind.u[22222], 3.603894954004558)

    def test_read_ascat_domain(self):
        wind = filament.Wind()
        r = wind.read_ascat(self.windfile, self.domain)
        self.assertTrue(r)

        self.assertEqual(len(wind.lon), 6134)
        self.assertEqual(len(wind.lat), 6134)
        self.assertEqual(wind.lon[-1], -47.16028)
        self.assertEqual(wind.lat[600], 4.96835)
        self.assertEqual(wind.speed.min(), 7.01)
        self.assertEqual(len(wind.speed.compressed()), 6134)
        self.assertFalse(np.ma.is_masked(wind.u[999]))
        self.assertFalse(np.ma.is_masked(wind.v[1999]))
        self.assertEqual(wind.u[124], -7.4646841483642)

    def test_read_ascat_domain(self):
        wind = filament.Wind()
        r = wind.read_ascat(self.windfile, self.domain2)
        self.assertFalse(r)

class TestWindCCMP(unittest.TestCase):

    def setUp(self):
        self.windfile = "./test/data/ascat_20180313_003000_metopa_59134_eps_o_coa_2401_ovw.l2.nc.gz.nc4"
        self.domain = (20.12, 40.98, -42.34, -20.)
        self.assertTrue(os.path.isfile(self.windfile))

    def test_read_ccmp(self):
        wind = filament.Wind()
        wind.read_from_ccmp(self.windfile)
        len(wind.lon) == 1440
        len(wind.lat) == 628
        wind.lon[0] == 0.125
        wind.lon[1] == 0.375
        wind.lat[0] == -78.375
        wind.lat[1] == -78.125
        wind.uwind.shape == (1, 628, 1440)
        wind.uwind.mean() == -0.11586497




if __name__ == '__main__':
    unittest.main()
