import os
import filament
import numpy as np
import unittest

class TestWindList(unittest.TestCase):
    """
    Tests for the functions returning the list of OPEnDAP URLs
    for QuikSCAT and ASCAT
    """
    def test_list_qs(self):
        qslist = filament.get_filelist_url_quikscat(2009, 154)
        self.assertEqual(len(qslist), 14)
        self.assertEqual(qslist[0], 'https://opendap.jpl.nasa.gov/opendap/OceanWinds/quikscat/L2B12/v4.0/2009/154/qs_l2b_51845_v4.0_200906030115.nc')
        self.assertEqual(qslist[-1], 'https://opendap.jpl.nasa.gov/opendap/OceanWinds/quikscat/L2B12/v4.0/2009/154/qs_l2b_51858_v4.0_200906032308.nc')

    def test_list_as(self):
        aslist = filament.get_filelist_url(2019, 3)
        self.assertEqual(len(aslist), 28)
        self.assertEqual(aslist[0], 'https://opendap.jpl.nasa.gov:443/opendap/OceanWinds/ascat/preview/L2/metop_a/coastal_opt/2019/003/ascat_20190103_013900_metopa_63340_eps_o_coa_3201_ovw.l2.nc.gz')
        self.assertEqual(aslist[-1],'https://opendap.jpl.nasa.gov:443/opendap/OceanWinds/ascat/preview/L2/metop_b/coastal_opt/2019/003/ascat_20190103_230000_metopb_32666_eps_o_coa_3201_ovw.l2.nc.gz')

class TestWindQuiscat(unittest.TestCase):

    def setUp(self):
        self.windfile = "./python/test/data/qs_l2b_52886_v4.0_200908150119.nc"
        self.domain = (-125., -100., 25., 50.)
        self.assertTrue(os.path.isfile(self.windfile))

    def test_read_quikcat(self):
        wind = filament.Wind()
        r = wind.read_from_quikscat(self.windfile)
        self.assertTrue(r)

        self.assertTrue(wind.lon.shape == (242935,))
        self.assertTrue(wind.lat.shape == (242935,))
        self.assertAlmostEqual(wind.lon.min(), -179.99792, places=5)
        self.assertAlmostEqual(wind.lon.mean(), -120.590515, places=6)
        self.assertAlmostEqual(wind.lat[0], 70.397896, places=6)
        self.assertAlmostEqual(wind.lat.max(), 89.866196, places=6)
        self.assertAlmostEqual(wind.u.max(), 29.148708, places=6)
        self.assertTrue(np.ma.is_masked(wind.u[0]))
        self.assertTrue(wind.u.mean(), 1.5385874209615187)
        self.assertTrue(wind.v.mean(), 1.2612295160130862)
        self.assertTrue(np.ma.is_masked(wind.angle[1]))

    def test_read_quikcat_domain(self):
        wind = filament.Wind()
        r = wind.read_from_quikscat(self.windfile, self.domain)
        self.assertTrue(r)

        self.assertTrue(wind.lon.shape == (29198,))
        self.assertTrue(wind.lat.shape == (29198,))
        self.assertAlmostEqual(wind.lon.min(), -124.99933, places=5)
        self.assertAlmostEqual(wind.lon.mean(), -115.39173, places=5)
        self.assertAlmostEqual(wind.lat[0], 49.949474, places=6)
        self.assertAlmostEqual(wind.lat.max(), 49.999313, places=6)
        self.assertAlmostEqual(wind.u.max(), 13.594069, places=6)
        self.assertTrue(np.ma.is_masked(wind.u[0]))
        self.assertTrue(wind.u.mean(), 2.214242427089597)
        self.assertTrue(wind.v.mean(), -5.515122331629585)
        self.assertTrue(np.ma.is_masked(wind.angle[1]))

class TestWindAscat(unittest.TestCase):

    def setUp(self):
        self.windfile = "./python/test/data/ascat_20180313_003000_metopa_59134_eps_o_coa_2401_ovw.l2.nc.gz.nc4"
        self.domain = (-50., -40., 0., 20.)
        self.domain2 = (100., 140., 0., 20.)
        self.assertTrue(os.path.isfile(self.windfile))

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
        self.windfile = "./python/test/data/CCMP_Wind_Analysis_201001_V02.0_L3.5_RSS.nc"
        # self.windfile = "https://opendap.jpl.nasa.gov/opendap/OceanWinds/ccmp/L3.5a/monthly/flk/2010/month_20100101_v11l35flk.nc.gz"
        self.domain = (20.12, 40.98, -42.34, -20.)
        # self.assertTrue(os.path.exists(self.windfile))

    def test_read_ccmp(self):
        wind = filament.Wind()
        wind.read_from_ccmp(self.windfile)
        self.assertEqual(len(wind.lon), 1440)
        self.assertEqual(len(wind.lat), 628)
        self.assertEqual(wind.lon[0], 0.125)
        self.assertEqual(wind.lon[1], 0.375)
        self.assertEqual(wind.lat[0], -78.375)
        self.assertEqual(wind.lat[1], -78.125)
        self.assertEqual(wind.u.shape, (1, 628, 1440))
        self.assertEqual(wind.u.mean(), -0.11586497)

        wind2 = filament.Wind()
        wind2.read_from_ccmp(self.windfile, self.domain)
        self.assertEqual(wind2.u.shape, (89, 84))
        self.assertEqual(wind2.u[10, 10], 4.51583)
        self.assertEqual(wind2.v[20, 20], 0.11376371)

class TestWindKNMI(unittest.TestCase):

    def setUp(self):
        self.windfile = "./data/GLO-WIND_L3-OBS_METOP-A_ASCAT_12_ASC_20190816.nc"


    def test_read_ccmp(self):
        wind = filament.Wind()
        wind.read_from_ccmp(self.windfile)
        self.assertEqual(wind.u.shape, (40, 96))
        self.assertEqual(wind.u.mean(), 0.2433392382639504)
        self.assertEqual(wind.v.mean(), 3.195155004428698)
        self.assertEqual(wind.u[10, 20], 1.6600000000000001)
        self.assertTrue(np.ma.is_masked(wind.v[32, 14]))
        self.assertEqual(wind.speed[10, 4], 2.52)
        self.assertEqual(wind.angle[-1, 40], 210.20000000000002)

if __name__ == '__main__':
    unittest.main()
