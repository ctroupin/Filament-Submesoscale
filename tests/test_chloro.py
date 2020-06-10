import os
import glob
import datetime
from filament import filament
import unittest
import logging


class TestChloro(unittest.TestCase):

    def setUp(self):
        self.fname = "./tests/data/A2019257141500.L2_LAC_OC.nc"
        self.assertTrue(os.path.isfile(self.fname))

    def test_read(self):

        chloro = filament.Chloro()
        chloro.read_from_oceancolorL2(self.fname)

        self.assertEqual(chloro.lon.shape, (2030, 1354))
        self.assertEqual(chloro.lat.shape, (2030, 1354))
        self.assertAlmostEqual(chloro.field.max(), 465.5181, places=4)
        self.assertAlmostEqual(chloro.field.min(), 0.001, places=3)

if __name__ == '__main__':
    unittest.main()
