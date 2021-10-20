import unittest

import sys

sys.path.append('pairformance')

import pairformance.utils as utils


class TestUtils(unittest.TestCase):
    def test_tau_to_percent_0(self):
        res = utils.tau_to_percent(0.)
        self.assertEqual(res, 0.5)

    def test_tau_to_percent_1(self):
        res = utils.tau_to_percent(-1.)
        self.assertEqual(res, 0.)

    def test_tau_to_percent_2(self):
        res = utils.tau_to_percent(1.)
        self.assertEqual(res, 1.)

    def test_tau_to_percent_3(self):
        res = utils.tau_to_percent(1.1)
        self.assertEqual(res, 1.)

    def test_tau_to_percent_4(self):
        res = utils.tau_to_percent(-1.1)
        self.assertEqual(res, 0.)

    def test_same_top_n_0(self):
        res = utils.same_top_n([], [], n=10)
        self.assertIsNone(res)

    def test_same_top_n_1(self):
        res = utils.same_top_n([1, 2, 3, 4], [], n=2)
        self.assertIsNone(res)

    def test_same_top_n_2(self):
        res = utils.same_top_n([1], [10], n=1)
        self.assertTrue(res)

    def test_same_top_n_3(self):
        res = utils.same_top_n([1, 2], [1.1, 0.9], n=1)
        self.assertFalse(res)

    def test_same_top_n_4(self):
        res = utils.same_top_n([1, 2, 3, 0.5, 0.3], [1, 2, 3, 0.2, 0.9], n=3)
        self.assertTrue(res)

    def test_same_top_n_5(self):
        res = utils.same_top_n([1, 2, 3, 0.5, 0.2], [2, 3, 1, 0.3, 0.4], n=3)
        self.assertTrue(res)
