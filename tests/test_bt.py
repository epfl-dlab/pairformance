import unittest
import pandas as pd
import numpy as np
import itertools
import sys
sys.path.append('pairformance')

import pairformance.bt as bt


class TestBT(unittest.TestCase):
    def test_empty_0(self):
        data = pd.DataFrame.from_dict({})
        res = bt.BT(data)
        self.assertIsNone(res)

    def test_empty_1(self):
        data = pd.DataFrame.from_dict({
            'A': [],
            'B': []
        })
        res = bt.BT(data)
        self.assertIsNone(res)

    def test_empty_2(self):
        data = pd.DataFrame.from_dict({
            'A': [1],
            'B': [0]
        })
        answer = np.array([1., 0.])
        res = bt.BT(data)
        self.assertTrue((res == answer).all())

    def test_same_inputs_0(self):
        data = pd.DataFrame.from_dict({
            'A': [0.5, 0.5],
            'B': [0.5, 0.5]
        })
        answer = np.array([0.5, 0.5])
        res = bt.BT(data)
        self.assertTrue((res == answer).all())

    def test_same_inputs_1(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 0.5, 0.2],
            'B': [1, 0.5, 0.2]
        })
        answer = np.array([0.5, 0.5])
        res = bt.BT(data)
        self.assertTrue((res == answer).all())

    def test_simple_behavior_0(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 1],
            'B': [0, 0]
        })
        answer = np.array([1., 0.])
        res = bt.BT(data)
        self.assertTrue((res == answer).all())

    def test_simple_behavior_1(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 2, 1],
            'B': [0, 1, 2]
        })
        res = bt.BT(data)
        self.assertTrue(res[0] > res[1])

    def test_simple_behavior_2(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 2, 1],
            'B': [0, 1, 2],
            'C': [0.5, 0, 0]
        })
        argsort_answer = np.array([0, 1, 2])
        res = bt.BT(data)
        argsort_res = np.argsort(-res)
        self.assertTrue((argsort_res == argsort_answer).all())

    def test_cirlucar_0(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 0],
            'B': [0, 1]
        })
        answer = np.array([0.5, 0.5])
        res = bt.BT(data)
        self.assertTrue((res == answer).all())

    def test_cirlucar_1(self):
        data = pd.DataFrame.from_dict({
            'A': [1, 0, 1],
            'B': [0, 1, 0.5],
            'C': [2, 0.5, 0]
        })
        # A beats B, B beats C, C beats A
        answer = np.array([0.3333, 0.3333, 0.3333])
        res = bt.BT(data)
        self.assertTrue(np.allclose(res, answer, rtol=1e-03))

    def test_randomness(self):
        player_a = [2, 5, 2, 3, 4]
        player_b = [1, 2, 3, 4, 1]
        player_c = [2, 4, 5, 2, 2]
        df = pd.DataFrame.from_dict({
            'player_a': player_a,
            'player_b': player_b,
            'player_c': player_c
        })
        results = []
        for _ in range(10):
            results.append(bt.BT(df))

        all_results_close = []
        for pair in itertools.combinations(results, 2):
            all_results_close.append(
                np.allclose(pair[0], pair[1], rtol=1e-03))
        self.assertTrue(np.array(all_results_close).all())


if __name__ == '__main__':
    print('here')
    unittest.main()