import unittest
import numpy as np
import sys

import pandas as pd

sys.path.append('pairformance')

import pairformance.core as pf


class TestPairformance(unittest.TestCase):
    def test_config_override_0(self):
        config = {
            'compute_pairwise': True,
            'n_bootstrap': 10,
            'epsilon_bt': 1e-9,
            'pval_threshold': 0.05,
            'aggregations': ['BT']
        }

        df = pd.DataFrame.from_dict({
            'A': [1, 2, 1],
            'B': [0, 1, 2],
            'C': [0.5, 0, 0]
        })

        pf_eval = pf.Pairformance(df, config=config)
        _ = pf_eval.eval()
        global_results = pf_eval.print_global_results()

        argsort_answer = np.array([0, 1, 2])
        argsort_res = np.argsort(-global_results['BT'])
        self.assertTrue((argsort_res == argsort_answer).all())

    def test_config_override_1(self):
        config = {
            'compute_pairwise': True,
            'n_bootstrap': 10,
            'epsilon_bt': 1e-9,
            'pval_threshold': 0.05,
            'aggregations': ['Mean']
        }
        player_a = [2, 5, 2, 3, 4]
        player_b = [1, 2, 3, 4, 1]
        player_c = [2, 4, 5, 2, 2]
        df = pd.DataFrame.from_dict({
            'player_a': player_a,
            'player_b': player_b,
            'player_c': player_c
        })

        pf_eval = pf.Pairformance(df, config=config)
        results = pf_eval.eval()
        res = (set(results['global-results'].keys()) == {'Mean'})
        self.assertTrue(res)
        self.assertEqual(results['global-results']['Mean'].shape[0], 10)