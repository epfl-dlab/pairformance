import itertools
import sys
import os
from typing import Dict, Union, List

import numpy as np
import scipy.stats as stats
from collections import defaultdict

import pandas as pd

from .bt import BT
from .utils import same_top_n, tau_to_percent, percent_outliers

DEFAULT_CONFIG: Dict[str, Union[float, List[str], int, bool]] = {
    'compute_pairwise': True,
    'n_bootstrap': 100,
    'epsilon_bt': 1e-9,
    'pval_threshold': 0.05,
    'aggregations': ['Mean', 'Median', 'BT']
}


class Pairformance(object):
    """
    Pairformance(df=eval_df, config=config)

    Tool to perform paired evaluation of automatic systems.
    It takes a dataframe as input with systems as columns and the system scores on test instances as rows (one row per test instance).

    The pairformance tool takes the pairing into account and provide:
    - Estimates of `mean`, `median`, and `BT` scores for each system
    - Confidence intervals for each estimate with bootstrap resampling of the data
    - Statistical testing for each of the three aggregation mechanisms
    - Pairwise comparison between each system. In particular, `pairformance` can produce the probability that any system has better score than any other system.
    - Diagnostic of the disagreement between `mean`, `median`, and `BT` about the ordering of systems
    - Diagnostic of the score distributions (outliers, measures of the strength of the pairing, score variances across systems and instances)
    - Plotting facilities to view global estimates, pairwise estimates, and the pairing structure for each pair of systems


    Attributes
    ----------
    df : pd.DataFrame
        evaluation dataframe with system scores (one column per system, one row per test instance)
    config: dict
        dictionary with configuration parameters


    Methods
    -------
    eval() -> dict
        Compute the main evaluation. Runs the global estimates with bootstrap resamples for all aggregation mechanisms considered.
        Run the pairwise comparison estimate with bootstrap resamples if the configuration parameter `compute_pairwise` is True.

    prob_a_better_b(system_a: str, system_b: str, df: pd.DataFrame = None) -> [Float, Float]
        Compute the probability that one system is better than other (special case of BT with 2 systems)
        It also reports the p-value testing whether the probability is significantly above 0.5.

    diagnostic_aggregation_agreement(system_results: dict = None) -> dict:
        Computes disagreement measures between aggregation mechanisms considered in the config file.
        The disagreement measures are the ones discussed in the paper.

    diagnostic_system_scores(eval_df: pd.DataFrame) -> dict:
        Computes diagnostic statistics from the distribution of system scores

    print_global_results() -> pd.DataFrame:
        Prints the global estimates for each system and each aggregation mechanism.

    print_pairwise_results() -> dict:
        Prints the pairwise comparison for each system and each aggregation mechanism.
    """
    def __init__(self, df: pd.DataFrame, config: dict = DEFAULT_CONFIG):
        """
        Intialization of Pairformance with evaluation dataframe and configuration parameters.

        The following configuration parameters are available:
        `aggregations`: a list of aggregation mechanisms to consider
        `compute_pairwise`: whether to run the analyses for each pair of systems
        `epsilon_bt`: when the iterative algorithm of BT does not vary by more than epsilon_bt, we stop.
        `n_bootstrap`: number of bootstrap resample of the data to take to compute confidence intervals
        `pval_threshold`: percentile to use for confidence interval. Example: 0.05 gives the (5, 95) percentiles.

        :param df: evaluation dataframe with system scores (one column per system, one row per test instance)
        :param config: dictionary with configuration parameters
        """
        self.df = df
        self.config = config

        if self.config['compute_pairwise']:
            self.data_matrices = {}
            self.pval_matrices = {}
            self.orders_of_systems = {}

    def _bootstrap_global_eval(self, df: pd.DataFrame) -> dict:
        """
        Runs the global estimates for all systems and aggregations considered with bootstrap resamples.

        :param df: evaluation dataframe with system scores (one column per system, one row per test instance)
        :return: dictionary of dataframes with bootstrap resamples of global estimates.
        """
        global_eval = defaultdict(list)

        for _ in range(self.config['n_bootstrap']):
            resample_df = df.sample(frac=1, replace=True)

            if 'Mean' in self.config['aggregations']:
                global_eval['Mean'].append(resample_df.mean(axis=0))

            if 'Median' in self.config['aggregations']:
                global_eval['Median'].append(resample_df.median(axis=0))

            if 'BT' in self.config['aggregations']:
                bt_results = BT(resample_df)
                global_eval['BT'].append(pd.Series(dict(zip(df.columns, bt_results))))

        self.global_results = {method: pd.concat(data, axis=1).T
                               for method, data in global_eval.items()}
        return self.global_results

    def _bootstrap_pairwise_eval(self, df: pd.DataFrame) -> dict:
        """
        Runs the pairwise comparison for all systems and aggregations considered with bootstrap resamples.

        :param df: evaluation dataframe with system scores (one column per system, one row per test instance)
        :return: dictionary of dataframes with bootstrap resample estimates of the pairwise comparisons
        """
        pairwise_data = defaultdict(dict)

        for system_a, system_b in itertools.combinations(df.columns, 2):
            if 'BT' in self.config['aggregations']:
                bt_scores = BT(df[[system_a, system_b]])
                bt_best_sys = (system_a if bt_scores[0] >= bt_scores[1] else system_b)
                bt_worst_sys = (system_b if bt_scores[0] >= bt_scores[1] else system_a)
                bt_median_of_differences = (df[bt_best_sys] - df[bt_worst_sys]) > 0
                p_a_better_b = bt_scores[0] / (bt_scores[0] + bt_scores[1])
                p_b_better_a = bt_scores[1] / (bt_scores[0] + bt_scores[1])
            else:
                bt_median_of_differences = None

            pval_mean, pval_median, pval_bt = \
                Pairformance._statistical_testing(df[system_a], df[system_b], bt_median_of_differences)

            # A -> B: information relative to whether A is better than B
            pairwise_data[system_a][system_b] = {}
            pairwise_data[system_b][system_a] = {}
            if 'Mean' in self.config['aggregations']:
                means_df = self.global_results['Mean']
                pairwise_data[system_a][system_b]['Difference of Means'] = means_df[system_a].mean(axis=0) - means_df[
                    system_b].mean(axis=0)
                pairwise_data[system_a][system_b]['p-val (paired t-test)'] = pval_mean

                pairwise_data[system_b][system_a]['Difference of Means'] = means_df[system_b].mean(axis=0) - means_df[
                    system_a].mean(axis=0)
                pairwise_data[system_b][system_a]['p-val (paired t-test)'] = pval_mean

            if 'Median' in self.config['aggregations']:
                medians_df = self.global_results['Median']
                pairwise_data[system_a][system_b]['Difference of Medians'] = medians_df[system_a].mean(axis=0) - \
                                                                             medians_df[system_b].mean(axis=0)
                pairwise_data[system_a][system_b]['p-val (Mood\'s test)'] = pval_median

                pairwise_data[system_b][system_a]['Difference of Medians'] = medians_df[system_b].mean(axis=0) - \
                                                                             medians_df[system_a].mean(axis=0)
                pairwise_data[system_b][system_a]['p-val (Mood\'s test)'] = pval_median

            if 'BT' in self.config['aggregations']:
                pairwise_data[system_a][system_b]['BT: P(A > B)'] = p_a_better_b
                pairwise_data[system_a][system_b]['p-val (Sign test)'] = pval_bt

                pairwise_data[system_b][system_a]['BT: P(A > B)'] = p_b_better_a
                pairwise_data[system_b][system_a]['p-val (Sign test)'] = pval_bt

        self.pairwise_results = pairwise_data
        return self.pairwise_results

    def eval(self) -> dict:
        """
        Compute the main evaluation. Runs the global estimates with bootstrap resamples for all aggregation mechanisms considered.
        Run the pairwise comparison estimate with bootstrap resamples if the configuration parameter `compute_pairwise` is True.

        :return: dictionary with global_result or both global_results and pairwise_results.
        """
        # sets the attribute self.global_results
        self._bootstrap_global_eval(self.df)

        if not self.config['compute_pairwise']:
            return {'global-results': self.global_results}

        # sets the attribute self.pairwise_results
        self._bootstrap_pairwise_eval(self.df)

        for agg_choice in self.config['aggregations']:
            self._prepare_data_matrix(agg_choice)

        return {'pairwise-results': self.pairwise_results,
                'global-results': self.global_results}

    def _prepare_data_matrix(self, aggreagation_choice: str = 'BT'):
        """
        Inner data structure management to cleanly organize the results of bootstrap resamples in the pairwise setup.

        :param aggreagation_choice: a choice of aggregation mechanism
        :return:
        """
        if aggreagation_choice == 'Mean':
            stat_key = 'Difference of Means'
            pval_key = 'p-val (paired t-test)'
        elif aggreagation_choice == 'Median':
            stat_key = 'Difference of Medians'
            pval_key = 'p-val (Mood\'s test)'
        else:
            stat_key = 'BT: P(A > B)'
            pval_key = 'p-val (Sign test)'

        nb_systems = len(self.pairwise_results)
        data_matrix = np.zeros((nb_systems, nb_systems))
        pval_matrix = np.zeros((nb_systems, nb_systems))

        # Order systems by performance
        order_of_systems = []
        for system, matches in self.pairwise_results.items():
            order_of_systems.append((system, np.sum([v[stat_key] for _, v in matches.items()])))

        sorted_systems = sorted(order_of_systems, key=lambda t: t[1])
        sorted_systems = [t[0] for t in sorted_systems]
        order_of_systems = dict(zip(sorted_systems, list(range(len(sorted_systems)))))

        for system_a, idx_a in order_of_systems.items():
            for system_b, idx_b in order_of_systems.items():
                if system_a == system_b:
                    # default difference between itself
                    data_matrix[idx_a][idx_b] = \
                        (0. if aggreagation_choice in ['Mean', 'Median'] else 0.5)

                    # Not significant by default
                    pval_matrix[idx_a][idx_b] = 1.
                    continue

                data_pair = self.pairwise_results[system_a][system_b]
                data_matrix[idx_a][idx_b] = data_pair[stat_key]
                pval_matrix[idx_a][idx_b] = data_pair[pval_key]

        self.data_matrices[aggreagation_choice] = data_matrix
        self.pval_matrices[aggreagation_choice] = pval_matrix
        self.orders_of_systems[aggreagation_choice] = order_of_systems

    def prob_a_better_b(self,
                        system_a: str,
                        system_b: str,
                        df: pd.DataFrame = None) -> [float, float]:

        """
        Compute the probability that one system is better than other (special case of BT with 2 systems)
        It also reports the p-value testing whether the probability is significantly above 0.5.

        :param system_a: name of the system a
        :param system_b: name of system b
        :param df: evaluation dataframe with system scores (one column per system, one row per test instance)
        :return: (P_A_betterthan_B, p_value)
        """
        if df is None:
            assert hasattr(self, 'df'), "No dataframe is available"
            df = self.df

        assert system_a in df.columns and system_b in df.columns, \
            "at least one of the systems provided is not in the dataframe"

        try:
            comparison_res = self.pairwise_results[system_a][system_b]
            return comparison_res['BT: P(A > B)'], comparison_res['p-val (Sign test)']
        except Exception as e:
            pass

        bt_scores = BT(df[[system_a, system_b]])
        p_a_better_b = bt_scores[0] / (bt_scores[0] + bt_scores[1])

        best_sys = (system_a if bt_scores[0] >= bt_scores[1] else system_b)
        worst_sys = (system_b if bt_scores[0] >= bt_scores[1] else system_a)
        median_of_differences = (df[best_sys] - df[worst_sys]) > 0

        pval_bt = stats.binom_test(np.sum(median_of_differences),
                                   n=len(median_of_differences), p
                                   =0.5)

        return p_a_better_b, pval_bt

    def diagnostic_aggregation_agreement(self, system_results: dict = None) -> dict:
        """
        Computes disagreement measures between aggregation mechanisms considered in the config file.
        The disagreement measures are the ones discussed in the paper.

        :param system_results: global system results as outputed by Pairformance.print_global_results()
        :return: dictionary with agreement statistics
        """
        if system_results is None:
            orig_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            system_results = self.print_global_results()
            sys.stdout.close()
            sys.stdout = orig_stdout

        print("\n\n===== Aggregation Agreement =====")
        print("\t {:^25} {:^15} {:^15} {:^15}".format("Pairs of Agg.",
                                                      "Percentage Aggr.",
                                                      "Same Sota",
                                                      "Same Top-3"))

        agreements = {}
        for pairs_of_agg in itertools.combinations(self.config['aggregations'], 2):
            agg_a, agg_b = pairs_of_agg
            tau_agreement = tau_to_percent(stats.kendalltau(system_results[agg_a], system_results[agg_b])[0])
            same_sota = same_top_n(system_results[agg_a], system_results[agg_b], 1)
            same_top_3 = same_top_n(system_results[agg_a], system_results[agg_b], 1)
            agreements[str(pairs_of_agg)] = {
                'Percentage-agreement': tau_agreement,
                'Same-sota': same_sota,
                'Same-top-3': same_top_3
            }
            pair = f"{agg_a} vs. {agg_b}"
            line = "\t {:^25} {:^15.2f} {:^15} {:^15}".format(pair,
                                                              tau_agreement,
                                                              str(same_sota),
                                                              str(same_top_3))
            print(line)

        return agreements

    @staticmethod
    def diagnostic_system_scores(eval_df: pd.DataFrame) -> dict:
        """
        Computes the following diagnostic statistics from the distribution of system scores:
        `global-outliers-percentage`: take the whole array of data and compute the percentage of outliers
        `per-system-outliers-percentage`: percentage of outliers for each system independently
        `mean-row-std`: measure the std per test instance, then average this std across test instances, i.e., how much variance is there across system
        `rank-std-across-rows`: computes the order of systems per test instance, and measure how much the ranking varies across test instances, i.e., how much variance is there across ranking of systems
        `std-of-row-means`: computes the mean score of each test instance (across systems) and measure the std across test instances, i.e., how much overall scores vary across test instances

        :param eval_df: evaluation dataframe with system scores (one column per system, one row per test instance)
        :return: dictionary with diagnostic statistics
        """

        # Compute % outliers on the whole dataframe
        overall_outlier_percent = percent_outliers(eval_df.to_numpy().flatten())

        # Compute % outliers per system
        outlier_per_system = {}
        for systems in eval_df.columns:
            outlier_per_system[systems] = \
                percent_outliers(eval_df[systems].to_numpy().flatten())

        # Compute score variance per row
        mean_row_std = eval_df.std(axis=1).mean()

        # Compute rank disagreement per instance average over instances
        rank_std_across_rows = eval_df.rank(axis=1).std(axis=0).to_dict()

        # Compute mean score per row variance
        std_of_row_means = eval_df.mean(axis=1).std()

        # Compute system pairwise harmony
        pairwise_harmony = {}
        for system_a, system_b in itertools.combinations(eval_df.columns, 2):
            key = f"{system_a}-{system_b}"
            pairwise_harmony[key] = stats.kendalltau(eval_df[system_a], eval_df[system_b])

        return {
            'global-outliers-percentage': overall_outlier_percent,
            'per-system-outliers-percentage': outlier_per_system,
            'mean-row-std': mean_row_std,
            'rank-std-across-rows': rank_std_across_rows,
            'std-of-row-means': std_of_row_means
        }

    @staticmethod
    def _statistical_testing(system_a_scores, system_b_scores, median_of_differences=None):
        """
        Computes statistical tests related to mean, median, and BT (if median_of_differences is not None).
        In particular, it computes the paired t-test, the Mood's median test, and the sign test.

        :param system_a_scores: list of scores
        :param system_b_scores: list of scores
        :param median_of_differences: list of differences
        :return: (p_value_t_test, p_value_mood_test, p_value_sign_test)
        """
        # Statistical testing related to mean
        _, p_value_t_test = stats.ttest_rel(system_a_scores, system_b_scores)

        # Statistical testing related to median
        _, p_value_mood_test, _, _ = stats.median_test(system_a_scores, system_b_scores)

        # Statistical testing related to BT scores
        if median_of_differences is not None:
            p_value_sign_test = stats.binom_test(np.sum(median_of_differences),
                                       n=len(median_of_differences),
                                       p=0.5)
        else:
            p_value_sign_test = None

        return p_value_t_test, p_value_mood_test, p_value_sign_test

    # Logging

    def print_global_results(self) -> pd.DataFrame:
        """
        Prints the global estimates for each system and each aggregation mechanism.
        It requires self.global_results to exists.
        This is done by first running eval().


        :return: a dataframe with the results printed to stdout.
        """

        assert hasattr(self, 'global_results'), " ".join(["self.global_results should exists,",
                                                          "please run .eval() first"])

        results = {}
        print("\n\n===== Global Results =====")
        for agg, aggregation_results in self.global_results.items():
            ci_type = "({:.0%} bootstrap CI)".format(1 - self.config['pval_threshold'])
            print("\n*** {} ***".format(agg))

            print("\t {:^15} {:^15} {:^15}".format("Systems", "Est. score", ci_type))

            #             print(" ".join([sys.rjust(20) for sys in aggregation_results.keys()]))
            for system, system_results in aggregation_results.items():
                estimated_score = system_results.mean()
                lo_CI = system_results.quantile(self.config['pval_threshold'])
                hi_CI = system_results.quantile(1 - self.config['pval_threshold'])
                ci = "({:.3f}, {:.3f})".format(lo_CI, hi_CI)
                line = "\t {:^15} {:^15.3f} {:^15}".format(system, estimated_score, ci)
                print(line)

                if system not in results:
                    results[system] = {}
                results[system][agg] = estimated_score

        return pd.DataFrame.from_dict(results, orient='index')

    def print_pairwise_results(self) -> dict:
        """
        Prints the pairwise comparison for each system and each aggregation mechanism.
        It requires self.pairwise_results to exists.
        This is done by first running eval() with the configuration parameter `compute_pairwise` set to True.


        :return: a dict object with the results printed to stdout.
        """

        assert hasattr(self, 'pairwise_results'), " ".join(["self.global_results should exists,",
                                                            "please run .eval() first with the configuration parameter",
                                                            "`compute_pairwise` set to True"])

        results = {}
        print("\n\n===== Pairwise Results =====")
        for system, system_comparisons in self.pairwise_results.items():
            print("\n*** {} ***".format(system))

            if len(system_comparisons.values()) == 0:
                return results

            top_line = "\t {:^20}".format("Systems [A vs. B]")
            for key_results in list(system_comparisons.values())[0].keys():
                top_line += " {:^20}".format(key_results)
            print(top_line)

            for compared_system, comparison_results in system_comparisons.items():
                systems_involved = "{} vs. {}".format(system, compared_system)
                line = "\t {:^20}".format(systems_involved)
                #                if systems_involved not in results:
                results[systems_involved] = comparison_results

                for k, result in comparison_results.items():
                    line += " {:^20.3f}".format(result)
                print(line)

        return results
