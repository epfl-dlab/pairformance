from pairformance import Pairformance
from pairformance import plotting_utils as pf_plot
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    import pandas as pd

    eval_df = pd.read_csv('dialogue_example_df.csv')

    pf_eval = Pairformance(df=eval_df)

    pp.pprint(pf_eval.diagnostic_system_scores(eval_df))

    # Probability that system M0 is better than system M1 (BT)
    p_m0_m1, pval = pf_eval.prob_a_better_b('M0', 'M1')
    print(f"Prob that M0 is better than M1 {p_m0_m1}, pval {pval}")

    # Compute the full analysis of the DataFrame
    pairwise_analysis, global_analysis = pf_eval.eval()

    # One score per systems, with confidence interval
    system_results = pf_eval.print_global_results()

    # One score per systems, with confidence interval
    system_results = pf_eval.print_pairwise_results()

    pf_eval.diagnostic_aggregation_agreement()

    pf_plot.plot_paired_graphs(data=eval_df,
                               system_a='M2',
                               system_b='M1',
                               save_path=None)
