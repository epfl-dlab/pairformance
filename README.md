# pairformance
Tool to perform paired evaluation of automatic systems.

See [published article](https://aclanthology.org/2021.acl-long.179/)

## Motivation

Evaluation in NLP is usually done by comparing the scores of competing systems independently averaged over a common set of test instances. 
In our recent ACL publication, we question the use of averages for aggregating evaluation scores into a final number used to decide which system is best, 
since the average, as well as alternatives such as the median, ignores the pairing arising from the fact that systems are evaluated on the same test instances. 
We illustrate the importance of taking the instancelevel pairing of evaluation scores into account and demonstrate, both theoretically and empirically, 
the advantages of aggregation methods based on pairwise comparisons, such as the Bradley–Terry (BT) model, a mechanism based on the estimated probability 
that a given system scores better than another on the test set. 
By re-evaluating 296 real NLP evaluation setups across four tasks and 18 evaluation metrics, 
we show that the choice of aggregation mechanism matters and yields different conclusions as to which systems are state of the art in about 30% of the setups.
To facilitate the adoption of pairwise evaluation, we release this tool called `pairformance` a tool aggregate scores with mean, median, and BT.

## Description

The pairformance tool takes a dataframe  as input with systems as columns
and the system scores on test instances as rows (one row per test instance)

In standard evaluation practice focused only focusing on `mean`, one would obtain 
mean estimates simply by running: `eval_df.mean(axis=0)` if `eval_df` is the variable 
containing the dataframe.

The pairformance tool takes the pairing into account and provide:
- Estimates of `mean`, `median`, and `BT` scores for each system
- Confidence intervals for each estimate with bootstrap resampling of the data
- Statistical testing for each of the three aggregation mechanisms
- Pairwise comparison between each system. In particular, `pairformance` can produce the probability that any system has better score than any other system.
- Diagnostic of the disagreement between `mean`, `median`, and `BT` about the ordering of systems
- Diagnostic of the score distributions (outliers, measures of the strength of the pairing, score variances across systems and instances)
- Plotting facilities to view global estimates, pairwise estimates, and the pairing structure for each pair of systems

## Default Usage

Suppose we have an evaluation data frame with evaluation scores for several systems
(one column per system, one row per test instance):

```python
import pandas as pd

eval_df = pd.read_csv('examples/dialogue_example_df.csv')
eval_df.head()
```

> | M0 | M1 | M2 | M3 | M4 |
> |----|----|----|----|----|
> | .495 | .506  | .483  | .506  |  .491  |
> | .508  | .500  | .502  | .488  |  .498 |
> | .483  |  .481 | .488  |  .491 |  .503 |


```python
from pairformance import Pairformance

pf_eval = Pairformance(df=eval_df)
pf_eval.eval()

global_results = pf_eval.print_global_results()    
```

will print to stdout:
```buildoutcfg
*** Mean ***
	     Systems       Est. score    (95% bootstrap CI)
	       M0             0.507      (0.504, 0.512) 
	       M1             0.507      (0.496, 0.513) 
	       M2             0.507      (0.497, 0.523) 
	       M3             0.502      (0.497, 0.508) 
	       M4             0.514      (0.509, 0.518) 

*** Median ***
	     Systems       Est. score    (95% bootstrap CI)
	       M0             0.503      (0.499, 0.508) 
	       M1             0.495      (0.489, 0.500) 
	       M2             0.497      (0.494, 0.499) 
	       M3             0.495      (0.492, 0.499) 
	       M4             0.507      (0.501, 0.512) 

*** BT ***
	     Systems       Est. score    (95% bootstrap CI)
	       M0             0.239      (0.191, 0.300) 
	       M1             0.133      (0.109, 0.180) 
	       M2             0.168      (0.121, 0.220) 
	       M3             0.151      (0.126, 0.174) 
	       M4             0.308      (0.259, 0.390) 
```
Furthermore, `global_results` contains the same information in a dictionary.

With BT, it is possible to estimate the probability that one system is better than another

```python
p_m0_m1, pval = pf_eval.prob_a_better_b('M0', 'M1')
print(f"Prob that M0 is better than M1 {p_m0_m1}, pval {pval}")
```

## Advanced Usage
Several capabilities are available to provide better insights in the structure of system performances.

#### Pairwise comparisons
The pairformance tool can analyze each pair of systems independently for all available aggregation mechanism:
```python
system_results = pf_eval.print_pairwise_results()
```

#### Plotting results
```python
from  pairformance import plotting_utils as pf_plot

# plotting global results
pf_plot.plot_global_results(pf_eval, aggregation='BT')

# plotting pairwise results
pf_plot.plot_matrix_results(pf_eval, aggregation='BT')
```

#### Diagnostics

```python
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Diagnostic statistics from the distribution of system scores 
pp.pprint(pf_eval.diagnostic_system_scores(eval_df))

# disagreement measures between aggregation mechanisms
pf_eval.diagnostic_aggregation_agreement()

# visualize the pairing structure for a pair of systems
pf_plot.plot_paired_graphs(eval_df, 'M2', 'M1')
```

More details are available in the docstrings and in the example [notebook](examples/example_usage.ipynb)


## Citation
```
@inproceedings{peyrard-etal-2021-better,
    title = "Better than Average: Paired Evaluation of {NLP} systems",
    author = "Peyrard, Maxime  and
      Zhao, Wei  and
      Eger, Steffen  and
      West, Robert",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.179",
    doi = "10.18653/v1/2021.acl-long.179",
    pages = "2301--2315"
}
```

#### Contact
Maxime Peyrard, maxime.peyrard@epfl.ch