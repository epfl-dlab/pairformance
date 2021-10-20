import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from .core import Pairformance


def plot_paired_graphs(data: pd.DataFrame,
                       system_a: str,
                       system_b: str,
                       save_path: str = None,
                       shade: bool = True,
                       dots: bool = False,
                       extended_lines: bool = True,
                       legend: bool = True):

    x_a, x_b = data[system_a], data[system_b]
    decision = (np.median(x_a - x_b) > 0)

    g = sns.JointGrid()
    sns.kdeplot(x=x_b, y=x_a, kind="kde", alpha=0.5, ax=g.ax_joint)

    sns.distplot(x_a, color='tab:blue', ax=g.ax_marg_y, vertical=True,
                 kde_kws={"alpha": 0.2, "bw_adjust": 1.5, "fill": True},
                 hist_kws={"alpha": 0.})
    sns.distplot(x_b, color='tab:green', ax=g.ax_marg_x,
                 kde_kws={"alpha": 0.2, "bw_adjust": 1.5, "fill": True},
                 hist_kws={"alpha": 0.})

    x_range = np.max(x_b) - np.min(x_b)
    y_range = np.max(x_a) - np.min(x_a)

    x_lim_low, x_lim_high = min(x_b) - 0.2 * x_range, max(x_b) + 0.2 * x_range
    y_lim_low, y_lim_high = min(x_a) - 0.2 * y_range, max(x_a) + 0.2 * y_range
    g.ax_joint.set_xlim(x_lim_low, x_lim_high)
    g.ax_joint.set_ylim(y_lim_low, y_lim_high)

    ## Graphical criterion demarcation
    line_x = np.linspace(x_lim_low, y_lim_high, num=100)
    g.ax_joint.plot(line_x, line_x, linestyle='dotted', linewidth=3, alpha=0.7, color='black')

    ## Plot dots
    if dots:
        y = data.iloc[:, 0]
        x = data.iloc[:, 1]
        g.ax_joint.scatter(x, y, marker='x', color='black', s=100)

    ## Shade the upper-left triangle
    if shade:
        color_shade = 'tab:gray'
        if decision:
            g.ax_joint.fill_between(line_x, line_x, [y_lim_high] * len(line_x), color=color_shade, alpha=0.2)
        else:
            g.ax_joint.fill_between(line_x, [y_lim_low] * len(line_x), line_x, color=color_shade, alpha=0.2)

    ## Draw Means
    mean_linestyle = '-'
    if extended_lines:
        mean_best = np.mean(x_a)
        mean_worst = np.mean(x_b)

        g.ax_joint.axhline(mean_best, linestyle=mean_linestyle, linewidth=2, color='tab:blue')
        g.ax_joint.axvline(mean_worst, linestyle=mean_linestyle, linewidth=2, color='tab:green')

        g.ax_marg_y.axhline(mean_best, linestyle=mean_linestyle, linewidth=2, color='tab:blue')
        g.ax_marg_x.axvline(mean_worst, linestyle=mean_linestyle, linewidth=2, color='tab:green')

    ## Draw Medians
    median_linestyle = 'dashed'
    if extended_lines:
        median_best = np.median(x_a)
        median_worst = np.median(x_b)

        g.ax_joint.axhline(median_best, linestyle=median_linestyle, linewidth=2, color='tab:blue')
        g.ax_joint.axvline(median_worst, linestyle=median_linestyle, linewidth=2, color='tab:green')

        g.ax_marg_y.axhline(median_best, linestyle=median_linestyle, linewidth=2, color='tab:blue')
        g.ax_marg_x.axvline(median_worst, linestyle=median_linestyle, linewidth=2, color='tab:green')

    ## Legend
    g.ax_joint.set_xlabel(f'Scores of {system_b}', color='tab:green')
    g.ax_joint.set_ylabel(f'Scores of {system_a}', color='tab:blue')
    if legend:
        legend_elem = [
            Line2D([0], [0], linestyle=median_linestyle, linewidth=2, c='tab:gray', label='Median'),
            Line2D([0], [0], linestyle=mean_linestyle, linewidth=2, c='tab:gray', label='Mean'),
            Line2D([0], [0], linestyle='dotted', linewidth=3, c="black", label='$X_A=X_B$'),
        ]

        g.fig.legend(handles=legend_elem, ncol=1, loc='upper center', frameon=False, bbox_to_anchor=(0.26, .82))

    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_global_results(pf: Pairformance,
                        aggregation: str = 'BT',
                        figsize: (int, int) = (10, 5),
                        symmetric_percentiles: int = 5,
                        save_path: str = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data = pf.global_results[aggregation]
    data = data.reindex(data.mean().sort_values().index, axis=1)

    parts = ax.violinplot(data, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('tab:blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    labels = data.columns.to_list()

    inds = np.arange(1, len(labels) + 1)

    percentile1, medians, percentile2 = \
        np.percentile(data.to_numpy(),
                      [symmetric_percentiles, 50, 100 - symmetric_percentiles],
                      axis=0)

    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, percentile1, percentile2, color='k', linestyle='-', lw=5)

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Systems')
    ax.set_ylabel(f'{aggregation} score distribution ' + \
                  f'({symmetric_percentiles}, {100 - symmetric_percentiles}) percentiles')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_matrix_results(pf: Pairformance,
                        aggregation: str,
                        fontsize: int = 13,
                        figsize: [int, int] = (10, 10),
                        save_path: str = None):

    assert aggregation in pf.config['aggregations'], \
        f"The aggregation mechanism {aggregation} is not available in " \
        f"pf.config: {pf.config['aggregation']}"
    assert pf.data_matrices is not None, \
        "The pairformance object does not have computed computed results. " \
        "Please run pf.eval() first with 'compute_pairwise' set to True."

    data = pf.data_matrices[aggregation]
    pval_matrix = pf.pval_matrices[aggregation]
    system_to_idx = pf.orders_of_systems[aggregation]

    fig, ax = plt.subplots(figsize=figsize)

    _plot_matrix(data, pval_matrix, system_to_idx, ax, fontsize)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def _plot_matrix(data: np.array,
                 pval_matrix: np.array,
                 system_to_idx: dict,
                 ax: object,
                 fontsize: int,
                 textcolors: list = ("black", "white"),
                 valfmt: str = "{x:.2f}"):
    labels = system_to_idx.keys()
    row_labels = labels
    col_labels = labels

    im = ax.imshow(data, cmap='RdBu')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    plt.setp(ax.get_yticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    default_value = im.norm(data.max()) / 2.
    threshold = (im.norm(data.max()) - default_value) / 3.

    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color_switch = int(abs(im.norm(data[i, j]) - default_value) > threshold)
            # kw.update(color=textcolors[int(abs(im.norm(data[i, j])) - default_value < threshold)])
            kw.update(color=textcolors[int(color_switch)])
            kw.update(fontsize=fontsize)
            if i != j:
                text_to_display = valfmt(data[i, j], None)
            else:
                text_to_display = '-'

            fontweight = 'normal'
            if pval_matrix[i][j] < 0.05:
                text_to_display += '*'
                fontweight = 'bold'
            text = im.axes.text(j, i, text_to_display, fontweight=fontweight, **kw)
            texts.append(text)
