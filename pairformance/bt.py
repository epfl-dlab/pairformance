import itertools
import numpy as np
import pandas as pd


def BT(df: pd.DataFrame, epsilon: float = 1e-9) -> np.array:
    """
    Computes the iterative BT algorithm to estimate relative strength of systems.
    It maximizes the likelihood of the observed data to predict which system is better than other for all pairs of systems.

    :param df: evaluation dataframe with system scores (one column per system, one row per test instance)
    :param epsilon: when the iterative algorithm of BT does not vary by more than epsilon_bt, we stop.
    :return: list of scores estimated by BT, one per system.
    """
    n_competitors = df.shape[1]
    competitors = df.columns

    # If there is nothing to compute, return None
    if n_competitors < 2 or df.shape[0] == 0:
        return None

    win_matrix = np.zeros((n_competitors, n_competitors))

    for pair in itertools.combinations(range(n_competitors), 2):
        idx_a, idx_b = pair
        competitor_a = competitors[idx_a]
        competitor_b = competitors[idx_b]

        win_ab = np.sum([int(score_a > score_b)
                         for score_a, score_b in zip(df[competitor_a], df[competitor_b])])
        win_ba = np.sum([int(score_b > score_a)
                         for score_a, score_b in zip(df[competitor_a], df[competitor_b])])

        # win_ba = df.shape[0] - win_ab

        win_matrix[idx_a][idx_b] = win_ab
        win_matrix[idx_b][idx_a] = win_ba

    W = np.sum(win_matrix, axis=1)
    p = [0.5] * n_competitors

    # if only ties, return equal probabilities to all systems
    if np.sum(W) == 0:
        return p

    while True:
        new_p = np.array([0.5] * n_competitors)
        for i in range(n_competitors):
            summing_term = 0
            for j in range(n_competitors):
                if i == j:
                    continue

                summing_term += (win_matrix[i][j] + win_matrix[j][i]) / (p[i] + p[j])

            new_p[i] = W[i] / summing_term

        new_p /= np.sum(new_p)
        diff = np.sum([(x - y) ** 2 for x, y in zip(p, new_p)])

        # stopping criterion
        if diff < epsilon:
            return new_p
        p = new_p
