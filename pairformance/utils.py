import numpy as np


def tau_to_percent(tau: float) -> float:
    if tau < -1 or tau > 1:
        print(f"Received a tau outside of (-1, 1) range: {tau}")
        tau = np.clip(tau, a_min=-1, a_max=1)
    return (tau + 1.) / 2.


def same_top_n(lst_a: list, lst_b: list, n: int = 3) -> bool:
    if n > min(len(lst_a), len(lst_b)):
        return None
    top_a = np.argsort(-np.array(lst_a))[:n]
    top_b = np.argsort(-np.array(lst_b))[:n]
    return set(top_a) == set(top_b)


def percent_outliers(data: np.array) -> float:
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return ((data < lower_bound).sum() + (data > upper_bound).sum()) / len(data)