"""Quartz and zircon mineral fertility calculations.

Extracted from NB06 (fertility analysis) cell 14.
"""

import csv
import numpy as np

THRESHOLD = 0.01  # Filter mixing coefficients below 1% or above 99%


def load_wct(csv_path):
    """Load two-column detritalPy-mix bootstrap coefficient CSV with filtering.

    Filters rows where either coefficient is outside (THRESHOLD, 1-THRESHOLD)
    to avoid near-zero denominator instability in ratio calculations.

    Parameters
    ----------
    csv_path : str or path-like

    Returns
    -------
    wa : np.ndarray
    wb : np.ndarray
    n_total : int
        Total rows in the CSV (excluding header).
    n_kept : int
        Rows passing the filter.
    """
    wa, wb = [], []
    n = 0
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            n += 1
            a, b = float(row[0]), float(row[1])
            if a > THRESHOLD and a < (1 - THRESHOLD) and b > THRESHOLD and b < (1 - THRESHOLD):
                wa.append(a)
                wb.append(b)
    return np.array(wa), np.array(wb), n, len(wa)


def quartz_fertility(E_a, E_b, E_c, A_a, A_b, A_c, P_a, P_b, P_c):
    """Compute quartz fertility ratio Qa/Qb from the sediment mixing equation.

    Equation: Qb/Qa = (Pc*Ea*Aa - Pa*Ec*Aa) / (Pb*Ec*Ab - Pc*Eb*Ab)
    Returns Qa/Qb (the inverse), keeping only positive values.

    Parameters
    ----------
    E_a, E_b, E_c : array-like
        Erosion rate sample arrays (m/Myr) for samples A, B, and child C.
    A_a, A_b, A_c : float
        Source areas (km^2).
    P_a, P_b, P_c : float
        Spallation production rates (atoms/g/yr).

    Returns
    -------
    Qa_Qb : np.ndarray
        Positive Qa/Qb values only.
    mask : np.ndarray of bool
        Boolean mask indicating which samples were retained.
    """
    num = P_c * E_a * A_a - P_a * E_c * A_a
    den = P_b * E_c * A_b - P_c * E_b * A_b
    Qb_Qa = num / den
    mask = Qb_Qa > 0
    return (1 / Qb_Qa[mask]), mask


def zircon_fertility(wa, wb, ER_A, ER_B, A_A, A_B, num_samples=10000, seed=17):
    """Compute zircon fertility ratio ZA/ZB using Equation 11.

    ZA/ZB = (wct_a / wct_b) * (ER_B * A_B) / (ER_A * A_A)

    Parameters
    ----------
    wa, wb : np.ndarray
        Coupled bootstrap mixing coefficient arrays from detritalPy-mix.
    ER_A, ER_B : np.ndarray
        Flux-ordered erosion rate arrays (m/Myr).
    A_A, A_B : float
        Source areas (km^2).
    num_samples : int
        Number of Monte Carlo draws.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{'ZA_ZB': {...}, 'ZB_ZA': {...}}`` each containing
        ``samples``, ``median``, ``p25``, ``p75``, and log10 equivalents.
    """
    np.random.seed(seed)

    idx = np.random.choice(len(wa), size=num_samples, replace=True)
    ratio = wa[idx] / wb[idx]

    ER_A_s = ER_A[:num_samples]
    ER_B_s = ER_B[:num_samples]

    ZA_ZB = ratio * (ER_B_s * A_B) / (ER_A_s * A_A)

    def stats(x):
        return {
            "samples": x,
            "median": np.median(x),
            "p25": np.percentile(x, 25),
            "p75": np.percentile(x, 75),
            "log10_samples": np.log10(x),
            "log10_median": np.median(np.log10(x)),
            "log10_p25": np.percentile(np.log10(x), 25),
            "log10_p75": np.percentile(np.log10(x), 75),
        }

    return {"ZA_ZB": stats(ZA_ZB), "ZB_ZA": stats(1 / ZA_ZB)}
