"""Monte Carlo erosion rate sampling and flux ordering.

Extracted from NB06 (fertility analysis) cells 6 and 8.
"""

import numpy as np


def generate_mc_samples(df, N=100_000, seed=17):
    """Generate *N* random erosion rate samples per site using external uncertainty.

    Parameters
    ----------
    df : pd.DataFrame
        Sample data with columns ``Sample_ID``, ``Erosion_rate``,
        ``Erosion_rate_uncertainty_external``.
    N : int
        Number of Monte Carlo draws per site.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{Sample_ID: np.ndarray of shape (N,)}``
    """
    np.random.seed(seed)
    ER_ext = {}
    for _, row in df.iterrows():
        sid = row["Sample_ID"]
        ER_ext[sid] = np.random.normal(
            row["Erosion_rate"],
            row["Erosion_rate_uncertainty_external"],
            N,
        )
    return ER_ext


def flux_order_samples(ER_ext, A, N=100_000, seed=17):
    """Apply flux ordering constraints through the flow network.

    Flux ordering retains only Monte Carlo draws where the downstream
    sediment flux exceeds the upstream flux, ensuring mass balance
    consistency.  After filtering, arrays are resampled back to *N*
    with replacement for downstream steps.

    Parameters
    ----------
    ER_ext : dict
        ``{Sample_ID: np.ndarray}`` from :func:`generate_mc_samples`.
    A : dict
        ``{Sample_ID: area_km2}`` source areas.
    N : int
        Target array length after resampling.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Named arrays for every flux-ordered stage.  Key names match
        the variables used in NB06 cell 8.
    """
    np.random.seed(seed)
    r = {}

    # -- CT-7 -> CT-5 --
    m_75 = np.where(ER_ext["CT-5"] * A["CT-5"] > ER_ext["CT-7"] * A["CT-7"])[0]
    r["CT5_ord"] = ER_ext["CT-5"][m_75]
    r["CT7_ord"] = ER_ext["CT-7"][m_75]
    r["m_75"] = m_75

    # -- CT-1 -> CT-6 --
    m_16 = np.where(ER_ext["CT-6"] * A["CT-6"] > ER_ext["CT-1"] * A["CT-1"])[0]
    r["CT6_ord"] = ER_ext["CT-6"][m_16]
    r["CT1_ord"] = ER_ext["CT-1"][m_16]
    r["m_16"] = m_16

    # Resample back to N with replacement
    CT5_res = np.random.choice(r["CT5_ord"], size=N, replace=True)
    CT6_res = np.random.choice(r["CT6_ord"], size=N, replace=True)
    r["CT5_res"] = CT5_res
    r["CT6_res"] = CT6_res

    # -- CT-5 + CT-6 -> CT-10 --
    m_10 = np.where(
        ER_ext["CT-10"] * A["CT-10"] >= CT5_res * A["CT-5"] + CT6_res * A["CT-6"]
    )[0]
    r["CT5_10"] = CT5_res[m_10]
    r["CT6_10"] = CT6_res[m_10]
    r["CT10_f"] = ER_ext["CT-10"][m_10]
    r["m_10"] = m_10

    # -- CT-5 + CT-6 -> CT-11 --
    m_11 = np.where(
        ER_ext["CT-11"] * A["CT-11"] >= CT5_res * A["CT-5"] + CT6_res * A["CT-6"]
    )[0]
    r["CT5_11"] = CT5_res[m_11]
    r["CT6_11"] = CT6_res[m_11]
    r["CT11_f"] = ER_ext["CT-11"][m_11]
    r["m_11"] = m_11

    # -- CT-5 + CT-6 -> CT-8 --
    m_8 = np.where(
        ER_ext["CT-8"] * A["CT-8"] >= CT5_res * A["CT-5"] + CT6_res * A["CT-6"]
    )[0]
    r["CT5_8"] = CT5_res[m_8]
    r["CT6_8"] = CT6_res[m_8]
    r["CT8_f"] = ER_ext["CT-8"][m_8]
    r["m_8"] = m_8

    # -- CT-4 -> CT-8 (CT-8 as parent) --
    m_48 = np.where(ER_ext["CT-8"] * A["CT-8"] > ER_ext["CT-4"] * A["CT-4"])[0]
    r["CT4_48"] = ER_ext["CT-4"][m_48]
    r["CT8_48"] = ER_ext["CT-8"][m_48]
    CT8_res = np.random.choice(r["CT8_48"], size=N, replace=True)
    r["CT8_res"] = CT8_res
    r["m_48"] = m_48

    # -- Independent resamples --
    r["CT3_res"] = np.random.choice(ER_ext["CT-3"], size=N, replace=True)
    r["CT4_res"] = np.random.choice(ER_ext["CT-4"], size=N, replace=True)
    r["CT10_res"] = np.random.choice(r["CT10_f"], size=N, replace=True)
    r["CT11_res"] = np.random.choice(r["CT11_f"], size=N, replace=True)

    return r
