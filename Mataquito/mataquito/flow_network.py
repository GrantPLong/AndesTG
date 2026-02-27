"""Mataquito River drainage topology and subwatershed mass-balance calculations.

The flow network encodes which upstream sampling sites contribute sediment
to each downstream site.  CT-4 is excluded from the hierarchy (problematic
data) and its area is lumped into the CT-10 subwatershed.
"""

# DAG: child -> list of upstream parents
FLOW_NETWORK = {
    "CT-5": ["CT-7"],
    "CT-6": ["CT-1"],
    "CT-10": ["CT-5", "CT-6"],  # CT-4 skipped
    "CT-11": ["CT-10"],
    "CT-8": ["CT-11"],
    "CT-2": ["CT-8"],
    "CT-9": ["CT-2", "CT-3"],
}

HEADWATERS = ["CT-1", "CT-3", "CT-7"]

# Topological sort order (each node appears after all its ancestors)
TRAVERSAL_ORDER = ["CT-5", "CT-6", "CT-10", "CT-11", "CT-8", "CT-2", "CT-9"]


def subwatershed_area(site, A):
    """Compute subwatershed area for *site*.

    A_sub = A[site] - sum(A[parent] for parent in parents)

    Headwater sites return A[site] unchanged.
    """
    if site not in FLOW_NETWORK:
        return A[site]
    return A[site] - sum(A[parent] for parent in FLOW_NETWORK[site])


def subwatershed_erosion_rate(site, E, A):
    """Deterministic subwatershed erosion rate via mass balance.

    E_sub = (E[site]*A[site] - sum(E[p]*A[p])) / A_sub

    Headwater sites return E[site] unchanged.
    """
    if site not in FLOW_NETWORK:
        return E[site]
    A_sub = subwatershed_area(site, A)
    upstream_flux = sum(E[p] * A[p] for p in FLOW_NETWORK[site])
    return (E[site] * A[site] - upstream_flux) / A_sub


def all_subwatershed_erosion_rates(E, A):
    """Compute subwatershed erosion rates for all sites in the network.

    Returns a dict ``{site: E_sub}``.
    """
    result = {}
    for site in HEADWATERS:
        result[site] = E[site]
    for site in TRAVERSAL_ORDER:
        result[site] = subwatershed_erosion_rate(site, E, A)
    return result
