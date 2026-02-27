"""Load and access Mataquito sample metadata from the canonical Excel file."""

import pandas as pd
from pathlib import Path

_DEFAULT_XLSX = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample_metadata"
    / "MataquitoSampleData.xlsx"
)


def load_samples(xlsx_path=None):
    """Load MataquitoSampleData.xlsx Sheet1 into a DataFrame.

    Parameters
    ----------
    xlsx_path : str or Path, optional
        Path to the Excel file.  Defaults to the canonical location
        ``data/sample_metadata/MataquitoSampleData.xlsx``.

    Returns
    -------
    pd.DataFrame
    """
    path = xlsx_path or _DEFAULT_XLSX
    return pd.read_excel(path)


def get_erosion_rates(df):
    """Return ``{Sample_ID: Erosion_rate}`` dict."""
    return df.set_index("Sample_ID")["Erosion_rate"].to_dict()


def get_areas(df):
    """Return ``{Sample_ID: Source_Area}`` dict."""
    return df.set_index("Sample_ID")["Source_Area"].to_dict()


def get_uncertainties(df):
    """Return ``{Sample_ID: Erosion_rate_uncertainty_external}`` dict."""
    return df.set_index("Sample_ID")["Erosion_rate_uncertainty_external"].to_dict()


def get_production_rates(df):
    """Return ``{Sample_ID: Surface_Production_Rate}`` dict."""
    return df.set_index("Sample_ID")["Surface_Production_Rate"].to_dict()
