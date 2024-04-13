# -*- coding: utf-8 -*-
"""
General description
-------------------
These are processing routines used in the power market model POMMES.

Installation requirements
-------------------------
Python version >= 3.8

@author: Johannes Kochems
"""
import pandas as pd
from oemof.solph import views


def process_demand_response_results(results):
    """Process the given demand response results to a harmonized format

    Parameters
    ----------
    results : pd.DataFrame
        The raw demand response results differing based on modelling
        approach chosen

    Returns
    -------
    processed_results : pd.DataFrame
        Demand response results in a proper and harmonized format
    """
    processed_results = pd.DataFrame(dtype="float64")
    upshift_columns = [
        col
        for col in results.columns
        if (
            "dsm_up" in col[1]
            and "balance" not in col[1]
            and "level" not in col[1]
        )
        or ("balance_dsm_do" in col)
    ]
    downshift_columns = [
        col
        for col in results.columns
        if (
            "dsm_do_shift" in col[1]
            and "balance" not in col[1]
            and "level" not in col[1]
        )
        or ("balance_dsm_up" in col)
    ]
    shedding_columns = [
        col for col in results.columns if "dsm_do_shed" in col[1]
    ]

    unique_keys = set([col[0][0] for col in results.columns])
    unique_keys.remove("DE_bus_el")
    cluster_identifier = unique_keys.pop()

    # Only upshift, downshift, shedding and storage level are required
    # Inflow is already considered in dispatch results
    processed_results[(cluster_identifier, "dsm_up")] = results[
        upshift_columns
    ].sum(axis=1)
    processed_results[(cluster_identifier, "dsm_do_shift")] = results[
        downshift_columns
    ].sum(axis=1)
    processed_results[(cluster_identifier, "dsm_do_shed")] = results[
        shedding_columns
    ].sum(axis=1)
    processed_results[(cluster_identifier, "dsm_storage_level")] = (
        processed_results[(cluster_identifier, "dsm_do_shift")]
        - processed_results[(cluster_identifier, "dsm_up")]
    ).cumsum()

    return processed_results


def filter_storage_results(results):
    """Filter the given storage results to isolate storage capacities

    Parameters
    ----------
    results : pd.DataFrame
        The raw storage results

    Returns
    -------
    filtered_results : pd.DataFrame
        Storage capacity results
    """
    capacity_idx = [idx for idx in results.index if idx[0][1] == "None"]
    filtered_results = results.loc[capacity_idx]

    return filtered_results


def filter_european_country_results(im, results):
    """Filter values for European countries from dispatch results

    Exclude dispatch for linking converters. These are already accounted
    for by considering the German electricity bus. Since transmission losses
    are neglected in favor of small transportation costs, flow values are equal.

    Parameters
    ----------
    im : :class:`InvestmentModel`
        The investment model that is considered

    results : pd.DataFrame
        The raw storage results

    Returns
    -------
    filtered_results : pd.DataFrame
        dispatch results of European countries excluding links to Germany
    """
    buses_el_views = [
        country + "_bus_el" for country in im.countries if country != "DE"
    ]
    filtered_results = pd.concat(
        [
            filter_dispatch(
                views.node(results, bus_el)["sequences"],
                exclude=["link_DE", "DE_link"],
            )
            for bus_el in buses_el_views
        ],
        axis=1,
    )
    return filtered_results


def filter_dispatch(df, exclude):
    """Filter data frame for columns not containing substrings from exclude

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be filtered

    exclude : list of str
        List of strings to exclude if they occur in DataFrames columns

    Returns
    -------
    df : pd.DataFrame
        filtered DataFrame excluding
    """
    cols_to_drop = [
        col
        for col in df.columns
        for substring in exclude
        if substring in col[0][0] or substring in col[0][1]
    ]

    return df.drop(columns=cols_to_drop)


def process_ev_bus_results(results):
    """Filter the given electric vehicle bus results

    Parameters
    ----------
    results : pd.DataFrame
        The raw ev bus results

    Returns
    -------
    filtered_results : pd.DataFrame
        ev bus results excluding duals
    """
    ev_cols = [col for col in results.columns if col[0][1] != "None"]
    filtered_results = results[ev_cols]

    return filtered_results
