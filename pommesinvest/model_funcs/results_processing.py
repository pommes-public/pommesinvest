import pandas as pd


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
