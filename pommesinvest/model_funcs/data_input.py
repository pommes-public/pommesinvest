# -*- coding: utf-8 -*-
"""
General description
-------------------
This file contains all function definitions for reading in input data
used for the investment variant of POMMES.

@author: Johannes Kochems (*), Johannes Giehl (*), Yannick Werner,
Benjamin Grosse

Contributors:
Julien Faist, Hannes Kachel, Sophie Westphal, Flora von Mikulicz-Radecki,
Carla Spiller, Fabian Büllesbach, Timona Ghosh, Paul Verwiebe,
Leticia Encinas Rosa, Joachim Müller-Kirchenbauer

(*) Corresponding authors
"""

import pandas as pd
import logging

from pommesinvest.model_funcs.subroutines import (
    load_input_data,
    create_buses,
    create_commodity_sources,
    create_shortage_sources,
    create_renewables,
    create_demand_response_units,
    create_demand,
    create_excess_sinks,
    create_existing_transformers,
    create_new_built_transformers,
    create_existing_storages,
    create_existing_storages_rolling_horizon,
    create_new_built_storages,
    create_new_built_storages_rolling_horizon,
    existing_transformers_exo_decom,
    new_transformers_exo,
    storages_exo,
    renewables_exo,
)
from pommesinvest.model_funcs import helpers


def parse_input_data(im):
    r"""Read in csv files as DataFrames and store them in a dict

    Parameters
    ----------
    im : :class:`InvestmentModel`
        The investment model that is considered

    Returns
    -------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys
    """
    buses = {
        "buses": "buses",
    }

    components = {
        "sinks_excess": "sinks_excess",
        "sinks_demand_el": "sinks_demand_el",
        "sources_shortage": "sources_shortage",
        "sources_commodity": "sources_commodity",
        "sources_renewables": "sources_renewables",
        "storages_el": "storages_el",
        "existing_transformers": "transformers",
        "new_built_transformers": "new_built_transformers",
        "exo_new_built_transformers": "exo_new_built_transformers",
    }

    time_series = {
        "sinks_demand_el_ts": "sinks_demand_el_ts",
        "sources_renewables_ts": "sources_renewables_ts",
        "transformers_minload_ts": "transformers_minload_ts",
        "transformers_availability_ts": "transformers_availability_ts",
        "costs_fuel": f"costs_fuel_{im.fuel_cost_pathway}_nominal",
        "costs_fuel_ts": "costs_fuel_ts",
        "costs_emissions": (f"costs_emissions_{im.emissions_cost_pathway}_nominal"),
        "costs_emissions_ts": "costs_emissions_ts",
        "costs_operation": "costs_operation_nominal",
        "costs_operation_storages": "costs_operation_storages_nominal",
        "costs_investment": (f"costs_investment_{im.investment_cost_pathway}_nominal"),
        "costs_storages_investment": (
            f"costs_storages_investment_{im.investment_cost_pathway}_nominal"
        ),
        "wacc": "wacc",
        "min_loads_dh": "min_loads_dh",
        "min_loads_ipp": "min_loads_ipp",
        "min_max_ts": "min_max_ts",
    }

    other_files = {
        "emission_limits": "emission_limits",
    }

    # Optionally use aggregated transformer data instead
    if im.aggregate_input:
        components["transformers"] = "transformers_clustered"

    # Add demand response units
    if im.activate_demand_response:
        components[
            "sinks_dr_el"
        ] = f"sinks_demand_response_el_{im.demand_response_scenario}"

        components[
            "sinks_dr_el_ts"
        ] = f"sinks_demand_response_el_ts_{im.demand_response_scenario}"

        components["sinks_dr_el_ava_pos_ts"] = (
            "sinks_demand_response_el_ava_pos_ts_" + im.demand_response_scenario
        )
        components["sinks_dr_el_ava_neg_ts"] = (
            "sinks_demand_response_el_ava_neg_ts_" + im.demand_response_scenario
        )

    # Combine all files
    input_files = {**buses, **components, **time_series}
    input_files = {**input_files, **other_files}

    return {
        key: load_input_data(filename=name, im=im) for key, name in input_files.items()
    }


def resample_input_data(input_data, im):
    """Adjust input data to investment model frequency

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered
    """
    transformer_data = ["existing_transformers", "new_built_transformers"]
    storage_data = ["storages_el", "new_built_storages_el"]
    annual_ts = [
        "transformers_minload_ts",
        "costs_fuel",
        "costs_emissions",
        "costs_operation",
        "costs_operation_storages",
    ]
    hourly_ts = [
        "transformers_availability_ts",
        "sinks_demand_el_ts",
        "sources_renewables_ts",
        "costs_fuel_ts",
        "costs_emissions_ts",
        "min_loads_dh",
        "min_loads_ipp",
    ]

    for key in input_data.keys():
        if key in transformer_data:
            input_data[key].loc[
                :, ["grad_pos", "grad_neg", "max_load_factor", "min_load_factor"]
            ] = (
                input_data[key]
                .loc[:, ["grad_pos", "grad_neg", "max_load_factor", "min_load_factor"]]
                .mul(im.multiplicator)
            )
        elif key in storage_data:
            input_data[key] = input_data[key].where(pd.notnull(input_data[key]), None)
            input_data[key].loc[
                :, ["max_storage_level", "min_storage_level", "nominal_storable_energy"]
            ] = (
                input_data[key]
                .loc[
                    :,
                    [
                        "max_storage_level",
                        "min_storage_level",
                        "nominal_storable_energy",
                    ],
                ]
                .mul(im.multiplicator)
            )
        elif key in annual_ts:
            input_data[key] = (
                helpers.resample_timeseries(
                    input_data[key], freq=im.freq, interpolation_rule="linear"
                )[:-1],
            )
        elif key in hourly_ts:
            input_data[key] = helpers.resample_timeseries(
                input_data[key], freq=im.freq, aggregation_rule="sum"
            )


def add_components(input_data, im):
    r"""Add the oemof components to a dictionary of nodes

    Note: Storages and new-built transformers are not included here.
    They have to be defined separately since the approaches differ
    between rolling horizon and simple model.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem
    """
    node_dict = {}
    node_dict = create_buses(input_data, node_dict)
    node_dict = create_commodity_sources(input_data, im, node_dict)
    node_dict = create_shortage_sources(input_data, node_dict)
    node_dict = create_renewables(input_data, im, node_dict)

    if im.activate_demand_response:
        node_dict, dr_overall_load_ts_df = create_demand_response_units(
            input_data, im, node_dict
        )

        node_dict = create_demand(input_data, im, node_dict, dr_overall_load_ts_df)
    else:
        node_dict = create_demand(input_data, im, node_dict)

    node_dict = create_excess_sinks(input_data, node_dict)
    node_dict = create_existing_transformers(input_data, im, node_dict)

    return node_dict


def add_limits(
    input_data,
    im,
):
    """Add further limits to the optimization model (emissions limit for now)

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    Returns
    -------
    emissions_limit : :obj:`float`
        The emissions limit to be used (converted)
    """
    return helpers.convert_annual_limit(
        input_data["emission_limits"][im.emission_pathway], im.starttime, im.endtime
    )


def nodes_from_csv(im):
    r"""Build oemof.solph components from input data

    Parameters
    ----------
    im : :class:`InvestmenthModel`
        The investment model that is considered

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    emissions_limit : int or None
        The overall emissions limit
    """
    input_data = parse_input_data(im)
    resample_input_data(input_data, im)

    node_dict = add_components(input_data, im)

    node_dict = create_new_built_transformers(input_data, im, node_dict)
    node_dict = create_existing_storages(input_data, im, node_dict)
    node_dict = create_new_built_storages(input_data, im, node_dict)

    emissions_limit = None
    if im.activate_emissions_limit:
        emissions_limit = add_limits(
            input_data,
            im,
        )

    return node_dict, emissions_limit


def nodes_from_csv_rolling_horizon(im, iteration_results):
    r"""Read in csv files and build components for a rolling horizon run

    Parameters
    ----------
    im : :class:`InvestmentModel`
        The investment model that is considered

    iteration_results : dict
        A dictionary holding the results of the previous rolling horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    emissions_limit : int or None
        The overall emissions limit

    storage_labels : :obj:`list` of :class:`str`
        A list of the labels of all storage elements included in the model
        used for assessing these and assigning initial states
    """
    frequency_used = {
        "60min": (
            getattr(im, "time_slice_length_with_overlap"),
            "h",
        ),
        "15min": (
            getattr(im, "time_slice_length_with_overlap") * 15,
            "min",
        ),
    }[im.freq]

    # Update start time and end time of the model for retrieving the right data
    im.start_time = getattr(im, "time_series_start").strftime("%Y-%m-%d %H:%M:%S")
    im.end_time = (
        getattr(im, "time_series_start")
        + pd.to_timedelta(frequency_used[0], frequency_used[1])
    ).strftime("%Y-%m-%d %H:%M:%S")

    input_data = parse_input_data(im)
    resample_input_data(input_data, im)

    node_dict = add_components(input_data, im)

    # create storages and new-built transformers (rolling horizon)
    node_dict, new_built_transformer_labels = create_existing_storages_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )

    node_dict, existing_storage_labels = create_existing_storages_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )
    node_dict, new_built_storage_labels = create_new_built_storages_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )

    storage_and_transformer_labels = (
        new_built_transformer_labels
        + new_built_transformer_labels
        + new_built_storage_labels
    )

    emissions_limit = None
    if im.activate_emissions_limit:
        emissions_limit = add_limits(input_data, im)

    return (node_dict, emissions_limit, storage_and_transformer_labels)


def exo_com_costs(
    startyear,
    endyear,
    existing_transformers_decom_df,
    new_transformers_de_com_df,
    investment_costs_df,
    WACC_df,
    new_built_storages_df,
    storage_turbine_investment_costs_df,
    storage_pump_investment_costs_df,
    storage_investment_costs_df,
    renewables_com_df,
    IR=0.02,
    discount=False,
):
    """Function takes the dataframes from the functions total_exo_decommissioning,
    transformers_exo_commissioning, storages_exo_commissioning and
    renewables_exo_commissioning and returns them

    Parameters
    ----------
    startyear : :obj:`int`
        Starting year of the overall optimization run

    endyear : :obj:`int`
        End year of the overall optimization run

    renewables_com_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the renewables data

    storage_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage capacity investment costs data

    storage_pump_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage infeed investment costs data

    storage_turbine_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage outfeed investment costs data

    new_built_storages_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the new built storage units data

    WACC_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the WACC data

    investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the transformers investment costs data

    new_transformers_de_com_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the new built transformers (exogeneous)
        commissioning and decommissioning data

    existing_transformers_decom_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the existing transformers (exogeneous)
        decommissioning data

    Returns
    -------
    total_exo_com_costs_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning costs

    exo_commissioned_capacity_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning capacity and year

    exo_decommissioned_capacity_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous decommissioning capacity and year

    """

    existing_transformers_decom_capacity_df = existing_transformers_exo_decom(
        existing_transformers_decom_df, startyear, endyear
    )

    (
        new_transformers_exo_com_costs_df,
        new_transformers_exo_com_capacity_df,
        new_transformers_exo_decom_capacity_df,
    ) = new_transformers_exo(
        new_transformers_de_com_df,
        investment_costs_df,
        WACC_df,
        startyear,
        endyear,
        IR=IR,
        discount=discount,
    )

    logging.info(
        "Exogenous transformers costs: {:,.0f}".format(
            new_transformers_exo_com_costs_df.sum().sum()
        )
    )

    (
        storages_exo_com_costs_df,
        storages_exo_com_capacity_df,
        storages_exo_decom_capacity_df,
    ) = storages_exo(
        new_built_storages_df,
        storage_turbine_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_investment_costs_df,
        WACC_df,
        startyear,
        endyear,
        IR=IR,
        discount=discount,
    )

    logging.info(
        "Exogenous storages costs: {:,.0f}".format(
            storages_exo_com_costs_df.sum().sum()
        )
    )

    (
        renewables_exo_com_costs_df,
        renewables_exo_com_capacity_df,
        renewables_exo_decom_capacity_df,
    ) = renewables_exo(
        renewables_com_df,
        investment_costs_df,
        WACC_df,
        startyear,
        endyear,
        IR=IR,
        discount=discount,
    )

    logging.info(
        "Exogenous renewables costs: {:,.0f}".format(
            renewables_exo_com_costs_df.sum().sum()
        )
    )

    total_exo_com_costs_df = pd.concat(
        [
            new_transformers_exo_com_costs_df,
            storages_exo_com_costs_df,
            renewables_exo_com_costs_df,
        ],
        axis=0,
        sort=False,
    )

    total_exo_com_capacity_df = pd.concat(
        [
            new_transformers_exo_com_capacity_df,
            storages_exo_com_capacity_df,
            renewables_exo_com_capacity_df,
        ],
        axis=0,
        sort=True,
    )

    total_exo_decom_capacity_df = pd.concat(
        [
            existing_transformers_decom_capacity_df,
            new_transformers_exo_decom_capacity_df,
            storages_exo_decom_capacity_df,
            renewables_exo_decom_capacity_df,
        ],
        axis=0,
        sort=True,
    )

    return (
        total_exo_com_costs_df,
        total_exo_com_capacity_df,
        total_exo_decom_capacity_df,
    )


def exo_com_costs_RH(
    startyear,
    endyear,
    counter,
    years_per_timeslice,
    total_exo_com_costs_df_RH,
    total_exo_com_capacity_df_RH,
    total_exo_decom_capacity_df_RH,
    existing_transformers_decom_df,
    new_transformers_de_com_df,
    investment_costs_df,
    WACC_df,
    new_built_storages_df,
    storage_turbine_investment_costs_df,
    storage_pump_investment_costs_df,
    storage_investment_costs_df,
    renewables_com_df,
    IR=0.02,
    discount=False,
):
    """Function takes the dataframes from the functions total_exo_decommissioning,
    transformers_exo_commissioning, storages_exo_commissioning and
    renewables_exo_commissioning and returns them

    Parameters
    ----------
    startyear : :obj:`int`
        Starting year of the overall optimization run

    endyear : :obj:`int`
        End year of the overall optimization run

    counter : :obj:`int`
        An integer counter variable counting the number of the rolling horizon run

    years_per_timeslice : :obj:`int`
        Useful length of optimization timeframe (t_intervall)

    IR : :obj:`pandas.DataFrame`
        A pd.DataFrame carrying the WACC information by technology / energy carrier

    total_exo_com_costs_df_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning costs

    total_exo_com_capacity_df_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning capacity and year

    total_exo_decom_capacity_df_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous decommissioning capacity and year

    discount : :obj:`boolean`
        If True, nominal values will be dicounted
        If False, real values have to be used as model inputs (default)

    renewables_com_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the renewables data

    storage_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage capacity investment costs data

    storage_pump_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage infeed investment costs data

    storage_turbine_investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage outfeed investment costs data

    new_built_storages_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the new built storage units data

    WACC_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the WACC data

    investment_costs_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the transformers investment costs data

    new_transformers_de_com_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the new built transformers (exogeneous)
        commissioning and decommissioning data

    existing_transformers_decom_df: :obj:`pandas.DataFrame`
        A pd.DataFrame containing the existing transformers (exogeneous)
        decommissioning data
    Returns
    -------
    exo_com_cost_df_total_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning costs including the
        data from the actual myopic run

    exo_commissioned_capacity_df_total_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous commissioning capacity and year
        including the data from the actual myopic run

    exo_decommissioned_capacity_df_total_RH : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the exogenous decommissioning capacity and year
        including the data from the actual myopic run
    """

    RH_startyear = startyear + (counter * years_per_timeslice)
    if (startyear + ((counter + 1) * years_per_timeslice) - 1) > endyear:
        RH_endyear = endyear
    else:
        RH_endyear = startyear + (((counter + 1) * years_per_timeslice) - 1)

    existing_transformers_decom_capacity_df_RH = existing_transformers_exo_decom(
        existing_transformers_decom_df, startyear=RH_startyear, endyear=RH_endyear
    )

    (
        new_transformers_exo_com_costs_df_RH,
        new_transformers_exo_com_capacity_df_RH,
        new_transformers_exo_decom_capacity_df_RH,
    ) = new_transformers_exo(
        new_transformers_de_com_df,
        investment_costs_df,
        WACC_df,
        startyear=RH_startyear,
        endyear=RH_endyear,
        IR=IR,
        discount=discount,
    )

    # Discount annuities to overall_startyear
    if discount:
        new_transformers_exo_com_costs_df_RH = discount_values(
            new_transformers_exo_com_costs_df_RH, IR, RH_startyear, startyear
        )

    logging.info(
        "Exogenous transformers costs for run {:d}: {:,.0f}".format(
            counter, new_transformers_exo_com_costs_df_RH.sum().sum()
        )
    )

    (
        storages_exo_com_costs_df_RH,
        storages_exo_com_capacity_df_RH,
        storages_exo_decom_capacity_df_RH,
    ) = storages_exo(
        new_built_storages_df,
        storage_turbine_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_investment_costs_df,
        WACC_df,
        startyear=RH_startyear,
        endyear=RH_endyear,
        IR=IR,
        discount=discount,
    )

    # Discount annuities to overall_startyear
    if discount:
        storages_exo_com_costs_df_RH = discount_values(
            storages_exo_com_costs_df_RH, IR, RH_startyear, startyear
        )

    logging.info(
        "Exogenous storages costs for run {:d}: {:,.0f}".format(
            counter, storages_exo_com_costs_df_RH.sum().sum()
        )
    )

    (
        renewables_exo_com_costs_df_total_RH,
        renewables_exo_com_capacity_renewables_df_RH,
        renewables_exo_decom_capacity_df_RH,
    ) = renewables_exo(
        renewables_com_df,
        investment_costs_df,
        WACC_df,
        startyear=RH_startyear,
        endyear=RH_endyear,
        IR=IR,
        discount=discount,
    )

    # Discount annuities to overall_startyear
    if discount:
        renewables_exo_com_costs_df_total_RH = discount_values(
            renewables_exo_com_costs_df_total_RH, IR, RH_startyear, startyear
        )

    logging.info(
        "Exogenous renewables costs for run {:d}: {:,.0f}".format(
            counter, renewables_exo_com_costs_df_total_RH.sum().sum()
        )
    )

    total_exo_com_costs_df_RH_iteration = pd.concat(
        [
            new_transformers_exo_com_costs_df_RH,
            storages_exo_com_costs_df_RH,
            renewables_exo_com_costs_df_total_RH,
        ],
        axis=0,
        sort=True,
    )

    total_exo_com_capacity_df_RH_iteration = pd.concat(
        [
            new_transformers_exo_com_capacity_df_RH,
            storages_exo_com_capacity_df_RH,
            renewables_exo_com_capacity_renewables_df_RH,
        ],
        axis=0,
        sort=True,
    )

    total_exo_decom_capacity_df_RH_iteration = pd.concat(
        [
            existing_transformers_decom_capacity_df_RH,
            new_transformers_exo_decom_capacity_df_RH,
            storages_exo_decom_capacity_df_RH,
            renewables_exo_decom_capacity_df_RH,
        ],
        axis=0,
        sort=True,
    )

    # Combine the results from previous iterations with the one of the current iteration
    total_exo_com_costs_df_RH = pd.concat(
        [total_exo_com_costs_df_RH, total_exo_com_costs_df_RH_iteration],
        axis=1,
        sort=True,
    )
    total_exo_com_capacity_df_RH = pd.concat(
        [total_exo_com_capacity_df_RH, total_exo_com_capacity_df_RH_iteration],
        axis=1,
        sort=True,
    )
    total_exo_decom_capacity_df_RH = pd.concat(
        [total_exo_decom_capacity_df_RH, total_exo_decom_capacity_df_RH_iteration],
        axis=1,
        sort=True,
    )

    return (
        total_exo_com_costs_df_RH,
        total_exo_com_capacity_df_RH,
        total_exo_decom_capacity_df_RH,
    )
