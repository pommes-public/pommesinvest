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

from pommesinvest.model_funcs.subroutines import (
    load_input_data,
    create_buses,
    create_commodity_sources,
    create_shortage_sources,
    create_renewables,
    create_demand_response_units,
    create_demand,
    create_excess_sinks,
    create_exogenous_transformers,
    create_new_built_transformers,
    create_new_built_transformers_rolling_horizon,
    create_exogenous_storages,
    create_exogenous_storages_rolling_horizon,
    create_new_built_storages,
    create_new_built_storages_rolling_horizon,
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
        "exogenous_transformers": "transformers",
        "new_built_transformers": "new_built_transformers",
    }

    time_series = {
        "sinks_demand_el_ts": "sinks_demand_el_ts",
        "sources_renewables_ts": "sources_renewables_ts",
        "transformers_minload_ts": "transformers_minload_ts",
        "transformers_availability_ts": "transformers_availability_ts",
        "exogenous_transformer_capacities": "transformer_capacities",
        "exogenous_storages_capacities": "storages_capacities",
        "costs_fuel": f"costs_fuel_{im.fuel_cost_pathway}_nominal",
        "costs_fuel_ts": "costs_fuel_ts",
        "costs_emissions": (
            f"costs_emissions_{im.emissions_cost_pathway}_nominal"
        ),
        "costs_emissions_ts": "costs_emissions_ts",
        "costs_operation": "costs_operation_nominal",
        "costs_operation_storages": "costs_operation_storages_nominal",
        "costs_investment": (
            f"costs_investment_{im.investment_cost_pathway}_nominal"
        ),
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
            "sinks_demand_response_el_ava_pos_ts_"
            + im.demand_response_scenario
        )
        components["sinks_dr_el_ava_neg_ts"] = (
            "sinks_demand_response_el_ava_neg_ts_"
            + im.demand_response_scenario
        )

    # Combine all files
    input_files = {**buses, **components, **time_series}
    input_files = {**input_files, **other_files}

    return {
        key: load_input_data(filename=name, im=im)
        for key, name in input_files.items()
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
        "min_max_ts",
        "exogenous_transformer_capacities",
        "exogenous_storages_capacities",
        "costs_fuel",
        "costs_emissions",
        "costs_operation",
        "costs_operation_storages",
    ]
    hourly_ts = [
        "transformers_availability_ts",
        "transformers_minload_ts",
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
                :,
                ["grad_pos", "grad_neg", "max_load_factor", "min_load_factor"],
            ] = (
                input_data[key]
                .loc[
                    :,
                    [
                        "grad_pos",
                        "grad_neg",
                        "max_load_factor",
                        "min_load_factor",
                    ],
                ]
                .mul(im.multiplicator)
            )
        elif key in storage_data:
            input_data[key] = input_data[key].where(
                pd.notnull(input_data[key]), None
            )
            input_data[key].loc[
                :,
                [
                    "max_storage_level",
                    "min_storage_level",
                    "nominal_storable_energy",
                ],
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

        node_dict = create_demand(
            input_data, im, node_dict, dr_overall_load_ts_df
        )
    else:
        node_dict = create_demand(input_data, im, node_dict)

    node_dict = create_excess_sinks(input_data, node_dict)
    node_dict = create_exogenous_transformers(input_data, im, node_dict)

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
        input_data["emission_limits"][im.emission_pathway],
        im.starttime,
        im.endtime,
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
    node_dict = create_exogenous_storages(input_data, im, node_dict)
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

    transformer_and_storage_labels : :obj:`list` of :class:`str`
        A list of the labels of all transformers and storages elements
        included in the model used for assessing these
        and assigning initial states
    """
    frequency_used = {
        "60min": (
            getattr(im, "time_slice_length_with_overlap"),
            "h",
        ),
        "4H": (
            getattr(im, "time_slice_length_with_overlap") * 4,
            "h",
        ),
        "8H": (
            getattr(im, "time_slice_length_with_overlap") * 8,
            "h",
        ),
        "24H": (
            getattr(im, "time_slice_length_with_overlap") * 24,
            "h",
        ),
        "48H": (
            getattr(im, "time_slice_length_with_overlap") * 48,
            "h",
        ),
    }[im.freq]

    # Update start time and end time of the model for retrieving the right data
    im.start_time = getattr(im, "time_series_start").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    im.end_time = (
        getattr(im, "time_series_start")
        + pd.to_timedelta(frequency_used[0], frequency_used[1])
    ).strftime("%Y-%m-%d %H:%M:%S")

    input_data = parse_input_data(im)
    resample_input_data(input_data, im)

    node_dict = add_components(input_data, im)

    # create storages and new-built transformers (rolling horizon)
    (
        node_dict,
        new_built_transformer_labels,
    ) = create_new_built_transformers_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )

    (
        node_dict,
        exogenous_storage_labels,
    ) = create_exogenous_storages_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )
    (
        node_dict,
        new_built_storage_labels,
    ) = create_new_built_storages_rolling_horizon(
        input_data, im, node_dict, iteration_results
    )

    transformer_and_storage_labels = (
        new_built_transformer_labels
        + exogenous_storage_labels
        + new_built_storage_labels
    )

    emissions_limit = None
    if im.activate_emissions_limit:
        emissions_limit = add_limits(input_data, im)

    return (node_dict, emissions_limit, transformer_and_storage_labels)
