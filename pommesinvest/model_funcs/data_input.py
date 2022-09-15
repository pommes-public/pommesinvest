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
import numpy as np
import pandas as pd

from pommesinvest.model_funcs import helpers
from pommesinvest.model_funcs.subroutines import (
    create_buses,
    create_commodity_sources,
    create_demand,
    create_demand_response_units,
    create_excess_sinks,
    create_exogenous_storages,
    create_exogenous_storages_myopic_horizon,
    create_exogenous_transformers,
    create_new_built_storages,
    create_new_built_storages_myopic_horizon,
    create_new_built_transformers,
    create_new_built_transformers_myopic_horizon,
    create_renewables,
    create_shortage_sources,
    load_input_data,
)


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
        "sources_renewables": "sources_renewables_investment_model",
        "exogenous_storages_el": "storages_el_exogenous",
        "new_built_storages_el": "storages_el_investment_options",
        "exogenous_transformers": "transformers_exogenous",
        "new_built_transformers": "transformers_investment_options",
    }

    hourly_time_series = {
        "sinks_demand_el_ts": "sinks_demand_el_ts_hourly",
        "sources_renewables_ts": "sources_renewables_ts_hourly",
        "transformers_minload_ts": "transformers_minload_ts_hourly",
        "transformers_availability_ts": "transformers_availability_ts_hourly",
    }

    annual_time_series = {
        "transformers_exogenous_max_ts": "transformers_exogenous_max_ts",
        "costs_fuel_ts": (
            f"costs_fuel_{im.fuel_cost_pathway}"
            + f"_{im.fuel_price_shock}_nominal_indexed_ts"
        ),
        "costs_emissions_ts": (
            f"costs_emissions_{im.emissions_cost_pathway}_nominal_indexed_ts"
        ),
        "costs_operation_ts": (
            f"variable_costs_{im.flexibility_options_scenario}%_nominal"
        ),
        "costs_operation_storages_ts": (
            f"variable_costs_storages_{im.flexibility_options_scenario}%_nominal"  # noqa: E501
        ),
        "costs_investment": (
            f"investment_expenses_{im.flexibility_options_scenario}%_nominal"
        ),
        "costs_storages_investment_capacity": (
            f"investment_expenses_storages_capacity_"
            + f"{im.flexibility_options_scenario}%_nominal"
        ),
        "costs_storages_investment_power": (
            f"investment_expenses_storages_power_"
            + f"{im.flexibility_options_scenario}%_nominal"
        ),
    }

    # Time-invariant data sets
    other_files = {
        "emission_limits": "emission_limits",
        "wacc": "wacc",
        "interest_rate": "interest_rate",
        "fixed_costs": (
            f"fixed_costs_{im.flexibility_options_scenario}%_nominal"
        ),
        "fixed_costs_storages": (
            f"fixed_costs_storages_{im.flexibility_options_scenario}%_nominal"
        ),
        "hydrogen_investment_maxima": "hydrogen_investment_maxima"
    }

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
    input_files = {
        **buses,
        **components,
        **annual_time_series,
        **hourly_time_series,
    }
    input_files = {**input_files, **other_files}

    # input_files = {
    #     k: v + "_2020" for k, v in input_files.items()
    # }

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
    transformer_data = ["exogenous_transformers", "new_built_transformers"]
    storage_data = ["exogenous_storages_el", "new_built_storages_el"]
    annual_ts = [
        "transformers_exogenous_max_ts",
        "costs_fuel_ts",
        "costs_emissions_ts",
        "costs_operation_ts",
        "costs_operation_storages_ts",
    ]
    hourly_ts = [
        "sinks_demand_el_ts",
        "sources_renewables_ts",
        "transformers_minload_ts",
        "transformers_availability_ts",
    ]

    transformer_columns = [
        "grad_pos",
        "grad_neg",
        "max_load_factor",
        "min_load_factor",
    ]
    storages_columns = [
        "max_storage_level",
        "min_storage_level",
        "loss_rate",
    ]
    existing_storages_add_columns = [
        "max_load_factor",
        "min_load_factor",
        "initial_storage_level",
    ]

    for key in input_data.keys():
        if key in transformer_data:
            input_data[key].loc[:, transformer_columns] = (
                input_data[key].loc[:, transformer_columns].mul(im.multiplier)
            )
        elif key in storage_data:
            # Add additional columns for existing units and remove later on
            if not "new_built" in key:
                storages_columns.extend(existing_storages_add_columns)
            input_data[key] = input_data[key].replace({np.nan: None})
            input_data[key].loc[:, storages_columns] = (
                input_data[key].loc[:, storages_columns].mul(im.multiplier)
            )
            if not "new_built" in key:
                for col in existing_storages_add_columns:
                    storages_columns.remove(col)
        elif key in annual_ts:
            input_data[key].loc["2051-01-01"] = input_data[key].loc[
                "2050-01-01"
            ]
            input_data[key] = helpers.resample_timeseries(
                input_data[key], freq=im.freq, interpolation_rule="linear"
            )[:-1]
        elif key in hourly_ts:
            input_data[key] = helpers.resample_timeseries(
                input_data[key], freq=im.freq, aggregation_rule="sum"
            )


def add_components(input_data, im):
    r"""Add the oemof components to a dictionary of nodes

    Note: Storages and new-built transformers are not included here.
    They have to be defined separately since the approaches differ
    between myopic horizon and simple model.

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

    emissions_limit : int, pd.Series or None
        An overall emissions budget limit or an emissions limit per period
        or None if not specified
    """
    input_data = parse_input_data(im)
    resample_input_data(input_data, im)

    node_dict = add_components(input_data, im)

    node_dict = create_new_built_transformers(input_data, im, node_dict)
    node_dict = create_exogenous_storages(input_data, im, node_dict)
    node_dict = create_new_built_storages(input_data, im, node_dict)

    emissions_limit = None
    if im.activate_emissions_budget_limit:
        emissions_limit = helpers.convert_annual_limit(
            input_data["emission_limits"][im.emissions_pathway],
            im.start_time,
            im.end_time,
        )
    elif im.activate_emissions_pathway_limit:
        emissions_limit = list(
            input_data["emission_limits"]
            .loc[f"{im.start_year}" : f"{im.end_year}", im.emissions_pathway]
            .values
        )

    return node_dict, emissions_limit


def nodes_from_csv_myopic_horizon(im, iteration_results):
    r"""Read in csv files and build components for a myopic horizon run

    Parameters
    ----------
    im : :class:`InvestmentModel`
        The investment model that is considered

    iteration_results : dict
        A dictionary holding the results of the previous myopic horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    emissions_limit : int or None
        The overall emissions limit

    transformer_and_storage_labels : :obj:`dict`
        dictionary containing the labels of all transformers and storages elements
        included in the model used for assessing these
        and assigning initial states
    """
    frequency_used = (
        (
            getattr(im, "time_slice_length_with_overlap") * im.multiplier,
            "h",
        ),
    )

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

    # create storages and new-built transformers (myopic horizon)
    (
        node_dict,
        new_built_transformer_labels,
    ) = create_new_built_transformers_myopic_horizon(
        input_data, im, node_dict, iteration_results
    )

    (
        node_dict,
        exogenous_storage_labels,
    ) = create_exogenous_storages_myopic_horizon(
        input_data, im, node_dict, iteration_results
    )
    (
        node_dict,
        new_built_storage_labels,
    ) = create_new_built_storages_myopic_horizon(
        input_data, im, node_dict, iteration_results
    )

    transformer_and_storage_labels = {
        "new_built_tranformers": new_built_transformer_labels,
        "exogenous_storages": exogenous_storage_labels,
        "new_built_storages": new_built_storage_labels,
    }

    emissions_limit = None
    if im.activate_emissions_budget_limit:
        emissions_limit = helpers.convert_annual_limit(
            input_data["emission_limits"][im.emissions_pathway],
            im.start_time,
            im.end_time,
        )

    return node_dict, emissions_limit, transformer_and_storage_labels
