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
    create_exogenous_converters,
    create_new_built_storages,
    create_new_built_storages_myopic_horizon,
    create_new_built_converters,
    create_new_built_converters_myopic_horizon,
    create_renewables,
    create_shortage_sources,
    load_input_data,
    create_linking_converters,
    create_electric_vehicles,
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
        "linking_transformers_ts": "linking_transformers_ts",
    }

    if im.include_artificial_shortage_units:
        components[
            "sources_shortage_el_artificial"
        ] = f"sources_el_artificial_{im.demand_response_scenario}"
        hourly_time_series[
            "sources_shortage_el_artificial_ts"
        ] = f"sources_el_artificial_ts_{im.demand_response_scenario}"

    annual_time_series = {
        "transformers_exogenous_max_ts": "transformers_exogenous_max_ts",
        "costs_fuel_ts": (
            f"costs_fuel_{im.fuel_cost_pathway}_nominal_indexed_ts"
        ),
        "costs_emissions_ts": (
            f"costs_emissions_{im.emissions_cost_pathway}_nominal_indexed_ts"
        ),
        "costs_operation_ts": (
            f"variable_costs_{im.flexibility_options_scenario}%_nominal"
        ),
        "costs_operation_storages_ts": (
            f"variable_costs_storages_"
            f"{im.flexibility_options_scenario}%_nominal"
        ),
        "costs_operation_linking_transformers_ts": (
            "costs_operation_linking_transformers_nominal_indexed_ts"
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
        "linking_transformers_annual_ts": "linking_transformers_annual_ts",
        "storages_el_exogenous_max_ts": "storages_el_exogenous_max_ts",
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
        "hydrogen_investment_maxima": "hydrogen_investment_maxima",
        "linking_transformers": "linking_transformers",
    }

    # Development factors for emissions; used for scaling minimum loads
    if (
        im.activate_emissions_pathway_limit
        or im.activate_emissions_budget_limit
    ):
        other_files[
            "emission_development_factors"
        ] = "emission_development_factors"

    # Add demand response units
    if im.activate_demand_response:
        # Overall demand = overall demand excluding demand response baseline
        hourly_time_series["sinks_demand_el_ts"] = (
            f"sinks_demand_el_excl_demand_response_ts_"
            f"{im.demand_response_scenario}_hourly"
        )
        components["sinks_demand_el"] = (
            f"sinks_demand_el_excl_demand_response_"
            f"{im.demand_response_scenario}"
        )

        # Obtain demand response clusters from file to avoid hard-coding
        components[
            "demand_response_clusters_eligibility"
        ] = "demand_response_clusters_eligibility"
        dr_clusters = load_input_data(
            filename="demand_response_clusters_eligibility", im=im
        )
        # Add demand response clusters information to the model itself
        im.add_demand_response_clusters(list(dr_clusters.index))
        for dr_cluster in dr_clusters.index:
            components[f"sinks_dr_el_{dr_cluster}"] = (
                f"{dr_cluster}_potential_parameters_"
                f"{im.demand_response_scenario}%"
            )
            annual_time_series[f"sinks_dr_el_{dr_cluster}_variable_costs"] = (
                f"{dr_cluster}_variable_costs_parameters_"
                f"{im.demand_response_scenario}%"
            )
            annual_time_series[
                f"sinks_dr_el_{dr_cluster}_fixed_costs_and_investments"
            ] = (
                f"{dr_cluster}_fixed_costs_and_investments_"
                f"parameters_{im.demand_response_scenario}%"
            )

        hourly_time_series[
            "sinks_dr_el_ts"
        ] = f"sinks_demand_response_el_ts_{im.demand_response_scenario}"

        hourly_time_series["sinks_dr_el_ava_pos_ts"] = (
            f"sinks_demand_response_el_ava_pos_ts_"
            f"{im.demand_response_scenario}"
        )
        hourly_time_series["sinks_dr_el_ava_neg_ts"] = (
            f"sinks_demand_response_el_ava_neg_ts_"
            f"{im.demand_response_scenario}"
        )

        # Electric vehicles
        components[
            "electric_vehicles"
        ] = f"components_electric_vehicles_{im.demand_response_scenario}"
        hourly_time_series[
            "electric_vehicles_ts"
        ] = f"electric_vehicles_ts_{im.demand_response_scenario}"
        ev_buses = load_input_data(
            filename=components["electric_vehicles"], im=im
        )
        ev_buses = list(ev_buses.loc[ev_buses["type"] == "bus"].index.values)
        im.add_ev_buses(ev_buses)

    # Combine all files
    input_files = {
        **buses,
        **components,
        **annual_time_series,
        **hourly_time_series,
    }
    input_files = {**input_files, **other_files}

    # Update files in case sensitivities are considered
    if im.sensitivity_parameter != "None":
        input_files = update_sensitivities(im, input_files)

    return {
        key: load_input_data(filename=name, im=im)
        for key, name in input_files.items()
    }


def update_sensitivities(im, input_files):
    """Update reference to files including respective sensitivities

    Parameters
    ----------
    im : :class:`InvestmentModel`
        The investment model that is considered

    input_files : dict
        Dictionary of all input file names

    Returns
    -------
    input_files : dict
        Dictionary of all input files with file names
        for considered sensitivity modified
    """
    sensitivities = {
        "pv": "sources_renewables_ts",
        "prices": ["costs_fuel_ts", "costs_emissions_ts"],
        "consumption": "sinks_demand_el",
    }
    if im.sensitivity_parameter not in ["pv", "prices", "consumption"]:
        raise ValueError(
            f"Invalid configuration given. 'sensitivity_parameter' "
            f"{im.sensitivity_parameter} is not implemented."
        )
    if im.sensitivity_value not in ["-50%", "-25%", "+25%", "+50%"]:
        if im.sensitivity_value == "None":
            raise ValueError(
                "'sensitivity_value' 'None' is only to be used if "
                "no sensitivity is considered, "
                "i.e. 'sensitivity_parameter' is 'None'."
            )
        else:
            raise ValueError(
                "Invalid value for 'sensitivity_value'. "
                "Must be one of ['-50%', '-25%', '+25%', '+50%']"
            )
    sensitivity = sensitivities[im.sensitivity_parameter]

    if isinstance(sensitivity, list):
        for value in sensitivity:
            input_files[
                value
            ] = f"{input_files[value]}_sensitivity_{im.sensitivity_value}"
    else:
        input_files[
            sensitivity
        ] = f"{input_files[sensitivity]}_sensitivity_{im.sensitivity_value}"

    return input_files


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
    converter_data = ["exogenous_transformers", "new_built_transformers"]
    storage_data = ["exogenous_storages_el", "new_built_storages_el"]
    annual_ts = [
        "transformers_exogenous_max_ts",
        "costs_fuel_ts",
        "costs_emissions_ts",
        "costs_operation_ts",
        "costs_operation_storages_ts",
        "costs_operation_linking_transformers_ts",
        "linking_transformers_annual_ts",
        "storages_el_exogenous_max_ts",
    ]
    hourly_ts = [
        "sinks_demand_el_ts",
        "sources_renewables_ts",
        "transformers_minload_ts",
        "transformers_availability_ts",
        "linking_transformers_ts",
    ]

    converter_columns = [
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

    # Integrate demand response in resampling activities
    if im.activate_demand_response:
        demand_response_potential_data = []
        for dr_cluster in input_data[
            "demand_response_clusters_eligibility"
        ].index:
            annual_ts.append(f"sinks_dr_el_{dr_cluster}_variable_costs")
            demand_response_potential_data.append(f"sinks_dr_el_{dr_cluster}")
        hourly_ts.extend(
            [
                "sinks_dr_el_ts",
                "sinks_dr_el_ava_pos_ts",
                "sinks_dr_el_ava_neg_ts",
                "electric_vehicles_ts",
            ]
        )
        demand_response_potential_columns = [
            "interference_duration_neg",
            "interference_duration_pos",
            "interference_duration_pos_shed",
            "maximum_activations_year",
            "maximum_activations_year_shed",
            "regeneration_duration",
            "shifting_duration",
        ]

    # Development factors for emissions; used for scaling minimum loads
    if im.activate_emissions_pathway_limit:
        annual_ts.append("emission_development_factors")

    for key in input_data.keys():
        if key in converter_data:
            input_data[key].loc[:, converter_columns] = (
                input_data[key].loc[:, converter_columns].mul(im.multiplier)
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
        # Note: The scaling down here may result in very unprecise results!
        elif (
            im.activate_demand_response
            and key in demand_response_potential_data
        ):
            input_data[key].loc[
                :, demand_response_potential_columns
            ] = np.ceil(
                input_data[key]
                .loc[:, demand_response_potential_columns]
                .div(im.multiplier)
            )
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

    Note: Storages and new-built converters are not included here.
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
    if im.countries != ["DE"]:
        node_dict = create_linking_converters(input_data, im, node_dict)
    node_dict = create_shortage_sources(input_data, im, node_dict)
    node_dict = create_renewables(input_data, im, node_dict)
    node_dict = create_demand(input_data, im, node_dict)

    if im.activate_demand_response:
        node_dict = create_demand_response_units(input_data, im, node_dict)
        node_dict = create_electric_vehicles(input_data, im, node_dict)

    node_dict = create_excess_sinks(input_data, node_dict)
    node_dict = create_exogenous_converters(input_data, im, node_dict)

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

    # Add new-built storages information to the model itself
    im.add_new_built_storages(list(input_data["new_built_storages_el"].index))

    node_dict = add_components(input_data, im)

    node_dict = create_new_built_converters(input_data, im, node_dict)
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
            .loc[f"{im.start_year}":f"{im.end_year}", im.emissions_pathway]
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

    converter_and_storage_labels : :obj:`dict`
        dictionary containing the labels of all converters and storages elements
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

    # create storages and new-built converters (myopic horizon)
    (
        node_dict,
        new_built_converter_labels,
    ) = create_new_built_converters_myopic_horizon(
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

    converter_and_storage_labels = {
        "new_built_tranformers": new_built_converter_labels,
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

    return node_dict, emissions_limit, converter_and_storage_labels
