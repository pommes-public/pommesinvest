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

from pommesinvest.model_funcs.helpers import discount_values
from pommesinvest.model_funcs.subroutines import (
    parse_input_sheets,
    create_buses,
    create_demand_response_units,
    create_sinks,
    create_excess_sinks,
    create_transformers,
    create_storages,
    existing_transformers_exo_decom,
    new_transformers_exo,
    storages_exo,
    renewables_exo,
    load_input_data,
)
from pommesinvest.model_funcs.helpers import convert_annual_limit, resample_timeseries

# TODO: Adjust to investment model needs
def parse_input_data(
    path_folder_input,
    AggregateInput,
    countries,
    fuel_cost_pathway="middle",
    year=2017,
    ActivateDemandResponse=False,
    scenario="50",
):
    """Read in csv files and build oemof components

    Parameters
    ----------
    path_folder_input : :obj:`str`
        The path_folder_output where the input data is stored

    AggregateInput: :obj:`boolean`
        boolean control variable indicating whether to use complete or aggregated
        transformer input data set

    countries : :obj:`list` of str
        List of countries to be simulated

    fuel_cost_pathway:  :obj:`str`
       The chosen pathway for commodity cost scenarios (lower, middle, upper)

    year: :obj:`str`
        Reference year for pathways depending on starttime

    ActivateDemandResponse : :obj:`boolean`
        If True, demand response input data is read in

    scenario : :obj:`str`
        Demand response scenario to be modeled;
        must be one of ['25', '50', '75'] whereby '25' is the lower,
        i.e. rather pessimistic estimate
    Returns
    -------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys
    """
    # save the input data in a dict; keys are names and values are DataFrames
    files = {
        "buses": "buses",
        # 'links': 'links',
        # 'links_ts': 'links_ts',
        "sinks_excess": "sinks_excess",
        "sinks_demand_el": "sinks_demand_el",
        "sinks_demand_el_ts": "sinks_demand_el_ts" + "_complete",
        "sources_shortage": "sources_shortage",
        "sources_commodity": "sources_commodity",
        # 'sources_renewables_fluc': 'sources_renewables_fluc',
        # 'costs_fuel': 'costs_fuel_' + fuel_cost_pathway,
        # 'costs_ramping': 'costs_ramping',
        # 'costs_fixed',
        # 'costs_carbon': 'costs_carbon',
        # 'costs_market_values': 'costs_market_values',
        # 'costs_operation': 'costs_operation',
        # 'costs_operation_storages': 'costs_operation_storages',
        "emission_limits": "emission_limits",
    }

    add_files = {  #'sources_renewables': 'sources_renewables',
        # 'sources_renewables_ts': 'sources_renewables_ts',
        #'storages_el': 'storages_el',
        # 'transformers': 'transformers',
        # 'transformers_minload_ts': 'transformers_minload_ts',
        # 'transformers_renewables': 'transformers_renewables',
        # 'min_loads_dh': 'min_loads_dh',
        # 'min_loads_ipp': 'min_loads_ipp',
        # 'costs_operation_renewables': 'costs_operation_renewables'
    }

    # Optionally use aggregated transformer data instead
    if AggregateInput:
        add_files["transformers"] = "transformers_clustered"

    # Addition: demand response units
    if ActivateDemandResponse:
        add_files["sinks_dr_el"] = "sinks_demand_response_el_" + scenario
        add_files["sinks_dr_el_ts"] = (
            "sinks_demand_response_el_ts_" + scenario + "_complete"
        )
        add_files["sinks_dr_el_ava_pos_ts"] = (
            "sinks_demand_response_el_ava_pos_ts_" + scenario + "_complete"
        )
        add_files["sinks_dr_el_ava_neg_ts"] = (
            "sinks_demand_response_el_ava_neg_ts_" + scenario + "_complete"
        )

    # Use dedicated 2030 data
    if year == 2030:
        add_files = {k: v + "" for k, v in add_files.items()}

    files = {**files, **add_files}

    input_data = {
        key: load_input_data(
            filename=name, path_folder_input=path_folder_input, countries=countries
        )
        for key, name in files.items()
    }

    return input_data


def add_limits(
    input_data,
    emission_pathway,
    starttime="2017-01-01 00:00:00",
    endtime="2017-01-01 23:00:00",
):
    """Add further limits to the optimization model (emissions limit for now)

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    emission_pathway : str
        The pathway for emissions reduction to be used

    starttime : :obj:`str`
        The starttime of the optimization run

    endtime : :obj:`str`
        The endtime of the optimization run

    Returns
    -------
    emissions_limit : :obj:`float`
        The emissions limit to be used (converted)
    """
    emissions_limit = convert_annual_limit(
        input_data["emission_limits"][emission_pathway], starttime, endtime
    )

    return emissions_limit


# TODO: Replace by method nodes_from_csv
def nodes_from_excel(
    path_folder_input,
    filename_node_data,
    filename_cost_data,
    filename_node_timeseries,
    filename_min_max_timeseries,
    filename_cost_timeseries,
    AggregateInput,
    startyear,
    endyear,
    MaxInvest,
    fuel_cost_pathway="middle",
    investment_cost_pathway="middle",
    starttime="2016-01-01 00:00:00",
    endtime="2017-01-01 00:00:00",
    freq="24H",
    multiplicator=24,
    optimization_timeframe=2,
    IR=0.02,
    discount=False,
    overlap_in_timesteps=0,
    RollingHorizon=False,
    ActivateEmissionsLimit=False,
    emission_pathway="100_percent_linear",
    ActivateDemandResponse=False,
    approach="DIW",
    scenario="50",
):
    """Reads in an Excel Workbook and builds the respective oemof components
    by going through the Worksheets of the file. The Worksheets are parsed
    to a pd.DataFrame. The DataFrames are iterrated through using the
    funcion pd.DataFrame.iterows(). The actual parameters are obtained
    from the entries in the respective columns. If booleanBinaries is True,
    variables and constraints for startup and shutdown as well as
    minimum uptime and downtime are activated. Method is used to build the
    nodes and flows for a dispatch optimization model.

    Parameters
    ----------
    path_folder_input : :obj:`str`
        The file path where input files are stored (common folder)

    filename_node_data : :obj:`str`
        Name of Excel Workbook containing all data
        for creating nodes (buses and oemof components)

    filename_cost_data : :obj:`str`
        Name of Excel Workbook containing cost pathways for oemof components

    filename_node_timeseries : :obj:`str`
        Filename of the node timeseries data, given in a separate .csv file

    filename_min_max_timeseries  : :obj:`str`
       Filename of the min / max transformer data, given in a separate .csv file

    filename_cost_timeseries : :obj:`str`
        Filename of the cost timeseries data, given in a separate .csv file

    AggregateInput: :obj:`boolean`
        boolean control variable indicating whether to use complete or aggregated
        transformer input data set

    startyear : :obj:`int`
        The startyear of the optimization run

    endyear : :obj:`int`
        The endyear of the optimization run

    MaxInvest : :obj:`boolean`
        If True, investment limits per technology are applied

    fuel_cost_pathway:  :obj:`str`
        The chosen pathway for commodity cost scenarios (lower, middle, upper)

    investment_cost_pathway:  :obj:`str`
        The chosen pathway for commodity cost scenarios (lower, middle, upper)

    starttime : :obj:`str`
        The starting timestamp of the optimization timeframe

    endtime : :obj:`str`
        The end timestamp of the optimization timeframe

    freq : :obj:`string`
        A string defining the timeseries target frequency; determined by the
        model configuration

    multiplicator : :obj:`int`
        A multiplicator to convert the input data given with an hourly resolution
        to another (usually a lower) one

    optimization_timeframe : :obj:`str`
        The length of the overall optimization timeframe in years
        (used for determining technology specific investment limits)

    IR : :obj:`float`
        The interest rate used for discounting

    discount : :obj:`boolean`
        Boolean parameter indicating whether or not to discount future investment costs

    overlap_in_timesteps : :obj:`int`
        the overlap in timesteps if a rolling horizon model is run
        (to prevent index out of bounds error)

    RollingHorizon: :obj:`boolean`
        If True a myopic (Rolling horizon) optimization run is carried out,
        elsewhise a simple overall optimization approach is chosen

    ActivateEmissionsLimit : :obj:`boolean`
        If True, an emission limit is introduced

    emission_pathway : str
        The pathway for emissions reduction to be used

    ActivateDemandResponse : :obj:`boolean`
        Boolean control parameter indicating whether or not to introduce
        demand response units

    approach : :obj:`str`
        Demand response modeling approach to be used;
        must be one of ['DIW', 'DLR', 'IER', 'TUD']

    scenario : :obj:`str`
        Demand response scenario to be modeled;
        must be one of ['25', '50', '75'] whereby '25' is the lower,
        i.e. rather pessimistic estimate

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    existing_storage_labels : :obj:`list`
        List of the labels of existing storages

    new_built_storage_labels : :obj:`list`
        List of the labels of new built storages

    total_exo_com_costs_df : :obj:`pd.DataFrame`
        A DataFrame containing the overall costs for exogeneous investements

    total_exo_com_capacity_df : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous investements

    total_exo_decom_capacity_df : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous decommissioning decisions
    """

    # Parse all data needed from input data sheets
    (
        buses_df,
        excess_df,
        shortage_df,
        commodity_sources_df,
        existing_transformers_df,
        new_built_transformers_df,
        renewables_df,
        demand_df,
        existing_storages_df,
        new_built_storages_df,
        existing_transformers_decom_df,
        new_transformers_de_com_df,
        renewables_com_df,
        node_timeseries_df,
        min_max_timeseries_df,
        fuel_costs_df,
        operation_costs_df,
        ramping_costs_df,
        startup_costs_df,
        storage_var_costs_df,
        investment_costs_df,
        storage_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_turbine_investment_costs_df,
        WACC_df,
        cost_timeseries_df,
    ) = parse_input_sheets(
        path_folder_input,
        filename_node_data,
        filename_cost_data,
        filename_node_timeseries,
        filename_min_max_timeseries,
        filename_cost_timeseries,
        fuel_cost_pathway,
        investment_cost_pathway,
        starttime,
        endtime,
        freq,
        multiplicator,
        overlap_in_timesteps=0,
    )

    # node_dict is a dictionary containing all nodes
    # (i.e. all oemof elements except for Flows) of the model
    node_dict = {}

    # Create Buses objects from table 'buses'
    # See subroutines_for_data_input.py for details on all following function calls
    node_dict = create_buses(buses_df, node_dict)

    # Create all source objects from tables 'commodity_sources', 'shortage', 'renewables'
    node_dict = create_sources(
        node_dict,
        commodity_sources_df,
        fuel_costs_df,
        fuel_cost_pathway,
        shortage_df,
        renewables_df,
        node_timeseries_df,
        cost_timeseries_df,
        starttime,
        endtime,
    )

    # TODO: Replace this temporary solution by final one
    # In the investment model, we only consider Germany
    countries = ["DE"]
    year = pd.to_datetime(starttime).year

    input_data = parse_input_data(
        "../data/Outputlisten_Test_Invest/",
        AggregateInput,
        countries,
        fuel_cost_pathway,
        year,
        ActivateDemandResponse,
        scenario,
    )

    if ActivateDemandResponse:

        ts_keys = [
            "sinks_dr_el_ts",
            "sinks_dr_el_ava_pos_ts",
            "sinks_dr_el_ava_neg_ts",
            "sinks_demand_el_ts",
        ]

        for key in ts_keys:
            input_data[key] = resample_timeseries(
                input_data[key], freq="48H", aggregation_rule="sum"
            )

        node_dict, dr_overall_load_ts_df = create_demand_response_units(
            input_data["sinks_dr_el"],
            input_data["sinks_dr_el_ts"],
            input_data["sinks_dr_el_ava_pos_ts"],
            input_data["sinks_dr_el_ava_neg_ts"],
            approach,
            starttime,
            endtime,
            node_dict,
        )

        node_dict = create_demand(
            input_data["sinks_demand_el"],
            input_data["sinks_demand_el_ts"],
            starttime,
            endtime,
            node_dict,
            # RollingHorizon,
            ActivateDemandResponse,
            dr_overall_load_ts_df,
        )

    else:

        ts_keys = ["sinks_demand_el_ts"]

        for key in ts_keys:
            input_data[key] = resample_timeseries(
                input_data[key], freq="48H", aggregation_rule="sum"
            )

        node_dict = create_demand(
            input_data["sinks_demand_el"],
            input_data["sinks_demand_el_ts"],
            starttime,
            endtime,
            node_dict,
            # RollingHorizon
        )

    # Create all sink objects from tables 'demand', 'excess_sinks'
    # node_dict = create_sinks(node_dict, demand_df, node_timeseries_df,
    #                          starttime, endtime, excess_df,
    #                          ActivateDemandResponse,
    #                          dr_overall_load_ts_df)

    # Create excess sinks
    node_dict = create_excess_sinks(excess_df, node_dict)

    # Create Transformer objects from 'transformers' table
    node_dict = create_transformers(
        node_dict,
        existing_transformers_df,
        new_built_transformers_df,
        AggregateInput,
        RollingHorizon,
        operation_costs_df,
        ramping_costs_df,
        investment_costs_df,
        WACC_df,
        cost_timeseries_df,
        min_max_timeseries_df,
        MaxInvest,
        starttime,
        endtime,
        endyear,
        optimization_timeframe=optimization_timeframe,
    )

    # Create Storage objects from 'storages' table
    node_dict = create_storages(
        node_dict,
        existing_storages_df,
        new_built_storages_df,
        RollingHorizon,
        MaxInvest,
        storage_var_costs_df,
        storage_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_turbine_investment_costs_df,
        WACC_df,
        starttime,
        endyear,
        optimization_timeframe=optimization_timeframe,
    )

    (
        total_exo_com_costs_df,
        total_exo_com_capacity_df,
        total_exo_decom_capacity_df,
    ) = exo_com_costs(
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
        IR=IR,
        discount=discount,
    )

    # Create existing_storage_labels and new_built_storage_labels
    # to iterate over them later to get investment information about capacity
    existing_storage_labels = list(existing_storages_df.index)
    new_built_storage_labels = list(new_built_storages_df.index)

    emissions_limit = None
    if ActivateEmissionsLimit:
        emissions_limit = add_limits(input_data, emission_pathway, starttime, endtime)

    return (
        node_dict,
        existing_storage_labels,
        new_built_storage_labels,
        total_exo_com_costs_df,
        total_exo_com_capacity_df,
        total_exo_decom_capacity_df,
        emissions_limit,
    )


def nodes_from_excel_rh(
    path_folder_input,
    filename_node_data,
    filename_cost_data,
    filename_node_timeseries,
    filename_min_max_timeseries,
    filename_cost_timeseries,
    AggregateInput,
    counter,
    storages_init_df,
    transformers_init_df,
    timeslice_length_with_overlap,
    RH_endyear,
    MaxInvest,
    timeseries_start,
    total_exo_com_costs_df_RH,
    total_exo_com_capacity_df_RH,
    total_exo_decom_capacity_df_RH,
    fuel_cost_pathway="middle",
    investment_cost_pathway="middle",
    freq="24H",
    multiplicator=24,
    overlap_in_timesteps=0,
    years_per_timeslice=1,
    IR=0.02,
    discount=False,
    RollingHorizon=True,
    ActivateEmissionsLimit=False,
    emission_pathway="100_percent_linear",
    ActivateDemandResponse=False,
    approach="DIW",
    scenario="50",
):

    """Function builds up an energy system from a given excel file,
    preparing it for a rolling horizon dispatch optimization.
    Functionality is more or less the same as for the simple method.
    The main difference is that a counter is used and initial status
    from the last optimization run is used in the next one.


    Parameters
    ----------
    path_folder_input : :obj:`str`
        The file path where input files are stored (common folder)

    filename_node_data : :obj:`str`
        Name of Excel Workbook containing all data for creating nodes (oemof components)

    filename_cost_data : :obj:`str`
        Name of Excel Workbook containing cost pathways for oemof components

    filename_node_timeseries : :obj:`str`
        Filename of the node timeseries data, given in a separate .csv file

    filename_min_max_timeseries  : :obj:`str`
       Filename of the min / max transformer data, given in a separate .csv file

    filename_cost_timeseries : :obj:`str`
        Filename of the cost timeseries data, given in a separate .csv file

    AggregateInput: :obj:`boolean`
        boolean control variable indicating whether to use complete or aggregated
        transformer input data set

    counter : :obj:`int`
        An integer counter variable counting the number of the rolling horizon run

    storages_init_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage data from previous model runs

    transformers_init_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the transformer data from previous model runs

    timeslice_length_with_overlap : :obj:`int`
        Overall length of optimization timeframe including overlap (t_step)

    RH_endyear : :obj:`int`
        End year of the current rolling horizon model run

    MaxInvest : :obj:`boolean`
        If True, investment limits per technology are applied

    timeseries_start : :obj:`pd.Timestamp`
        The starting timestamp of the optimization timeframe

    total_exo_com_costs_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall costs for exogeneous investements

    total_exo_com_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous investements

    total_exo_decom_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous decommissioning decisions

    fuel_cost_pathway:  :obj:`str`
        The chosen pathway for commodity cost scenarios (lower, middle, upper)

    investment_cost_pathway:  :obj:`str`
        The chosen pathway for investment cost scenarios (lower, middle, upper)

    freq : :obj:`string`
        A string defining the timeseries target frequency; determined by the
        model configuration

    multiplicator : :obj:`int`
        A multiplicator to convert the input data given with an hourly resolution
        to another (usually a lower) one

    overlap_in_timesteps : :obj:`int`
        the overlap in timesteps if a rolling horizon model is run
        (to prevent index out of bounds error)

    years_per_timeslice : :obj:`int`
        Number of years of a timeslice (a given myopic iteration); may differ
        for the last timesteps since it does not necessarily have to be
        completely covered

    IR : :obj:`float`
        The interest rate used for discounting

    discount : :obj:`boolean`
        Boolean parameter indicating whether or not to discount future investment costs

    RollingHorizon: :obj:`boolean`
        If True a myopic (Rolling horizon) optimization run is carried out,
        elsewhise a simple overall optimization approach is chosen

    ActivateEmissionsLimit : :obj:`boolean`
        If True, an emission limit is introduced

    emission_pathway : str
        The pathway for emissions reduction to be used

    ActivateDemandResponse : :obj:`boolean`
        Boolean control parameter indicating whether or not to introduce
        demand response units

    approach : :obj:`str`
        Demand response modeling approach to be used;
        must be one of ['DIW', 'DLR', 'IER', 'TUD']

    scenario : :obj:`str`
        Demand response scenario to be modeled;
        must be one of ['25', '50', '75'] whereby '25' is the lower,
        i.e. rather pessimistic estimate

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    new_built_transformer_labels : :obj:`list` of :class:`str`
        A list of the labels of all transformer elements included in the model
        used for assessing these and assigning existing capacities (via the
        function initial_states_RH from functions_for_model_control_invest)

    new_built_storage_labels : :obj:`list` of :class:`str`
        A list of the labels of all storage elements included in the model
        used for assessing these and assigning initial states as well
        as existing caoacities (via the
        function initial_states_RH from functions_for_model_control_invest)

    endo_exo_exist_df : :obj:`pd.DataFrame`
        A DataFrame containing the endogeneous and exogeneous transformers commissioning
        information for setting initial states

    endo_exo_exist_stor_df : :obj:`pd.DataFrame`
        A DataFrame containing the endogeneous and exogeneous storages commissioning
        information for setting initial states

    total_exo_com_costs_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall costs for exogeneous investements

    total_exo_com_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous investements

    total_exo_decom_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous decommissioning decisions

    """

    # The rolling horizon approach differs by choosing
    # starttime as well as endtime and initial capacities for transformers.
    starttime_RH = timeseries_start.strftime("%Y-%m-%d %H:%M:%S")
    endtime_RH = (
        timeseries_start + timeslice_length_with_overlap * timeseries_start.freq
    ).strftime("%Y-%m-%d %H:%M:%S")

    logging.info("Start of iteration {} : {}".format(counter, starttime_RH))
    logging.info("End of iteration {} : {}".format(counter, endtime_RH))

    # Parse all data needed from input data sheets
    # NOTE: Parsing from starttime to endtime of overall optimization horizon
    # would be needed if constraints were introduced which cover more than one
    # timeslice (such as expected power plant rentability constraints)
    (
        buses_df,
        excess_df,
        shortage_df,
        commodity_sources_df,
        existing_transformers_df,
        new_built_transformers_df,
        renewables_df,
        demand_df,
        existing_storages_df,
        new_built_storages_df,
        existing_transformers_decom_df,
        new_transformers_de_com_df,
        renewables_com_df,
        node_timeseries_df,
        min_max_timeseries_df,
        fuel_costs_df,
        operation_costs_df,
        ramping_costs_df,
        startup_costs_df,
        storage_var_costs_df,
        investment_costs_df,
        storage_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_turbine_investment_costs_df,
        WACC_df,
        cost_timeseries_df,
    ) = parse_input_sheets(
        path_folder_input,
        filename_node_data,
        filename_cost_data,
        filename_node_timeseries,
        filename_min_max_timeseries,
        filename_cost_timeseries,
        fuel_cost_pathway,
        investment_cost_pathway,
        starttime=starttime_RH,
        endtime=endtime_RH,
        freq=freq,
        multiplicator=multiplicator,
        overlap_in_timesteps=overlap_in_timesteps,
    )

    node_dict = {}

    node_dict = create_buses(buses_df, node_dict)

    node_dict = create_sources(
        node_dict,
        commodity_sources_df,
        fuel_costs_df,
        fuel_cost_pathway,
        shortage_df,
        renewables_df,
        node_timeseries_df,
        cost_timeseries_df,
        starttime=starttime_RH,
        endtime=endtime_RH,
    )

    # TODO: Replace this temporary solution by final one
    # In the investment model, we only consider Germany
    countries = ["DE"]
    year = pd.to_datetime(starttime_RH).year

    input_data = parse_input_data(
        "../data/Outputlisten_Test_Invest/",
        AggregateInput,
        countries,
        fuel_cost_pathway,
        year,
        ActivateDemandResponse,
        scenario,
    )

    if ActivateDemandResponse:

        ts_keys = [
            "sinks_dr_el_ts",
            "sinks_dr_el_ava_pos_ts",
            "sinks_dr_el_ava_neg_ts",
            "sinks_demand_el_ts",
        ]

        for key in ts_keys:
            input_data[key] = resample_timeseries(
                input_data[key], freq="48H", aggregation_rule="sum"
            )

        node_dict, dr_overall_load_ts_df = create_demand_response_units(
            input_data["sinks_dr_el"],
            input_data["sinks_dr_el_ts"],
            input_data["sinks_dr_el_ava_pos_ts"],
            input_data["sinks_dr_el_ava_neg_ts"],
            approach,
            starttime_RH,
            endtime_RH,
            node_dict,
        )

        node_dict = create_demand(
            input_data["sinks_demand_el"],
            input_data["sinks_demand_el_ts"],
            starttime_RH,
            endtime_RH,
            node_dict,
            # RollingHorizon,
            ActivateDemandResponse,
            dr_overall_load_ts_df,
        )

    else:

        ts_keys = ["sinks_demand_el_ts"]

        for key in ts_keys:
            input_data[key] = resample_timeseries(
                input_data[key], freq="48H", aggregation_rule="sum"
            )

        node_dict = create_demand(
            input_data["sinks_demand_el"],
            input_data["sinks_demand_el_ts"],
            starttime_RH,
            endtime_RH,
            node_dict,
            # RollingHorizon)
        )

    # node_dict = create_sinks(node_dict, demand_df, node_timeseries_df,
    #                          starttime = starttime_RH,
    #                          endtime = endtime_RH,
    #                          excess_df = excess_df,
    #                          RollingHorizon = True)

    # Create excess sinks
    node_dict = create_excess_sinks(excess_df, node_dict)

    node_dict, new_built_transformer_labels, endo_exo_exist_df = create_transformers(
        node_dict,
        existing_transformers_df,
        new_built_transformers_df,
        AggregateInput,
        RollingHorizon,
        operation_costs_df,
        ramping_costs_df,
        investment_costs_df,
        WACC_df,
        cost_timeseries_df,
        min_max_timeseries_df,
        MaxInvest,
        starttime=starttime_RH,
        endtime=endtime_RH,
        counter=counter,
        transformers_init_df=transformers_init_df,
        years_per_timeslice=years_per_timeslice,
        endyear=RH_endyear,
    )

    node_dict, new_built_storage_labels, endo_exo_exist_stor_df = create_storages(
        node_dict,
        existing_storages_df,
        new_built_storages_df,
        RollingHorizon,
        MaxInvest,
        storage_var_costs_df,
        storage_investment_costs_df,
        storage_pump_investment_costs_df,
        storage_turbine_investment_costs_df,
        WACC_df,
        starttime=starttime_RH,
        counter=counter,
        storages_init_df=storages_init_df,
        years_per_timeslice=years_per_timeslice,
        endyear=RH_endyear,
    )

    (
        total_exo_com_costs_df_RH,
        total_exo_com_capacity_df_RH,
        total_exo_decom_capacity_df_RH,
    ) = exo_com_costs_RH(
        timeseries_start.year,
        RH_endyear,
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
        IR=IR,
        discount=discount,
    )

    # Create existing_storage_labels to iterate over them later to get investment information about capacity
    existing_storage_labels = list(existing_storages_df.index)

    emissions_limit = None
    if ActivateEmissionsLimit:
        emissions_limit = add_limits(
            input_data, emission_pathway, starttime_RH, endtime_RH
        )

    return (
        node_dict,
        new_built_transformer_labels,
        new_built_storage_labels,
        endo_exo_exist_df,
        endo_exo_exist_stor_df,
        existing_storage_labels,
        total_exo_com_costs_df_RH,
        total_exo_com_capacity_df_RH,
        total_exo_decom_capacity_df_RH,
        emissions_limit,
    )


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
