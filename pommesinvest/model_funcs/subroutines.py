# -*- coding: utf-8 -*-
"""
General description
-------------------
This file contains all subroutines used for reading in input data
for the investment variant of POMMES.

Functions build_XX_transformer represent a hierarchical structure:
    build_XX_transformer builds a single transformer element of a given type
    and returns this to create_XX_transformers as node_dict[i], so the i_th
    element to be build

@author: Johannes Kochems (*), Johannes Giehl (*), Yannick Werner,
Benjamin Grosse

Contributors:
Julien Faist, Hannes Kachel, Sophie Westphal, Flora von Mikulicz-Radecki,
Carla Spiller, Fabian Büllesbach, Timona Ghosh, Paul Verwiebe,
Leticia Encinas Rosa, Joachim Müller-Kirchenbauer

(*) Corresponding authors
"""

# Import necessary packages for function definitions
import math

import numpy as np
import oemof.solph as solph
import pandas as pd
from oemof.tools import economics


def load_input_data(filename=None, im=None):
    r"""Load input data from csv files.

    Parameters
    ----------
    filename : :obj:`str`
        Name of CSV file containing data

    im : :class:`InvestmentModel`
        The investment model that is considered

    Returns
    -------
    df : :class:`pandas.DataFrame`
        DataFrame containing information about nodes or time series.
    """
    if "ts_hourly" not in filename:
        df = pd.read_csv(im.path_folder_input + filename + ".csv", index_col=0)
    # Load slices for hourly data to reduce computational overhead
    else:
        df = load_time_series_data_slice(filename + ".csv", im)

    if "country" in df.columns and im.countries is not None:
        df = df[df["country"].isin(im.countries)]

    if df.isna().any().any() and "_ts" in filename:
        print(
            f"Attention! Time series input data file {filename} contains NaNs."
        )
        print(df.loc[df.isna().any(axis=1)])

    return df


def load_time_series_data_slice(filename=None, im=None):
    """Load slice of input time series data from csv files.

    Determine index range to read in from reading in index
    separately.

    Parameters
    ----------
    filename : :obj:`str`
        Name of CSV file containing data

    im : :class:`InvestmentModel`
        The investment model that is considered

    Returns
    -------
    df : :class:`pandas.DataFrame`
        DataFrame containing sliced time series.
    """
    time_series_start = pd.read_csv(
        im.path_folder_input + filename,
        parse_dates=True,
        index_col=0,
        usecols=[0],
    )
    start_index = pd.Index(time_series_start.index).get_loc(im.start_time)
    time_series_end = pd.Timestamp(im.end_time)
    end_index = pd.Index(time_series_start.index).get_loc(
        (
            time_series_end
            + im.overlap_in_time_steps
            * pd.Timedelta(hours=int(im.freq.split("H")[0]))
        ).strftime("%Y-%m-%d %H:%M:%S")
    )

    return pd.read_csv(
        im.path_folder_input + filename,
        parse_dates=True,
        index_col=0,
        skiprows=range(1, start_index),
        nrows=end_index - start_index + 1,
    )


def create_buses(input_data, node_dict):
    r"""Create buses and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the buses elements
    """
    for i, b in input_data["buses"].iterrows():
        node_dict[i] = solph.buses.Bus(label=i)

    return node_dict


# TODO: Provide (annual) cost input data in the neccessary format (2020 = 1.0)
def create_commodity_sources(input_data, im, node_dict):
    r"""Create commodity sources and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the commodity source elements
    """
    # Regular commodity sources
    for i, cs in input_data["sources_commodity"].iterrows():
        node_dict[i] = solph.components.Source(
            label=i,
            outputs={
                node_dict[cs["to"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_fuel_ts"]
                        .loc[im.start_time : im.end_time, i]
                        .to_numpy()
                        + input_data["costs_emissions_ts"]
                        .loc[im.start_time : im.end_time, i]
                        .to_numpy()
                        * cs["emission_factors"]
                    ),
                    emission_factor=cs["emission_factors"],
                )
            },
        )

    return node_dict


def create_shortage_sources(input_data, node_dict):
    r"""Create shortage sources and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the shortage source elements
    """
    for i, s in input_data["sources_shortage"].iterrows():
        node_dict[i] = solph.components.Source(
            label=i,
            outputs={
                node_dict[s["to"]]: solph.flows.Flow(
                    variable_costs=s["shortage_costs"]
                )
            },
        )

    return node_dict


def create_renewables(input_data, im, node_dict):
    r"""Create renewable sources and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the renewables source elements
    """
    for i, re in input_data["sources_renewables"].iterrows():
        try:
            node_dict[i] = solph.components.Source(
                label=i,
                outputs={
                    node_dict[re["to"]]: solph.flows.Flow(
                        fix=np.array(
                            input_data["sources_renewables_ts"][i][
                                im.start_time : im.end_time
                            ]
                        ),
                        nominal_value=re["capacity"],
                    )
                },
            )
        except KeyError:
            raise KeyError(
                f"Renewable source {i} not specified or causing trouble!"
            )

    return node_dict


def create_demand(input_data, im, node_dict, dr_overall_load_ts_df=None):
    r"""Create demand sinks and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    dr_overall_load_ts_df : :class:`pandas.Series`
        The overall load time series from demand response units which is
        used to decrement overall electrical load for Germany
        NOTE: This shall be substituted through a version which already
        includes this in the data preparation

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the demand sink elements
    """
    for i, d in input_data["sinks_demand_el"].iterrows():
        kwargs_dict = {
            "label": i,
            "inputs": {
                node_dict[d["from"]]: solph.flows.Flow(
                    fix=np.array(
                        input_data["sinks_demand_el_ts"][i][
                            im.start_time : im.end_time
                        ]
                    ),
                    nominal_value=d["maximum"],
                )
            },
        }

        # Adjusted demand here means the difference between overall demand
        # and the baseline load profile for demand response units
        if im.activate_demand_response and i == "DE_sink_el_load":
            kwargs_dict["inputs"] = {
                node_dict[d["from"]]: solph.flows.Flow(
                    fix=np.array(
                        input_data["sinks_demand_el_ts"][i][
                            im.start_time : im.end_time
                        ]
                        .mul(d["maximum"])
                        .sub(
                            dr_overall_load_ts_df[im.start_time : im.end_time]
                        )
                    ),
                    nominal_value=1,
                )
            }

        node_dict[i] = solph.components.Sink(**kwargs_dict)

    return node_dict


def create_demand_response_units(input_data, im, node_dict):
    r"""Create demand response units and add them to the dict of nodes.

    The demand response modeling approach can be chosen from different
    approaches that have been implemented.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the demand response sink elements

    dr_overall_load_ts : :class:`pandas.Series`
        The overall baseline load time series from demand response units
        which is used to decrement overall electrical load for Germany
        NOTE: This shall be substituted through a version which already
        includes this in the data preparation
    """

    for i, d in input_data["sinks_dr_el"].iterrows():
        # kwargs for all demand response modeling approaches
        kwargs_all = {
            "demand": np.array(
                input_data["sinks_dr_el_ts"][i].loc[
                    im.start_time : im.end_time
                ]
            ),
            "capacity_up": np.array(
                input_data["sinks_dr_el_ava_neg_ts"][i].loc[
                    im.start_time : im.end_time
                ]
            ),
            "flex_share_up": d["flex_share_up"],
            "capacity_down": np.array(
                input_data["sinks_dr_el_ava_pos_ts"][i].loc[
                    im.start_time : im.end_time
                ]
            ),
            "flex_share_down": d["flex_share_down"],
            "delay_time": math.ceil(d["shifting_duration"]),
            "shed_time": 1,
            "recovery_time_shed": 0,
            "cost_dsm_up": d["variable_costs"] / 2,
            "cost_dsm_down_shift": d["variable_costs"] / 2,
            "cost_dsm_down_shed": 10000,
            "efficiency": 1,
            "shed_eligibility": False,
            "shift_eligibility": True,
        }

        # kwargs dependent on demand response modeling approach chosen
        kwargs_dict = {
            "DIW": {
                "approach": "DIW",
                "recovery_time_shift": math.ceil(d["regeneration_duration"]),
            },
            "DLR": {
                "approach": "DLR",
                "shift_time": d["interference_duration_pos"],
                "ActivateYearLimit": True,
                "ActivateDayLimit": False,
                "n_yearLimit_shift": np.max(
                    [round(d["maximum_activations_year"]), 1]
                ),
                "n_yearLimit_shed": 1,
                "t_dayLimit": 24,
                "addition": True,
                "fixes": True,
            },
            "oemof": {"approach": "oemof", "shift_interval": 24},
        }

        invest_kwargs = {
            "minimum": 0,
            "maximum": min(
                d["max_cap"] + d["potential_neg_overall"],
                d["installed_cap"],
            ),
            "ep_costs": economics.annuity(
                # TODO: Add individual entries for investment expenses to data basis
                capex=input_data["investment_costs"].loc[i, im.start_year],
                n=d["unit_lifetime"],
                wacc=input_data["wacc"].loc[i, "wacc in p.u."],
            ),
        }

        if im.multi_period:
            invest_kwargs["ep_costs"] = (
                input_data["investment_costs"].loc[i, im.start_year],
            )
            multi_period_invest_kwargs = {
                "lifetime": d["lifetime"],
                "age": 0,
                "interest_rate": d["interest_rate"],
                "fixed_costs": d["fixed_costs"],
                "overall_maximum": d["overall_maximum"],
            }
            invest_kwargs = {**invest_kwargs, **multi_period_invest_kwargs}

        # TODO: Critically check min and max invest params
        approach_dict = {
            "DLR": solph.components.experimental.SinkDSM(
                label=i,
                inputs={
                    node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)
                },
                **kwargs_all,
                **kwargs_dict["DLR"],
                invest=solph.Investment(**invest_kwargs),
            ),
            "DIW": solph.components.experimental.SinkDSM(
                label=i,
                inputs={
                    node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)
                },
                **kwargs_all,
                **kwargs_dict["DIW"],
                invest=solph.Investment(**invest_kwargs),
            ),
            "oemof": solph.components.experimental.SinkDSM(
                label=i,
                inputs={
                    node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)
                },
                **kwargs_all,
                **kwargs_dict["oemof"],
                invest=solph.Investment(**invest_kwargs),
            ),
        }

        node_dict[i] = approach_dict[im.demand_response_approach]

    # Calculate overall electrical baseline load from demand response units
    dr_overall_load_ts_df = (
        input_data["sinks_dr_el_ts"]
        .mul(input_data["sinks_dr_el"]["max_cap"])
        .sum(axis=1)
    )

    return node_dict, dr_overall_load_ts_df


def create_excess_sinks(input_data, node_dict):
    r"""Create excess sinks and add them to the dict of nodes.

    The German excess sink is additionally connected to the renewable buses
    including penalty costs, which is needed to model negative prices.
    It is therefore excluded in the DataFrame that is read in.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the excess sink elements
    """
    for i, e in input_data["sinks_excess"].iterrows():
        node_dict[i] = solph.components.Sink(
            label=i,
            inputs={
                node_dict[e["from"]]: solph.flows.Flow(
                    variable_costs=e["excess_costs"]
                )
            },
        )

    return node_dict


def build_condensing_transformer(i, t, node_dict, outflow_args_el):
    r"""Build a regular condensing transformer

    Parameters
    ----------
    i : :obj:`str`
        label of current transformer (within iteration)

    t : :obj:`pd.Series`
        pd.Series containing attributes for transformer component
        (row-wise data entries)

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    outflow_args_el: :obj:`dict`
        Dictionary holding the values for electrical outflow arguments

    Returns
    -------
    node_dict[i] : `transformer <oemof.solph.components.Transformer>`
        The transformer element to be added to the dict of nodes
        as i-th element
    """
    node_dict[i] = solph.components.Transformer(
        label=i,
        inputs={node_dict[t["from"]]: solph.flows.Flow()},
        outputs={node_dict[t["to_el"]]: solph.flows.Flow(**outflow_args_el)},
        conversion_factors={node_dict[t["to_el"]]: t["efficiency_el"]},
    )

    return node_dict[i]


def create_exogenous_transformers(
    input_data,
    im,
    node_dict,
):
    """Create exogeneous transformers and add them to the dict of nodes

    exogeneous transformers (fleets) are created for which no investments
    are considered and which are instead phased in / out by increasing / reducing
    capacities based on commissioning date / unit age

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the exogenous transformer elements
    """
    for i, t in input_data["exogenous_transformers"].iterrows():
        # HACK: Use annual capacity values and max timeseries
        # to depict commissioning of new / decommissioning existing units

        # TODO: Define minimum investment requirement for CHP units (also heat pumps etc as providers)
        maximum = (
            (
                input_data["transformers_exogenous_max_ts"].loc[
                    im.start_time : im.end_time, i
                ]
            )
            .mul(im.multiplier)
            .to_numpy()
        )

        outflow_args_el = {
            # Nominal value is the maximum capacity
            # Actual existing capacity is derived from max time series
            "nominal_value": t["capacity_max"],
            "variable_costs": (
                input_data["costs_operation_ts"]
                .loc[im.start_time : im.end_time, t["tech_fuel"]]
                .to_numpy()
            ),
            "min": t["min_load_factor"],
            "max": maximum,
        }

        # Assign minimum loads for German CHP and IPP plants
        # TODO: Create aggregated minimum load profiles!
        if t["type"] == "chp":
            if t["fuel"] in ["natgas", "hardcoal", "lignite"]:
                outflow_args_el["min"] = (
                    input_data["transformers_minload_ts"]
                    .loc[
                        im.start_time : im.end_time,
                        "chp_" + t["fuel"],
                    ]
                    .to_numpy()
                )
            else:
                outflow_args_el["min"] = (
                    input_data["transformers_minload_ts"]
                    .loc[
                        im.start_time : im.end_time,
                        "chp",
                    ]
                    .to_numpy()
                )

        if t["type"] == "ipp":
            outflow_args_el["min"] = (
                input_data["transformers_minload_ts"]
                .loc[
                    im.start_time : im.end_time,
                    "ipp",
                ]
                .to_numpy()
            )

        # Correct minimum load by maximum capacities of particular time
        outflow_args_el["min"] *= (
            input_data["transformers_exogenous_max_ts"]
            .loc[
                im.start_time : im.end_time,
                i,
            ]
            .to_numpy()
        )

        node_dict[i] = build_condensing_transformer(
            i, t, node_dict, outflow_args_el
        )

    return node_dict


def create_new_built_transformers(
    input_data,
    im,
    node_dict,
):
    """Create new-built transformers and add them to the dict of nodes

    New built units are modelled as power plant fleets per energy carrier /
    technology.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the new-built transformer elements
    """
    for i, t in input_data["new_built_transformers"].iterrows():

        # overall maximum is the amount of capacity that can be installed
        # at maximum, i.e. the potential limit of a given technology
        overall_maximum = t["overall_maximum"]
        annual_invest_limit = t["max_invest"]

        if (
            not im.impose_investment_maxima
            and "water" in i
            or im.impose_investment_maxima
        ):
            invest_max = np.min(
                [
                    overall_maximum,
                    annual_invest_limit * im.optimization_timeframe,
                ]
            )
        else:
            invest_max = float(1e10)

        invest_kwargs = {
            "maximum": invest_max,
            "existing": 0,
            # New built plants are installed at capacity costs for start year
            # (of each myopic iteration = investment possibility)
            "ep_costs": economics.annuity(
                capex=input_data["costs_investment"].loc[
                    f"{im.start_year}-01-01", t["tech_fuel"]
                ],
                n=t["unit_lifetime"],
                wacc=input_data["wacc"].loc[t["tech_fuel"], "wacc in p.u."],
            ),
        }
        if im.multi_period:
            invest_max = annual_invest_limit
            invest_kwargs["maximum"] = invest_max
            invest_kwargs["ep_costs"] = np.array(
                input_data["costs_investment"].loc[
                    im.start_year : im.end_year,
                    t["tech_fuel"],
                ]
            )
            multi_period_invest_kwargs = {
                "lifetime": t["unit_lifetime"],
                "age": 0,
                "interest_rate": input_data["interest_rate"].loc["value"][0],
                "fixed_costs": input_data["fixed_costs"].loc[
                    t["tech_fuel"], "fixed_costs_percent_per_year"
                ],
                "overall_maximum": overall_maximum,
            }
            invest_kwargs = {**invest_kwargs, **multi_period_invest_kwargs}

        # TODO: Define minimum investment / output requirement for CHP units
        # (also heat pumps etc as providers of district heating)
        outflow_args_el = {
            "variable_costs": (
                input_data["costs_operation_ts"].loc[
                    im.start_time : im.end_time,
                    t["tech_fuel"],
                ]
            ).to_numpy(),
            "min": t["min_load_factor"],
            "max": t["max_load_factor"],
            "investment": solph.Investment(**invest_kwargs),
        }

        node_dict[i] = build_condensing_transformer(
            i, t, node_dict, outflow_args_el
        )

    return node_dict


def create_new_built_transformers_myopic_horizon(
    input_data,
    im,
    node_dict,
    iteration_results,
):
    """Create new-built transformers and add them to the dict of nodes

    New built units are modelled as power plant fleets per energy carrier /
    technology.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    iteration_results : dict
        A dictionary holding the results of the previous myopic horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the new-built transformer elements

    new_built_transformer_labels : list
        Labels of new-built transformers
    """
    # Use a list to store capacities installed for new built alternatives
    new_built_transformer_labels = []

    for i, t in input_data["new_built_transformers"].iterrows():

        new_built_transformer_labels.append(i)

        if not iteration_results["new_built_transformers"].empty:
            existing_capacity = iteration_results[
                "new_built_transformers"
            ].loc[i, "existing_capacity"]
        else:
            # Set existing capacity for 0th iteration
            existing_capacity = 0

        overall_maximum = t["overall_maximum"]
        annual_invest_limit = t["max_invest"]

        # invest_max is the amount of capacity that can maximally be installed
        # within the optimization timeframe (including existing capacity)
        if (
            not im.impose_investment_maxima
            and "water" in i
            or im.impose_investment_maxima
        ):
            invest_max = np.min(
                [
                    overall_maximum,
                    annual_invest_limit * im.years_per_timeslice,
                ]
            )
        else:
            invest_max = 10000000.0

        outflow_args_el = {
            "variable_costs": (
                input_data["operation_costs_ts"].loc[
                    im.start_time : im.end_time,
                    ("operation_costs", t["bus_technology"]),
                ]
                * input_data["operation_costs"].loc[
                    t["bus_technology"], "2020"
                ]
            ).to_numpy(),
            "min": t["min"],
            "max": t["max"],
            "investment": solph.Investment(
                maximum=invest_max,
                # New built plants are installed at capacity costs for the start year
                # (of each myopic iteration = investment possibility)
                ep_costs=economics.annuity(
                    capex=input_data["investment_costs"].loc[
                        t["bus_technology"], im.start_year
                    ],
                    n=t["unit_lifetime"],
                    wacc=input_data["wacc"].loc[
                        t["tech_fuel"], "wacc in p.u."
                    ],
                ),
                existing=existing_capacity,
            ),
        }

        node_dict[i] = build_condensing_transformer(
            i, t, node_dict, outflow_args_el
        )

    return node_dict, new_built_transformer_labels


def create_exogenous_storages(input_data, im, node_dict):
    r"""Create exogenous storages and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements
    """
    for i, s in input_data["exogenous_storages_el"].iterrows():

        if s["type"] == "phes":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={
                    node_dict[s["bus_inflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_pump"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                    )
                },
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=s["initial_storage_level"],
                max_storage_level=s["max_storage_level"],
                min_storage_level=s["min_storage_level"],
                inflow_conversion_factor=s["efficiency_pump"],
                outflow_conversion_factor=s["efficiency_turbine"],
                balanced=True,
            )

        if s["type"] == "reservoir":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={node_dict[s["bus_inflow"]]: solph.flows.Flow()},
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                        min=s["min_load_factor"],
                        max=s["max_load_factor"],
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=s["initial_storage_level"],
                max_storage_level=s["max_storage_level"],
                min_storage_level=s["min_storage_level"],
                inflow_conversion_factor=s["efficiency_pump"],
                outflow_conversion_factor=s["efficiency_turbine"],
                balanced=True,
            )

    return node_dict


def create_exogenous_storages_myopic_horizon(
    input_data, im, node_dict, iteration_results
):
    r"""Create exogenous storages and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmenthModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    iteration_results : dict
        A dictionary holding the results of the previous myopic horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements

    exogenous_storage_labels : list
        list of exogenous storage labels (existing + new-built ones)
    """
    exogenous_storage_labels = []

    for i, s in input_data["storages_el"].iterrows():

        exogenous_storage_labels.append(i)
        if not iteration_results["exogenous_storages"].empty:
            initial_storage_level_last_iteration = (
                iteration_results["exogenous_storages"].loc[
                    i, "initial_storage_level_last_iteration"
                ]
                / s["nominal_storable_energy"]
            )
        else:
            initial_storage_level_last_iteration = s["initial_storage_level"]

        minimum = {}
        maximum = {}
        for key in ["capacity", "pump", "turbine"]:
            minimum[key] = (
                input_data["min_max_ts"].loc[
                    im.start_time : im.end_time, (f"i_{key}", "min")
                ]
            ).to_numpy()
            maximum[key] = (
                input_data["min_max_ts"].loc[
                    im.start_time : im.end_time, (f"i_{key}", "max")
                ]
            ).to_numpy()

        if s["type"] == "phes":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={
                    node_dict[s["bus_inflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_pump"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                        min=minimum["pump"],
                        max=maximum["pump"],
                    )
                },
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                        min=minimum["turbine"],
                        max=maximum["turbine"],
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=initial_storage_level_last_iteration,
                max_storage_level=s["max_storage_level"] * maximum["turbine"],
                min_storage_level=s["min_storage_level"] * minimum["turbine"],
                inflow_conversion_factor=s["efficiency_pump"],
                outflow_conversion_factor=s["efficiency_turbine"],
                balanced=True,
            )

        if s["type"] == "reservoir":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={node_dict[s["bus_inflow"]]: solph.flows.Flow()},
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[
                                im.start_time : im.end_time, i
                            ]
                        ).to_numpy(),
                        min=s["min_load_factor"] * minimum["turbine"],
                        max=s["max_load_factor"] * maximum["turbine"],
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=initial_storage_level_last_iteration,
                max_storage_level=s["max_storage_level"] * maximum["capacity"],
                min_storage_level=s["min_storage_level"] * minimum["capacity"],
                inflow_conversion_factor=s["efficiency_pump"],
                outflow_conversion_factor=s["efficiency_turbine"],
                balanced=True,
            )

    return node_dict, exogenous_storage_labels


def create_new_built_storages(input_data, im, node_dict):
    """Create new-built storages and add them to the dict of nodes

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmentModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements

    new_built_storage_labels : list
        list of new-built storage labels
    """
    for i, s in input_data["new_built_storages_el"].iterrows():
        # overall invest limit is the amount of capacity that can be installed
        # at maximum, i.e. the potential limit of a given technology
        overall_maximum_pump = s["overall_invest_limit_pump"]
        overall_maximum_turbine = s["overall_invest_limit_turbine"]
        overall_maximum = s["overall_invest_limit"]

        annual_invest_limit_pump = s["max_invest_pump"]
        annual_invest_limit_turbine = s["max_invest_turbine"]
        annual_invest_limit = s["max_invest"]

        if im.impose_investment_maxima:
            invest_max_pump = np.min(
                [
                    overall_maximum_pump,
                    annual_invest_limit_pump * im.optimization_timeframe,
                ]
            )
            invest_max_turbine = np.min(
                [
                    overall_maximum_turbine,
                    annual_invest_limit_turbine * im.optimization_timeframe,
                ]
            )
            invest_max = np.min(
                [
                    overall_maximum,
                    annual_invest_limit * im.optimization_timeframe,
                ]
            )
        else:
            invest_max_pump = 10000000.0
            invest_max_turbine = 10000000.0
            invest_max = 10000000.0

        wacc = input_data["wacc"].loc[
            f"storage_el_{s['type']}", "wacc in p.u."
        ]

        invest_kwargs = {
            "inflow": {
                "maximum": invest_max_pump,
                "ep_costs": economics.annuity(
                    # Adjust capex by storage inflow efficiency
                    # (more inflow capacity needs to be build)
                    capex=input_data["costs_storages_investment_power"].loc[
                        f"{im.start_year}-01-01", f"storage_el_{s['type']}"
                    ]
                    * s["efficiency_pump"],
                    n=s["unit_lifetime_pump"],
                    wacc=wacc,
                ),
                "existing": 0,
            },
            "outflow": {
                "maximum": invest_max_turbine,
                "ep_costs": economics.annuity(
                    # Use symbolic cost values and attribute actual costs
                    # to storage capacity itself resp. inflow
                    capex=1e-8,
                    n=s["unit_lifetime_turbine"],
                    wacc=wacc,
                ),
                "existing": 0,
            },
            "capacity": {
                "maximum": invest_max,
                "ep_costs": economics.annuity(
                    capex=input_data["costs_storages_investment_capacity"].loc[
                        f"{im.start_year}-01-01", f"storage_el_{s['type']}"
                    ],
                    n=s["unit_lifetime"],
                    wacc=wacc,
                ),
                "existing": 0,
            },
        }

        if im.multi_period:
            invest_max_pump = annual_invest_limit_pump
            invest_max_turbine = annual_invest_limit_turbine
            invest_max = annual_invest_limit
            invest_kwargs["inflow"]["maximum"] = invest_max_pump
            invest_kwargs["outflow"]["maximum"] = invest_max_turbine
            invest_kwargs["capacity"]["maximum"] = invest_max

            invest_kwargs["inflow"]["ep_costs"] = (
                input_data["costs_storages_investment_power"].loc[
                    f"{im.start_year}-01-01":f"{im.end_year}-01-01",
                    f"storage_el_{s['type']}",
                ]
                * s["efficiency_pump"]
            ).to_numpy()
            invest_kwargs["outflow"]["ep_costs"] = 1e-8
            invest_kwargs["capacity"]["ep_costs"] = (
                input_data["costs_storages_investment_capacity"].loc[
                    f"{im.start_year}-01-01":f"{im.end_year}-01-01",
                    f"storage_el_{s['type']}",
                ]
            ).to_numpy()

            multi_period_invest_kwargs = {
                "inflow": {
                    "lifetime": s["unit_lifetime_pump"],
                    "age": 0,
                    "interest_rate": input_data["interest_rate"].loc["value"][
                        0
                    ],
                    # "fixed_costs": s["fixed_costs_pump"],
                    "overall_maximum": overall_maximum_pump,
                },
                "outflow": {
                    "lifetime": s["unit_lifetime_turbine"],
                    "age": 0,
                    "interest_rate": input_data["interest_rate"].loc["value"][
                        0
                    ],
                    # "fixed_costs": s["fixed_costs_turbine"],
                    "overall_maximum": overall_maximum_turbine,
                },
                "capacity": {
                    "lifetime": s["unit_lifetime"],
                    "age": 0,
                    "interest_rate": input_data["interest_rate"].loc["value"][
                        0
                    ],
                    "fixed_costs": input_data["fixed_costs"].loc[
                        f"storage_el_{s['type']}",
                        "fixed_costs_percent_per_year",
                    ],
                    "overall_maximum": overall_maximum_turbine,
                },
            }

            invest_kwargs = {
                key: {**invest_kwargs[key], **multi_period_invest_kwargs[key]}
                for key in invest_kwargs.keys()
            }

        node_dict[i] = solph.components.GenericStorage(
            label=i,
            inputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages_ts"].loc[
                            im.start_time : im.end_time, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(**invest_kwargs["inflow"]),
                )
            },
            outputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages_ts"].loc[
                            im.start_time : im.end_time, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(**invest_kwargs["outflow"]),
                )
            },
            loss_rate=s["loss_rate"],
            max_storage_level=s["max_storage_level"],
            min_storage_level=s["min_storage_level"],
            inflow_conversion_factor=s["efficiency_pump"],
            outflow_conversion_factor=s["efficiency_turbine"],
            invest_relation_input_output=s["invest_relation_input_output"],
            invest_relation_input_capacity=s["invest_relation_input_capacity"],
            invest_relation_output_capacity=s[
                "invest_relation_output_capacity"
            ],
            investment=solph.Investment(**invest_kwargs),
        )

    return node_dict


def create_new_built_storages_myopic_horizon(
    input_data,
    im,
    node_dict,
    iteration_results,
):
    """Create new-built storages and add them to the dict of nodes

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmenthModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    iteration_results : dict
        A dictionary holding the results of the previous myopic horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements

    new_built_storage_labels : list
        list of new-built storage units
    """
    new_built_storage_labels = []

    for i, s in input_data["new_built_storages"].iterrows():

        if not iteration_results["new_built_storages"].empty:
            existing_pump = iteration_results["new_built_storages"].loc[
                i, "existing_inflow_power"
            ]
            existing_turbine = iteration_results["new_built_storages"].loc[
                i, "existing_outflow_power"
            ]
            existing = iteration_results["new_built_storages"].loc[
                i, "existing_capacity_storage"
            ]

        else:
            # Set values for 0th iteration (empty DataFrame)
            existing_pump = 0
            existing_turbine = 0
            existing = 0

        # Prevent potential zero division for storage states
        if existing != 0:
            initial_storage_level_last_iteration = (
                iteration_results["new_built_storages"].loc[
                    i, "initial_storage_level_last_iteration"
                ]
                / existing
            )
        else:
            initial_storage_level_last_iteration = s["initial_storage_level"]

        # overall invest limit is the amount of capacity that can at maximum be installed
        # i.e. the potential limit of a given technology
        overall_maximum_pump = s["overall_maximum_pump"]
        overall_maximum_turbine = s["overall_maximum_turbine"]
        overall_maximum = s["overall_maximum"]

        annual_invest_limit_pump = s["max_invest_pump"]
        annual_invest_limit_turbine = s["max_invest_turbine"]
        annual_invest_limit = s["max_invest"]

        # invest_max is the amount of capacity that can maximally be installed
        # within the optimization timeframe
        if im.impose_investment_maxima:
            invest_max_pump = np.min(
                [
                    overall_maximum_pump,
                    annual_invest_limit_pump * im.years_per_timeslice,
                ]
            )
            invest_max_turbine = np.min(
                [
                    overall_maximum_turbine,
                    annual_invest_limit_turbine * im.years_per_timeslice,
                ]
            )
            invest_max = np.min(
                [
                    overall_maximum,
                    annual_invest_limit * im.years_per_timeslice,
                ]
            )
        else:
            invest_max_pump = 10000000.0
            invest_max_turbine = 10000000.0
            invest_max = 10000000.0

        node_dict[i] = solph.components.GenericStorage(
            label=i,
            inputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages_ts"].loc[
                            im.start_time : im.end_time, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_pump,
                        ep_costs=economics.annuity(
                            capex=input_data["costs_storages_investment"].loc[
                                i + "_pump", im.start_year
                            ],
                            n=s["unit_lifetime_pump"],
                            wacc=input_data["wacc"].loc[i, im.start_year],
                        ),
                        existing=existing_pump,
                    ),
                )
            },
            outputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages_ts"].loc[
                            im.start_time : im.end_time, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_turbine,
                        ep_costs=economics.annuity(
                            capex=input_data["costs_storages_investment"].loc[
                                i + "_turbine", im.start_year
                            ],
                            n=s["unit_lifetime_turbine"],
                            wacc=input_data["wacc"].loc[i, im.start_year],
                        ),
                        existing=existing_turbine,
                    ),
                )
            },
            loss_rate=s["loss_rate"] * s["max_storage_level"],
            initial_storage_level=initial_storage_level_last_iteration,
            balanced=True,
            max_storage_level=s["max_storage_level"],
            min_storage_level=s["min_storage_level"],
            inflow_conversion_factor=s["efficiency_pump"],
            outflow_conversion_factor=s["efficiency_turbine"],
            invest_relation_input_output=s["invest_relation_input_output"],
            invest_relation_input_capacity=s["invest_relation_input_capacity"],
            invest_relation_output_capacity=s[
                "invest_relation_output_capacity"
            ],
            investment=solph.Investment(
                maximum=invest_max,
                ep_costs=economics.annuity(
                    capex=input_data["costs_storages_investment"].loc[
                        i, im.start_year
                    ],
                    n=s["unit_lifetime"],
                    wacc=input_data["wacc"].loc[i, im.start_year],
                ),
                existing=existing,
            ),
        )

    return node_dict, new_built_storage_labels
