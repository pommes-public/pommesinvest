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
import pandas as pd
import numpy as np

from oemof.tools import economics
import oemof.solph as solph

from pommesinvest.model_funcs.helpers import resample_timeseries, calc_annuity


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
    df = pd.DataFrame()
    if not "_ts" in filename:
        df = pd.read_csv(im.path_folder_input + filename + ".csv", index_col=0)
    else:
        df = load_time_series_data_slice(filename, im)

    if "country" in df.columns and im.countries is not None:
        df = df[df["country"].isin(im.countries)]

    if df.isna().any().any() and "_ts" in filename:
        print(f"Attention! Time series input data file " f"{filename} contains NaNs.")
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
    ts_start = pd.read_csv(
        im.path_folder_input + filename,
        parse_dates=True,
        index_col=0,
        usecols=[0],
    )
    start_index = pd.Index(ts_start.index).get_loc(im.starttime)
    timeseries_end = pd.Timestamp(im.endtime, im.freq)
    end_index = pd.Index(ts_start.index).get_loc(
        (timeseries_end + im.overlap_in_timesteps * timeseries_end.freq).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    df = pd.read_csv(
        im.path_folder_input + filename,
        parse_dates=True,
        index_col=0,
        skiprows=range(1, start_index),
        nrows=end_index - start_index + 1,
    )

    return df


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
                        input_data["costs_fuel"].loc[i, 2020]
                        * np.array(
                            input_data["costs_fuel_ts"]["price"][
                                im.start_time : im.end_time
                            ]
                        )
                        + input_data["costs_emissions"].loc[i, 2020]
                        * np.array(
                            input_data["costs_emissions_ts"]["price"][
                                im.start_time : im.end_time
                            ]
                        )
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
                node_dict[s["to"]]: solph.flows.Flow(variable_costs=s["shortage_costs"])
            },
        )

    return node_dict


# TODO: Normalize infeed timeseries for RES (2020 = 1.0)
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
                        nominal_value=re["capacity_2020"],
                    )
                },
            )
        except KeyError:
            print(re)

    return node_dict


# NOTE: As long as renewables capacity is given model endogeneously, there is no reason why
# another approach as for transformers is applied. Methods could be combined to one.
def renewables_exo(
    renewables_com_df,
    investment_costs_df,
    WACC_df,
    startyear,
    endyear,
    IR=0.02,
    discount=False,
):
    """Function calculates dicounted investment costs for exogenously
    new built renewables from renewables_exo_commissioning_df
    which are built after the model-endogeneous investment year. Furthermore the commissioned
    and decommissioned capacity from startyear until endyear for all new renewables
    are stored in a seperate DataFrame

    Parameters
    ----------
    renewables_com_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the renewables' data

    investment_costs_df: :obj:`pandas.DataFrame`
        pd.DataFrame containing the Investment costs for new built renewables

    WACC_df : :obj:`pd.DataFrame`
        pd.DataFrame containing the WACC data

    startyear : :obj:`int`
        Starting year of the overall optimization run

    endyear : :obj:`int`
        End year of the overall optimization run

    IR : :obj:`float`
        The interest rate to be applied for discounting (only if discount is True)

    discount : :obj:`boolean`
        If True, nominal values will be dicounted
        If False, real values have to be used as model inputs (default)

    Returns
    -------
    renewables_exo_com_costs_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing total discounted cost for new built renewables
        from startyear until endyear

    renewables_exo_com_capacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the commissioned capacity of all exogenously new built
        renewables in between startyear and endyear

    renewables_exo_decom_capacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the decommissioned capacity of all exogenously new built
        renewables in between startyear and endyear
    """

    renewables_exo_com_costs_df = pd.DataFrame(index=renewables_com_df.index)
    renewables_exo_com_capacity_df = pd.DataFrame(index=renewables_com_df.index)
    renewables_exo_decom_capacity_df = pd.DataFrame(index=renewables_com_df.index)

    # Use temporary DataFrames with transformer type as index,
    # merge these together and apply function calc_annuity
    # implementation of oemof.economics.annuity is not applicable for pd.Series
    costs_df = pd.merge(
        investment_costs_df,
        WACC_df,
        on="transformer_type",
        suffixes=["_invest", "_WACC"],
    )
    all_cols_df = pd.merge(
        renewables_com_df,
        costs_df,
        left_on=renewables_com_df.index,
        right_on="transformer_type",
    ).set_index("transformer_type")

    for year in range(startyear, endyear + 1):
        year_str = str(year)
        renewables_exo_com_costs_df["exo_cost_" + year_str] = renewables_com_df[
            "commissioned_" + year_str
        ] * calc_annuity(
            capex=all_cols_df[year_str + "_invest"],
            n=all_cols_df["unit_lifetime"],
            wacc=all_cols_df[year_str + "_WACC"],
        )

        if discount:
            renewables_exo_com_costs_df[
                "exo_cost_" + year_str
            ] = renewables_exo_com_costs_df["exo_cost_" + year_str].div(
                ((1 + IR) ** (year - startyear))
            )

        # extract commissioning resp. decommissioning data
        renewables_exo_com_capacity_df["commissioned_" + year_str] = renewables_com_df[
            "commissioned_" + year_str
        ]
        renewables_exo_decom_capacity_df[
            "decommissioned_" + year_str
        ] = renewables_com_df["decommissioned_" + year_str]

    return (
        renewables_exo_com_costs_df,
        renewables_exo_com_capacity_df,
        renewables_exo_decom_capacity_df,
    )


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
                        input_data["sinks_demand_el_ts"][i][im.start_time : im.end_time]
                    ),
                    nominal_value=d["maximum_2020"],
                )
            },
        }

        # Adjusted demand here means the difference between overall demand
        # and the baseline load profile for demand response units
        if im.activate_demand_response and i == "DE_sink_el_load":
            kwargs_dict["inputs"] = {
                node_dict[d["from"]]: solph.flows.Flow(
                    fix=np.array(
                        input_data["sinks_demand_el_ts"][i][im.start_time : im.end_time]
                        .mul(d["maximum_2020"])
                        .sub(dr_overall_load_ts_df[im.start_time : im.end_time])
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
                input_data["sinks_dr_el_ts"][i].loc[im.start_time : im.end_time]
            ),
            "capacity_up": np.array(
                input_data["sinks_dr_el_ava_neg_ts"][i].loc[im.start_time : im.end_time]
            ),
            "flex_share_up": d["FLEX_SHARE_UP"],
            "capacity_down": np.array(
                input_data["sinks_dr_el_ava_pos_ts"][i].loc[im.start_time : im.end_time]
            ),
            "flex_share_down": d["FLEX_SHARE_DOWN"],
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
                "n_yearLimit_shift": np.max([round(d["maximum_activations_year"]), 1]),
                "n_yearLimit_shed": 1,
                "t_dayLimit": 24,
                "addition": True,
                "fixes": True,
            },
            "oemof": {"approach": "oemof", "shift_interval": 24},
        }

        # TODO: Critically check min and max invest params
        approach_dict = {
            "DLR": solph.components.experimental.SinkDSM(
                label=i,
                inputs={node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)},
                **kwargs_all,
                **kwargs_dict["DLR"],
                invest=solph.Investment(
                    minimum=0,
                    maximum=min(
                        d["max_cap"] + d["potential_neg_overall"], d["installed_cap"]
                    ),
                    ep_costs=d["specific_investments"] * 1e3,
                ),
            ),
            "DIW": solph.components.experimental.SinkDSM(
                label=i,
                inputs={node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)},
                **kwargs_all,
                **kwargs_dict["DIW"],
                invest=solph.Investment(
                    minimum=0,
                    maximum=min(
                        d["max_cap"] + d["potential_neg_overall"], d["installed_cap"]
                    ),
                    ep_costs=d["specific_investments"] * 1e3,
                ),
            ),
            "oemof": solph.components.experimental.SinkDSM(
                label=i,
                inputs={node_dict[d["from"]]: solph.flows.Flow(variable_costs=0)},
                **kwargs_all,
                **kwargs_dict["oemof"],
                invest=solph.Investment(
                    minimum=0,
                    maximum=min(
                        d["max_cap"] + d["potential_neg_overall"], d["installed_cap"]
                    ),
                    ep_costs=d["specific_investments"] * 1e3,
                ),
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
                node_dict[e["from"]]: solph.flows.Flow(variable_costs=e["excess_costs"])
            },
        )

    return node_dict


def build_chp_transformer(i, t, node_dict, outflow_args_el, outflow_args_th):
    r"""Build a CHP transformer (fixed relation heat / power)

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

    outflow_args_th: :obj:`dict`
        Dictionary holding the values for thermal outflow arguments

    Returns
    -------
    node_dict[i] : `transformer <oemof.network.Transformer>`
        The transformer element to be added to the dict of nodes
        as i-th element
    """
    node_dict[i] = solph.components.Transformer(
        label=i,
        inputs={node_dict[t["from"]]: solph.flows.Flow()},
        outputs={
            node_dict[t["to_el"]]: solph.flows.Flow(**outflow_args_el),
            node_dict[t["to_th"]]: solph.flows.Flow(**outflow_args_th),
        },
        conversion_factors={
            node_dict[t["to_el"]]: t["efficiency_el_CC"],
            node_dict[t["to_th"]]: t["efficiency_th_CC"],
        },
    )

    return node_dict[i]


def build_var_chp_units(i, t, node_dict, outflow_args_el, outflow_args_th):
    r"""Build variable CHP units

    These are modeled as extraction turbine CHP units and can choose
    between full condensation mode, full coupling mode
    and any allowed state in between.

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

    outflow_args_th: :obj:`dict`
        Dictionary holding the values for thermal outflow arguments

    Returns
    -------
    node_dict[i] : oemof.solph.components.ExtractionTurbineCHP
        The extraction turbine element to be added to the dict of nodes
        as i-th element
    """
    node_dict[i] = solph.components.ExtractionTurbineCHP(
        label=i,
        inputs={node_dict[t["from"]]: solph.flows.Flow()},
        outputs={
            node_dict[t["to_el"]]: solph.flows.Flow(**outflow_args_el),
            node_dict[t["to_th"]]: solph.flows.Flow(**outflow_args_th),
        },
        conversion_factors={
            node_dict[t["to_el"]]: t["efficiency_el_CC"],
            node_dict[t["to_th"]]: t["efficiency_th_CC"],
        },
        conversion_factor_full_condensation={node_dict[t["to_el"]]: t["efficiency_el"]},
    )

    return node_dict[i]


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


def create_existing_transformers(
    input_data,
    im,
    node_dict,
):
    """Create existing transformers and add them to the dict of nodes

    Existing transformers (fleets) are created for which no investments
    are considered and which are instead phased out by reducing capacities
    based on unit age

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
        the existing transformer elements
    """
    startyear_str = str(pd.to_datetime(im.starttime).year)

    for i, t in input_data["existing_transformers"].iterrows():
        # Don't build units whose lifetime is reached and whose cap is 0
        if t[f"existing_{startyear_str}"] == 0:
            continue

        # Minimum and maximum loads for decommissioning of existing units
        minimum_el_load = (
            input_data["min_max_timeseries"]
            .loc[im.starttime : im.endtime, (i, "min")]
            .to_numpy()
        )
        maximum_el_load = (
            input_data["min_max_timeseries"]
            .loc[im.starttime : im.endtime, (i, "max")]
            .to_numpy()
        )

        outflow_args_el = {
            "nominal_value": t[f"existing_{startyear_str}"],
            "variable_costs": (
                input_data["costs_operation_ts"].loc[
                    im.starttime : im.endtime, ("operation_costs", t["from"])
                ]
                * input_data["costs_operation"].loc[t["from"], "2020"]
            ).to_numpy(),
            "min": minimum_el_load,
            "max": maximum_el_load,
        }

        # Assign minimum loads for German CHP and IPP plants
        # TODO: Create aggregated minimum load profiles!
        if t["type"] == "chp":
            if t["identifier"] in input_data["min_loads_dh"].columns:
                outflow_args_el["min"] = (
                    input_data["min_loads_dh"]
                    .loc[
                        im.start_time : im.end_time,
                        t["identifier"],
                    ]
                    .to_numpy()
                )
            elif t["fuel"] in ["natgas", "hardcoal", "lignite"]:
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
            if t["identifier"] in input_data["min_loads_ipp"].columns:
                outflow_args_el["min"] = (
                    input_data["min_loads_ipp"]
                    .loc[
                        im.start_time : im.end_time,
                        t["identifier"],
                    ]
                    .to_numpy()
                )
            else:
                outflow_args_el["min"] = (
                    input_data["transformers_minload_ts"]
                    .loc[
                        im.start_time : im.end_time,
                        "ipp",
                    ]
                    .to_numpy()
                )

    return node_dict


def existing_transformers_exo_decom(
    transformers_decommissioning_df, startyear, endyear
):
    """Function takes the decommissioned capacity of transformers and groups
    them by bus_technology

    Parameters
    ----------
    transformers_decommissioning_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the transformers' decommissioning data

    startyear : :obj:`int`
        Starting year of the optimization run

    endyear : :obj:`int`
        End year of the optimization run

    Returns
    -------
    existing_transformers_exo_decom_capacity_df : :obj:`pandas.DataFrame
        DataFrame containing the decommissioned capacity of each bus_technology
        from the existing transformers from startyear until endyear.
    """
    existing_transformers_exo_decom_capacity_df = (
        transformers_decommissioning_df.groupby("bus_technology").sum()
    )
    existing_transformers_exo_decom_capacity_df = (
        existing_transformers_exo_decom_capacity_df.loc[
            :, "decommissioned_" + str(startyear) : "decommissioned_" + str(endyear)
        ]
    )

    return existing_transformers_exo_decom_capacity_df


def calc_exist_cap(t, startyear, endyear, col_name):
    """Calculate existing capacity for a given oemof element / column name"""
    # JF 10.01.2020 initialize existing_capacity
    existing_capacity = t[col_name + str(startyear)]
    # JF 10.01.2020 iterate over all upcoming years and increase increase exisiting_capaacity by the exisiting capacity of each year
    for year in range(startyear + 1, endyear + 1):
        existing_capacity += t[col_name + str(year)] - t[col_name + str(year - 1)]
    if existing_capacity <= t[col_name + str(startyear)]:
        existing_capacity = t[col_name + str(startyear)]
    return existing_capacity


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

        existing_capacity = calc_exist_cap(t, im.startyear, im.endyear, "existing_")
        # overall invest limit is the amount of capacity that can be installed
        # at maximum, i.e. the potential limit of a given technology
        overall_invest_limit = t["overall_invest_limit"]
        annual_invest_limit = t["max_invest"]

        max_bound_transformer = overall_invest_limit - existing_capacity
        if overall_invest_limit - existing_capacity <= 0:
            max_bound_transformer = 0

        if (
            not im.impose_investment_maxima
            and "water" in i
            or im.impose_investment_maxima
        ):
            invest_max = np.min(
                [
                    max_bound_transformer,
                    annual_invest_limit * im.optimization_timeframe,
                ]
            )
        else:
            invest_max = 10000000.0

        # TODO: Define minimum investment requirement for CHP units (also heat pumps etc as providers)
        minimum = (
            input_data["min_max_timeseries"].loc[im.starttime : im.endtime, (i, "min")]
        ).to_numpy()
        maximum = (
            input_data["min_max_timeseries"].loc[im.starttime : im.endtime, (i, "max")]
        ).to_numpy()

        outflow_args_el = {
            "variable_costs": (
                input_data["operation_costs_ts"].loc[
                    im.starttime : im.endtime, ("operation_costs", t["bus_technology"])
                ]
                * input_data["operation_costs"].loc[t["bus_technology"], "2020"]
            ).to_numpy(),
            "min": minimum,
            "max": maximum,
            "investment": solph.Investment(
                maximum=invest_max,
                # New built plants are installed at capacity costs for the start year
                # (of each myopic iteration = investment possibility)
                ep_costs=economics.annuity(
                    capex=input_data["investment_costs"].loc[
                        t["bus_technology"], im.startyear
                    ],
                    n=t["unit_lifetime"],
                    wacc=input_data["wacc"].loc[t["bus_technology"], im.startyear],
                ),
                existing=existing_capacity,
            ),
        }

        node_dict[i] = build_condensing_transformer(i, t, node_dict, outflow_args_el)

    return node_dict


def create_new_built_transformers_rolling_horizon(
    input_data,
    im,
    node_dict,
    iteration_results,
):
    """FCreate new-built transformers and add them to the dict of nodes

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
        A dictionary holding the results of the previous rolling horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem including
        the new-built transformer elements
    """
    # Use a list to store capacities installed for new built alternatives
    new_built_transformer_labels = []
    # TODO: Move this df one or two levels up and simplify!
    endo_exo_exist_df = pd.DataFrame(
        columns=["existing_capacity_endogenously", "old_capacity_exogenously"],
        index=input_data["new_built_transformers"].index.values,
    )

    for i, t in input_data["new_built_transformers"].iterrows():

        new_built_transformer_labels.append(i)

        exo_exist = calc_exist_cap(t, im.startyear, im.endyear, "existing_")

        if not iteration_results["new_built_transformers"].empty:
            existing_capacity = (
                exo_exist
                + iteration_results["new_built_transformers"].loc[
                    i, "endogenous_capacity_cumulated"
                ]
            )
        else:
            # Set existing capacity for 0th iteration
            existing_capacity = exo_exist

        endo_exo_exist_df.loc[i, "old_capacity_exogenously"] = exo_exist
        endo_exo_exist_df.loc[i, "existing_capacity_endogenously"] = (
            existing_capacity - exo_exist
        )

        overall_invest_limit = t["overall_invest_limit"]
        annual_invest_limit = t["max_invest"]

        max_bound_transformer = overall_invest_limit - existing_capacity
        if overall_invest_limit - existing_capacity <= 0:
            max_bound_transformer = 0
        # invest_max is the amount of capacity that can maximally be installed
        # within the optimization timeframe (including existing capacity)
        if (
            not im.impose_investment_maxima
            and "water" in i
            or im.impose_investment_maxima
        ):
            invest_max = np.min(
                [
                    max_bound_transformer,
                    annual_invest_limit * im.years_per_timeslice,
                ]
            )
        else:
            invest_max = 10000000.0

        minimum = (
            input_data["min_max_timeseries"].loc[im.starttime : im.endtime, (i, "min")]
        ).to_numpy()
        maximum = (
            input_data["min_max_timeseries"].loc[im.starttime : im.endtime, (i, "max")]
        ).to_numpy()

        outflow_args_el = {
            "variable_costs": (
                input_data["operation_costs_ts"].loc[
                    im.starttime : im.endtime, ("operation_costs", t["bus_technology"])
                ]
                * input_data["operation_costs"].loc[t["bus_technology"], "2020"]
            ).to_numpy(),
            "min": minimum,
            "max": maximum,
            "investment": solph.Investment(
                maximum=invest_max,
                # New built plants are installed at capacity costs for the start year
                # (of each myopic iteration = investment possibility)
                ep_costs=economics.annuity(
                    capex=input_data["investment_costs"].loc[
                        t["bus_technology"], im.startyear
                    ],
                    n=t["unit_lifetime"],
                    wacc=input_data["wacc"].loc[t["bus_technology"], im.startyear],
                ),
                existing=existing_capacity,
            ),
        }

        node_dict[i] = build_condensing_transformer(i, t, node_dict, outflow_args_el)

    return node_dict, new_built_transformer_labels, endo_exo_exist_df


def new_transformers_exo(
    new_transformers_de_com_df,
    investment_costs_df,
    WACC_df,
    startyear,
    endyear,
    IR=0.02,
    discount=False,
):
    """Function calculates dicounted investment costs from exogenously
    new built transformers from new_transformers_de_com_df
    which are built after the model-endogeneous investment year. Furthermore, the commissioned
    and decommissioned capacity from startyear until endyear for each new built
    transformer are stored in a seperate DataFrame

    Parameters
    ----------
    new_transformers_de_com_df : :obj:`pandas.DataFrame`
        pd.DataFrame containing the new built transformers' (de)commissioning data

    investment_costs_df: :obj:`pandas.DataFrame`
        pd.DataFrame containing the Investment costs for new built transformers

    WACC_df : :obj:`pd.DataFrame`
        pd.DataFrame containing the WACC data

    startyear : :obj:`int`
        Starting year of the optimization run

    endyear : :obj:`int`
        End year of the optimization run

    IR : :obj:`float`
        The interest rate to be applied for discounting (only if discount is True)

    discount : :obj:`boolean`
        If True, nominal values will be dicounted
        If False, real values have to be used as model inputs (default)

    Returns
    -------
    new_transformers_exo_com_costs_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing total discounted cost for new built transformers
        from startyear until endyear

    new_transformers_exo_com_cacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the commissioned capacity of all exogenously new built
        transformers in between startyear and endyear

    new_transformers_exo_decom_capacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the decommissioned capacity of all exogenously new built
        transformers in between startyear and endyear

    """

    new_transformers_exo_com_costs_df = pd.DataFrame(
        index=new_transformers_de_com_df["bus_technology"]
    )
    new_transformers_exo_com_costs_df[
        "labels"
    ] = new_transformers_de_com_df.index.values

    # Use temporary DataFrames with bus technology as index,
    # merge these together and apply function calc_annuity
    # implementation of oemof.economics.annuity is not applicable for pd.Series
    costs_df = pd.merge(
        investment_costs_df,
        WACC_df,
        on="transformer_type",
        suffixes=["_invest", "_WACC"],
    )
    all_cols_df = pd.merge(
        new_transformers_de_com_df,
        costs_df,
        left_on="bus_technology",
        right_on="transformer_type",
    ).set_index("bus_technology")

    for year in range(startyear, endyear + 1):
        year_str = str(year)
        new_transformers_exo_com_costs_df["exo_cost_" + year_str] = all_cols_df[
            "commissioned_" + year_str
        ] * calc_annuity(
            capex=all_cols_df[year_str + "_invest"],
            n=all_cols_df["unit_lifetime"],
            wacc=all_cols_df[year_str + "_WACC"],
        )

        if discount:
            new_transformers_exo_com_costs_df[
                "exo_cost_" + year_str
            ] = new_transformers_exo_com_costs_df["exo_cost_" + year_str].div(
                ((1 + IR) ** (year - startyear))
            )

    new_transformers_exo_com_costs_df = new_transformers_exo_com_costs_df.reset_index(
        drop=True
    ).set_index("labels")

    # extract commissioning resp. decommissioning data
    new_transformers_exo_com_cacity_df = new_transformers_de_com_df.loc[
        :, "commissioned_" + str(startyear) : "commissioned_" + str(endyear)
    ]
    new_transformers_exo_decom_capacity_df = new_transformers_de_com_df.loc[
        :, "decommissioned_" + str(startyear) : "decommissioned_" + str(endyear)
    ]

    return (
        new_transformers_exo_com_costs_df,
        new_transformers_exo_com_cacity_df,
        new_transformers_exo_decom_capacity_df,
    )


def create_existing_storages(input_data, im, node_dict):
    r"""Create existing storages and add them to the dict of nodes.

    Parameters
    ----------
    input_data: :obj:`dict` of :class:`pd.DataFrame`
        The input data given as a dict of DataFrames
        with component names as keys

    im : :class:`InvestmenthModel`
        The investment model that is considered

    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Dictionary containing all nodes of the EnergySystem

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements
    """
    for i, s in input_data["storages_el"].iterrows():

        if s["type"] == "phes":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={
                    node_dict[s["bus_inflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_pump"],
                        variable_costs=(
                            input_data["costs_operation_storages_ts"].loc[i, 2020]
                            * input_data["costs_operation_storages"].loc[
                                im.starttime : im.endtime, i
                            ]
                        ).to_numpy(),
                    )
                },
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages"].loc[i, 2020]
                            * input_data["costs_operation_storages"].loc[
                                im.starttime : im.endtime, i
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
                            input_data["costs_operation_storages"].loc[i, 2020]
                            * input_data["costs_operation_storages"].loc[
                                im.starttime : im.endtime, i
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


def create_existing_storages_rolling_horizon(
    input_data, im, node_dict, iteration_results
):
    r"""Create existing storages and add them to the dict of nodes.

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
        A dictionary holding the results of the previous rolling horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements
    """
    existing_storage_labels = []

    for i, s in input_data["storages_el"].iterrows():

        existing_storage_labels.append(i)
        if not iteration_results["storages_initial"].empty:
            initial_storage_level_last_iteration = (
                iteration_results["storages_initial"].loc[
                    i, "initial_storage_level_last_iteration"
                ]
                / s["nominal_storable_energy"]
            )
        else:
            initial_storage_level_last_iteration = s["initial_storage_level"]

        if s["type"] == "phes":
            node_dict[i] = solph.components.GenericStorage(
                label=i,
                inputs={
                    node_dict[s["bus_inflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_pump"],
                        variable_costs=(
                            input_data["costs_operation_storages"].loc[i, im.year]
                        ),
                    )
                },
                outputs={
                    node_dict[s["bus_outflow"]]: solph.flows.Flow(
                        nominal_value=s["capacity_turbine"],
                        variable_costs=(
                            input_data["costs_operation_storages"].loc[i, im.year]
                        ),
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=initial_storage_level_last_iteration,
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
                            input_data["costs_operation_storages"].loc[i, im.year]
                        ),
                        min=s["min_load_factor"],
                        max=s["max_load_factor"],
                    )
                },
                nominal_storage_capacity=s["nominal_storable_energy"],
                loss_rate=s["loss_rate"],
                initial_storage_level=initial_storage_level_last_iteration,
                max_storage_level=s["max_storage_level"],
                min_storage_level=s["min_storage_level"],
                inflow_conversion_factor=s["efficiency_pump"],
                outflow_conversion_factor=s["efficiency_turbine"],
                balanced=True,
            )

    return node_dict, existing_storage_labels


def create_new_built_storages(
    input_data,
    im,
    node_dict,
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

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements
    """
    for i, s in input_data["new_built_storages"].iterrows():
        # Add upcoming commissioned capacity to existing
        existing_pump = calc_exist_cap(s, im.startyear, im.endyear, "existing_pump_")
        existing_turbine = calc_exist_cap(
            s, im.startyear, im.endyear, "existing_turbine_"
        )
        existing = calc_exist_cap(s, im.startyear, im.endyear, "existing_")

        # overall invest limit is the amount of capacity that can be installed
        # at maximum, i.e. the potential limit of a given technology
        overall_invest_limit_pump = s["overall_invest_limit_pump"]
        overall_invest_limit_turbine = s["overall_invest_limit_turbine"]
        overall_invest_limit = s["overall_invest_limit"]

        annual_invest_limit_pump = s["max_invest_pump"]
        annual_invest_limit_turbine = s["max_invest_turbine"]
        annual_invest_limit = s["max_invest"]

        max_bound1 = max(overall_invest_limit_pump - existing_pump, 0)
        max_bound2 = max(overall_invest_limit_turbine - existing_turbine, 0)
        max_bound3 = max(overall_invest_limit - existing, 0)

        if im.impose_investment_maxima:
            invest_max_pump = np.min(
                [max_bound1, annual_invest_limit_pump * im.optimization_timeframe]
            )
            invest_max_turbine = np.min(
                [max_bound2, annual_invest_limit_turbine * im.optimization_timeframe]
            )
            invest_max = np.min(
                [max_bound3, annual_invest_limit * im.optimization_timeframe]
            )
        else:
            invest_max_pump = 10000000.0
            invest_max_turbine = 10000000.0
            invest_max = 10000000.0

        wacc = input_data["wacc"].loc[i, im.startyear]

        node_dict[i] = solph.components.GenericStorage(
            label=i,
            inputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages"].loc[i, 2020]
                        * input_data["costs_operation_storages_ts"].loc[
                            im.starttime : im.endtime, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_pump,
                        ep_costs=economics.annuity(
                            capex=input_data["storage_pump_investment_costs"].loc[
                                i, im.startyear
                            ],
                            n=s["unit_lifetime_pump"],
                            wacc=wacc,
                        ),
                        existing=existing_pump,
                    ),
                )
            },
            outputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages"].loc[i, 2020]
                        * input_data["costs_operation_storages_ts"].loc[
                            im.starttime : im.endtime, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_turbine,
                        ep_costs=economics.annuity(
                            capex=input_data["storage_turbine_investment_costs"].loc[
                                i, im.startyear
                            ],
                            n=s["unit_lifetime_turbine"],
                            wacc=wacc,
                        ),
                        existing=existing_turbine,
                    ),
                )
            },
            loss_rate=s["loss_rate"],
            max_storage_level=s["max_storage_level"],
            min_storage_level=s["min_storage_level"],
            inflow_conversion_factor=s["efficiency_pump"],
            outflow_conversion_factor=s["efficiency_turbine"],
            invest_relation_input_output=s["invest_relation_input_output"],
            invest_relation_input_capacity=s["invest_relation_input_capacity"],
            invest_relation_output_capacity=s["invest_relation_output_capacity"],
            investment=solph.Investment(
                maximum=invest_max,
                ep_costs=economics.annuity(
                    capex=input_data["storage_investment_costs"].loc[i, im.startyear],
                    n=s["unit_lifetime"],
                    wacc=wacc,
                ),
                existing=existing,
            ),
        )

    return node_dict


def create_new_built_storages_rolling_horizon(
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
        A dictionary holding the results of the previous rolling horizon
        iteration

    Returns
    -------
    node_dict : :obj:`dict` of :class:`nodes <oemof.network.Node>`
        Modified dictionary containing all nodes of the EnergySystem
        including the storage elements
    """
    new_built_storage_labels = []
    # TODO: Move this df one or two levels up and simplify!
    endo_exo_exist_stor_df = pd.DataFrame(
        columns=[
            "capacity_endo",
            "old_exo_cap",
            "turbine_endo",
            "old_exo_turbine",
            "pump_endo",
            "old_exo_pump",
        ],
        index=input_data["new_built_storages"].index.values,
    )

    for i, s in input_data["new_built_storages"].iterrows():

        new_built_storage_labels.append(i)
        # Get values from previous iteration
        exo_exist_pump = calc_exist_cap(s, im.startyear, im.endyear, "existing_pump_")
        exo_exist_turbine = calc_exist_cap(
            s, im.startyear, im.endyear, "existing_turbine_"
        )
        exo_exist_cap = calc_exist_cap(s, im.startyear, im.endyear, "existing_")

        if not iteration_results["storages_new_built"].empty:
            existing_pump = (
                exo_exist_pump
                + iteration_results["storages_new_built"].loc[
                    i, "existing_inflow_power"
                ]
            )
            existing_turbine = (
                exo_exist_turbine
                + iteration_results["storages_new_built"].loc[
                    i, "existing_outflow_power"
                ]
            )
            existing = (
                exo_exist_cap
                + iteration_results["storages_new_built"].loc[
                    i, "existing_capacity_storage"
                ]
            )

        else:
            # Set values for 0th iteration (empty DataFrame)
            existing_pump = exo_exist_pump
            existing_turbine = exo_exist_turbine
            existing = exo_exist_cap

        # Prevent potential zero division for storage states
        if existing != 0:
            initial_storage_level_last_iteration = (
                iteration_results["storages_new_built"].loc[
                    i, "initial_storage_level_last_iteration"
                ]
                / existing
            )
        else:
            initial_storage_level_last_iteration = s["initial_storage_level"]

        endo_exo_exist_stor_df.loc[i, "old_exo_cap"] = exo_exist_cap
        endo_exo_exist_stor_df.loc[i, "old_exo_turbine"] = exo_exist_turbine
        endo_exo_exist_stor_df.loc[i, "old_exo_pump"] = exo_exist_pump

        endo_exo_exist_stor_df.loc[i, "capacity_endo"] = existing - exo_exist_cap
        endo_exo_exist_stor_df.loc[i, "turbine_endo"] = (
            existing_turbine - exo_exist_turbine
        )
        endo_exo_exist_stor_df.loc[i, "pump_endo"] = existing_pump - exo_exist_pump

        # overall invest limit is the amount of capacity that can at maximum be installed
        # i.e. the potential limit of a given technology
        overall_invest_limit_pump = s["overall_invest_limit_pump"]
        overall_invest_limit_turbine = s["overall_invest_limit_turbine"]
        overall_invest_limit = s["overall_invest_limit"]

        annual_invest_limit_pump = s["max_invest_pump"]
        annual_invest_limit_turbine = s["max_invest_turbine"]
        annual_invest_limit = s["max_invest"]

        max_bound1 = max(overall_invest_limit_pump - existing_pump, 0)
        max_bound2 = max(overall_invest_limit_turbine - existing_turbine, 0)
        max_bound3 = max(overall_invest_limit - existing, 0)

        # invest_max is the amount of capacity that can maximally be installed
        # within the optimization timeframe
        if im.impose_investment_maxima:
            invest_max_pump = np.min(
                [max_bound1, annual_invest_limit_pump * im.years_per_timeslice]
            )
            invest_max_turbine = np.min(
                [max_bound2, annual_invest_limit_turbine * im.years_per_timeslice]
            )
            invest_max = np.min(
                [max_bound3, annual_invest_limit * im.years_per_timeslice]
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
                        input_data["costs_operation_storages"].loc[i, 2020]
                        * input_data["costs_operation_storages_ts"].loc[
                            im.starttime : im.endtime, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_pump,
                        ep_costs=economics.annuity(
                            capex=input_data["costs_storages_investment"].loc[
                                i + "_pump", im.startyear
                            ],
                            n=s["unit_lifetime_pump"],
                            wacc=input_data["wacc"].loc[i, im.startyear],
                        ),
                        existing=existing_pump,
                    ),
                )
            },
            outputs={
                node_dict[s["bus"]]: solph.flows.Flow(
                    variable_costs=(
                        input_data["costs_operation_storages"].loc[i, 2020]
                        * input_data["costs_operation_storages_ts"].loc[
                            im.starttime : im.endtime, i
                        ]
                    ).to_numpy(),
                    max=s["max_storage_level"],
                    investment=solph.Investment(
                        maximum=invest_max_turbine,
                        ep_costs=economics.annuity(
                            capex=input_data["costs_storages_investment"].loc[
                                i + "_turbine", im.startyear
                            ],
                            n=s["unit_lifetime_turbine"],
                            wacc=input_data["wacc"].loc[i, im.startyear],
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
            invest_relation_output_capacity=s["invest_relation_output_capacity"],
            investment=solph.Investment(
                maximum=invest_max,
                ep_costs=economics.annuity(
                    capex=input_data["costs_storages_investment"].loc[i, im.startyear],
                    n=s["unit_lifetime"],
                    wacc=input_data["wacc"].loc[i, im.startyear],
                ),
                existing=existing,
            ),
        )

    return node_dict, new_built_storage_labels, endo_exo_exist_stor_df


def storages_exo(
    new_built_storages_df,
    storage_turbine_inv_costs_df,
    storage_pump_inv_costs_df,
    storage_cap_inv_costs_df,
    WACC_df,
    startyear,
    endyear,
    IR=0.02,
    discount=False,
):
    """Function calculates dicounted investment costs for exogenously
    new built storages from new_built_storages_df which are built after the
    model-endogeneous investment year. Furthermore the commissioned capacity
    from startyear until endyear for new built storages is stored in a
    seperate DataFrame

    Parameters
    ----------
    new_built_storages_df : :obj:`pandas.DataFrame`
        pd.DataFrame containing the new built storages' data

    storage_turbine_inv_costs_df: :obj:`pandas.DataFrame`
        pd.DataFrame containing the turbine investment costs for new built storages

    storage_pump_inv_costs_df: :obj:`pandas.DataFrame`
        pd.DataFrame containing the pump investment costs for new built storages

    storage_cap_inv_costs_df: :obj:`pandas.DataFrame`
        pd.DataFrame containing the capacity investment costs for new built storages

    WACC_df : :obj:`pd.DataFrame`
        pd.DataFrame containing the WACC data

    startyear : :obj:`int`
        Starting year of the (current) optimization run

    endyear : :obj:`int`
        End year of the (current) optimization run

    IR : :obj:`float`
        The interest rate to be applied for discounting (only if discount is True)

    discount : :obj:`boolean`
        If True, nominal values will be dicounted
        If False, real values have to be used as model inputs (default)

    Returns
    -------
    storages_exo_com_costs_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing total discounted cost for new built storages
        from startyear until endyear

    storages_exo_com_capacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the commissioned capacity of all exogenously new built
        storages in between startyear and endyear

    storages_exo_decom_capacity_df : :obj:`pandas.core.frame.DataFrame`
        Dataframe containing the decommissioned capacity of all exogenously new built
        storages in between startyear and endyear
    """

    # JF 14.12.2019 create storage_part for easier access to the needed values
    # and creating a new index for a new Dataframe
    storage_part = ["_turbine", "_pump", "_cap"]
    # 27.03.2020, JK: Mistake below: zip does not create the cartesian product which is demanded
    new_index = [
        i + j for i in new_built_storages_df.index.values for j in storage_part
    ]
    # new_index = [i + j for i, j in zip(new_built_storages_df.index.values, storage_part)]
    storages_exo_com_costs_df = pd.DataFrame(index=new_index)
    storages_exo_com_capacity_df = pd.DataFrame(index=new_index)
    storages_exo_decom_capacity_df = pd.DataFrame(index=new_index)

    # go through all storages
    for storage in new_built_storages_df.index.values:
        # go through all years, for example range(2016+1=2017, 2050+1=2051)
        for year in range(startyear, endyear + 1):
            year_str = str(year)
            # JF 15.12.2019 define (de)commissioning as net (de)commissioning in between the years
            if year == 2016:
                turbine_commissioning = new_built_storages_df.loc[
                    storage, "existing_turbine_" + year_str
                ]
                pump_commissioning = new_built_storages_df.loc[
                    storage, "existing_pump_" + year_str
                ]
                cap_commissioning = new_built_storages_df.loc[
                    storage, "existing_" + year_str
                ]
                turbine_decommissioning = 0
                pump_decommissioning = 0
                cap_decommissioning = 0
            else:
                turbine_commissioning = (
                    new_built_storages_df.loc[storage, "existing_turbine_" + year_str]
                    - new_built_storages_df.loc[
                        storage, "existing_turbine_" + str(year - 1)
                    ]
                )
                pump_commissioning = (
                    new_built_storages_df.loc[storage, "existing_pump_" + year_str]
                    - new_built_storages_df.loc[
                        storage, "existing_pump_" + str(year - 1)
                    ]
                )
                cap_commissioning = (
                    new_built_storages_df.loc[storage, "existing_" + year_str]
                    - new_built_storages_df.loc[storage, "existing_" + str(year - 1)]
                )
                turbine_decommissioning = -turbine_commissioning
                pump_decommissioning = -pump_commissioning
                cap_decommissioning = -cap_commissioning

            # JF 15.12.2019 save discounted cost data for the turbine in the storages costs' dataframe
            # check if commissioning is greater or equal to zero, if yes then
            # store it as commissioning and calculate the discounted commissioning cost,
            # if not store the result as decommissioning
            if turbine_commissioning >= 0:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ] = turbine_commissioning * economics.annuity(
                    capex=storage_turbine_inv_costs_df.loc[storage, year],
                    n=new_built_storages_df.loc[storage, "unit_lifetime_turbine"],
                    wacc=WACC_df.loc[storage, year],
                )

                storages_exo_com_capacity_df.loc[
                    storage + storage_part[0], "commissioned_" + year_str
                ] = turbine_commissioning
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[0], "decommissioned_" + year_str
                ] = 0
            else:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ] = 0
                storages_exo_com_capacity_df.loc[
                    storage + storage_part[0], "commissioned_" + year_str
                ] = 0
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[0], "decommissioned_" + year_str
                ] = turbine_decommissioning

            # save discounted cost data for the pump in the storages costs' dataframe
            if pump_commissioning >= 0:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[1], "exo_cost_" + year_str
                ] = pump_commissioning * economics.annuity(
                    capex=storage_pump_inv_costs_df.loc[storage, year],
                    n=new_built_storages_df.loc[storage, "unit_lifetime_pump"],
                    wacc=WACC_df.loc[storage, year],
                )

                storages_exo_com_capacity_df.loc[
                    storage + storage_part[1], "commissioned_" + year_str
                ] = pump_commissioning
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[1], "decommissioned_" + year_str
                ] = 0

            else:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[1], "exo_cost_" + year_str
                ] = 0
                storages_exo_com_capacity_df.loc[
                    storage + storage_part[1], "commissioned_" + year_str
                ] = 0
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[1], "decommissioned_" + year_str
                ] = pump_decommissioning

            # save discounted cost data for the capacity in the storages costs' dataframe
            if cap_commissioning >= 0:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[2], "exo_cost_" + year_str
                ] = cap_commissioning * economics.annuity(
                    capex=storage_cap_inv_costs_df.loc[storage, year],
                    n=new_built_storages_df.loc[storage, "unit_lifetime"],
                    wacc=WACC_df.loc[storage, year],
                )

                storages_exo_com_capacity_df.loc[
                    storage + storage_part[2], "commissioned_" + year_str
                ] = cap_commissioning
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[2], "decommissioned_" + year_str
                ] = 0
            else:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[2], "exo_cost_" + year_str
                ] = 0
                storages_exo_com_capacity_df.loc[
                    storage + storage_part[2], "commissioned_" + year_str
                ] = 0
                storages_exo_decom_capacity_df.loc[
                    storage + storage_part[2], "decommissioned_" + year_str
                ] = cap_decommissioning

            if discount:
                storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ] = storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ].div(
                    ((1 + IR) ** (year - startyear))
                )
                storages_exo_com_costs_df.loc[
                    storage + storage_part[1], "exo_cost_" + year_str
                ] = storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ].div(
                    ((1 + IR) ** (year - startyear))
                )
                storages_exo_com_costs_df.loc[
                    storage + storage_part[2], "exo_cost_" + year_str
                ] = storages_exo_com_costs_df.loc[
                    storage + storage_part[0], "exo_cost_" + year_str
                ].div(
                    ((1 + IR) ** (year - startyear))
                )

    return (
        storages_exo_com_costs_df,
        storages_exo_com_capacity_df,
        storages_exo_decom_capacity_df,
    )
