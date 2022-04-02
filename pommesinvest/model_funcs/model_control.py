# -*- coding: utf-8 -*-
"""
General description
-------------------
This file contains all class and function definitions for controlling the model
workflow of the investment variant of POMMES.

@author: Johannes Kochems (*), Johannes Giehl (*), Yannick Werner,
Benjamin Grosse

Contributors:
Julien Faist, Hannes Kachel, Sophie Westphal, Flora von Mikulicz-Radecki,
Carla Spiller, Fabian Büllesbach, Timona Ghosh, Paul Verwiebe,
Leticia Encinas Rosa, Joachim Müller-Kirchenbauer

(*) Corresponding authors
"""

import logging
import math

from oemof.solph import constraints, views, models, network, processing
from oemof.tools import logger

from pommesinvest.model_funcs import helpers
from pommesinvest.model_funcs.data_input import (
    nodes_from_csv,
    nodes_from_csv_rolling_horizon,
)
import warnings

import pandas as pd
import math
import logging

import oemof.solph as solph


FREQUENCY_TO_TIMESTEPS = {
    "60min": {"timesteps": 8760, "multiplicator": 1},
    "4H": {"timesteps": 2190, "multiplicator": 4},
    "8H": {"timesteps": 1095, "multiplicator": 8},
    "24H": {"timesteps": 365, "multiplicator": 24},
    "36H": {"timesteps": 244, "multiplicator": 36},
    "48H": {"timesteps": 182, "multiplicator": 48},
}


def show_meta_logging_info(model_meta):
    """Show some logging information on model meta data"""
    logging.info("***** MODEL RUN TERMINATED SUCCESSFULLY :-) *****")
    logging.info(f"Overall objective value: {model_meta['overall_objective']:,.0f}")
    logging.info(f"Overall solution time: {model_meta['overall_solution_time']:.2f}")
    logging.info(f"Overall time: {model_meta['overall_time']:.2f}")


class InvestmentModel(object):
    r"""A class that holds an investment model.

    An investment model is a container for all the model parameters as well
    as for methods for controlling the model workflow.

    Attributes
    ----------
    rolling_horizon : boolean
        boolean control variable indicating whether to run a rolling horizon
        optimization or an integral optimization run (a simple model).
        Note: For the rolling_horizon optimization run, additionally the
        parameters `time_slice_length_wo_overlap_in_hours` and
        `overlap_in_hours` (both of type int) have to be defined.

    aggregate_input : boolean
        boolean control variable indicating whether to use complete
        or aggregated transformer input data set

    interest_rate : float
        Interest rate used for discounting

    solver : str
        The solver to be used for solving the mathematical optimization model.
        Must be one of the solvers oemof.solph resp. pyomo support, e.g.
        'cbc', 'gplk', 'gurobi', 'cplex'.

    fuel_cost_pathway :  str
        A predefined pathway for commodity cost development until 2050

        .. csv-table:: Pathways and explanations
            :header: "pathway", "explanation", "description"
            :widths: 10 45 45

            "NZE", "| Net Zero Emissions Scenario
            | from IEA's world energy outlook 2021", "comparatively
            low commodity prices"
            "SDS", "| Sustainable Development Scenario
            | from IEA's world energy outlook 2021", "| comparatively low commodity prices;
            | slightly higher than NZE"
            "APS", "| Announced Pledges Scenario
            | from IEA's world energy outlook 2021", "| medium price development,
            | decline in prices between
            | 2030 and 2050"
            "STEPS", "| Stated Policies Scenario
            | from IEA's world energy outlook 2021", "| highest price development,
            | esp. for oil and natgas"
            "regression", "| Linear regression based on historic
            | commodity prices from 1991-2020", "| compared to IEA's scenarios,
            | close to upper range of projections"

    emissions_cost_pathway : str
        A predefined pathway for emissions cost development until 2030 or 2050

        .. csv-table:: Pathways and explanations
            :header: "pathway", "explanation", "description"
            :widths: 10 45 45

            "Fit_for_55_split_high", "| Emissions split according to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| high estimate,
            | values until 2030"
            "Fit_for_55_split_medium", "| Emissions split according to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| medium estimate,
            | values until 2030"
            "Fit_for_55_split_low", "| Emissions split according to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| low estimate,
            | values until 2030"
            "ESR_reduced_high", "| Higher emission reduction
            | in ETS compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| high estimate,
            | values until 2030"
            "ESR_reduced_medium", "| Higher emission reduction
            | in ETS compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| medium estimate,
            | values until 2030"
            "ESR_reduced_low", "| Higher emission reduction
            | in ETS compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| low estimate,
            | values until 2030"
            "reductions_in_ETS_only_high", "| Reductions only in ETS
            | compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| high estimate,
            | values until 2030"
            "reductions_in_ETS_only_medium", "| Reductions only in ETS
            | compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| medium estimate,
            | values until 2030"
            "reductions_in_ETS_only_low", "| Reductions only in ETS
            | compared to
            | Fit for 55 split between
            | ETS and ESR (non-ETS)", "| low estimate,
            | values until 2030"
            "long-term", "| Long-term emissions cost pathway
            | according to medium estimate", "| medium estimate,
            | values until 2050"

    investment_cost_pathway : str
        A predefined pathway for investment cost development until 2050

    activate_emissions_limit : boolean
        boolean control variable indicating whether to introduce an overall
        emissions limit
        Note: Combining an emissions limit with comparatively high minimum
        loads of conventionals may lead to an infeasible model configuration
        since either one of the restrictions may not be reached.

    emissions_pathway : str
        A predefined pathway for emissions reduction until 2045
        Options: '100_percent_linear', '95_percent_linear', '80_percent_linear'
        or 'BAU'

    activate_demand_response : boolean
        boolean control variable indicating whether to introduce
        demand response to the model

    demand_response_approach : str
        The approach used for demand response modeling
        Options: 'DLR', 'DIW', 'oemof'
        See the documentation of the custom SinkDSM in oemof.solph as well
        as the presentation by Johannes Kochems from the INREC 2020
        for further information

    demand_response_scenario : str
        A predefined demand response scenario to be modeled
        Options: '25', '50', '75', whereby '25' is the lower,
        i.e. rather pessimistic estimate

    save_production_results : boolean
        boolean control variable indicating whether to save the dispatch
        results of the model run to a .csv file

    save_investment_results : boolean
        boolean control variable indicating whether to save the investment
        decision results of the model run to a .csv file

    write_lp_file : boolean
        boolean control variable indicating whether to save an lp file
        *CAUTION*: Only use for debugging when simulating small time frames

    start_time : str
        A date string of format "YYYY-MM-DD hh:mm:ss" defining the start time
        of the simulation

    end_time : str
        A date string of format "YYYY-MM-DD hh:mm:ss" defining the end time
        of the simulation

    freq : str
        Frequency of the simulation, i.e. freqeuncy of the pandas.date_range
        object

    path_folder_input : str
        The path to the folder where the input data is stored

    path_folder_output : str
        The path to the folder where the output data is to be stored

    om : :class:`oemof.solph.models.Model`
        The mathematical optimization model itself

    time_slice_length_wo_overlap_in_hours : int (optional, for rolling horizon)
        The length of a time slice for a rolling horizon model run in hours,
        not including an overlap

    overlap_in_hours : int (optional, for rolling horizon)
        The length of the overlap for a rolling horizon model run in hours
    """  # noqa: E501

    def __init__(self):
        """Initialize an empty InvestmentModel object"""
        self.rolling_horizon = None
        self.aggregate_input = None
        self.interest_rate = None
        self.solver = None
        self.fuel_cost_pathway = None
        self.emissions_cost_pathway = None
        self.investment_cost_pathway = None
        self.activate_emissions_limit = None
        self.emissions_pathway = None
        self.activate_demand_response = None
        self.demand_response_approach = None
        self.demand_response_scenario = None
        self.save_production_results = None
        self.save_investment_results = None
        self.write_lp_file = None
        self.start_time = None
        self.end_time = None
        self.freq = None
        self.path_folder_input = None
        self.path_folder_output = None
        self.om = None

    def update_model_configuration(self, *model_parameters, nolog=False):
        """Set the main model parameters by extracting them from dicts

        Parameters
        ----------
        *model_parameters : dict
            An arbitrary amount of dicts holding the model parameterization
            information

        nolog : boolean
            Show no logging ingo if True; else show logs for updating resp.
            adding attributes to the dispatch model
        """
        for param_dict in model_parameters:
            for k, v in param_dict.items():
                if not nolog:
                    if hasattr(self, k):
                        print(f"Updating attribute `{k}` with value '{v}'.")
                    else:
                        print(
                            f"Adding attribute `{k}` with value '{v}' "
                            + "to the model."
                        )
                setattr(self, k, v)
                if k == "freq":
                    self.set_multiplicator()

        if hasattr(self, "start_time"):
            setattr(self, "start_year", str(pd.to_datetime(self.start_time).year))
        if hasattr(self, "end_time"):
            setattr(self, "end_year", str(pd.to_datetime(self.end_time).year))

    def set_multiplicator(self):
        """Set multiplicator and timesteps dependent on frequency attribute"""
        self.multiplicator = FREQUENCY_TO_TIMESTEPS[self.freq]["multiplicator"]

    def check_model_configuration(self):
        """Checks if any necessary model parameter hasn't been set yet"""
        missing_parameters = []

        for entry in dir(self):
            if (
                not entry.startswith("_")
                and entry != "om"
                and getattr(self, entry) is None
            ):
                missing_parameters.append(entry)
                logging.warning(
                    f"Necessary model parameter `{entry}` "
                    + "has not yet been specified!"
                )
            if entry == "fuel_cost_pathway":
                logging.info(f"Using fuel cost pathway: {getattr(self, entry)}")
            elif entry == "emissions_cost_pathway":
                logging.info(f"Using emissions cost pathway: {getattr(self, entry)}")
            elif entry == "investment_cost_pathway":
                logging.info(f"Using investment cost pathway: {getattr(self, entry)}")

        return missing_parameters

    def add_rolling_horizon_configuration(
        self, rolling_horizon_parameters, nolog=False
    ):
        r"""Add a rolling horizon configuration to the dispatch model

        .. _note:

            The amount of time steps is limited in such a way that only
            complete time slices are used. If the time series do not
            allow for adding another time slice, the last couple of time
            steps of the time series are not used.
        """
        self.update_model_configuration(rolling_horizon_parameters, nolog=nolog)

        setattr(self, "time_series_start", pd.Timestamp(self.start_time, self.freq))
        setattr(self, "time_series_end", pd.Timestamp(self.end_time, self.freq))

        setattr(
            self,
            "time_slice_length_wo_overlap_in_time_steps",
            (
                FREQUENCY_TO_TIMESTEPS[self.freq]["timesteps"]
                * getattr(self, "myopic_horizon_in_years")
            ),
        )
        setattr(
            self,
            "overlap_in_time_steps",
            (
                FREQUENCY_TO_TIMESTEPS[self.freq]["timesteps"]
                * getattr(self, "overlap_in_years")
            ),
        )
        setattr(
            self,
            "time_slice_length_with_overlap",
            (
                getattr(self, "time_slice_length_wo_overlap_in_time_steps")
                + getattr(self, "overlap_in_time_steps")
            ),
        )
        setattr(
            self,
            "overall_time_steps",
            helpers.time_steps_between_timestamps(
                getattr(self, "time_series_start"),
                getattr(self, "time_series_end"),
                self.freq,
            ),
        )
        setattr(
            self,
            "amount_of_time_slices",
            math.ceil(
                getattr(self, "overall_time_steps")
                / getattr(self, "time_slice_length_wo_overlap_in_time_steps")
            ),
        )

    def initialize_logging(self):
        """Initialize logging by deriving a filename from the configuration"""
        optimization_timeframe = helpers.days_between(self.start_time, self.end_time)

        if not self.rolling_horizon:
            rh = "simple_"
        else:
            rh = "RH_"
        if self.aggregate_input:
            agg = "clustered"
        else:
            agg = "complete"

        filename = (
            "dispatch_LP_start-"
            + self.start_time[:10]
            + "_"
            + str(optimization_timeframe)
            + "-days_"
            + rh
            + agg
        )

        setattr(self, "filename", filename)
        logger.define_logging(logfile=filename + ".log")

        return filename

    def show_configuration_log(self):
        """Show some logging info dependent on model configuration"""
        if self.aggregate_input:
            agg_string = "Using the AGGREGATED POWER PLANT DATA SET"
        else:
            agg_string = "Using the COMPLETE POWER PLANT DATA SET."

        if self.activate_demand_response:
            dr_string = (
                f"Using approach '{self.demand_response_approach}' "
                f"for DEMAND RESPONSE modeling\n"
                f"Considering a {self.demand_response_scenario}% scenario"
            )

        else:
            dr_string = "Running a model WITHOUT DEMAND RESPONSE"

        logging.info(agg_string)
        logging.info(dr_string)

        return agg_string, dr_string

    def build_simple_model(self):
        r"""Set up and return a simple model

        Construct a model for an overall optimization run
        not including any measures for complexity reduction.
        """
        logging.info("Starting optimization")
        logging.info("Running an integrated INVESTMENT AND DISPATCH OPTIMIZATION")

        datetime_index = pd.date_range(self.start_time, self.end_time, freq=self.freq)
        es = network.EnergySystem(timeindex=datetime_index)

        nodes_dict, emissions_limit = nodes_from_csv(self)

        logging.info("Creating a LP model for INVESTMENT AND DISPATCH OPTIMIZATION.")

        es.add(*nodes_dict.values())
        setattr(self, "om", models.Model(es))

        self.add_further_constrs(emissions_limit)

    def add_further_constrs(self, emissions_limit, countries=None, fuels=None):
        r"""Integrate further constraints into the optimization model

        For now, an additional overall emissions limit can be imposed.

        Note that setting an emissions limit may conflict with high minimum
        loads from conventional transformers.
        Be aware that this may lead to model infeasibility
        if commodity bus balances cannot be met.

        Parameters
        ----------
        emissions_limit : float
            The actual emissions limit to be used

        countries : :obj:`list` of `str`
            The countries for which an emissions limit shall be imposed
            (Usually only Germany, so ["DE"])

        fuels : :obj:`list` of `str`
            The fuels for which an emissions limit shall be imposed
        """
        if countries is None:
            countries = ["DE"]

        if fuels is None:
            fuels = [
                "biomass",
                "hardcoal",
                "lignite",
                "natgas",
                "uranium",
                "oil",
                "otherfossil",
                "waste",
                "mixedfuels",
            ]

        # Emissions limit is imposed for flows from commodity source to bus
        emission_flow_labels = [
            f"{country}_bus_{fuel}" for country in countries for fuel in fuels
        ]

        emission_flows = {}

        for (i, o) in self.om.flows:
            if any(x in o.label for x in emission_flow_labels):
                emission_flows[(i, o)] = self.om.flows[(i, o)]

        if self.activate_emissions_limit:
            constraints.emission_limit(
                self.om, flows=emission_flows, limit=emissions_limit
            )
            logging.info(f"Adding an EMISSIONS LIMIT of {emissions_limit} t CO2")

    def build_rolling_horizon_model(self, counter, iteration_results):
        r"""Set up and return a rolling horizon LP dispatch model

        Track the storage labels in order to obtain and pass initial
        storage levels for each iteration. Set the end time of an iteration
        excluding the overlap to the start of the next iteration.

        Parameters
        ----------
        counter : int
            A counter for the rolling horizon optimization iterations

        iteration_results : dict
            A dictionary holding the results of the previous rolling horizon
            iteration
        """
        logging.info(f"Starting optimization for optimization run {counter}")
        logging.info(
            f"Start of iteration {counter}: " + f"{getattr(self, 'time_series_start')}"
        )
        logging.info(
            f"End of iteration {counter}: " + f"{getattr(self, 'time_series_end')}"
        )

        datetime_index = pd.date_range(
            start=getattr(self, "time_series_start"),
            periods=getattr(self, "time_slice_length_with_overlap"),
            freq=self.freq,
        )
        es = network.EnergySystem(timeindex=datetime_index)

        (
            node_dict,
            emissions_limit,
            storage_and_transformer_labels,
        ) = nodes_from_csv_rolling_horizon(self, iteration_results)
        # Only set storage and transformer labels attribute for the 0th iteration
        if not hasattr(self, "storage_and_transformer_labels"):
            setattr(
                self, "storage_and_transformer_labels", storage_and_transformer_labels
            )

        # Update model start time for the next iteration
        setattr(self, "time_series_start", getattr(self, "time_series_end"))

        es.add(*node_dict.values())
        logging.info(f"Successfully set up energy system for iteration {counter}")

        self.om = models.Model(es)

        self.add_further_constrs(emissions_limit)

    def solve_rolling_horizon_model(
        self, counter, iter_results, model_meta, no_solver_log=False
    ):
        """Solve a rolling horizon optimization model

        Parameters
        ----------
        counter : int
            A counter for the rolling horizon optimization iterations

        iter_results : dict
            A dictionary holding the results of the previous rolling horizon
            iteration

        model_meta : dict
            A dictionary holding meta information on the model, such as
             solution times and objective value

        no_solver_log : boolean
            Show no solver logging if set to True
        """
        if self.write_lp_file:
            self.om.write(
                (
                    self.path_folder_output
                    + "pommesinvest_model_iteration_"
                    + str(counter)
                    + ".lp"
                ),
                io_options={"symbolic_solver_labels": True},
            )

        if no_solver_log:
            solve_kwargs = {"tee": False}
        else:
            solve_kwargs = {"tee": True}

        self.om.solve(solver=self.solver, solve_kwargs=solve_kwargs)
        print("********************************************************")
        logging.info(f"Model run {counter} done!")

        iter_results["model_results"] = processing.results(self.om)
        electricity_bus = views.node(iter_results["model_results"], "DE_bus_el")
        sliced_dispatch_results = pd.DataFrame(
            data=electricity_bus["sequences"].iloc[
                0 : getattr(self, "time_slice_length_wo_overlap_in_time_steps")
            ]
        )
        iter_results["dispatch_results"] = iter_results["dispatch_results"].append(
            sliced_dispatch_results
        )
        iter_results["investment_results"] = electricity_bus["scalars"]

        meta_results = processing.meta_results(self.om)
        # Objective is weighted in order to take overlap into account
        model_meta["overall_objective"] += int(
            meta_results["objective"]
            * (
                getattr(self, "time_slice_length_wo_overlap_in_time_steps")
                / getattr(self, "time_slice_length_with_overlap")
            )
        )
        model_meta["overall_solution_time"] += meta_results["solver"]["Time"]

    def retrieve_initial_states_rolling_horizon(self, iteration_results):
        r"""Retrieve the initial states for the upcoming rolling horizon run

        Parameters
        ----------
        iteration_results : dict
            A dictionary holding the results of the previous rolling horizon
            iteration
        """
        iteration_results["storages_existing"] = pd.DataFrame(
            columns=["initial_storage_level_last_iteration"],
            index=getattr(self, "existing_storage_labels"),
        )

        for i, s in iteration_results["storages_existing"].iterrows():
            storage = views.node(iteration_results["model_results"], i)

            iteration_results["storages_existing"].at[
                i, "initial_storage_level_last_iteration"
            ] = storage["sequences"][((i, "None"), "storage_content")].iloc[
                getattr(self, "time_slice_length_wo_overlap_in_time_steps") - 1
            ]

        iteration_results["storages_new_built"] = pd.DataFrame(
            columns=[
                "initial_storage_level_last_iteration",
                "existing_inflow_power",
                "existing_outflow_power",
                "existing_capacity_storage",
            ],
            index=getattr(self, "new_built_storage_labels"),
        )

        for i, s in iteration_results["storages_new_built"].iterrows():
            storage = views.node(iteration_results["model_results"], i)

            iteration_results["storages_new_built"].at[
                i, "initial_storage_level_last_iteration"
            ] = storage["sequences"][((i, "None"), "storage_content")].iloc[
                getattr(self, "time_slice_length_wo_overlap_in_time_steps") - 1
            ]

            iteration_results["storages_new_built"].at[
                i, "existing_inflow_power"
            ] = storage["scalars"].loc[(("DE_bus_el", i), "invest")]

            iteration_results["storages_new_built"].at[
                i, "existing_outflow_power"
            ] = storage["scalars"].loc[((i, "DE_bus_el"), "invest")]

            iteration_results["storages_new_built"].at[
                i, "existing_capacity_storage"
            ] = storage["scalars"].loc[((i, "None"), "invest")]

        logging.info("Obtained initial (storage) levels for next iteration")


def determine_timeslices_RH(
    starttime,
    endtime,
    freq,
    freq_timesteps,
    myopic_horizon_in_years,
    overlap_in_timesteps,
):
    """Functions determines timeslice lengths for a RH model run.
    It takes start and end time as well as myopic optimization length as inputs
    and returns a dict of timeslice lengths taking into acount leap years.

    Parameters
    ----------
    starttime : :obj:`str`
        The starttime of the optimization run

    endtime : :obj:`str`
        The endtime of the optimization run

    freq : :obj:`str`
        The frequency used for the datetimeindex of the optimization run

    freq_timesteps : obj:`dict` of :class:`tuple`
        A dictionary mapping amount of annual timeslices and a multiplicator
        for modifying (hourly) input data to the given frequency (freq)

    myopic_horizon_in_years : :obj:`int`
        The length of intervalls used for myopic optimization in years

    overlap_in_timesteps : :obj:`int`
        An overlap may in principle be set, but is set to 0 in the first place

    Returns
    -------
    timeseries_start : :obj:`pd.Timestamp`
        starttime given as a pd.Timestamp object

    amount_of_timeslices : :obj:`int`
        The amount of myopic optimization iterations

    timeslice_length_dict : :obj:`dict` of :class:`tuple`
        A dictionary with the iteration (i.e. timeslice) number as key and
        the amount of timesteps for that iteration as value; the value in turn
        is a tuple consisting of number of years, number of timesteps wo overlap
        and number of timesteps with overlap

    """

    timeseries_start = pd.Timestamp(starttime, freq)
    timeseries_end = pd.Timestamp(endtime, freq)

    # Consideration of leap years -> adjust amount of timeslices if year is a leap year.
    startyear = timeseries_start.year
    endyear = timeseries_end.year

    years = range(startyear, endyear + 1)
    leap_years = MultipleLeapYears(years)

    # Determine amount of timeslices per year for every year considered
    timeslice_year_dict = {}

    for year in years:
        if not year in leap_years:
            timeslice_year_dict[year] = freq_timesteps[0]
        else:
            timeslice_year_dict[year] = freq_timesteps[1]

    # Calculate amount of timeslices needed
    # (Consideration of complete years only)
    overall_years = endyear - startyear + 1
    amount_of_timeslices = math.ceil(overall_years / myopic_horizon_in_years)

    # Amount of timeslices within one iteration may vary due to leap years.
    # Therefore, it is stored as a dict with the iteration number as key
    timeslice_length_dict = {}
    start = startyear

    for i in range(amount_of_timeslices):

        # Initialize timeslice length without overlap
        timeslice_length_dict[i] = 0

        # If we are not in the last iteration
        if i != amount_of_timeslices - 1:
            for year in range(start, start + myopic_horizon_in_years):
                if not year == start + myopic_horizon_in_years - 1:
                    timeslice_length_dict[i] += timeslice_year_dict[year]
                else:
                    timeslice_length_dict[i] = (
                        myopic_horizon_in_years,
                        timeslice_length_dict[i] + timeslice_year_dict[year],
                        timeslice_length_dict[i]
                        + timeslice_year_dict[year]
                        + overlap_in_timesteps,
                    )

        # Last iteration
        else:

            for year in range(start, endyear + 1):
                if not year == endyear:
                    timeslice_length_dict[i] += timeslice_year_dict[year]
                else:
                    timeslice_length_dict[i] = (
                        len(range(start, endyear + 1)),
                        timeslice_length_dict[i] + timeslice_year_dict[year],
                        timeslice_length_dict[i]
                        + timeslice_year_dict[year]
                        + overlap_in_timesteps,
                    )

        start += myopic_horizon_in_years

    return timeseries_start, amount_of_timeslices, timeslice_length_dict


def add_further_constrs(
    om,
    ActivateEmissionsLimit,
    ActivateInvestmentBudgetLimit,
    emissions_limit,
    investment_budget,
    countries=None,
    fuels=None,
):
    """Integrate further constraints into the optimization model

    For now, an additional overall emissions limit can be imposed.

    Note that setting an emissions limit may conflict with high minimum
    loads from conventional transformers. This may lead to model infeasibility
    if commodity bus balances cannot be met.

    Parameters
    ----------
    om : :class:`oemof.solph.models.Model`
        The original mathematical optimisation model to be solved

    ActivateEmissionsLimit : :obj:`boolean`
        If True, an emission limit is introduced

    ActivateInvestmentBudgetLimit : :obj:`boolean`
        If True, an overall investment budget limit is introduced

    emissions_limit : float
        The actual emissions limit to be used

    investment_budget : float
        The overall investment budget limit to be used

    countries : :obj:`list` of `str`
        The countries for which an emissions limit shall be imposed
        (Usually only Germany)

    fuels : :obj:`list` of `str`
        The fuels for which an emissions limit shall be imposed

    """

    if countries is None:
        countries = ["DE"]

    if fuels is None:
        fuels = [
            "biomass",
            "hardcoal",
            "lignite",
            "natgas",
            "uranium",
            "oil",
            "otherfossil",
            "waste",
            "mixedfuels",
        ]

    # Emissions limit is imposed for flows from commodity source to commodity bus
    emission_flow_labels = [
        country + "_bus_" + fuel for country in countries for fuel in fuels
    ]

    emission_flows = {}

    for (i, o) in om.flows:
        if any(x in o.label for x in emission_flow_labels):
            emission_flows[(i, o)] = om.flows[(i, o)]

    if ActivateEmissionsLimit:
        solph.constraints.emission_limit(
            om, flows=emission_flows, limit=emissions_limit
        )
        logging.info(f"Adding an EMISSIONS LIMIT of {emissions_limit} t CO2")

    # TODO: Revise!
    if ActivateInvestmentBudgetLimit:
        om = solph.constraints.investment_limit(om, limit=investment_budget)

        logging.info(f"Adding an INVESTMENT BUDGET LIMIT of {investment_budget} €")

    return om


def build_simple_model(
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
    fuel_cost_pathway,
    investment_cost_pathway,
    starttime,
    endtime,
    freq,
    multiplicator,
    optimization_timeframe,
    IR=0.02,
    discount=False,
    ActivateEmissionsLimit=False,
    emission_pathway="100_percent_linear",
    ActivateInvestmentBudgetLimit=False,
    investment_budget=None,
    ActivateDemandResponse=False,
    approach="DIW",
    scenario="50",
):
    """Set up and return a simple model (i.e. an overall optimization run
    not including any measures for complexity reduction).

    Parameters
    ----------
    path_folder_input : :obj:`str`
        The file path where input files are stored (common folder)

    filename_node_data : :obj:`str`
        Name of Excel Workbook containing all data for creating nodes (buses and oemof components)

    filename_cost_data : :obj:`str`
        Name of Excel Workbook containing cost pathways for oemof components

    filename_node_timeseries : :obj:`str`
        Filename of the node timeseries data, given in a separate .csv file

    filename_min_max_timeseries  : :obj:`str`
       Filename of the min / max transformer data, given in a separate .csv file

    filename_cost_timeseries : :obj:`str`
        Filename of the cost timeseries data, given in a separate .csv file

    AggregateInput: :obj:`boolean`
        If True an aggregated transformers input data set is used, elsewhise
        the full transformers input data set is used

    startyear : :obj:`int`
        The startyear of the optimization run

    endyear : :obj:`int`
        The endyear of the optimization run

    MaxInvest : :obj:`boolean`
        If True, investment limits per technology are applied

    fuel_cost_pathway : :obj:`str`
        variable indicating which fuel cost pathway to use
        Possible values 'lower', 'middle', 'upper'

    investment_cost_pathway : :obj:`str`
        variable indicating which investment cost pathway to use
        Possible values 'lower', 'middle', 'upper'

    starttime : :obj:`str`
        The starttime of the optimization run

    endtime : :obj:`str`
        The endtime of the optimization run

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

    ActivateEmissionsLimit : :obj:`boolean`
        If True, an emission limit is introduced

    emission_pathway : str
        The pathway for emissions reduction to be used

    ActivateInvestmentBudgetLimit : :obj:`boolean`
        If True, an overall investment budget limit is introduced

    investment_budget : float
        The overall investment budget limit to be used

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
    om : :class:`oemof.colph.models.Model`
        The mathematical optimisation model solved including the results

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

    datetime_index = pd.date_range(starttime, endtime, freq=freq)

    logging.info("Starting optimization")

    # initialisation of the energy system with its timeindex attribute
    es = solph.EnergySystem(timeindex=datetime_index)

    logging.info("Running an integrated INVESTMENT and dispatch OPTIMIZATION")
    logging.info("Time frequency used is {}".format(freq))

    # Create all nodes from excel sheet and store costs data
    (
        nodes_dict,
        existing_storage_labels,
        new_built_storage_labels,
        total_exo_com_costs_df,
        total_exo_com_capacity_df,
        total_exo_decom_capacity_df,
        emissions_limit,
    ) = nodes_from_excel(
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
        fuel_cost_pathway,
        investment_cost_pathway,
        starttime,
        endtime,
        freq,
        multiplicator,
        optimization_timeframe,
        IR=IR,
        discount=discount,
        ActivateEmissionsLimit=ActivateEmissionsLimit,
        emission_pathway=emission_pathway,
        ActivateDemandResponse=ActivateDemandResponse,
        approach=approach,
        scenario=scenario,
    )

    logging.info(
        "Creating a LP optimization model for integrated\n"
        "INVESTMENT and dispatch OPTIMIZATION."
    )

    # Add all nodes to the EnergySystem
    es.add(*nodes_dict.values())

    # Create a least cost model from the energy system (builds mathematical model)
    om = solph.Model(es)

    om = add_further_constrs(
        om,
        ActivateEmissionsLimit,
        ActivateInvestmentBudgetLimit,
        emissions_limit,
        investment_budget,
    )

    return (
        om,
        existing_storage_labels,
        new_built_storage_labels,
        total_exo_com_costs_df,
        total_exo_com_capacity_df,
        total_exo_decom_capacity_df,
    )


def initial_states_RH(
    om,
    timeslice_length_wo_overlap_in_timesteps,
    new_built_transformer_labels,
    new_built_storage_labels,
    endo_exo_exist_df,
    endo_exo_exist_stor_df,
):
    """Obtain the initial states / existing capacities for the upcoming rolling horizon (resp.
    myopic optimization window) model run for a LP INVESTMENT MODEL configuration by iterating
    over all nodes of the energy system using lists of storage resp. transformer input data obtained
    from the input data file. Existing capacities have to be determined for both, transformers and
    storages to determine how much capacity was installed in the previous model run (myopic optimization
    window timeslice) and how existing capacity changes. Initial states only have to be set for
    storage units since there are none for transformer units when a LP INVESTMENT model is run.

    Parameters
    ----------
    om : :class:`oemof.colph.models.Model`
        The mathematical optimisation model solved including the results

    timeslice_length_wo_overlap_in_timesteps: :obj:`int`
        length of a rolling horizon timeslice excluding overlap

    new_built_transformer_labels: :obj:`list` of ::class:`str`
        list of transformer labels (obtained from input data)

    new_built_storage_labels: :obj:`list` of ::class:`str`
        list of storage labels (obtained from input data)

    endo_exo_exist_df : :obj:`pd.DataFrame`
        A DataFrame containing the endogeneous and exogeneous transformers commissioning
        information for setting initial states

    endo_exo_exist_stor_df : :obj:`pd.DataFrame`
        A DataFrame containing the endogeneous and exogeneous storages commissioning
        information for setting initial states

    Returns
    -------
    transformers_init_df : :obj:`pd.DataFrame`
        A pd.DataFrame containing the storage data (i.e. statuses for
        the last timestep of the optimization window - excluding overlap)

    storages_init_df : :obj:`pd.DataFrame`
        A pd.DataFrame containing the storage data (i.e. statuses for
        the last timestep of the optimization window - excluding overlap)

    """
    # DataFrame for storing transformer resp. storage initial timestep data
    transformers_init_df = pd.DataFrame(
        columns=[
            "Existing_Capacity_Transformer",
            "Existing_Capacity_endo",
            "endo_cumulated",
            "old_exo",
        ],
        index=new_built_transformer_labels,
    )

    storages_init_df = pd.DataFrame(
        columns=[
            "Capacity_Last_Timestep",
            "Existing_Inflow_Power",
            "Existing_Outflow_Power",
            "Existing_Capacity_Storage",
        ],
        index=new_built_storage_labels,
    )

    #    results_df = solph.processing.create_dataframe(om)
    model_results = solph.processing.results(om)

    for i, t in transformers_init_df.iterrows():

        transformer = solph.views.node(model_results, i)
        # TODO, JK: Find a more elegant solution than doing this here... -> obtain country info before
        try:
            transformers_init_df.loc[i, "Existing_Capacity_Transformer"] = transformer[
                "scalars"
            ][((i, "DE_bus_el"), "invest")]
            transformers_init_df.loc[
                i, "Existing_Capacity_endo"
            ] = endo_exo_exist_df.loc[i, "Existing_Capacity_endo"]
            transformers_init_df.loc[i, "endo_cumulated"] = (
                transformers_init_df.loc[i, "Existing_Capacity_Transformer"]
                + transformers_init_df.loc[i, "Existing_Capacity_endo"]
            )
            transformers_init_df.loc[i, "old_exo"] = endo_exo_exist_df.loc[i, "old_exo"]

        except:
            transformers_init_df.loc[i, "Existing_Capacity_Transformer"] = transformer[
                "scalars"
            ][((i, "AT_bus_el"), "invest")]

    # Iterate over all storages and set parameters for initial timestep of next timeslice
    for i, s in storages_init_df.iterrows():

        storage = solph.views.node(model_results, i)

        try:

            #  Obtain data for last timestep of storage unit during the optimization run (excluding overlap)
            storages_init_df.loc[i, "Capacity_Last_Timestep"] = storage["sequences"][
                ((i, "None"), "storage_content")
            ][timeslice_length_wo_overlap_in_timesteps - 1]
            storages_init_df.loc[i, "Existing_Inflow_Power"] = storage["scalars"][
                (("DE_bus_el", i), "invest")
            ]
            storages_init_df.loc[i, "Existing_Outflow_Power"] = storage["scalars"][
                ((i, "DE_bus_el"), "invest")
            ]
            storages_init_df.loc[i, "Existing_Capacity_Storage"] = storage["scalars"][
                ((i, "None"), "invest")
            ]

            storages_init_df.loc[
                i, "Existing_Capacity_endo"
            ] = endo_exo_exist_stor_df.loc[i, "capacity_endo"]
            storages_init_df.loc[
                i, "Existing_turbine_endo"
            ] = endo_exo_exist_stor_df.loc[i, "turbine_endo"]
            storages_init_df.loc[i, "Existing_pump_endo"] = endo_exo_exist_stor_df.loc[
                i, "pump_endo"
            ]

            storages_init_df.loc[i, "capacity_endo_cumulated"] = (
                storages_init_df.loc[i, "Existing_Capacity_Storage"]
                + storages_init_df.loc[i, "Existing_Capacity_endo"]
            )
            storages_init_df.loc[i, "turbine_endo_cumulated"] = (
                storages_init_df.loc[i, "Existing_Outflow_Power"]
                + storages_init_df.loc[i, "Existing_turbine_endo"]
            )
            storages_init_df.loc[i, "pump_endo_cumulated"] = (
                storages_init_df.loc[i, "Existing_Inflow_Power"]
                + storages_init_df.loc[i, "Existing_Capacity_endo"]
            )

            storages_init_df.loc[i, "old_exo_cap"] = endo_exo_exist_stor_df.loc[
                i, "old_exo_cap"
            ]
            storages_init_df.loc[i, "old_exo_turbine"] = endo_exo_exist_stor_df.loc[
                i, "old_exo_turbine"
            ]
            storages_init_df.loc[i, "old_exo_pump"] = endo_exo_exist_stor_df.loc[
                i, "old_exo_pump"
            ]

        except:

            storages_init_df.loc[i, "Capacity_Last_Timestep"] = storage["sequences"][
                ((i, "None"), "storage_content")
            ][timeslice_length_wo_overlap_in_timesteps - 1]
            storages_init_df.loc[i, "Existing_Inflow_Power"] = storage["scalars"][
                (("AT_bus_el", i), "invest")
            ]
            storages_init_df.loc[i, "Existing_Outflow_Power"] = storage["scalars"][
                ((i, "AT_bus_el"), "invest")
            ]
            storages_init_df.loc[i, "Existing_Capacity_Storage"] = storage["scalars"][
                ((i, "None"), "invest")
            ]

    return transformers_init_df, storages_init_df


def build_RH_model(
    path_folder_input,
    filename_node_data,
    filename_cost_data,
    filename_node_timeseries,
    filename_min_max_timeseries,
    filename_cost_timeseries,
    AggregateInput,
    fuel_cost_pathway,
    investment_cost_pathway,
    endyear,
    MaxInvest,
    myopic_horizon_in_years,
    timeseries_start,
    timeslice_length_with_overlap,
    counter,
    transformers_init_df,
    storages_init_df,
    freq,
    multiplicator,
    overlap_in_timesteps,
    years_per_timeslice,
    total_exo_com_costs_df_RH,
    total_exo_com_capacity_df_RH,
    total_exo_decom_capacity_df_RH,
    IR=0.02,
    discount=False,
    ActivateEmissionsLimit=False,
    emission_pathway="100_percent_linear",
    ActivateInvestmentBudgetLimit=False,
    investment_budget=None,
    ActivateDemandResponse=False,
    approach="DIW",
    scenario="50",
):
    """Set up and return a rolling horizon LP dispatch model

    Parameters
    ----------
    path_folder_input : :obj:`str`
        The file path where input files are stored (common folder)

    filename_node_data : :obj:`str`
        Name of Excel Workbook containing all data for creating nodes (buses and oemof components)

    filename_cost_data : :obj:`str`
        Name of Excel Workbook containing cost pathways for oemof components

    filename_node_timeseries : :obj:`str`
        Filename of the node timeseries data, given in a separate .csv file

    filename_min_max_timeseries  : :obj:`str`
       Filename of the min / max transformer data, given in a separate .csv file

    filename_cost_timeseries : :obj:`str`
        Filename of the cost timeseries data, given in a separate .csv file

    AggregateInput: :obj:`boolean`
        If True an aggregated transformers input data set is used, elsewhise
        the full transformers input data set is used

    fuel_cost_pathway : :obj:`str`
        variable indicating which fuel cost pathway to use
        Possible values 'lower', 'middle', 'upper'

    investment_cost_pathway : :obj:`str`
        variable indicating which investment cost pathway to use
        Possible values 'lower', 'middle', 'upper'

    endyear : :obj:`int`
        The endyear of the optimization run

    MaxInvest : :obj:`boolean`
        If True, investment limits per technology are applied

    myopic_horizon_in_years : :obj:`int`
        The length of a myopic iteration in years

    timeseries_start : :obj:`pd.Timestamp`
        the starting timestep for used for the iteration

    timeslice_length_with_overlap : :obj:`int`
        the duration of a timeslice in timesteps including overlap
        (Usually no overlap used in investment model formulation)

    counter : :obj:`int`
        A counter for the myopic iteration

    transformers_init_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the transformer data from previous model runs

    storages_init_df : :obj:`pandas.DataFrame`
        A pd.DataFrame containing the storage data from previous model runs

    freq : :obj:`str`
        The frequency used for the datetimeindex of the optimization run

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

    total_exo_com_costs_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall costs for exogeneous investements

    total_exo_com_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous investements

    total_exo_decom_capacity_df_RH : :obj:`pd.DataFrame`
        A DataFrame containing the overall capacity for exogeneous decommissioning decisions

    IR : :obj:`float`
        The interest rate used for discounting

    discount : :obj:`boolean`
        Boolean parameter indicating whether or not to discount future investment costs

    ActivateEmissionsLimit : :obj:`boolean`
        If True, an emission limit is introduced

    emission_pathway : str
        The pathway for emissions reduction to be used

    ActivateInvestmentBudgetLimit : :obj:`boolean`
        If True, an overall investment budget limit is introduced

    investment_budget : float
        The overall investment budget limit to be used

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
    om : :class:`oemof.solph.models.Model`
        The mathematical optimisation model to be solved

    es : :class:`oemof.solph.network.EnergySystem`
        The energy system itself (used for determining initial states for the
        next rolling horizon iteration)

    timeseries_start : :obj:`pd.Timestamp`
        the adjusted starting timestep for used the next iteration

    new_built_transformer_labels : :obj:`list` of `str` values
        List of transformer labels
        (passed to a DataFrame in function initial_states_RH and
        used for next iteration)

    new_built_storage_labels :obj:`list` of `str` values
        List of storages labels
        (passed to a DataFrame in function initial_states and
        used for next iteration)

    datetime_index : :obj:`pd.DateRange`
        datetime index for the current iteration

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
    # Set date_range object from start to endtime of each iteration
    datetime_index = pd.date_range(
        start=timeseries_start, periods=timeslice_length_with_overlap, freq=freq
    )

    startyear = timeseries_start.year

    if (startyear + myopic_horizon_in_years - 1) >= endyear:
        RH_endyear = endyear
    else:
        RH_endyear = startyear + myopic_horizon_in_years - 1

    logging.info("Starting optimization for optimization run " + str(counter))

    es = solph.EnergySystem(timeindex=datetime_index)

    # Crate all nodes of the energy system and return labels
    # as well as information on existing capacities
    (
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
    ) = nodes_from_excel_rh(
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
        fuel_cost_pathway,
        investment_cost_pathway,
        freq,
        multiplicator,
        overlap_in_timesteps,
        years_per_timeslice,
        IR=IR,
        discount=discount,
        ActivateEmissionsLimit=ActivateEmissionsLimit,
        emission_pathway=emission_pathway,
        ActivateDemandResponse=ActivateDemandResponse,
        approach=approach,
        scenario=scenario,
    )

    # Increase starting point for the next model run (do not lose time information)
    # Instead of incrementing, set as start point of the next full year (needed for frequencies > 24 H)
    timeseries_start = pd.Timestamp(
        year=RH_endyear + 1, month=1, day=1, hour=0, freq=freq
    )

    es.add(*node_dict.values())
    logging.info("Sucessfully set up energy system for iteration " + str(counter))

    om = solph.Model(es)

    om = add_further_constrs(
        om,
        ActivateEmissionsLimit,
        ActivateInvestmentBudgetLimit,
        emissions_limit,
        investment_budget,
    )

    return (
        om,
        es,
        timeseries_start,
        new_built_transformer_labels,
        new_built_storage_labels,
        datetime_index,
        endo_exo_exist_df,
        endo_exo_exist_stor_df,
        existing_storage_labels,
        total_exo_com_costs_df_RH,
        total_exo_com_capacity_df_RH,
        total_exo_decom_capacity_df_RH,
    )


def solve_RH_model(
    om,
    datetime_index,
    counter,
    startyear,
    myopic_horizon_in_years,
    timeslice_length_wo_overlap_in_timesteps,
    results_sequences,
    results_scalars,
    overall_objective,
    overall_solution_time,
    new_built_storage_labels,
    existing_storage_labels,
    IR=0.02,
    discount=False,
    solver="gurobi",
):
    """Function for solving the rolling horizon optimization for a given window.
    Returns the results as well as updated information on the overall_solution
    and overall solution_time.

    Parameters
    ----------
    om : :class:`oemof.solph.models.Model`
        The mathematical optimisation model to be solved

    datetime_index : :obj:`pd.date_range`
        The datetime index of the energy system

    startyear : :obj:`int`
        The start year of the optimization window (iteration)

    counter : :obj:`int`
        A counter for rolling horizon optimization windows (iterations)

    timeslice_length_wo_overlap_in_timesteps : :obj:`int`
        Determines the length of a single (rolling horizon) optimization window
        (excluding overlap)

    results_sequences : :obj:`pd.DataFrame`
        A DataFrame to store the results of every optimization window
        (for processing the results as well as a model comparison)

    results_scalars :obj:`pd.DataFrame`
        A DataFrame to store the investment decisions taken

    overall_objective : :obj:`float`
        The overall objective value

    overall_solution_time :obj:`float`
        The overall solution time

    IR : :obj:`pandas.DataFrame`
        A pd.DataFrame carrying the WACC information by technology / energy carrier

    discount : :obj:`boolean`
        If True, nominal values will be dicounted
        If False, real values have to be used as model inputs (default)

    solver : :obj:`str`
        The solver to be used (defaults to 'gurobi')

    Returns
    -------
    df_rcut : :obj:`pd.DataFrame`
        A DataFrame needed to store results of the current optimization window
        (for a model comparison)

    results_sequences : :obj:`pd.DataFrame`
        A DataFrame to store the results of every optimization window
        (for processing the results as well as a model comparison)

    results_scalars :obj:`pd.DataFrame`
        A DataFrame to store the investment decisions taken

    overall_objective : :obj:`float`
        The overall objective value

    overall_solution_time :obj:`float`
        The overall solution time

    """
    RH_startyear = startyear + (counter * myopic_horizon_in_years)
    # Solve the mathematical optimization model using the given solver
    om.solve(solver=solver, solve_kwargs={"tee": True})
    print("********************************************************")

    logging.info("Model run %s done!" % (str(counter)))

    # JFG: Not necesary anymore after update to v.0.4.1
    model_results = solph.processing.results(om)

    # Obtain electricity results
    electricity_bus = solph.views.node(model_results, "DE_bus_el")
    electricity_bus_scalars = electricity_bus["scalars"]
    # Save results (excluding overlap)
    df_rcut = pd.DataFrame(
        data=electricity_bus["sequences"][0:timeslice_length_wo_overlap_in_timesteps]
    )

    # Create storage_labels to iterate over them to get the capacity scalars and sequences
    storage_labels = new_built_storage_labels + existing_storage_labels

    storage_results_dict = {}
    for i in storage_labels:
        storage_results_dict[i] = solph.views.node(model_results, i)
        storage_capacity_results = pd.DataFrame(
            data=storage_results_dict[i]["sequences"][
                0:timeslice_length_wo_overlap_in_timesteps
            ][[((i, "None"), "storage_content")]]
        )
        df_rcut = pd.concat([df_rcut, storage_capacity_results], axis=1, sort=True)
        if i in new_built_storage_labels:
            capacity_invest_results = storage_results_dict[i]["scalars"][
                [((i, "None"), "invest")]
            ]
            electricity_bus_scalars = pd.concat(
                [electricity_bus_scalars, capacity_invest_results], axis=0, sort=True
            )

    results_scalars = pd.concat(
        [
            results_scalars,
            electricity_bus_scalars,
        ],
        axis=1,
        sort=True,
    )
    results_sequences = results_sequences.append(df_rcut, sort=True)

    meta_results = solph.processing.meta_results(om)

    # NOTE: objective is calculated including overlap -> No slicing possible here
    if not discount:
        overall_objective += int(meta_results["objective"])
    else:
        overall_objective += int(
            (meta_results["objective"]) / ((1 + IR) ** (RH_startyear - startyear))
        )

    overall_solution_time += meta_results["solver"]["Time"]

    return (
        om,
        results_sequences,
        results_scalars,
        overall_objective,
        overall_solution_time,
    )


# TODO: Resume here, JK / YW
def reconstruct_objective_value(om):
    """WORK IN PROGRESS; NO WARRANTY, THERE MAY BE BUGS HERE!"""
    variable_costs = 0
    gradient_costs = 0
    investment_costs = 0

    for i, o in om.FLOWS:
        if om.flows[i, o].variable_costs[0] is not None:
            for t in om.TIMESTEPS:
                variable_costs += (
                    om.flow[i, o, t]
                    * om.objective_weighting[t]
                    * om.flows[i, o].variable_costs[t]
                )

        if om.flows[i, o].positive_gradient["ub"][0] is not None:
            for t in om.TIMESTEPS:
                gradient_costs += (
                    om.flows[i, o].positive_gradient[i, o, t]
                    * om.flows[i, o].positive_gradient["costs"]
                )

        if om.flows[i, o].negative_gradient["ub"][0] is not None:
            for t in om.TIMESTEPS:
                gradient_costs += (
                    om.flows[i, o].negative_gradient[i, o, t]
                    * om.flows[i, o].negative_gradient["costs"]
                )

        if om.flows[i, o].investment.ep_costs is not None:
            investment_costs += (
                om.flows[i, o].invest[i, o] * om.flows[i, o].investment.ep_costs
            )

    return variable_costs + gradient_costs + investment_costs


# TODO, JK
def dump_es(om, es, path, timestamp):
    """Function creates a dump of the given energy system including its
    results as well as meta results"""

    # 17.05.2019, JK: Add results to the energy system to make it possible to store them.
    es.results["main"] = solph.processing.results(om)
    es.results["meta"] = solph.processing.meta_results(om)

    filename = "es_dump_" + timestamp + ".oemof"

    # 17.05.2019, JK: dump the energy system at the path defined
    es.dump(dpath=path, filename=filename)

    return None
