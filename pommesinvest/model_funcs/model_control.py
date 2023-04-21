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

import oemof.solph as solph
import pandas as pd
# constraints, views, models, network, processing
from oemof.tools import logger

from pommesinvest.model_funcs import helpers
from pommesinvest.model_funcs.data_input import (
    nodes_from_csv,
    nodes_from_csv_myopic_horizon,
)
from pommesinvest.model_funcs.helpers import is_leap_year


def show_meta_logging_info(model_meta):
    """Show some logging information on model meta data"""
    logging.info("***** MODEL RUN TERMINATED SUCCESSFULLY :-) *****")
    logging.info(
        f"Overall objective value: {model_meta['overall_objective']:,.0f}"
    )
    logging.info(
        f"Overall solution time: {model_meta['overall_solution_time']:.2f}"
    )
    logging.info(f"Overall time: {model_meta['overall_time']:.2f}")


def adjust_datetime_index(periods):
    """Create a datetime index ignoring the leap days

    Parameters
    ----------
    periods : dict
        pd.date_ranges defining the time stamps for the respective period,
        starting with period 0

    Returns
    -------
    datetime_index : pd.date_range
        Actual datetime index of the model ignoring leap days
    """
    datetime_index = periods[0]
    for period, timeindex in periods.items():
        if period >= 1:
            datetime_index = datetime_index.append(timeindex)

    return datetime_index


class InvestmentModel(object):
    r"""A class that holds an investment model.

    An investment model is a container for all the model parameters as well
    as for methods for controlling the model workflow.

    Attributes
    ----------
    myopic_horizon : boolean
        boolean control variable indicating whether to run a myopic horizon
        optimization or an integral optimization run (a simple model).
        Note: For the myopic_horizon optimization run, additionally the
        parameters `time_slice_length_wo_overlap_in_hours` and
        `overlap_in_hours` (both of type int) have to be defined.

    interest_rate : float
        Interest rate used for discounting

    countries : list of str
        List of countries to be simulated

    solver : str
        The solver to be used for solving the mathematical optimization model.
        Must be one of the solvers oemof.solph resp. pyomo support, e.g.
        'cbc', 'gplk', 'gurobi', 'cplex'.

    solver_commandline_options: bool
        If True, use solver command line option; If False, use solver defaults

    solver_tmp_dir: str
        Directory for solver to store tmp files;
        Will be adjusted if set other than 'default'
    
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

    fuel_price_shock : str
        The extend to which the Ukraine war price shock is assumed to
        influence near-term future fuel price development

        .. csv-table:: Price shocks and explanations
            :header: "price shock", "explanation"
            :widths: 20 80

            "high", "price peaks in 2026 and remains high until around 2030"
            "low"", "price quickly stabilizes and reaches pre-war levels by 2026"

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

    flexibility_options_scenario : str
        A predefined scenario for flexibility options parameters
        Options: '5', '50', '95', whereby '5' is the lower,
        i.e. rather pessimistic estimate

    activate_emissions_budget_limit : boolean
        boolean control variable indicating whether to introduce an overall
        emissions budget limit
        Note: Combining an emissions limit with comparatively high minimum
        loads of conventionals may lead to an infeasible model configuration
        since either one of the restrictions may not be reached.

    activate_emissions_pathway_limit : boolean
        boolean control variable indicating whether to introduce an
        emissions pathway limit
        Note: Combining an emissions limit with comparatively high minimum
        loads of conventionals may lead to an infeasible model configuration
        since either one of the restrictions may not be reached.

    emissions_pathway : str
        A predefined pathway for emissions reduction until 2045
        Options: '100_percent_linear', '95_percent_linear',
        '80_percent_linear', 'KNS_2035' or 'BAU'
        If an emissions budget limit is chosen, it is calculated based on the
        given pathway.

    use_technology_specific_wacc: boolean
        If True, discriminate wacc among technologies, else use interest_value
        as wacc (social planner with ideal level-playing field)
    
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
        Options: '5', '50', '95', whereby '5' is the lower,
        i.e. rather pessimistic estimate
    
    use_subset_of_delay_times : boolean
        If True, only allow for a subset of the given maximum delay time
        of demand response units. The allowed subset hereby is defined as
        [1, 2, delay_time/2, delay_time] since this reflects short-, medium-
        and long-term shifting processes while limiting model complexity.

    save_production_results : boolean
        boolean control variable indicating whether to save the dispatch
        results of the model run to a .csv file

    save_investment_results : boolean
        boolean control variable indicating whether to save the investment
        decision results of the model run to a .csv file

    write_lp_file : boolean
        boolean control variable indicating whether to save an lp file
        *CAUTION*: Only use for debugging when simulating small time frames

    extract_duals : boolean
        boolean control variable indicating whether to save dual values of bus
        balance constraint
    
    sensitivity_parameter: str
        Parameter for which to consider sensitivities; 
        Supported sensitivities are
        - "None": no sensitivity
        - "PV": Different split between solar PV and wind onshore for Germany
          (higher / lower PV generation)
        - "prices": Combined variation of fuel and CO2 prices
        - "consumption": variation of inflexible baseline consumption
    
    sensitivity_value: str
        Sensitivity to consider;
        Supported values are
        - "-50%": 50% lower value compared to normal one
        - "-25%": 25% lower value compared to normal one
        - "+25%": 25% higher value compared to normal one
        - "+50%": 50% higher value compared to normal one
    
    start_time : str
        A date string of format "YYYY-MM-DD hh:mm:ss" defining the start time
        of the simulation

    end_time : str
        A date string of format "YYYY-MM-DD hh:mm:ss" defining the end time
        of the simulation

    optimization_timeframe: int
        Model horizon in years

    freq : str
        Frequency of the simulation, i.e. frequency of the pandas.date_range
        object

    multiplier : int
        multiplier to transform parameters defined for hourly frequency

    path_folder_input : str
        The path to the folder where the input data is stored

    path_folder_output : str
        The path to the folder where the output data is to be stored

    om : :class:`oemof.solph.models.Model`
        The mathematical optimization model itself

    time_slice_length_wo_overlap_in_hours : int (optional, for myopic horizon)
        The length of a time slice for a myopic horizon model run in hours,
        not including an overlap

    overlap_in_time_steps : int (optional, for myopic horizon)
        The length of the overlap for a myopic horizon model run in hours;
        defaults to 0 for a regular, i.e. non-myopic model

    new_built_storages : list
        A list specifying all potentially new-built storages

    demand_response_clusters : list (optional, only for demand response)
        A list specifying the names of the demand response clusters introduced
    """  # noqa: E501

    def __init__(self):
        """Initialize an empty InvestmentModel object"""
        self.multi_period = None
        self.myopic_horizon = None
        self.interest_rate = None
        self.countries = None
        self.solver = None
        self.solver_commandline_options = None
        self.solver_tmp_dir = None
        self.fuel_cost_pathway = None
        self.fuel_price_shock = None
        self.emissions_cost_pathway = None
        self.flexibility_options_scenario = None
        self.activate_emissions_budget_limit = None
        self.activate_emissions_pathway_limit = None
        self.emissions_pathway = None
        self.use_technology_specific_wacc = None
        self.activate_demand_response = None
        self.demand_response_approach = None
        self.demand_response_scenario = None
        self.use_subset_of_delay_times = None
        self.save_production_results = None
        self.save_investment_results = None
        self.write_lp_file = None
        self.extract_duals = None
        self.sensitivity_parameter = None
        self.sensitivity_value = None
        self.start_time = None
        self.end_time = None
        self.optimization_timeframe = None
        self.freq = None
        self.multiplier = None
        self.path_folder_input = None
        self.path_folder_output = None
        self.om = None
        self.overlap_in_time_steps = 0

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
                    self.set_multiplier()

        if hasattr(self, "start_time"):
            setattr(
                self, "start_year", str(pd.to_datetime(self.start_time).year)
            )
        if hasattr(self, "end_time"):
            setattr(self, "end_year", str(pd.to_datetime(self.end_time).year))

    def set_multiplier(self):
        """Set multiplier and timesteps dependent on frequency attribute"""
        self.multiplier = helpers.FREQUENCY_TO_TIMESTEPS[self.freq][
            "multiplier"
        ]

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
                logging.info(
                    f"Using fuel cost pathway: {getattr(self, entry)}"
                )
            elif entry == "emissions_cost_pathway":
                logging.info(
                    f"Using emissions cost pathway: {getattr(self, entry)}"
                )
            elif entry == "investment_cost_pathway":
                logging.info(
                    f"Using investment cost pathway: {getattr(self, entry)}"
                )

        if (
            self.activate_emissions_budget_limit
            and self.activate_emissions_pathway_limit
        ):
            raise ValueError(
                "You set both, 'activate_emissions_budget_limit' and "
                "'activate_emissions_pathway_limit' to True.\n"
                "They are mutually exclusive. "
                "Choose either one of both and set the other to False."
            )

        return missing_parameters

    def add_myopic_horizon_configuration(
        self, myopic_horizon_parameters, nolog=False
    ):
        r"""Add a myopic horizon configuration to the dispatch model

        .. _note:

            The amount of time steps is limited in such a way that only
            complete time slices are used. If the time series do not
            allow for adding another time slice, the last couple of time
            steps of the time series are not used.
        """
        self.update_model_configuration(myopic_horizon_parameters, nolog=nolog)

        setattr(
            self, "time_series_start", pd.Timestamp(self.start_time, self.freq)
        )
        setattr(
            self, "time_series_end", pd.Timestamp(self.end_time, self.freq)
        )

        setattr(
            self,
            "time_slice_length_wo_overlap_in_time_steps",
            (
                helpers.FREQUENCY_TO_TIMESTEPS[self.freq]["timesteps"]
                * getattr(self, "myopic_horizon_in_years")
            ),
        )
        setattr(
            self,
            "overlap_in_time_steps",
            (
                helpers.FREQUENCY_TO_TIMESTEPS[self.freq]["timesteps"]
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

    def add_demand_response_clusters(self, demand_response_clusters):
        """Append the information on demand response clusters to the model

        Parameters
        ----------
        demand_response_clusters : list
            Demand response clusters to be considered
        """
        setattr(self, "demand_response_clusters", demand_response_clusters)

    def add_new_built_storages(self, new_built_storages):
        """Append the information on new-built storage units to the model

        Parameters
        ----------
        new_built_storages : list
            Potentially new-built storage units
        """
        setattr(self, "new_built_storages", new_built_storages)

    def initialize_logging(self):
        """Initialize logging by deriving a filename from the configuration"""
        setattr(
            self,
            "optimization_timeframe",
            helpers.years_between(self.start_time, self.end_time) + 1,
        )

        if not self.myopic_horizon:
            rh = "simple"
        else:
            rh = "RH"
        if not self.activate_demand_response:
            dr = "no_dr"
        else:
            dr = "with_dr"

        filename = (
            f"investment_LP_start-{self.start_time[:10]}_"
            f"{self.optimization_timeframe}-years_{rh}_freq_{self.freq}_"
            f"{dr}_{self.demand_response_scenario}_"
            f"fuel_price-{self.fuel_cost_pathway}_{self.fuel_price_shock}_"
            f"co2_price-{self.emissions_cost_pathway}"
        )
        if self.sensitivity_parameter != "None":
            filename += (
                f"_sensitivity_{self.sensitivity_parameter}_"
                f"{self.sensitivity_value}"
            )

        setattr(self, "filename", filename)
        logger.define_logging(logfile=f"{filename}.log")

        return filename

    def show_configuration_log(self):
        """Show some logging info dependent on model configuration"""
        if self.activate_demand_response:
            dr_string = (
                f"Using approach '{self.demand_response_approach}' "
                f"for DEMAND RESPONSE modeling\n"
                f"Considering a {self.demand_response_scenario}% scenario"
            )
            if self.use_subset_of_delay_times:
                dr_string += " using only a SUBSET of DELAY TIMES"

        else:
            dr_string = "Running a model WITHOUT DEMAND RESPONSE"

        if self.sensitivity_parameter != "None":
            sensitivity_string = (
                f"Considering PARAMETRIC SENSITIVITY for parameter "
                f"{self.sensitivity_parameter}, altering the default "
                f"values by {self.sensitivity_value}."
            )
            logging.info(sensitivity_string)

        logging.info(dr_string)

    def build_simple_model(self):
        r"""Set up and return a simple model

        Construct a model for an overall optimization run
        not including any measures for complexity reduction.
        """
        logging.info("Starting optimization")
        logging.info(
            "Running an integrated INVESTMENT AND DISPATCH OPTIMIZATION"
        )

        datetime_index = pd.date_range(
            self.start_time, self.end_time, freq=self.freq
        )
        periods = None
        if self.freq != "48H":
            periods = self.determine_periods(datetime_index)
            datetime_index = adjust_datetime_index(periods)

        if self.multi_period:
            es = solph.EnergySystem(
                timeindex=datetime_index,
                timeincrement=[self.multiplier] * len(datetime_index),
                periods=periods,
                freq=self.freq,
                multi_period=True,
                infer_last_interval=False,
            )
        else:
            es = solph.EnergySystem(timeindex=datetime_index)

        nodes_dict, emissions_limit = nodes_from_csv(self)

        logging.info(
            "Creating a LP model for INVESTMENT AND DISPATCH OPTIMIZATION."
        )

        es.add(*nodes_dict.values())
        setattr(self, "om", solph.Model(es))

        self.add_further_constrs(emissions_limit)

    def add_further_constrs(self, emissions_limit, countries=None, fuels=None):
        r"""Integrate further constraints into the optimization model

        For now, an additional overall emissions budget limit or an
        emissions pathway limit can be imposed.

        Note that setting an emissions limit may conflict with high minimum
        loads from conventional transformers.
        Be aware that this may lead to model infeasibility
        if commodity bus balances cannot be met.

        Also note that since flows minima and maxima are multiplied with
        the multiplier accounting for the model frequency, considering
        the model time increment (equal to the multiplier) would lead
        to double accounting for emissions. Thus, emission factors are
        scaled down using the multiplier.

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
                # Correct emission factors by multiplier
                # in order not to double account emissions
                self.om.flows[(i, o)].emission_factor = (
                    self.om.flows[(i, o)].emission_factor / self.multiplier
                )
                emission_flows[(i, o)] = self.om.flows[(i, o)]

        if self.activate_emissions_budget_limit:
            solph.constraints.emission_limit(
                self.om, flows=emission_flows, limit=emissions_limit
            )
            logging.info(
                f"Adding an EMISSIONS BUDGET LIMIT of {emissions_limit:,.0f} "
                f"t CO2"
            )
        elif self.activate_emissions_pathway_limit:
            solph.constraints.emission_limit_per_period(
                self.om, flows=emission_flows, limit=emissions_limit
            )
            logging.info(
                f"Introducing an EMISSIONS PATHWAY LIMIT using pathway "
                f"{self.emissions_pathway}."
            )

    def get_power_prices_from_duals(self):
        r"""Obtain the power price results for the dispatch model

        The power prices are obtained from the dual value of the
        Bus.balance constraint of the German electricity bus.

        Returns
        -------
        power_prices: :obj:`pd.DataFrame`
        """
        constr = self.om.BusBlock.balance

        power_prices_list = [
            self.om.dual[constr[index]]
            for index in constr
            if index[0].label == "DE_bus_el"
        ]
        power_prices = pd.DataFrame(
            data=power_prices_list,
            index=self.om.es.timeindex,
            columns=["Power price"],
        )

        return power_prices

    def determine_periods(self, datetimeindex):
        """Explicitly define and return periods of the energy system

        Ignore leap years, i.e. treat them as if they had 8 760 hours as well.
        The remainder is the same as oemof.solph._energy_system._add_periods().

        Parameters
        ----------
        datetimeindex : pd.date_range
            DatetimeIndex of the model comprising all time steps

        Returns
        -------
        periods : dict
            pd.date_ranges defining the time stamps for the respective period,
            starting with period 0
        """
        years = sorted(list(set(getattr(datetimeindex, "year"))))
        periods = {}
        filter_series = datetimeindex.to_series()
        for number, year in enumerate(years):
            start = filter_series.loc[filter_series.index.year == year].min()
            if not is_leap_year(year):
                end = filter_series.loc[filter_series.index.year == year].max()
            else:
                # Exclude last day / resp. leap day
                end = filter_series.loc[
                    (filter_series.index.year == year)
                    & (filter_series.index.month == 12)
                    & (filter_series.index.day != 31)
                ].max()
            periods[number] = pd.date_range(start, end, freq=self.freq)

        return periods

    def build_myopic_horizon_model(self, counter, iteration_results):
        r"""Set up and return a myopic horizon LP dispatch model

        Track the transformer and storage labels in order to obtain and pass
        transformer and storage investments as well as initial storage levels
        for each iteration. Set the end time of an iteration
        excluding the overlap to the start of the next iteration.

        Parameters
        ----------
        counter : int
            A counter for the myopic horizon optimization iterations

        iteration_results : dict
            A dictionary holding the results of the previous myopic horizon
            iteration
        """
        if self.multi_period:
            msg = (
                "A model cannot be a myopic horizon model "
                "and a multi-period model at once.\n"
                "Please choose one of both in the configuration and rerun."
            )
            raise ValueError(msg)

        logging.info(f"Starting optimization for optimization run {counter}")
        logging.info(
            f"Start of iteration {counter}: "
            + f"{getattr(self, 'time_series_start')}"
        )
        logging.info(
            f"End of iteration {counter}: "
            + f"{getattr(self, 'time_series_end')}"
        )

        datetime_index = pd.date_range(
            start=getattr(self, "time_series_start"),
            periods=getattr(self, "time_slice_length_with_overlap"),
            freq=self.freq,
        )
        es = solph.EnergySystem(timeindex=datetime_index)

        (
            node_dict,
            emissions_limit,
            transformer_and_storage_labels,
        ) = nodes_from_csv_myopic_horizon(self, iteration_results)
        # Only set storage and transformer labels attribute for 0th iteration
        if not hasattr(self, "transformer_and_storage_labels"):
            setattr(
                self,
                "transformer_and_storage_labels",
                transformer_and_storage_labels,
            )

        # Update model start time for the next iteration
        setattr(self, "time_series_start", getattr(self, "time_series_end"))

        es.add(*node_dict.values())
        logging.info(
            f"Successfully set up energy system for iteration {counter}"
        )

        self.om = solph.Model(es)

        self.add_further_constrs(emissions_limit)

    def solve_myopic_horizon_model(
        self, counter, iter_results, model_meta, no_solver_log=False
    ):
        """Solve a myopic horizon optimization model

        Parameters
        ----------
        counter : int
            A counter for the myopic horizon optimization iterations

        iter_results : dict
            A dictionary holding the results of the previous myopic horizon
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

        iter_results["model_results"] = solph.processing.results(self.om)
        electricity_bus = solph.views.node(
            iter_results["model_results"], "DE_bus_el"
        )
        sliced_dispatch_results = pd.DataFrame(
            data=electricity_bus["sequences"].iloc[
                0 : getattr(self, "time_slice_length_wo_overlap_in_time_steps")
            ]
        )
        iter_results["dispatch_results"] = iter_results[
            "dispatch_results"
        ].append(sliced_dispatch_results)
        iteration_investments = electricity_bus["scalars"]
        iter_results["investment_results"] = iter_results[
            "dispatch_results"
        ].append(iteration_investments, axis=1)

        meta_results = solph.processing.meta_results(self.om)
        # Objective is weighted in order to take overlap into account
        model_meta["overall_objective"] += int(
            meta_results["objective"]
            * (
                getattr(self, "time_slice_length_wo_overlap_in_time_steps")
                / getattr(self, "time_slice_length_with_overlap")
            )
        )
        model_meta["overall_solution_time"] += meta_results["solver"]["Time"]

    def retrieve_initial_states_myopic_horizon(self, iteration_results):
        r"""Retrieve the initial states for the upcoming myopic horizon run

        Parameters
        ----------
        iteration_results : dict
            A dictionary holding the results of the previous myopic horizon
            iteration
        """
        if iteration_results["new_built_transformers"].empty:
            iteration_results["new_built_tansformers"] = pd.DataFrame(
                columns=["existing_capacity"],
                index=getattr(self, "transformer_and_storage_labels")[
                    "new_built_transformers"
                ],
                # Intialize with zero value
                data=0,
            )

        for i, s in iteration_results["new_built_tansformers"].iterrows():
            transformer = solph.views.node(
                iteration_results["model_results"], i
            )

            # Increase capacity by results of each iteration
            iteration_results["new_built_tansformers"].at[
                i, "existing_capacity"
            ] += transformer["scalars"][((i, "DE_bus_el"), "invest")]

        if iteration_results["storages_exogenous"].empty:
            iteration_results["storages_exogenous"] = pd.DataFrame(
                columns=["initial_storage_level_last_iteration"],
                index=getattr(self, "transformer_and_storage_labels")[
                    "exogenous_storages"
                ],
            )

        for i, s in iteration_results["storages_exogenous"].iterrows():
            storage = solph.views.node(iteration_results["model_results"], i)

            iteration_results["storages_existing"].at[
                i, "initial_storage_level_last_iteration"
            ] = storage["sequences"][((i, "None"), "storage_content")].iloc[
                getattr(self, "time_slice_length_wo_overlap_in_time_steps") - 1
            ]

        if iteration_results["storages_new_built"].empty:
            iteration_results["storages_new_built"] = pd.DataFrame(
                columns=[
                    "initial_storage_level_last_iteration",
                    "existing_inflow_power",
                    "existing_outflow_power",
                    "existing_capacity_storage",
                ],
                index=getattr(self, "transformer_and_storage_labels")[
                    "new_built_storages"
                ],
                data=0,
            )

        for i, s in iteration_results["storages_new_built"].iterrows():
            storage = solph.views.node(iteration_results["model_results"], i)

            iteration_results["storages_new_built"].at[
                i, "initial_storage_level_last_iteration"
            ] = storage["sequences"][((i, "None"), "storage_content")].iloc[
                getattr(self, "time_slice_length_wo_overlap_in_time_steps") - 1
            ]

            iteration_results["storages_new_built"].at[
                i, "existing_inflow_power"
            ] += storage["scalars"].loc[(("DE_bus_el", i), "invest")]

            iteration_results["storages_new_built"].at[
                i, "existing_outflow_power"
            ] += storage["scalars"].loc[((i, "DE_bus_el"), "invest")]

            iteration_results["storages_new_built"].at[
                i, "existing_capacity_storage"
            ] += storage["scalars"].loc[((i, "None"), "invest")]

        logging.info(
            "Obtained initial (transformer and storage) levels for next iteration"
        )
