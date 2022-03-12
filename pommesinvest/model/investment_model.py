# coding: utf-8
"""
General description
-------------------
This is the investment variant of POMMES, the POwer Market Model
of Energy and reSources Department at TU Berlin.
The fundamental power market model has been originally developed
at TU Berlin and is now maintained by a developer group of alumni.
The source code is freely available under MIT license.
Usage of the model is highly encouraged. Contributing is welcome as well.

Repository, Documentation, Installation
---------------------------------------
All founds are hosted on
`GitHub <https://github.com/pommes-public/pommesinvest>`_

To install, simply type ``pip install pommesinvest``

Please find the documentation `here <https://pommesinvest.readthedocs.io/>`_

Licensing information and Disclaimer
------------------------------------
This software is provided under MIT License (see licensing file).

A special thank you goes out to all the developers creating,
maintaining, and expanding packages used in this model,
especially to the oemof and pyomo developer groups!

In addition to that, a special thank you goes to all students
and student assistants which have contributed to the model itself
or its data inputs.

Input Data
----------
Input data can be compiled using the ``pommesdata`` package.
A precompiled version is distributed with the investment model.

Installation requirements
-------------------------
See `environments.yml` file

@author: Johannes Kochems (*), Johannes Giehl (*), Yannick Werner,
Benjamin Grosse

Contributors:
Julien Faist, Hannes Kachel, Sophie Westphal, Flora von Mikulicz-Radecki,
Carla Spiller, Fabian Büllesbach, Timona Ghosh, Paul Verwiebe,
Leticia Encinas Rosa, Joachim Müller-Kirchenbauer

(*) Corresponding authors
"""


import time, calendar
import logging
import pandas as pd
from matplotlib import pyplot as plt

import oemof.solph as solph
from oemof.solph import processing
from oemof.solph import views
from oemof.tools import logger

# Import all functions necessary for a model run.
# A complete enumeration of functions is used here, so it can be seen which functions are imported
from functions_for_model_control_invest import determine_timeslices_RH, \
    add_further_constrs, \
    build_simple_model, build_RH_model, initial_states_RH, solve_RH_model, \
    dump_es, reconstruct_objective_value

from helper_functions_invest import years_between

from functions_for_processing_of_outputs_invest import \
    extract_model_results, create_aggregated_energy_source_results, create_aggregated_investment_results, \
    create_aggregated_investment_results_RH, create_results_to_save, \
    draw_production_plot, draw_investment_decisions_plot, draw_investment_decisions_plot_RH, \
    draw_exo_investment_decisions_plot, draw_decommissioning_plot

# %%

###############################################################################
### MODEL SETTINGS ############################################################
###############################################################################

# %%

### 1) Determine model configuration through control variables

# Set file version
file_version = '_2050'
lignite_phase_out = 2038
hardcoal_phase_out = 2038
MaxInvest = True

# Determine major model configuration
RollingHorizon = True
AggregateInput = True
Europe = False
solver = 'gurobi'

# Determine fuel and investment cost pathways (options: lower, middle, upper) 
# as well emission limits (options: BAU, 80_percent_linear, 95_percent_linear, 100_percent_linear, customized)
fuel_cost_pathway = 'middle'
investment_cost_pathway = 'middle'

# Interest rate and discounting
# Important Note: By default, exclusively real cost values are applied, so no discounting is needed.
# Hence, the interest rate is only effective if discounting is explicitly activated.
# Please make sure to be using nominal cost data if you do so (but best thing is, you don't use it at all).
IR = 0.04
discount = False

# Control Demand response modeling
# options for approach: ['DIW', 'DLR', 'IER', 'TUD']
# options for scenario: ['25', '50', '75']
ActivateDemandResponse = False
approach = 'DLR'
scenario = '50'

# Control emissions limit (options: BAU, 80_percent_linear,
# 95_percent_linear, 100_percent_linear)
ActivateEmissionsLimit = False
emission_pathway = '100_percent_linear'

ActivateInvestmentBudgetLimit = False

# 11.05.2019, JK: Determine processing of outputs

# Decide which grouping to apply on investment decision results;
# possibilities: "sources", "technologies", None
grouping = "sources"

Dumps = False
PlotProductionResults = True
PlotInvestmentDecisionsResults = True

SaveProductionPlot = False
SaveInvestmentDecisionsPlot = False

SaveProductionResults = True
SaveInvestmentDecisionsResults = True

# %%

### 2) Set model optimization time and frequency for simple model runs

# Define starting and end year of (overall) optimization run and define which 
# frequency shall be used. (End year is included.)
startyear = 2016
endyear = 2036
freq = '48H'  # for alternatives see dict below

# Determine the number of timesteps per year (no leap year, leap year, multiplicator)
# Multiplicator is used to adapt input data given for hourly timesteps
freq_timesteps = {'60min': (8760, 8784, 1),
                  '4H': (2190, 2196, 4),
                  '8H': (1095, 1098, 8),
                  '24H': (365, 366, 24),
                  '36H': (244, 244, 36),
                  '48H': (182, 183, 48),
                  '72H': (122, 122, 72),
                  '96H': (92, 92, 96),
                  '120H': (73, 73, 120),
                  '240H': (37, 37, 240)}[freq]

# Multiplicator used for getting frequency adjusted input data
# NOTE: Time shift (winter time CET / summer time CEST) is ignored
# Instead, UTC timestamps are used
multiplicator = freq_timesteps[2]

# Set myopic optimization horizon here
# NOTE: Rolling horizon approach is not a typical rolling horizon optimization,
# but a rolling annual window used for optimization modelling with myopic foresight
if RollingHorizon:
    # NOTE: Myopic horizon must be <= overall timeframe length and
    # furthermore, amount of years must be a multiple of myopic horizon length.
    # Else a KeyError is thrown and execution terminates and has to be started
    # again with parameters set according to the prerequisites here.
    myopic_horizon_in_years = 4

    # Overlap is set to 0 in the first place.
    overlap_in_timesteps = 0

# %%

# Values used for model control only (No need to change anything here)

# Create date strings including day, month and hour (used for the simulation)
# Date strings respresent the start of the first year and the end of the last one
starttime = str(startyear) + '-01-01 00:00:00'
endtime = str(endyear) + '-12-31 23:00:00'

# Get the current time and convert it to a human readable form
# 08.12.2018, JK: time.localtime() is actually the one to be used in combination with time.mktime()
# since here, there are no further arguments, it is identical to time.localtime() as well as time.time()
ts = time.gmtime()
# ts_2 = time.localtime()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", ts)
# 27.11.2018, CS: Calculate overall time and overall solution time 
overall_objective = 0
overall_time = 0
overall_solution_time = 0

# Get the length of the (overall) optimization timeframe in years
optimization_timeframe = years_between(starttime, endtime) + 1

# Create string amendments in such a way that strings can be used to be concatenated
# to filenames indicating which model configuration has been considered
basic_filename = 'invest_'

# This formulation is not really nice. Dicts don't work because keys must be unique.
if not RollingHorizon:
    RH = 'overall_'
    horizon_overlap = ''
else:
    RH = 'myopic_'
    horizon_overlap = str(myopic_horizon_in_years) + "years_" + str(overlap_in_timesteps) + "overlap_"
if AggregateInput:
    Agg = 'clustered_'
else:
    Agg = 'complete_'
if not Europe:
    EU = 'DE-only'
else:
    EU = 'EU'
if MaxInvest:
    max_inv = ''
else:
    max_inv = '_wo_invest_limits'

# Filename concatenates all infos relevant for the model run
# (timestamp of model creation is no longer included)

# filename = basic_filename + freq +"_" + RH + horizon_overlap + Agg + str(lignite_phase_out) + '_endyear' +str(endyear) + max_inv
filename = basic_filename + "start-" + starttime[:10] + "_" + str(optimization_timeframe) + "-years_" + RH + Agg + EU

# Initialize logger for logging information
# NOTE: Created an additional path to store .log files
logger.define_logging(logfile=filename + '.log')

# %%

### 3) Set input data

path_folder_input = './inputs/'
path_folder_input_csv = '../data/Outputlisten_Test_Invest/'

# Use aggregated input data if respective control parameter is
# set to True -> It is highly recommended to use aggregated input data
if AggregateInput:
    filename_min_max_timeseries = '4transformers_clustered_min_max_timeseries_' + 'BK' + str(
        lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.csv'
    # filename_min_max_timeseries = 'transformers_min_max_timeseries_clustered_2019-10-13.csv'
    if Europe:
        filename_node_data = '4power_market_input_data_invest_new_annually_clustered_' + 'BK' + str(
            lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.xlsx'
        # filename_node_data= 'node_input_data_invest_clustered_2019-10-13.xlsx'
        logging.info('Using the AGGREGATED POWER PLANT DATA SET for EUROPE')
    else:
        filename_node_data = '4power_market_input_data_invest_new_annually_clustered_' + 'BK' + str(
            lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.xlsx'
        # filename_node_data = 'node_input_data_invest_clustered_2019-10-13.xlsx'
        logging.info('Using the AGGREGATED POWER PLANT DATA SET for Germany')
# NOTE: This will not be applicable for investment modelling and might as well be dropped
else:
    filename_min_max_timeseries = '4transformers_min_max_timeseries_' + 'BK' + str(
        lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.csv'
    # filename_min_max_timeseries = 'transformers_min_max_timeseries_complete_2019-10-13.csv'
    if Europe:
        filename_node_data = '4power_market_input_data_invest_new_annually_' + 'BK' + str(
            lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.xlsx'
        # filename_node_data= 'node_input_data_invest_complete_2019-10-13.xlsx'
        logging.info('Using the COMPLETE POWER PLANT DATA SET for EUROPE. \n'
                     'Minimum power output constraint of (individual) \n'
                     'transformers will be neglected.')
    else:
        filename_node_data = '4power_market_input_data_invest_new_annually_' + 'BK' + str(
            lignite_phase_out) + '_SK' + str(hardcoal_phase_out) + file_version + '_JK.xlsx'
        # filename_node_data = 'node_input_data_invest_complete_2019-10-13.xlsx'
        logging.info('Using the COMPLETE POWER PLANT DATA SET for GERMANY. \n'
                     'Minimum power output constraint of (individual) \n'
                     'transformers will be neglected.')

# Input data containing timeseries information for nodes (except for cost data)
filename_node_timeseries = '5node_timeseries_' + 'BK' + str(lignite_phase_out) + '_SK' + str(
    hardcoal_phase_out) + file_version + '.csv'
# filename_node_timeseries = 'node_timeseries_2019-10-13.csv'

# Input data containing costs data
filename_cost_data = '2power_market_input_data_complete_cost' + file_version + '.xlsx'
# filename_cost_data = 'cost_input_data_invest_2019-10-13.xlsx'
filename_cost_timeseries = '3cost_timeseries' + file_version + '_JK.csv'
# filename_cost_timeseries = 'cost_timeseries_2019-10-13.csv'

# Initialize an investment budget.
investment_budget = 0

if ActivateInvestmentBudgetLimit:

    investment_budget_per_year = 1000000

    if RollingHorizon:
        investment_budget = investment_budget_per_year * myopic_horizon_in_years
    else:
        investment_budget = investment_budget_per_year * optimization_timeframe

if ActivateDemandResponse:
    logging.info('Using approach from {} for DEMAND RESPONSE modeling\n'
                 'Considering a {}% scenario'.format(approach, scenario))
else:
    logging.info('Running a model WITHOUT DEMAND RESPONSE')

# %%

### Calculate timeslice and model control information for Rolling horizon model runs

if RollingHorizon:
    # Set amount of timeslices and determine timeslice lengths for every iteration
    # timeslice_length_dict has tuples as values: (timeslice_length_wo_overlap, timeslice_length_with_overlap)
    timeseries_start, amount_of_timeslices, timeslice_length_dict \
        = determine_timeslices_RH(starttime,
                                  endtime,
                                  freq,
                                  freq_timesteps,
                                  myopic_horizon_in_years,
                                  overlap_in_timesteps)

# %%

###############################################################################
### MODEL RUN #################################################################
###############################################################################

# %%

### Model run for simple model set up

if not RollingHorizon:
    # Build the mathematical optimization model and obtain var_costs
    # In addition to that, calculate exo commissioning cost and total de/commissioned capacity
    # See function definition for details
    om, existing_storage_labels, new_built_storage_labels, \
    total_exo_com_costs_df, total_exo_com_capacity_df, total_exo_decom_capacity_df \
        = build_simple_model(path_folder_input,
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
                             ActivateInvestmentBudgetLimit=ActivateInvestmentBudgetLimit,
                             investment_budget=investment_budget,
                             ActivateDemandResponse=ActivateDemandResponse,
                             approach=approach,
                             scenario=scenario)

    # Solve the problem using the given solver
    om.solve(solver=solver, solve_kwargs={'tee': True})

    # Calculate overall objective and optimization time
    meta_results = processing.meta_results(om)
    overall_solution_time += meta_results['solver']['Time']

    # get the current time and calculate the overall time
    ts_2 = time.gmtime()
    # 08.12.2018, JK: See above at definition of ts...
    #    ts_2 = time.localtime()
    overall_time = time.mktime(ts_2) - time.mktime(ts)

    print("********************************************************")
    logging.info("Done!")
    print('Overall solution time: {:.2f}'.format(overall_solution_time))
    print('Overall time: {:.2f}'.format(overall_time))

# %%

### Myopic optimization with rolling window: Run invest model
# NOTE: This is a quasi rolling horizon approach

if RollingHorizon:

    logging.info('Creating a LP optimization model for INVESTMENT optimization \n'
                 'using a MYOPIC OPTIMIZATION ROLLING WINDOW approach for model solution.')

    # Initialization of RH model run 
    counter = 0
    transformers_init_df = pd.DataFrame()
    storages_init_df = pd.DataFrame()
    results_sequences = pd.DataFrame()
    results_scalars = pd.DataFrame()
    total_exo_com_costs_df_RH = pd.DataFrame()
    total_exo_com_capacity_df_RH = pd.DataFrame()
    total_exo_decom_capacity_df_RH = pd.DataFrame()

    # 14.10.2018, JK: For loop controls rolling horizon model run
    for counter in range(amount_of_timeslices):

        # (re)build optimization model in every iteration
        # Return the model, the energy system as well as the information on existing transformer and storage
        # capacity for each iteration.
        # In addition to that, calculate exo commissioning cost and total de/commissioned capacity
        # See function definitions for details
        om, es, timeseries_start, new_built_transformer_labels, \
        new_built_storage_labels, datetime_index, endo_exo_exist_df, \
        endo_exo_exist_stor_df, existing_storage_labels, \
        total_exo_com_costs_df_RH, total_exo_com_capacity_df_RH, total_exo_decom_capacity_df_RH \
            = build_RH_model(path_folder_input,
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
                             timeslice_length_with_overlap=timeslice_length_dict[counter][2],
                             counter=counter,
                             transformers_init_df=transformers_init_df,
                             storages_init_df=storages_init_df,
                             freq=freq,
                             multiplicator=multiplicator,
                             overlap_in_timesteps=overlap_in_timesteps,
                             years_per_timeslice=timeslice_length_dict[counter][0],
                             total_exo_com_costs_df_RH=total_exo_com_costs_df_RH,
                             total_exo_com_capacity_df_RH=total_exo_com_capacity_df_RH,
                             total_exo_decom_capacity_df_RH=total_exo_decom_capacity_df_RH,
                             IR=IR,
                             discount=discount,
                             ActivateEmissionsLimit=ActivateEmissionsLimit,
                             emission_pathway=emission_pathway,
                             ActivateInvestmentBudgetLimit=ActivateInvestmentBudgetLimit,
                             investment_budget=investment_budget,
                             ActivateDemandResponse=ActivateDemandResponse,
                             approach=approach,
                             scenario=scenario
                             )

        # 14.05.2019, JK: Solve model and return results
        om, results_sequences, results_scalars, \
        overall_objective, overall_solution_time \
            = solve_RH_model(om,
                             datetime_index,
                             counter,
                             startyear,
                             myopic_horizon_in_years,
                             timeslice_length_wo_overlap_in_timesteps=timeslice_length_dict[counter][1],
                             results_sequences=results_sequences,
                             results_scalars=results_scalars,
                             overall_objective=overall_objective,
                             overall_solution_time=overall_solution_time,
                             new_built_storage_labels=new_built_storage_labels,
                             existing_storage_labels=existing_storage_labels,
                             solver=solver)

        # 14.05.2019, JK: Set initial states for the next model run
        # See function definition for details
        transformers_init_df, storages_init_df = \
            initial_states_RH(om,
                              timeslice_length_wo_overlap_in_timesteps=timeslice_length_dict[counter][1],
                              new_built_transformer_labels=new_built_transformer_labels,
                              new_built_storage_labels=new_built_storage_labels,
                              endo_exo_exist_df=endo_exo_exist_df,
                              endo_exo_exist_stor_df=endo_exo_exist_stor_df)

        # 05.06.2019, JK: To Do: Check whether this dump works properly or 
        # if objects are still kept in memory.
        if Dumps:
            # 17.05.2019, JK: Dump energy system including results
            # (Pickling in order to reuse results later)
            dump_es(om, es, "./dumps/", timestamp)

        # TODO: 20.08.2019, JK: Get overall objective value for model comparison
        # overall_objective = reconstruct_objective_value(om)

        # TODO, JK: Get from experimental state to senseful integration
        # Do some garbage collection in every iteration
        gc.collect()

    # The following is carried out when all model runs are carried out

    # 27.11.2018, CS: get the current time and calculate the overall time
    # 08.12.2018, JK: Seems to work here, although according to documentation of package time, the function time.mktime()
    # should be used in combination with time.localtime(), not time.gmtime()... Commented this out below.
    ts_2 = time.gmtime()
    #    ts_2 = time.localtime()
    overall_time = calendar.timegm(ts_2) - calendar.timegm(ts)

    print('*************************************FINALLY DONE*************************************')
    print('Overall objective value: {:,.0f}'.format(overall_objective))
    print('Overall solution time: {:.2f}'.format(overall_solution_time))
    print('Overall time: {:.2f}'.format(overall_time))

# %%

###############################################################################
### PROCESS MODEL RESULTS #####################################################
###############################################################################

# %%

# 15.05.2019, JK: Store results
if not RollingHorizon:
    results_scalars, results_sequences = extract_model_results(om,
                                                               new_built_storage_labels,
                                                               existing_storage_labels)

# %%

### Create and visualize production schedule

# 16.05.2019, JK: Power contains the aggregated production results per energy source
# as well as information on storage infeed and outfeed and load; see function definition for details
Power = create_aggregated_energy_source_results(results_sequences)

# Create a nice stackplot of power plant production schedule as well as load
if PlotProductionResults:

    # 16.05.2019, JK: Draw a stackplot of the production results
    draw_production_plot(Power,
                         Europe)

    if SaveProductionPlot:
        path = "./results/"
        plt.savefig(path + filename + '_production.png', dpi=150, bbox_inches="tight")

    plt.show()

if SaveProductionResults:
    path = "./results/"
    results_sequences.to_csv(path + filename + '_production.csv', sep=';', decimal=',', header=True)

# %%

### Create and visualize investment decisions taken

if not RollingHorizon:
    Invest = create_aggregated_investment_results(results_scalars,
                                                  starttime,
                                                  grouping=grouping)

    print('Total exogenous commissioning costs: {:,.0f}'.format(total_exo_com_costs_df.sum().sum()))
    print('Objective + Total exogenous commissioning costs: {:,.0f}'.format(
        total_exo_com_costs_df.sum().sum() + meta_results['objective']))

    results_to_save = create_results_to_save(meta_results["objective"],
                                             total_exo_com_costs_df,
                                             total_exo_com_capacity_df,
                                             total_exo_decom_capacity_df,
                                             overall_solution_time,
                                             overall_time,
                                             Invest)

else:
    Cumulated_Invest, Invest \
        = create_aggregated_investment_results_RH(results_scalars,
                                                  amount_of_timeslices,
                                                  timeslice_length_dict,
                                                  starttime,
                                                  freq,
                                                  grouping=grouping)

    print('Total exogenous commissioning costs: {:,.0f}'.format(total_exo_com_costs_df_RH.sum().sum()))
    print('Objective + Total exogenous commissioning costs: {:,.0f}'.format(
        total_exo_com_costs_df_RH.sum().sum() + overall_objective))

    results_to_save = create_results_to_save(overall_objective,
                                             total_exo_com_costs_df_RH,
                                             total_exo_com_capacity_df_RH,
                                             total_exo_decom_capacity_df_RH,
                                             overall_solution_time,
                                             overall_time,
                                             Invest)

# In the first place, only new installations are shown.
# Later on, it seems reasonable to include decommissioning decisions as well
# on an annual basis using a stacked bar plot and two y-axis directions
if PlotInvestmentDecisionsResults:

    if not RollingHorizon:
        draw_investment_decisions_plot(Invest)
        draw_exo_investment_decisions_plot(total_exo_com_capacity_df)
        draw_decommissioning_plot(total_exo_decom_capacity_df)
        draw_investment_decisions_plot(results_to_save['net_commissioning'])

    else:
        draw_investment_decisions_plot_RH(Cumulated_Invest, Invest)
        draw_exo_investment_decisions_plot(total_exo_com_capacity_df_RH)
        draw_decommissioning_plot(total_exo_decom_capacity_df_RH)
        draw_investment_decisions_plot(results_to_save['net_commissioning'])

    if SaveInvestmentDecisionsPlot:
        path = "./results/"
        plt.savefig(path + filename + '_investments.png', dpi=150, bbox_inches="tight")

if SaveInvestmentDecisionsResults:
    path = "./results/"
    results_to_save.to_csv(path + filename + '_investments.csv', sep=';', decimal=',', header=True)
