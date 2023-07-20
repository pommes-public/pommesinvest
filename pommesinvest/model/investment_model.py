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
import argparse
import logging
import time

import pandas as pd
import yaml
from oemof.solph import processing
from oemof.solph import views
from pyomo.common.tempfiles import TempfileManager
from yaml.loader import SafeLoader

from pommesinvest.model_funcs import model_control
from pommesinvest.model_funcs.helpers import make_directory_if_missing
from pommesinvest.model_funcs.results_processing import (
    process_demand_response_results,
    filter_storage_results,
    process_ev_bus_results,
    filter_european_country_results,
)


def run_investment_model(config_file="./config.yml"):
    """
    Run a pommesinvest model.

    Read in config information from a yaml file, initialize and run a
    investment model and process results.

    Parameters
    ----------
    config_file: str
        A file holding the necessary configuration information for
        a pommesinvest model
    """
    # ---- MODEL CONFIGURATION ----

    # Import model config from yaml config file
    with open(config_file) as file:
        config = yaml.load(file, Loader=SafeLoader)

    im = model_control.InvestmentModel()
    im.update_model_configuration(
        config["control_parameters"],
        config["time_parameters"],
        config["input_output_parameters"],
        nolog=True,
    )

    if im.myopic_horizon:
        im.add_myopic_horizon_configuration(
            config["myopic_horizon_parameters"], nolog=True
        )

    im.initialize_logging()
    im.check_model_configuration()
    im.show_configuration_log()

    # ---- MODEL RUN ----

    # Initialize model meta information and results DataFrames
    model_meta = {
        "overall_objective": 0,
        "overall_time": 0,
        "overall_solution_time": 0,
    }
    ts = time.gmtime()
    investment_results = pd.DataFrame()
    dispatch_results = pd.DataFrame()

    # Model run for integral optimization horizon (simple model set up)
    if not im.myopic_horizon:
        im.build_simple_model()

        if im.extract_duals:
            im.om.receive_duals()
            logging.info(
                "Obtaining dual values and reduced costs from the model\n"
                "in order to calculate power prices."
            )

        if im.write_lp_file:
            im.om.write(
                f"{im.path_folder_output}pommesinvest_model.lp",
                io_options={"symbolic_solver_labels": True},
            )
        if im.solver_tmp_dir != "default":
            logging.info(
                f"Adjusting directory to store tmp solver files. "
                f"Temporary files are stored at {im.solver_tmp_dir}"
            )
            if -1 < im.solver_tmp_dir.find("./") <= 1:
                make_directory_if_missing(im.solver_tmp_dir, relative=True)
            else:
                make_directory_if_missing(im.solver_tmp_dir)
            TempfileManager.tempdir = im.solver_tmp_dir
        if im.solver_commandline_options:
            logging.info(
                "Using solver command line options.\n"
                "Ensure that these are the correct ones for your solver of "
                "choice since otherwise, this might lead to "
                "the solver to run into an Error."
            )
            im.om.solve(
                solver=im.solver,
                solve_kwargs={"tee": True},
                cmdline_options=config["solver_cmdline_options"],
            )
        else:
            im.om.solve(solver=im.solver, solve_kwargs={"tee": True})

        if im.extract_duals:
            power_prices = im.get_power_prices_from_duals()

        meta_results = processing.meta_results(im.om)

        model_meta["overall_objective"] = meta_results["objective"]
        model_meta["overall_solution_time"] += meta_results["solver"]["Time"]

    # Model run for myopic horizon optimization
    if im.myopic_horizon:
        logging.info(
            "Creating an LP optimization model for investment optimization\n"
            "using a MYOPIC HORIZON approach for model solution."
        )

        # Initialization of myopic horizon model run
        iteration_results = {
            "new_built_transformers": pd.DataFrame(),
            "exogenous_storages": pd.DataFrame(),
            "new_built_storages": pd.DataFrame(),
            "model_results": {},
            "dispatch_results": dispatch_results,
            "investment_results": investment_results,
        }

        for counter in range(getattr(im, "amount_of_time_slices")):
            # rebuild the EnergySystem in each iteration
            im.build_myopic_horizon_model(counter, iteration_results)

            # Solve myopic horizon model
            im.solve_myopic_horizon_model(
                counter, iteration_results, model_meta
            )

            # Get initial states for the next model run from results
            im.retrieve_initial_states_myopic_horizon(iteration_results)

        dispatch_results = iteration_results["dispatch_results"]
        investment_results = iteration_results["investment_results"]

    model_meta["overall_time"] = time.mktime(time.gmtime()) - time.mktime(ts)

    # ---- MODEL RESULTS PROCESSING ----

    model_control.show_meta_logging_info(model_meta)

    if not im.myopic_horizon:
        model_results = processing.results(im.om)

        dispatch_results = views.node(model_results, "DE_bus_el")["sequences"]
        if im.extract_other_countries_production:
            dispatch_results = pd.concat(
                [
                    dispatch_results,
                    filter_european_country_results(im, model_results),
                ],
                axis=1,
            )
        investment_results = views.node(model_results, "DE_bus_el")[
            "period_scalars"
        ]
        electrolyzer_investment_results = views.node(
            model_results, "DE_bus_hydrogen"
        )["period_scalars"]

        investments_to_concat = [
            investment_results,
            electrolyzer_investment_results,
        ]

        for storage in im.new_built_storages:
            filtered_storage_results = filter_storage_results(
                views.node(model_results, storage)["period_scalars"]
            )
            investments_to_concat.append(filtered_storage_results)

        if im.activate_demand_response:
            dispatch_to_concat = [dispatch_results]
            for cluster in im.demand_response_clusters:
                investments_to_concat.append(
                    views.node(model_results, cluster)["period_scalars"]
                )
                processed_demand_response_results = (
                    process_demand_response_results(
                        views.node(model_results, cluster)["sequences"]
                    )
                )
                dispatch_to_concat.append(processed_demand_response_results)
            for bus in im.ev_buses:
                processed_ev_bus_results = process_ev_bus_results(
                    views.node(model_results, bus)["sequences"]
                )
                dispatch_to_concat.append(processed_ev_bus_results)
            dispatch_results = pd.concat(dispatch_to_concat, axis=1)

        investment_results = pd.concat(investments_to_concat)

    if im.save_investment_results:
        investment_results = investment_results.round(
            im.results_rounding_precision
        )
        investment_results.to_csv(
            im.path_folder_output
            + getattr(im, "filename")
            + "_investment.csv",
            sep=",",
            decimal=".",
        )

    if im.save_production_results:
        dispatch_results = dispatch_results.round(
            im.results_rounding_precision
        )
        dispatch_results.to_csv(
            im.path_folder_output
            + getattr(im, "filename")
            + "_production.csv",
            sep=",",
            decimal=".",
        )

    if im.extract_duals:
        power_prices = power_prices.round(im.results_rounding_precision)
        power_prices.to_csv(
            im.path_folder_output
            + getattr(im, "filename")
            + "_power-prices.csv",
            sep=",",
            decimal=".",
        )


def add_args():
    """Add command line argument for config file"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        required=False,
        default="./config.yml",
        help="Specify input config file",
    )
    parser.add_argument(
        "--init",
        required=False,
        action="store_true",
        help="Automatically generate default config",
    )
    return parser.parse_args()
