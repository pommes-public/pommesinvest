Installation and User's guide
=============================

.. contents::


Installation
------------
To set up ``pommesinvest``, you have to set up a virtual environment
(e.g. using conda) or add the required packages to your python installation.
This is taken care of by the pip installation. When you clone the environment,
you have to install the packages needed from the requirements file (see :ref:`setup`).

Additionally, you have to install a solver in order to solve
the mathematical optimization problem (see :ref:`solver`).

.. _setup:

Setting up the environment
++++++++++++++++++++++++++
``pommesinvest`` is hosted on `PyPI <https://pypi.org/projects/pommesinvest/>`_.
To install it, please use the following command

.. code::

    pip install pommesinvest


If you want to contribute as a developer, you fist have to
`fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_
it and then clone the repository, in order to copy the files locally by typing

.. code::

    git clone https://github.com/your-github-username/pommesinvest.git

| After cloning the repository, you have to install the required dependencies.
 In case you have conda (or a suitable alternative) installed as a package manager,
 you may follow along.
 If not, you can download conda `here <https://www.anaconda.com/>`_.
| Open a command shell and navigate to the folder
 where you copied the environment to.
| Use the following command to install dependencies

.. code::

    conda env create -f environment.yml

Activate your environment by typing

.. code::

    conda activate pommesinvest

.. _solver:

Installing a solver
+++++++++++++++++++
In order to solve a ``pommesinvest`` model instance,
you need a solver installed.
Please see
`oemof.solph's information on solvers <https://github.com/oemof/oemof-solph#installing-a-solver>`_.
As a default, gurobi is used for ``pommesinvest`` models.
It is a commercial solver, but provides academic licenses, though,
if this applies to you. Else, we recommend to use CBC
as the solver oemof recommends. To test your solver
and oemof.solph installation,
again see information from
`oemof.solph <https://github.com/oemof/oemof-solph#installation-test>`_.

.. _using:

Using pommesinvest
---------------------

Providing input data
++++++++++++++++++++

The input data is not stored within the ``pommesinvest`` repository.
You can obtain ``pommesinvest`` **input data** either by

* downloading and extracting data sets created with ``pommesdata`` which
  you can find on `zenodo <https://zenodo.org/>`_ or
* from directly running ``pommesdata`` which you can find
  `in this repository <https://github.com/pommes-public/pommesdata>`_.

``pommesdata`` provides input datat covering the simulation horizon 2020 to 2045 (or even 2050).
Once you have run it, you have to copy the output of ``pommesdata`` which
is stored in the folder "prepared_data"
to the "inputs" folder of ``pommesinvest``. You should now have a bunch
of .csv files with the oemof.solph components names in their file name.

.. note::

    ``pommesdata`` only includes time series information for 2017
    which is reindexed to another year if needed.

.. _config:

Configuring the model
+++++++++++++++++++++

We provide a default model configuration coming with the installation.
To make use of this, you have to run a console script by typing

.. code::

    run_pommes_invest --init

Once you have created the config file, you can adjust it to your needs.
If you do not wish to do so and use the default, you can skip this section
and move right to the next one, :ref:`running`.

You'll find dictionary-alike hierarchical entries in the ``config.yml``
file which control the simulation.
In the first section, you can change general model settings, e.g. if
you want to use another solver or if you want to run a rolling horizon
model. You can play around with the boolean values, but we recommend to
at least keep the parameters for storing result files, i.e.
``save_production_results`` and ``save_investment_results`` set to True and
``write_lp_file`` set to False.

Pay attention to the allowed values for the string values:

- ``countries``: The maximum of countries allowed is the default. You can just
  remove countries if you wish to have a smaller coverage
- ``fuel_cost_pathway``: allowed values are *NZE*, *APS*, *SDS*, *STEPS* and *regression*,
  reflecting scenario assumptions from the IEA's world energy outlook as well as
  a linear regression performed on historic data.
- ``emissions_pathway``: allowed values are *BAU*, *80_percent_linear*,
  *95_percent_linear*, *KNS_2035* or *100_percent_linear*,
  describing the emissions reduction path for the German power sector
  by a historic trend extrapolation (linear regression), an 80%
  reduction path until 2050, a 95% reduction path until 2050, the path from
  the study climate-neutral power sector by 2035 (Prognos on behalf of Agora Energiewende)
  or a 100% reduction path until 2045.
- ``demand_response_approach``: allowed values are *DLR*, *DIW* and *oemof*.
  These describe different options for demand response modeling implemented in
  oemof.solph, see `this oemof.solph module <https://github.com/oemof/oemof-solph/blob/dev/src/oemof/solph/custom/sink_dsm.py>`_
  and an `comparison of approaches from the INREC 2020 <https://github.com/jokochems/DR_modeling_oemof/blob/master/Kochems_Demand_Response_INREC.pdf>`_
  for details.

.. code:: yaml

    # 1) Set overall workflow control parameters
    control_parameters:
        multi_period: True
        myopic_horizon: False
        interest_rate: 0.02
        countries: # ["DE"]
            [
                "AT",
                "BE",
                "CH",
                "CZ",
                "DE",
                "DK1",
                "DK2",
                "FR",
                "NL",
                "NO1",
                "NO2",
                "NO3",
                "NO4",
                "NO5",
                "PL",
                "SE1",
                "SE2",
                "SE3",
                "SE4",
                "IT",
            ]
        solver: "gurobi"
        solver_commandline_options: False
        solver_tmp_dir: "default" # absolute or relative path; standard: "default"
        fuel_cost_pathway: "NZE"
        fuel_price_shock: "high"
        emissions_cost_pathway: "long-term"
        flexibility_options_scenario: "50"
        activate_emissions_budget_limit: False
        activate_emissions_pathway_limit: True
        emissions_pathway: "KNS_2035"
        use_technology_specific_wacc: True
        activate_demand_response: True
        demand_response_approach: "DLR"
        demand_response_scenario: "50"
        use_subset_of_delay_times: False
        impose_investment_maxima: True
        include_artificial_shortage_units: False
        save_production_results: True
        save_investment_results: True
        write_lp_file: False
        extract_duals: True
        extract_other_countries_production: True
        results_rounding_precision: 2
        sensitivity_parameter: "None"  # "None", "pv", "prices", "consumption"
        sensitivity_value: "None"  # "None", "-50%", "-25%", "+25%", "+50%"

In the next section, you can control the simulation time. Please stick
to the date format (pre-)defined. You have to ensure that the input data
time series matches time frame you want to simulate. ``pommesdata`` takes
care of that by reindexing your time series data accordingly.

.. code:: yaml

    # 2) Set model optimization time and frequency
    time_parameters:
        start_time: "2020-01-01 00:00:00"
        end_time: "2020-12-30 23:00:00"
        freq: "1H"  # "4H", "8H", "24H", "36H", "48H"

In the third section, you specify where your inputs and outputs are stored.
You can use the default values here. Please ensure that you have provided
the necessary input data.

.. code:: yaml

    # 3) Set input and output data paths
    input_output_parameters:
        path_folder_input: "./inputs/"
        path_folder_output: "./results/"

The next section is only applicable if you want to run a myopic
horizon simulation, see :ref:`myopic-horizon` for background information
if you are not familiar with the concept.

- ``myopic_horizon_in_years`` defines the length of a time slice
  excluding the overlap in years
- ``overlap_in_hours`` is the length of the overlap in hours, i.e. the number
  of hours that will be dropped and are only introduced to prevent end-time
  effects.

.. code:: yaml

    # 4) Set rolling horizon parameters (optional)
    rolling_horizon_parameters:
        myopic_horizon_in_years: 4
        overlap_in_years: 0

The last section is for controlling the solver behaviour. Parameters will only
be applied in case the respective control parameter above
(`solver_commandline_options`) is set to True. Note that the parameters are
solver specific. The following parameters have been applied using the CPLEX solver.

.. code:: yaml

    # 4) Set rolling horizon parameters (optional)
    solver_cmdline_options:
        lpmethod: 4
        preprocessing dual: -1
        solutiontype: 2
        threads: 12
        barrier convergetol: 1.0e-6

.. _running:

Running the model
+++++++++++++++++
Once you have configured your model, running it is fairly simple.

You can directly run the console script ``run_pommes_invest``
in a command line shell by typing

.. code::

    run_pommes_invest <-f "path-to-your-config-file.yml">

You may leave out the specification for the YAML file.
This will lead to using the ``config.yml`` file you have created when
initializing the config.

When you run the script, you'll see
some logging information on the console when your run the model.
Once the model run is finished, you can find, inspect, analyze and plot your
results in the results folder (or the folder you have specified to store
model results).

Another way is to run ``cli.run_pommes_invest`` in your python editor of choice
(e.g. `PyCharm <https://www.jetbrains.com/pycharm/>`_ or `VSCodium <https://vscodium.com/>`_).
In this case, you have to specify the path to your config file as a run
argument ``-f ../config.yml``.
Also, in the config file, you have to specify the relative
relations to the input and output folder, so you probably have to replace
``./inputs`` with ``../inputs`` and ``./outputs`` with ``../outputs``.
