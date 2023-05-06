Reference docs
==============

.. contents::

pommesinvest.cli
------------------
`cli.py` is the command line interface. The method `run_pommes_invest` triggers
a ``pommesinvest`` model run. For usage please see :ref:`using`.

pommesinvest.model
---------------------
The ``pommesinvest.model`` package contains the actual model which
is a single file ``investment_model.py``.

pommesinvest.model_funcs
---------------------------
The ``pommesinvest.model_funcs`` package contains a collection of functions,
classes and methods that are needed to create a investment model in
``pommesinvest.model.investment_model.py``.

Modules
+++++++

* ``model_control.py``: Controls a model workflow consisting of defining a model
  configuration, controlling logging information, reading in input data, building
  and solving an oemof.solph model and a pyomo model. The class ``InvestmentModel``
  holds parameterization information as well as methods in order to control the
  model workflow. Makes use of the modules ``data_input.py`` as well as ``helpers.py``.
* ``data_input.py``: Holds functions to parse input data from .csv files and to
  create oemof.solph components (see
  `oemof.solph's user's guide <https://oemof-solph.readthedocs.io/en/latest/usage.html#>`_)
  out of it. Optionally includes an emissions limit.
* ``subroutines.py``: Includes all the functions to actually build the
  oemof.solph components given the provided input data. Is imported in the
  module ``data_input.py``.
* ``helpers.py``: Includes some helper routines to support the model workflow
  that are imported in ``model_control.py`` resp. ``data_input.py`` in order
  to perform some calculations not directly related to the actual mathematical
  model, but facilitating its setup and parameterization.

For an in-depth documentation of all modules, functions, classes and methods,
see the :doc:`api/api`.