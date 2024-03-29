Model description
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   background
   categorization
   formulation

Feel free to directly jump to the section of interest.

The **investment variant** of the power market model *POMMES* ``pommesinvest``
enables the user to optimize the **investments and dispatch of backup power plants, storages as well as demand response units for the Federal Republic of Germany**
for long-term time frames spanning from 2020 to up to 2045 (or even 2050).
The dispatch of renewable power plants is exogenously determined
by normalized infeed time series which are multiplied with capacity values
(maximum capacities for foreign countries, installed ones for Germany). Also,
Investments for countries other than Germany are exogenously given and for these countries,
only the dispatch of capacities is optimized.
The models' overall goal is to minimize power system costs
occurring from wholesale markets, whereby no network constraints
are considered except for the existing bidding zone configuration
used for modeling electricity exchange.
Thus, the model purpose is to simulate **investments and dispatch decisions**.

You can find the following information on the subpages:

- :ref:`background`: economical, mathematical and technical background information
    - :ref:`economics`: Some basics of power wholesale markets design and how our model
      fits in there
    - :ref:`granularity`: Information on the default resolution in terms of technologies,
      time and space
    - :ref:`maths`: A brief mathematical characterisation of our linear programming
      model
    - :ref:`techs`: Information on the main python packages used, such as ``oemof.solph``
- :ref:`characteristics`: An in-depth characterization of our model, following
  the proposed scheme from Hall and Buckley (2016).
- :ref:`formulas`: A complete mathematical description including all (in)equations of the model.




