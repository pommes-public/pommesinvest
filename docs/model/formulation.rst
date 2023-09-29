
.. _formulas:

Mathematical formulation
------------------------

All constraints formulations can be found in the
`oemof.solph documentation <https://oemof-solph.readthedocs.io/en/latest/reference/oemof.solph.html>`_ since ``pommesinvest`` makes use of this module.
The breakdown of the mathematical model formulation at hand focuses on the sets, parameters, variables, target function and
constraints actually applied within ``pommesinvest``.

Nomenclature
++++++++++++

.. csv-table:: Sets (S), Variables (V) and Parameters (P)
    :header: **name**, **type**, **description**
    :widths: 20, 10, 70

    ":math:`N`", "S", "| all components of the energy system.
    | This comprises Sources, Sinks, Buses, Transformers,
    | GenericStorages and optionally DSMSinks"
    ":math:`T`", "S", "| all time steps within the optimization time frame
    | (and time increment, i.e. frequency) chosen"
    ":math:`P`", "S", "| all periods (i.e. years) within the optimization time frame
    | chosen"
    ":math:`PT`", "S", "| all periods and time steps, whereby the first value
    | denotes the period and the second one the time step"
    ":math:`F`", "S", "| all flows of the energy system.
    | A flow is a directed connection between node A and B
    | and has a (non-negative) value (i.e. capacity flow) for every time step"
    ":math:`IF`", "S", "| all flows or nodes of the energy system that can be invested into.
    | Comprises InvestmentFlows, GenericStorages and DSMSinks"
    ":math:`RES`", "S", "all renewable generators"
    ":math:`TF`", "S", "all transformers (i.e. conversion units, such as generators)"
    ":math:`B`", "S", "all buses (fictitious bus bars to connect capacity resp. energy flows)"
    ":math:`S`", "S", "all storage units"
    ":math:`IC`", "S", "all interconnector units"
    ":math:`I(n)`", "S", "all inputs for node n"
    ":math:`O(n)`", "S", "all outputs for node n"
    ":math:`DR`", "S", "all demand response clusters (eligible for investment)"
    ":math:`P_{invest}(n, p)`", "V", "investment into new capacity for node n in period p"
    ":math:`P_{total}(n, p)`", "V", "total installed capacity for node n in period p"
    ":math:`P_{old}(n, p)`", "V", "| old installed capacity for node n
    | to be decommissioned at the beginning of period p"
    ":math:`P_{old,exo}(n, p)`", "V", "| old installed capacity from exogenous investments,
    | i.e. from capacity that has initially been existing,
    | for node n to be decommissioned at the beginning of period p"
    ":math:`P_{old,end}(n, p)`", "V", "| old installed capacity from endogenous investments,
    | i.e. from investments that have been chosen by the optimization model
    | and reached their lifetime within the optimization time frame,
    | for node n to be decommissioned at the beginning of period p"
    ":math:`f(i,o,p,t)`", "V", "| Flow from node i (input) to node o (output)
    | in period p and at time step t"
    ":math:`C`", "V", "system costs"
    ":math:`P_{i}(n, p, t)`", "V", "inflow into transformer n in period p and at time step t"
    ":math:`P_{o}(n, p, t)`", "V", "outflow from transformer n in period p and at time step t"
    ":math:`E(s, t)`", "V", "energy currently stored in storage s"
    ":math:`A(c_{invest}(n, p), l(n), i(n))`", "P", "| annualised investment costs for investments into node or flow n
    | in period p, with lifetime l and interest rate i"
    ":math:`l(n)`", "P", "| lifetime of investments into flow or node n
    | (varied per technology)"
    ":math:`a(n)`", "P", "initial age of existing capacity of flow or node n"
    ":math:`i(n)`", "P", "| interest rate for investments into node resp. flow n
    | (varied per technology)"
    ":math:`dr`", "P", "discount rate (same across all technologies)"
    ":math:`D_{max}(o)`", "P", "maximum demand for market area or node o"
    ":math:`year(p)`", "P", "number of 'real-life' calendric year corresponding to the start of period p"
    ":math:`c_{var}(i, o, t)`", "P", "V costs for flow from input i to output o at time step t"
    ":math:`cf(i, t)`", "P", "| time-dependent capacity factor for renewable generator i"
    ":math:`d(o, t)`", "P", "normalized demand for time step t of node o"
    ":math:`\tau(t)`", "P", "time increment of the model for time step t"
    ":math:`\eta_{o}(n, t)`", "P", "conversion efficiency for outflow from node n at time step t"
    ":math:`\eta_{i}(n, t)`", "P", "conversion efficiency for inflow into node n at time step t"
    ":math:`P_{nom}(i, o)`", "P", "| installed capacity (all except RES outside Germany)
    | or maximum achievable output value (RES outside Germany)
    | for exogenously defined capacities"
    ":math:`P_{overall,max}(n)`", "P", "| overall maximum allowed installations for node or flow n
    | (varied per technology)"
    ":math:`f_{min}(i, o, t)`", "P", "normalized minimum output for flow from input i to output o"
    ":math:`f_{max}(i, o, t)`", "P", "normalized maximum output for flow from input i to output o"
    ":math:`E_{nom}(s)`", "P", "| nominal capacity of storage s (maximum achievable capacity
    | based on historic utilization, not the installed one)"
    ":math:`E(s,-1)`", "P", "initial storage content for storage s"
    ":math:`E_{min}(s, t)`", "P", "minimum allowed storage level for storage s"
    ":math:`E_{max}(s, t)`", "P", "maximum allowed storage level for storage s"
    ":math:`\beta(s, t)`", "P", "fraction of lost energy as share of :math:`E(s, t)`"
    ":math:`\dot{E}_i(s, p, t)`", "P", "energy flowing into storage s in period p and at time step t"
    ":math:`\dot{E}_o(s, p, t)`", "P", "energy extracted from storage s in period p and at time step t"
    ":math:`\eta_i(s, t)`", "P", "conversion factor (i.e. efficiency) of storage s for storing energy"
    ":math:`\eta_o(s, t)`", "P", "| conversion factor (i.e. efficiency) of storage s
    | for withdrawing stored energy"
    ":math:`t_u`", "P", "time unit of losses :math:`\beta(t)` and time increment :math:`\tau(t)`"
    ":math:`e(i, o)`", "P", "emission factor in :math:`t \space \frac {CO_2}{MWh}`"
    ":math:`EL`", "P", "overall emission limit in :math:`t \space CO_2`"
    ":math:`EL(p)`", "P", "annual overall emission limit in :math:`t \space CO_2`"


Target function
+++++++++++++++
The target function is build together by the ``_objective_expression`` terms of all
oemof.solph components used (`see the oemof.solph.models module <https://github.com/oemof/oemof-solph/blob/dev/src/oemof/solph/models.py>`_):


**System costs**: Sum of

    * annualised investment costs for flows that can be invested into,
    * fixed costs for flows associated with a fixed costs value (only flows eligible for investment) as well as
    * variable costs for all flows (commodity resp. fuel, emissions and operation costs):

.. math::

    Min \space C = & \sum_{n \in \mathrm{IF}} ((\sum_{p \in \mathrm{P}} P_{invest}(n, p) \cdot A(c_{invest}(n, p), l(n), i(n)) \cdot l(n) \\
    & + \sum_{pp=p}^{p+l(n)} P_{invest}(n, p) \cdot c_{fixed}(n, pp) \cdot DF^{-pp}) \\
    & + \sum_{(i,o) \in \mathrm{F}} \sum_{p \in \mathrm {P}} \sum_{t \in \mathrm {T}} f(i, o, p, t) \cdot c_{var}(i, o, t)) \cdot DF^{-p} \\

whereby

.. math::

    & A(c_{invest}(n, p), l(n), i(n)) = c_{invest}(n, p) \cdot
    \frac {(1+i(n))^{l(n)} \cdot i(n)} {(1+i(n))^{l(n)} - 1} \\
    & \\
    & DF=(1+dr)

Constraints of the core model
+++++++++++++++++++++++++++++

The following constraints apply to a model in its very basic formulation (i.e.
not including demand response and emissions limits):

Investment variables interrelation
==================================

* Investment bounds:

.. math::
    & P_{invest, min}(n, p) \leq P_{invest}(n, p) \leq P_{invest,max}(n, p) \\
    & \forall \space n \in \mathrm{IF}, \space p \in \mathrm{P}


* Total capacity (resp. total energy in case of storage energy content):

.. math::
    &
        P_{total}(n, p) = \left\{\begin{array}{11} P_{invest}(n, p) + P_{exist}(n, p), & p=0 \\
        P_{total}(n, p-1) + P_{invest}(n, p) - P_{old}(n, p), & p\not=0\end{array}\right. \\
    & \forall \space n \in \mathrm{IF}, p \in \mathrm{P}

* Old capacity to be decommissioned in period p

.. math::
    &
    P_{old}(n, p) = P_{old,exo}(n, p) + P_{old,end}(n, p) \\
    & \forall \space n \in \mathrm{IF}, p \in \mathrm{P} \\
    &\\
    &
    P_{old,end}(n, p) =
        \begin{cases} 0, & p=0 \\
        P_{invest}(n, p_{comm}), & l(n) \leq year(p) \quad (*) \\
        0, & else \\
        \end{cases} \\
    & \forall \space n \in \mathrm{IF}, p \in \mathrm{P} \\
    &\\
    &
    P_{old,exo}(n, p) =
        \begin{cases} 0, & p=0 \\
        P_{exist}(n), & l(n) - a(n) \leq year(p) \quad (**) \\
        0, & else \\
        \end{cases} \\
    & \forall \space n \in \mathrm{IF}, p \in \mathrm{P} \\

whereby:

* (*) is only performed for the first period the condition
  is True. This is achieved by a matrix that keeps track of the unit
  age per period and serves to determine commissioning periods.
* (**) is only performed for the first period the condition
  is True. A decommissioning flag is then set to True
  to prevent having falsely added old capacity in future periods.
* :math:`year(p)` is the year corresponding to period p
* :math:`p_{comm}` is the commissioning period of the flow
  (which is determined by the model itself). For determining the commissioning
  period, a matrix is used that keeps track of unit age per period. This is used
  to check for the first period, in which the lifetime of an investment is reached
  or exceeded that is than selected as decommissioning period for this particular
  investment.

\

* Overall maximum of total installed capacity (resp. energy)

.. math::
    &
    P_{total}(n, p) \leq P_{overall,max}(n) \\
    & \forall \space n \in \mathrm{IF}, \space p \in \mathrm{P}

Power balance
=============

* Flow balance(s):

.. math::

    & \sum_{i \in I(n)} f(i, n, p, t) \cdot \tau(t)
    = \sum_{o \in O(n)} f(n, o, p, t) \cdot \tau(t) \\
    & \forall \space n \in \mathrm{B}, \space (p, t) \in \mathrm{PT}

with :math:`\tau(t)` equalling to the time increment (defaults to 1 hour)

.. note::

    This is equal to an overall energy balance requirement, but build up
    decentrally from a balancing requirement of every bus, thus allowing for
    a flexible expansion of the system size.

Power Transmission
==================

There are two kinds of power transmission options between market areas:
AC transmission with a time-dependent maximum capacity and DC transmission with a fixed maximum capacity

* Maximum exchange between market areas:

.. math::

    & f(i, o, p, t) \leq f_{max}(i, o, t) \cdot P_{nom}(i, o) \\
    & \space \forall \space (i, o) \in \mathrm{IC}, \space (p, t) \in \mathrm{PT}

whereby :math:`f(i, o, p, t)` denotes the flow via an interconnector that connects
the exporting market area on the input side :math:`i` with the importing market area on the output
side :math:`o`.

Demand
======

The baseline inflexible demand is given as a fixed time series per market area. In case of the presence of demand response,
this time series is decreased accordingly for Germany by the baseline demand for demand response applications.

* Fixed demand:

.. math::

    & f(i, o, p, t) = d(o, t) \cdot D_{max}(o) \\
    & \forall \space o \in \mathrm{D}, \space i \in I(o), \space (p, t) \in \mathrm{PT}

Renewable Generators
====================

The installed capacity as well as the output of renewable energies is fixed. The
model may decide on curtailing excessive amounts by activating
a sink to collect the excess generation, though.

* Renewables output:

.. math::

    & f(i, o, p, t) = cf(i, t) \cdot P_{nom}(i) \\
    & \forall \space i \in \mathrm{RES}, \space o \in O(i), \space (p, t) \in \mathrm{PT}

The capacity factor :math:`cf(i, t)` is scaled accordingly to account for
renewable capacity expansion.

Backup Generators
=================

* Energy transformation:

.. math::
    & P_{i}(n, p, t) \cdot \eta_{o}(n, t) =
    P_{o}(n, p, t) \cdot \eta_{i}(n, t), \\
    & \forall \space n \in \mathrm{TF},
    \space i \in \mathrm{I(n)}, \space o \in \mathrm{O(n)}, \space (p, t) \in \mathrm{PT}

with

* :math:`P_{i}(n, p, t)` as the inflow into the transformer node n,
* :math:`P_{o}(n, p, t)` as the transformer outflow,
* :math:`\eta_{o}(n, t)` the conversion efficiency for outputs and
* :math:`\eta_{i}(n, t)` the conversion factors for inflows. We only use the conversion factor for outflows to account
  for losses from the conversion (within the power plant).
* :math:`\mathrm{TF}` is the set of transformers, i.e. any kind of energy conversion
  unit. We use this for conventional or carbon-neutral controllable backup generators
  as well as interconnection lines (see above), where we apply negligible losses.

\

* Minimum and maximum load requirements

.. math::

    & f(i, o, p, t) \geq f_{min}(i, o, t) \cdot P_{nom}(i, o) \\
    & \forall \space (i, o) \in \mathrm{F} \setminus \mathrm{IF},
    \space (p, t) \in \mathrm{PT} \\
    & \\
    & f(i, o, t) \leq f_{max}(i, o, t) \cdot P_{nom}(i, o) \\
    & \forall \space (i, o) \in \mathrm{F} \setminus \mathrm{IF},
    \space (p, t) \in \mathrm{PT}

with :math:`P_{nom}(i, o)` equalling to the installed resp. maximum capacity,
:math:`f_{min}(i, o, t)` as the normalized minimum flow value
and :math:`f_{max}(i, o, t)` as the normalized maximum flow value.

.. note::

    Both, the maximum and the minimum output may vary over time.
    This is e.g. used for modelling combined heat and power (CHP) plants
    and industrial power plants (IPP), where a minimum load pattern
    applies, or for exogenous installations or decommissionings, where
    the maximum is increased or decreased on an annual basis.

For investment flows, :math:`P_{nom}(i, o)` is replaced by the total capacity,
which leads to:

.. math::

    & f(i, o, p, t) \geq f_{min}(i, o, t) \cdot P_{total}(i, o, p) \\
    & \forall \space (i, o) \in \mathrm{IF},
    \space(p, t) \in \mathrm{PT} \\
    & \\
    & f(i, o, p, t) \leq f_{max}(i, o, t) \cdot P_{total}(i, o, p) \\
    & \forall \space (i, o) \in \mathrm{IF},
    \space (p, t) \in \mathrm{PT}

Storages
========

* Storage roundtrip (existing units):

.. math::

    & E(s, |\mathrm{T}|) = E(s, -1) \\
    & \forall \space s \in \mathrm{S}

with the last storage level :math:`E(s, |\mathrm{T}|)` equalling the
initial storage content :math:`E(s, -1)`.

.. note::

    The storage roundtrip condition is only applied to existing storage units.
    Storages that are invested into by the model, initially have a storage content of
    0. Since it would be costly for the model, not to withdraw all energy from the storage
    until the last time point of the optimization, no additional roundtrip balancing
    constraint is introduced.

* Storage balance:

.. math::

    E(s, t + 1) = & E(s, t) \cdot (1 - \beta(s, t)) ^{\frac {\tau(t)}{(t_u)}} \\
    & - \frac{\dot{E}_o(s, p, t)}{\eta_o(s, t)} \cdot \tau(t)
    + \dot{E}_i(s, p, t) \cdot \eta_i(s, t) \cdot \tau(t) \\
    & \forall \space s \in \mathrm{S}, \space (p, t) \in \mathrm{PT}

with :math:`E_{nom}(s)` as the nominal storage capacity,
:math:`\beta(t)` as the relative loss of stored energy and
:math:`t_u` the time unit to create dimensionless factors resp. exponents.

    * Storage level limits:

    .. math::

        & E_{min}(s, t) \leq E(s, t) \leq E_{max}(s, t) \\
        & \forall \space s \in \mathrm{S}, \space t \in \mathrm{T}

with :math:`E_{min}(s, t)` as the minimum and :math:`E_{max}(s, t)`
as the maximum allowed storage content for time step t.

Constraints for core model extensions
+++++++++++++++++++++++++++++++++++++

The following constraints can be optionally included in the model
formulation if the respective control parameter in the configuration file
are set accordingly, see :ref:`config`.

Emissions limit
===============

``pommesinvest`` allows to select between two optional investment limits:

* an overall emissions budget limit for the entire optimization timeframe that
  the model is free to distribute over time and
* an annual emissions limit that is defined on a periodical, i.e. annual basis.
  The latter is used as a default.

\

* Overall emissions budget:

.. math::

    & \sum_{(i,o)} \sum_t f(i, o, p, t) \cdot \tau(t) \cdot e(i, o) \leq EL \\
    & \space (i, o) \in \mathrm{F}

with :math:`e(i, o)` as the specific emission factor and :math:`EL` as the
overall emission budget (cap) for the overall optimization time frame.

* Annual emissions limit:

.. math::

    & \sum_{(i,o)} \sum_t f(i, o, p, t) \cdot \tau(t) \cdot e(i, o) \leq EL(p) \\
    & \space (i, o) \in \mathrm{F}, \space \forall p \in \mathrm{P}

with :math:`EL(p)` as the emission budget (cap) for period :math:`p`.

Demand Response
===============

Since demand response is one of the key interest points of *POMMES*, there
are three different implementations which can be chosen from:

    * *DIW*: Based on a paper by Zerrahn and Schill (2015), pp. 842-843,
    * *DLR*: Based on the PhD thesis of Gils (2015) or
    * *oemof*: Created by Julian Endres. A fairly simple DSM representation
      which demands the energy balance to be levelled out in fixed cycles.

    An evaluation of different modeling approaches has been carried out and
    presented at the INREC 2020 (Kochems 2020). Some of the results are as follows:

    * *DLR*: An extensive modeling approach for demand response which neither
      leads to an over- nor underestimization of potentials and balances
      modeling detail and computation intensity.
    * *DIW*: A solid implementation with the tendency of slight overestimization
      of potentials since a shift time :math:`t_{shift}` is not included. It may get
      computationally expensive due to a high time-interlinkage in constraint
      formulations.
    * *oemof*: A very computationally efficient approach which only requires the
      energy balance to be levelled out in certain intervals. If demand
      response is not at the center of the research and/or parameter
      availability is limited, this approach should be chosen.
      Note that approach `oemof` does allow for load shedding,
      but does not impose a limit on maximum amount of shedded energy.

One of the approaches has to be selected by the user upfront. It does not
make sense to mix different approaches, though this would be technically feasible.

.. note::

    Since the contraints around the definition of the relationship between the
    investment-related parameters :math:`P_{total}(n, p)`, :math:`P_{invest}(n, p)`
    and :math:`P_{old}(n, p)` with :math:`n` denoting the node (e.g. the demand response cluster)
    and :math:`p` denoting the respective period are basically identical to those for other
    investments (InvestmentFlows, GenericStorages), these are not explicitly stated
    here, but of course are incorporated in the model. Instead, only the differences
    is focussed upon in the following section.

For the sake of readability, the variables and parameters used for demand
response modeling are listed separately in the following table:

.. table:: Sets (S), Variables (V) and Parameters (P)
    :widths: 20, 10, 60, 10

    ================================= ==== ==================================================================== ==============
    symbol                            type explanation                                                          approach
    ================================= ==== ==================================================================== ==============
    :math:`DSM_{t}^{up}`              V    DSM up shift (capacity shifted upwards)                              oemof, DIW
    :math:`DSM_{h, t}^{up}`           V    DSM up shift (additional load) in hour t with delay time h           DLR
    :math:`DSM_{t}^{do, shift}`       V    DSM down shift (capacity shifted downwards)                          oemof
    :math:`DSM_{t, tt}^{do, shift}`   V    | DSM down shift (less load) in hour tt                              DIW
                                           | to compensate for upwards shifts in hour t
    :math:`DSM_{h, t}^{do, shift}`    V    DSM down shift (less load) in hour t with delay time h               DLR
    :math:`DSM_{h, t}^{balanceUp}`    V    | DSM down shift (less load) in hour t with delay time h             DLR
                                           | to balance previous upshift
    :math:`DSM_{h, t}^{balanceDo}`    V    | DSM up shift (additional load) in hour t with delay time h         DLR
                                           | to balance previous downshift
    :math:`DSM_{t}^{do, shed}`        V    DSM shedded (capacity shedded, i.e. not compensated for)             all
    :math:`\dot{E}_{t}`               V    Energy flowing in from (electrical) inflow bus                       all
    :math:`demand_{t}`                P    (Electrical) demand series (normalized)                              all
    :math:`demand_{max}`              P    Maximum demand value                                                 all
    :math:`h`                         P    | Maximum delay time for load shift (integer value                   DLR
                                           | from set of feasible delay times per DSM portfolio;
                                           | time until the energy balance has to be levelled out again;
                                           | roundtrip time of one load shifting cycle, i.e. time window
                                           | for upshift and compensating downshift)
    :math:`H_{DR}`                    S    | Set of feasible delay times for load shift                         DLR
                                           | of a certain DSM portfolio
    :math:`t_{shift}`                 P    | Maximum time for a shift in one direction,                         DLR
                                           | i. e. maximum time for an upshift *or* a downshift
                                           | in a load shifting cycle
    :math:`L`                         P    | Maximum delay time for load shift                                  DIW
                                           | (time until the energy balance has to be levelled out again;
                                           | roundtrip time of one load shifting cycle, i.e. time window
                                           | for upshift and compensating downshift)
    :math:`t_{she}`                   P    Maximum time for one load shedding process                           DLR, DIW
    :math:`E_{t}^{do}`                P    | Capacity  allowed for a load adjustment downwards                  all
                                           | (normalized; shifting + shedding)
    :math:`E_{t}^{up}`                P    Capacity allowed for a shift upwards (normalized)                    all
    :math:`E_{do, max}`               P    | Maximum capacity allowed for a load adjustment downwards           all
                                           | (shifting + shedding)
    :math:`E_{up, max}`               P    Maximum capacity allowed for a shift upwards                         all
    :math:`\tau`                      P    | interval (time within which the                                    oemof
                                           | energy balance must be levelled out)
    :math:`\eta`                      P    Efficiency for load shifting processes                               all
    :math:`\mathrm{T}`                P    Time steps of the model                                              all
    :math:`e_{shift}`                 P    | Boolean parameter indicating if unit can be used                   all
                                           | for load shifting
    :math:`e_{shed}`                  P    | Boolean parameter indicating if unit can be used                   all
                                           | for load shedding
    :math:`cost_{t}^{dsm, up}`        P    Variable costs for an upwards shift                                  all
    :math:`cost_{t}^{dsm, do, shift}` P    Variable costs for a downwards shift (load shifting)                 all
    :math:`cost_{t}^{dsm, do, shed}`  P    Variable costs for shedding load                                     all
    :math:`\Delta t`                  P    The time increment of the model                                      DLR, DIW
    :math:`\omega_{t}`                P    Objective weighting of the model for time step t                     all
    :math:`R_{shi}`                   P    | Minimum time between the end of one load shifting process          DIW
                                           | and the start of another
    :math:`R_{she}`                   P    | Minimum time between the end of one load shedding process          DIW
                                           | and the start of another
    :math:`n_{yearLimitShift}`        P    | Maximum allowed number of load shifts (at full capacity)           DLR
                                           | in the optimization timeframe
    :math:`n_{yearLimitShed}`         P    | Maximum allowed number of load sheds (at full capacity)            DLR
                                           | in the optimization timeframe
    :math:`t_{dayLimit}`              P    | Maximum duration of load shifts at full capacity per day           DLR
                                           | resp. in the last hours before the current"
    ================================= ==== ==================================================================== ==============


In the following, the constraint formulations and objective terms
are given separately for each approach:

.. note::

    * The constraints and objective terms hold for all demand response units which are
      aggregated to demand response clusters (with homogeneous costs and delay resp. shifting times).
    * For the sake of readability, the technology index is not displayed except for the target function term
      which sums across the different demand response clusters.
    * Furthermore, for some constraints there may be index violations which are taken care of by
      limiting to the feasible time indices :math:`{0, 1, .., |T|}`. This is also not displayed for the sake of readability.
    * For the complete implementation and details, please refer to `the sink_dsm module of oemof.solph <https://github.com/oemof/oemof-solph/blob/master/src/oemof/solph/experimental/_sink_dsm.py>`_.

**approach `oemof`**:

* Constraints:

.. math::
    &
    (1) \quad DSM_{t}^{up} = 0 \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shift} = \textrm{False} \\
    & \\
    &
    (2) \quad DSM_{t}^{do, shed} = 0 \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shed} = \textrm{False} \\
    & \\
    &
    (3) \quad \dot{E}_{t} = demand_{t} \cdot demand_{max}(p)
    + DSM_{t}^{up}
    - DSM_{t}^{do, shift} - DSM_{t}^{do, shed} \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (4) \quad  DSM_{t}^{up} \leq E_{t}^{up} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (5) \quad DSM_{t}^{do, shift} +  DSM_{t}^{do, shed} \leq
    E_{t}^{do} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (6) \quad  \sum_{t=t_s}^{t_s+\tau} DSM_{t}^{up} \cdot \eta =
    \sum_{t=t_s}^{t_s+\tau} DSM_{t}^{do, shift} \\
    & \quad \quad \quad \quad \forall t_s \in
    \{k \in \mathrm{T} \mid k \mod \tau = 0\} \\

* Objective function term (added to objective function above):

.. math::

    \sum_{n \in \mathrm{DR}} & (\sum_{p \in \mathrm{P}} P_{invest}(n, p) \cdot A(c_{invest}(n, p), l(n), i(n)) \cdot l(n) \cdot DF^{-p} \\
    &
    + \sum_{pp=year(p)}^{year(p)+l(n)} P_{invest}(n, p) \cdot c_{fixed}(n, pp) \cdot DF^{-pp} \cdot DF^{-p} \\
    &
    + \sum_{p \in \mathrm{P}} \sum_{t \in \mathrm{T}} (DSM_{n, t}^{up} \cdot cost_{n, t}^{dsm, up} + DSM_{n, t}^{do, shift} \cdot cost_{n, t}^{dsm, do, shift} \\
    &
    + DSM_{n, t}^{do, shed} \cdot cost_{n, t}^{dsm, do, shed}) \cdot \omega_{t} \cdot DF^{-p}) \\

**approach `DIW`**:

* Constraints:

.. math::
    &
    (1) \quad DSM_{t}^{up} = 0 \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shift} = \textrm{False} \\
    & \\
    &
    (2) \quad DSM_{t}^{do, shed} = 0 \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shed} = \textrm{False} \\
    & \\
    &
    (3) \quad \dot{E}_{t} = demand_{t} \cdot demand_{max}(p)
    + DSM_{t}^{up} -
    \sum_{tt=t-L}^{t+L} DSM_{tt,t}^{do, shift} - DSM_{t}^{do, shed} \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (4) \quad DSM_{t}^{up} \cdot \eta =
    \sum_{tt=t-L}^{t+L} DSM_{t,tt}^{do, shift} \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T} \\
    & \\
    &
    (5) \quad DSM_{t}^{up} \leq E_{t}^{up} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T} \\
    & \\
    &
    (6) \quad \sum_{t=tt-L}^{tt+L} DSM_{t,tt}^{do, shift}
    + DSM_{tt}^{do, shed} \leq E_{tt}^{do} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (7) \quad DSM_{tt}^{up} + \sum_{t=tt-L}^{tt+L} DSM_{t,tt}^{do, shift}
    + DSM_{tt}^{do, shed} \leq max \{ E_{tt}^{up}, E_{tt}^{do} \}
    \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (8) \quad \sum_{tt=t}^{t+R-1} DSM_{tt}^{up}
    \leq E_{t}^{up} \cdot P_{total}(p)
    \cdot L \cdot \Delta t \\
    & \quad \quad \quad \quad \forall (p, t)  \in \mathrm{PT} \\
    & \\
    &
    (9) \quad \sum_{tt=t}^{t+R-1} DSM_{tt}^{do, shed}
    \leq E_{t}^{do} \cdot P_{total}(p)
    \cdot t_{shed}
    \cdot \Delta t \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\

* Objective function term (added to objective function above):

.. math::

    \sum_{n \in \mathrm{DR}} & (\sum_{p \in \mathrm{P}} P_{invest}(n, p) \cdot A(c_{invest}(n, p), l(n), i(n)) \cdot l(n) \cdot DF^{-p} \\
    &
    + \sum_{pp=year(p)}^{year(p)+l(n)} P_{invest}(n, p) \cdot c_{fixed}(n, pp) \cdot DF^{-pp} \cdot DF^{-p} \\
    &
    + \sum_{p \in \mathrm{P}} \sum_{t \in \mathrm{T}} (DSM_{n, t}^{up} \cdot cost_{n, t}^{dsm, up} + DSM_{n, t}^{do, shift} \cdot cost_{n, t}^{dsm, do, shift} \\
    &
    + DSM_{n, t}^{do, shed} \cdot cost_{n, t}^{dsm, do, shed}) \cdot \omega_{t} \cdot DF^{-p}) \\

**approach `DLR`**:

* Constraints:

.. math::
    &
    (1) \quad DSM_{h, t}^{up} = 0 \\
    & \quad \quad \quad \quad \forall h \in H_{DR}, t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shift} = \textrm{False} \\
    &
    (2) \quad DSM_{t}^{do, shed} = 0 \\
    & \quad \quad \quad \quad \forall t \in \mathrm{T}
    \quad \textrm{if} \quad e_{shed} = \textrm{False} \\
    & \\
    &
    (3) \quad \dot{E}_{t} = demand_{t} \cdot demand_{max}(p) \\
    & + \displaystyle\sum_{h=1}^{H_{DR}} (DSM_{h, t}^{up}
    + DSM_{h, t}^{balanceDo} - DSM_{h, t}^{do, shift}
    - DSM_{h, t}^{balanceUp}) - DSM_{t}^{do, shed} \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (4) \quad DSM_{h, t}^{balanceDo} =
    \frac{DSM_{h, t - h}^{do, shift}}{\eta} \\
    & \quad \quad \quad \quad \forall h \in H_{DR}, \space t \in [h..T] \\
    & \\
    &
    (5) \quad DSM_{h, t}^{balanceUp} =
    DSM_{h, t-h}^{up} \cdot \eta \\
    & \quad \quad \quad \quad \forall h \in H_{DR}, \space t \in [h..T] \\
    & \\
    &
    (6) \quad DSM_{h, t}^{do, shift} = 0 \\
    & \quad \quad \quad \quad \forall h \in H_{DR}, \space t \in [T - h..T] \\
    & \\
    &
    (7) \quad DSM_{h, t}^{up} = 0 \\
    & \quad \quad \quad \quad \forall h \in H_{DR}, \space t \in [T - h..T] \\
    & \\
    &
    (8) \quad \displaystyle\sum_{h=1}^{H_{DR}} (DSM_{h, t}^{do, shift}
    + DSM_{h, t}^{balanceUp}) + DSM_{t}^{do, shed}
    \leq E_{t}^{do} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad  \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (9) \quad \displaystyle\sum_{h=1}^{H_{DR}} (DSM_{h, t}^{up}
    + DSM_{h, t}^{balanceDo})
    \leq E_{t}^{up} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad  \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (10) \quad \Delta t \cdot \displaystyle\sum_{h=1}^{H_{DR}}
    (DSM_{h, t}^{do, shift} - DSM_{h, t}^{balanceDo} \cdot \eta)
    = W_{t}^{levelDo} - W_{t-1}^{levelDo} \\
    & \quad \quad \quad \quad \forall t \in [1..T] \\
    & \\
    &
    (11) \quad \Delta t \cdot \displaystyle\sum_{h=1}^{H_{DR}}
    (DSM_{h, t}^{up} \cdot \eta - DSM_{h, t}^{balanceUp})
    = W_{t}^{levelUp} - W_{t-1}^{levelUp} \\
    & \quad \quad \quad \quad \forall t \in [1..T] \\
    & \\
    &
    (12) \quad W_{t}^{levelDo} \leq \overline{E}_{t}^{do}
    \cdot P_{total}(p) \cdot t_{shift} \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (13) \quad W_{t}^{levelUp} \leq \overline{E}_{t}^{up}
    \cdot P_{total}(p)  \cdot t_{shift} \\
    & \quad \quad \quad \quad \forall (p, t) \in \mathrm{PT} \\
    & \\
    &
    (14) \quad \displaystyle\sum_{t=0}^{T} DSM_{t}^{do, shed}
    \leq P_{total}(p) \cdot \overline{E}_{t}^{do}
    \cdot t_{shed}
    \cdot n^{yearLimitShed} \\
    & \\
    &
    (15) \quad \displaystyle\sum_{t=0}^{T} \sum_{h=1}^{H_{DR}}
    DSM_{h, t}^{do, shift}
    \leq P_{total}(p)
    \cdot \overline{E}_{t}^{do}
    \cdot t_{shift}
    \cdot n^{yearLimitShift} \\
    & \quad \quad \textrm{(optional constraint)} \\
    & \\
    &
    (16) \quad \displaystyle\sum_{t=0}^{T} \sum_{h=1}^{H_{DR}}
    DSM_{h, t}^{up}
    \leq P_{total}(p)
    \cdot \overline{E}_{t}^{up}
    \cdot t_{shift}
    \cdot n^{yearLimitShift} \\
    & \quad \quad \textrm{(optional constraint)} \\
    &
    (17) \quad \displaystyle\sum_{h=1}^{H_{DR}} DSM_{h, t}^{do, shift}
    \leq P_{total}(p)
    \cdot \overline{E}_{t}^{do}
    \cdot t_{shift} -
    \displaystyle\sum_{t'=1}^{t_{dayLimit}} \sum_{h=1}^{H_{DR}}
    DSM_{h, t - t'}^{do, shift} \\
    & \quad \quad \quad \quad \forall t \in [t-t_{dayLimit}..T] \\
    & \quad \quad \textrm{(optional constraint)} \\
    & \\
    &
    (18) \quad \displaystyle\sum_{h=1}^{H_{DR}} DSM_{h, t}^{up}
    \leq (invest + E_{exist})
    \cdot \overline{E}_{t}^{up}
    \cdot t_{shift} -
    \displaystyle\sum_{t'=1}^{t_{dayLimit}} \sum_{h=1}^{H_{DR}}
    DSM_{h, t - t'}^{up} \\
    & \quad \quad \quad \quad \forall t \in [t-t_{dayLimit}..T] \\
    & \quad \quad \textrm{(optional constraint)} \\
    & \\
    &
    (19) \quad \displaystyle\sum_{h=1}^{H_{DR}} (DSM_{h, t}^{up}
    + DSM_{h, t}^{balanceDo}
    + DSM_{h, t}^{do, shift} + DSM_{h, t}^{balanceUp}) \\
    & \quad \quad \quad + DSM_{t}^{shed} \leq \max \{E_{t}^{up}, E_{t}^{do} \} \cdot P_{total}(p) \\
    & \quad \quad \quad \quad \forall (p, t) \in \textrm{TIMEINDEX} \\
    & \quad \quad \textrm{(optional constraint)} \\
    &

* Objective function term (added to objective function above):

.. math::

    \sum_{n \in \mathrm{DR}} & (\sum_{p \in \mathrm{P}} P_{invest}(n, p) \cdot A(c_{invest}(n, p), l(n), i(n)) \cdot l(n) \cdot DF^{-p} \\
    &
    + \sum_{pp=year(p)}^{year(p)+l(n)} P_{invest}(n, p) \cdot c_{fixed}(n, pp) \cdot DF^{-pp} \cdot DF^{-p} \\
    &
    + \sum_{p \in \mathrm{P}} \sum_{t \in \mathrm{T}} \sum_{h \in H_{DR}} ((DSM_{n, h, t}^{up} + DSM_{n, h, t}^{balanceDo}) \cdot cost_{n, t}^{dsm, up} \\
    &
    + (DSM_{n, h, t}^{do, shift} + DSM_{n, h, t}^{balanceUp}) \cdot cost_{n, t}^{dsm, do, shift} \\
    &
    + DSM_{n, t}^{do, shed} \cdot cost_{n, t}^{dsm, do, shed}) \cdot \omega_{t} \cdot DF^{-p}) \\

Electric Vehicles
=================

The deployment of electric vehicles is exogenously defined. In ``pommesinvst``, three categories of electric vehicles are modelled:
uncontrolled charging (fixed demand time series), unidirectional controlled charging as well as bilateral controlled charging.

.. table:: Sets (S), Variables (V) and Parameters (P) additionally to the ones defined above :ref:`formulas`
    :widths: 20, 10, 70

    ================================= ==== =====================================================================
    symbol                            type explanation
    ================================= ==== =====================================================================
    :math:`EV_{UC}`                   S    all electric vehicles demand sinks eligible for uncontrolled charging
    :math:`EV_{CC,bi}`                S    all electric vehicles demand sinks eligible for bidirectional controlled charging
    :math:`EV_{CC,uni}`               S    all electric vehicles demand sinks eligible for unidirectional controlled charging
    :math:`S_{CC,bi}`                 S    fleet battery for bidirectional controlled charging
    :math:`S_{CC,uni}`                S    fleet battery for unidirectional controlled charging
    :math:`B_{CC,bi}`                 S    bus for bidirectional controlled charging
    :math:`avail_{CC}(o, t)`          P    charging availability for node o :math:`\in [0;1]`
    :math:`P_{in,max}(o)`             P    maximum inflow power for node o
    ================================= ==== =====================================================================

* Uncontrolled charging

.. math::

    & f(i, o, p, t) = d(o, t) \cdot D_{max}(o) \\
    & \forall \space o \in \mathrm{EV_{UC}}, \space i \in I(o), \space (p, t) \in \mathrm{PT}

* Unidirectional controlled charging

.. math::

    & (1) \quad E(s, |\mathrm{T}|) = E(s, -1) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,uni}} \\
    & \\
    & (2) \quad E(s, t + 1) = E(s, t) \cdot (1 - \beta(s, t)) ^{\frac {\tau(t)}{(t_u)}} \\
    & \quad \quad - \frac{\dot{E}_o(s, p, t)}{\eta_o(s, t)} \cdot \tau(t)
    + \dot{E}_i(s, p, t) \cdot \eta_i(s, t) \cdot \tau(t) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,uni}}, \space (p, t) \in \mathrm{PT} \\
    & \\
    & (3) \quad E_{min}(s, t) \leq E(s, t) \leq E_{max}(s, t) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,uni}}, \space t \in \mathrm{T} \\
    & \\
    & (4) \quad f(i, o, p, t) \leq avail_{CC}(o, t) \cdot P_{in,max}(o) \\
    & \quad \quad \forall \space o \in \mathrm{S_{CC,uni}}, \space i \in I(o), \space (p, t) \in \mathrm{PT} \\
    & \\
    & (5) \quad f(i, o, p, t) = d(o, t) \cdot D_{max}(o) \\
    & \quad \quad \forall \space o \in \mathrm{EV_{CC,uni}}, \space i \in I(o), \space (p, t) \in \mathrm{PT}

.. note::

    * Time-dependent state of charge limits :math:`E_{min}(s, t)` and :math:`E_{max}(s, t)` are used to
      account for the driving demand to be fulfilled (Eq. (3)).
    * Charging power is limited by the connection rate :math:`avail_{CC}(o, t)` of vehicles (Eq. (4)).
    * The demand to be satisfied is fixed (Eq. (5)), but charging is flexible.

* Bidirectional controlled charging

.. math::

    & (1) \quad E(s, |\mathrm{T}|) = E(s, -1) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,bi}} \\
    & \\
    & (2) \quad E(s, t + 1) = E(s, t) \cdot (1 - \beta(s, t)) ^{\frac {\tau(t)}{(t_u)}} \\
    & \quad \quad - \frac{\dot{E}_o(s, p, t)}{\eta_o(s, t)} \cdot \tau(t)
    + \dot{E}_i(s, p, t) \cdot \eta_i(s, t) \cdot \tau(t) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,bi}}, \space (p, t) \in \mathrm{PT} \\
    & \\
    & (3) \quad E_{min}(s, t) \leq E(s, t) \leq E_{max}(s, t) \\
    & \quad \quad \forall \space s \in \mathrm{S_{CC,bi}}, \space t \in \mathrm{T} \\
    & \\
    & (4) \quad f(i, o, p, t) \leq avail_{cc}(o, t) \cdot P_{in,max}(o) \\
    & \quad \quad \forall \space o \in \mathrm{S_{CC,bi}}, \space i \in I(o), \space (p, t) \in \mathrm{PT} \\
    & \\
    & (5) \quad f(i, o, p, t) = d(o, t) \cdot D_{max}(o) \\
    & \quad \quad \forall \space o \in \mathrm{EV_{CC,uni}}, \space i \in I(o), \space (p, t) \in \mathrm{PT} \\
    & \\
    & (6) \quad f(i, o, p, t) \leq avail_{CC}(o, t) \cdot P_{in,max}(o) \\
    & \quad \quad \forall \space o \in \mathrm{B_{CC,bi}}, \space i \in I(o), \space (p, t) \in \mathrm{PT} \\

.. note::

    * Eq. (1)-(5) are the same as for uncontrolled charging.
    * Eq. (6) ensures that the power fed back into the grid is limited by the connection rate :math:`avail_{CC}(o, t)`.
      Note that through Eq. (3) also the allowed energy to be fed back is limited.

References
++++++++++
Gils, Hans Christian (2015): `Balancing of Intermittent Renewable Power Generation by Demand Response and Thermal Energy Storage`, Stuttgart,
`http://dx.doi.org/10.18419/opus-6888 <http://dx.doi.org/10.18419/opus-6888>`_, accessed 24.09.2021, pp. 67-70.

Kochems, Johannes (2020): Demand response potentials for Germany: potential clustering and comparison of modeling approaches, presentation at the 9th international Ruhr Energy Conference (INREC 2020), 10th September 2020,
`https://github.com/jokochems/DR_modeling_oemof/blob/master/Kochems_Demand_Response_INREC.pdf <https://github.com/jokochems/DR_modeling_oemof/blob/master/Kochems_Demand_Response_INREC.pdf>`_, accessed 24.09.2021.

Zerrahn, Alexander and Schill, Wolf-Peter (2015): On the representation of demand-side management in power system models,
in: Energy (84), pp. 840-845, `10.1016/j.energy.2015.03.037 <https://doi.org/10.1016/j.energy.2015.03.037>`_,