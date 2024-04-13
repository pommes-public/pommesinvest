![PyPI](https://img.shields.io/pypi/v/pommesdispatch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pommesdispatch)
![Documentation Status](https://readthedocs.org/projects/pommesdispatch/badge/?version=latest)
![PyPI - License](https://img.shields.io/pypi/l/pommesdispatch)

# pommesinvest

**A bottom-up fundamental investment model for the German electricity sector**

This is the **investment variant** of the fundamental power market model *POMMES* (**PO**wer **M**arket **M**odel of **E**nergy and re**S**ources).
Please navigate to the section of interest to find out more.

## Contents
* [Introduction](#introduction)
* [Documentation](#documentation)
* [Installation](#installation)
    * [Setting up pommesinvest](#setting-up-pommesinvest)
    * [Installing a solver](#installing-a-solver)
* [Contributing](#contributing)
* [Citing](#citing)
* [License](#license)

## Introduction
*POMMES* itself is a cosmos consisting of a **dispatch model** (stored in this repository and described here), a **data preparation routine** and an **investment model** for the German wholesale power market. The model was originally developed by a group of researchers and students at the [chair of Energy and Resources Management of TU Berlin](https://www.er.tu-berlin.de/menue/home/) and is now maintained by a group of alumni and open for other contributions.

If you are interested in the data preparation routines used or investment modeling, please find more information here:
- [pommesdata](https://github.com/pommes-public/pommesdata): A full-featured transparent data preparation routine from raw data to POMMES model inputs
- [pommesdispatch](https://github.com/pommes-public/pommesdispatch): A bottom-up fundamental power market model for the German power sector.

### Purpose and model characterization
The **invest variant** of the power market model *POMMES* `pommesinvest` enables the user to simulate the **investment into backup power plants, storages as well as demand response units for the Federal Republic of Germany** for long-term horizons (until 2045). The expansion of renewable power plants is exogeneously determined by expansion pathways as well as normalized infeed time series.
The models' overall goal is to minimize power system costs occuring from wholesale markets whereby no network constraints are considered. Thus, the model purpose is to simulate **investment decisions** and the resulting **capacity mix**. A brief categorization of the model is given in the following table. An extensive categorization can be found in the [model documentation]().

| **criterion** | **manifestation** |
| ---- | ---- |
| Purpose | - simulation of power plant dispatch and day-ahead prices for DE (scenario analysis) |
| Spatial coverage | - Germany (DE-LU) + electrical neighbours (NTC approach) |
| Time horizon | - usually 1 year in hourly resolution |
| Technologies | - conventional power plants, storages, demand response (optimized)<br> - renewable generators (fixed)<br> - demand: exogenous time series |
| Data sources | - input data not shipped out, but can be obtained from [pommesdata](https://github.com/pommes-public/pommesdata); OPSD, BNetzA, ENTSO-E, others |
| Implementation | - graph representation & linear optimization: [oemof.solph](https://github.com/oemof/oemof-solph) / [pyomo](https://github.com/Pyomo/pyomo) <br> - data management: python / .csv |

### Mathematical and technical implementation
The models' underlying mathematical method is a **linear programming** approach, seeking to minimize overall 
power system costs under constraints such as satisfying power demand at all times and not violating power generation 
capacity or storage limits. Thus, binary variables such as units' status, startups and shutdowns are not accounted for.

The model builds on the framework **[oemof.solph](https://github.com/oemof/oemof-solph)** which allows modeling
energy systems in a graph-based representation with the underlying mathematical constraints and objective function 
terms implemented in **[pyomo](https://pyomo.readthedocs.io/en/stable/)**. Some of the required oemof.solph featuresm - such as demand response modeling - have been provided by the *POMMES* main developers which are also active in the 
oemof community. Users not familiar with oemof.solph may find further information in the 
[oemof.solph documentation](https://oemof-solph.readthedocs.io/en/latest/readme.html).

## Documentation
An extensive **[documentation of pommesinvest](https://pommesinvest.readthedocs.io/)** can be found on readthedocs. It contains a user's guide, a model categorization, some energy economic and technical background information, a complete model formulation as well as documentation of the model functions and classes. 

## Installation
To set up `pommesinvest`, set up a virtual environment (e.g. using conda) or add the required packages to your python installation. Additionally, you have to install a solver in order to solve the mathematical optimization problem.

### Setting up pommesinvest
`pommesinvest` is hosted on [PyPI](https://pypi.org/project/pommesinvest/). 
To install it, please use the following command
```
pip install pommesinvest
```

If you want to contribute as a developer, you fist have to
[fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo>)
it and then clone the repository, in order to copy the files locally by typing
```
git clone https://github.com/your-github-username/pommesinvest.git
```
After cloning the repository, you have to install the required dependencies.
Make sure you have conda installed as a package manager.
If not, you can download it [here](https://www.anaconda.com/).
Open a command shell and navigate to the folder
where you copied the environment to.

Use the following command to install dependencies
```
conda env create -f environment.yml
```
Activate your environment by typing
```
conda activate pommesinvest
```

### Installing a solver
In order to solve a `pommesinvest` model instance, you need a solver installed. Please see [oemof.solph's information on solvers](https://github.com/oemof/oemof-solph#installing-a-solver). As a default, gurobi is used for `pommesinvest` models. It is a commercial solver, but provides academic licenses, though, if this applies to you. Elsewhise, we recommend to use CBC as the solver oemof recommends. To test your solver and oemof.solph installation, again see information from [oemof.solph](https://github.com/oemof/oemof-solph#installation-test).

## Contributing
Every kind of contribution or feedback is warmly welcome.<br>
We use the [GitHub issue management](https://github.com/pommes-public/pommesinvest/issues) as well as 
[pull requests](https://github.com/pommes-public/pommesinvest/pulls) for collaboration. We try to stick to the PEP8 coding standards.

### Authors
* Authors of `pommesinvest` are Johannes Kochems and Yannick Werner. It is maintained by Johannes Kochems.
* All people mentioned below contributed to early-stage versions or predecessors of POMMES or ideally supported it.

### List of contributors to POMMES
The following people have contributed to *POMMES*.
Most of these contributions belong to early-stage versions and are not part
of the actual source code. Nonetheless, all contributions shall be acknowledged and the full list is provided for transparency reasons.

The main contributors are stated on top, the remainder
is listed in alphabetical order.

| Name                                       | Contribution                                                                                                                                                                                                                                                                                         |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Johannes Kochems                           | major development & conceptualization<br>conceptualization, development of all investment-related parts; development of main data preparation routines (esp. future projection for all components, RES tender data and LCOE estimates, documentation), architecture, publishing process, maintenance |
| Yannick Werner                             | major development & conceptualization<br>conceptualization, development of main data preparation routines (status quo data for all components, detailed RES, interconnector and hydro data), architecture                                                                                            |
| Benjamin Grosse                            | data collection for conventional power plants in early development stage, ideal support and conceptionel counseling                                                                                                                                                                                  |
| Carla Spiller                              | data collection for conventional power plants in early stage development as an input to *pommesdata*; co-development of rolling horizon dispatch modelling in predecessor of *pommesdispatch*                                                                                                        |
| Christian Fraatz                           | data collection for conventional power plants in early stage development as an input to *pommesdata*                                                                                                                                                                                                 |
| Conrad Nicklisch                           | data collection for RES in early stage development as an input to *pommesdata*                                                                                                                                                                                                                       |
| Daniel Peschel                             | data collection on CHP power plants as an input to *pommesdata*                                                                                                                                                                                                                                      |
| Dr. Johannes Giehl                         | conceptionel support and research of data licensing; conceptionel support for investment modelling in *pommesinvest*                                                                                                                                                                                 |
| Dr. Paul Verwiebe                          | development of small test models as a predecessor of POMMES                                                                                                                                                                                                                                          |
| Fabian Büllesbach                          | development of a predecessor of the rolling horizon modeling approach in *pommesdispatch*                                                                                                                                                                                                            |
| Flora von Mikulicz-Radecki                 | extensive code and functionality testing in an early development stage for predecessors of *pommesdispatch* and *pommesinvest*                                                                                                                                                                       |
| Florian Maurer                             | support with / fix for python dependencies                                                                                                                                                                                                                                                           |
| Hannes Kachel                              | development and analysis of approaches for complexity reduction in a predecessor of *pommesinvest*                                                                                                                                                                                                   |
| Julian Endres                              | data collection for costs and conventional power plants in early stage development                                                                                                                                                                                                                   |
| Julien Faist                               | data collection for original coal power plant shutdown and planned installation of new power plants for *pommesdata*; co-development of a predecessor of *pommesinvest*                                                                                                                              |
| Leticia Encinas Rosa                       | ata collection for conventional power plants in early stage development as an input to *pommesdata*                                                                                                                                                                                                  |
| Prof. Dr.-Ing. Joachim Müller-Kirchenbauer | funding, enabling and conceptual support                                                                                                                                                                                                                                                             |
| Robin Claus                                | data collection for RES in early stage development as an input to *pommesdata*                                                                                                                                                                                                                       |
| Sophie Westphal                            | data collection for costs and conventional power plants in early stage development as an input for *pommesdata*                                                                                                                                                                                      |
| Timona Ghosh                               | data collection for interconnector data as an input to *pommesdata*                                                                                                                                                                                                                                  |


## Citing
If you are using `pommesinvest` for your own analyses, we recommend citing as:<br>
*Kochems, J. and Werner, Y. (2024): pommesinvest. A bottom-up fundamental power market model for the German electricity sector. https://github.com/pommes-public/pommesinvest, accessed YYYY-MM-DD.*

We furthermore recommend naming the version tag or the commit hash used for the sake of transparency and reproducibility.

Also see the *CITATION.cff* file for citation information.

## License
This software is licensed under MIT License.

Copyright 2024 pommes developer group

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
