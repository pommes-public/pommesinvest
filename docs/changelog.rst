Changelog
=========

v0.1.0 (2023-XX-XX)
-------------------

Initial release of ``pommeinvest``

Welcome to the *POMMES* cosmos!

**A bottom-up fundamental investment model for the German electricity sector**

Features:

* ``pommesinvest`` is the **investment** variant of *POMMES* that allows
  to simulate investment and dispatch decisions for Germany in up to hourly
  resolution for the time horizon 2020 to 2045 (or shorter time frames).
* ``pommesinvest`` uses a multi-period dynamic investment approach. Hence
  investments may occur at the beginning of each period while decommisionings
  occur at the beggining of a period in case the asset reached its assumed
  lifetime.
* Consistent input data sets for *POMMES* models can be obtained from
  `pommesdata <https://github.com/pommes-public/pommesdata>`_,
  taking into account various open data sources.
* All *POMMES* models are easy to adjust and extend
  because they are build on top of `oemof.solph <https://github.com/oemof/oemof-solph>`_.
