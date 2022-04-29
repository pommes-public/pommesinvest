# -*- coding: utf-8 -*-
"""
General description
-------------------
These are supplementary routines used in the power market model POMMES.

Installation requirements
-------------------------
Python version >= 3.8

@author: Johannes Kochems
"""

# Import necessary packages for function definitions
import pandas as pd
import math

from pommesinvest.model_funcs.model_control import FREQUENCY_TO_TIMESTEPS

# Import datetime for conversion between date string and its element (year, month, ...)
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset

from itertools import compress


def years_between(y1, y2):

    """Calculate the difference in years between two dates using the dateutil.relativedelta package

    Parameters:
    ----------
    y1: :obj:`str`
        The first date string
    y2: :obj:`str`
        The second date string

    Returns
    -------
    year_diff: :obj:`int`
        The difference between the two dates in years

    """

    y1 = datetime.strptime(y1, "%Y-%m-%d %H:%M:%S")
    y2 = datetime.strptime(y2, "%Y-%m-%d %H:%M:%S")
    return abs(relativedelta(y1, y2).years)


def time_steps_between_timestamps(ts1, ts2, freq):
    """Calculate the difference between two time steps ignoring leap years

    Parameters
    ----------
    ts1 : pd.Timestamp
        The first timestamp
    ts2 : pd.Timestamp
        The second timestamp
    freq: str
        The frequency information, e.g. '60min', '4H'

    Returns
    -------
    hour_diff: int
        The difference between the two dates in time steps
    """
    diff = ts2 - ts1
    start_year = ts1.year
    end_year = ts2.year
    no_of_leap_years = len(
        multiple_leap_years(list(range(start_year, end_year + 1)))
    )

    diff_in_hours = (
        diff.days * 24
        + math.floor(diff.seconds / 3600)
        - no_of_leap_years * 24
    )

    return math.floor(
        diff_in_hours / FREQUENCY_TO_TIMESTEPS[freq]["multiplicator"]
    )


def is_leap_year(year):
    """Check whether given year is a leap year or not

    Parameters:
    -----------
    year: :obj:`int`
        year which shall be checked

    Returns:
    --------
    leap_year: :obj:`boolean`
        True if year is a leap year and False else
    """
    leap_year = False

    if year % 4 == 0:
        leap_year = True
    if year % 100 == 0:
        leap_year = False
    if year % 400 == 0:
        leap_year = True

    return leap_year


def multiple_leap_years(years):
    """Check a list of multiple years to find out which are leap years

    Parameters:
    -----------
    years: :obj:`list`
        list of years which shall be checked

    Returns:
    --------
    list of leap years
    """
    leap_years_boolean = [is_leap_year(el) for el in years]
    return list(compress(years, leap_years_boolean))


def resample_timeseries(
    timeseries, freq, aggregation_rule="sum", interpolation_rule="linear"
):

    """Resample a timeseries to the frequency provided

    The frequency of the given timeseries is determined at first and upsampling
    resp. downsampling are carried out afterwards. For upsampling linear
    interpolation (default) is used, but another method may be chosen.

    Parameters
    ----------
    timeseries : :obj:`pd.DataFrame`
        The timeseries to be resampled stored in a pd.DataFrame

    freq : :obj:`str`
        The target frequency

    interpolation_rule : :obj:`str`
        Method used for interpolation in upsampling

    Returns
    -------
    resampled_timeseries : :obj:`pd.DataFrame`
        The resampled timeseries stored in a pd.DataFrame

    """
    try:
        original_freq = pd.infer_freq(timeseries.index, warn=True)
    except ValueError:
        original_freq = "AS"

    if to_offset(freq) > to_offset(original_freq):
        # do downsampling
        resampled_timeseries = timeseries.resample(freq).agg(aggregation_rule)

    else:
        # do upsampling
        resampled_timeseries = timeseries.resample(freq).interpolate(
            method=interpolation_rule
        )

    return resampled_timeseries


# 07.06.2019, JK: Generic function for combining functions taken from Stack Overflow:
# https://stackoverflow.com/questions/13865009/have-multiple-commands-when-button-is-pressed
# accessed 07.06.2019
def combine_funcs(*funcs):
    """Take an arbitrary number of functions as argument and combine them,
    i.e. carry out the functions successively.

    Parameters
    ----------
    *funcs:
        The functions to be combined

    Returns
    -------
    combined_func:
        A method that successively runs the functions that shall be combined.
    """

    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return combined_func


# Taken from oemof.tools.economics, but removed check for infeasible values
def calc_annuity(capex, n, wacc, u=None, cost_decrease=0):
    r"""Calculates the annuity of an initial investment 'capex', considering the
    cost of capital 'wacc' during a project horizon 'n'

    In case of a single initial investment, the employed formula reads:

    .. math::
    annuity = capex \cdot \frac{(wacc \cdot (1+wacc)^n)}
              {((1 + wacc)^n - 1)}

    In case of repeated investments (due to replacements) at fixed intervals
    'u', the formula yields:

    .. math::
        annuity = capex \cdot \frac{(wacc \cdot (1+wacc)^n)}
                {((1 + wacc)^n - 1)} \cdot \left(
                \frac{1 - \left( \frac{(1-cost\_decrease)}
                {(1+wacc)} \right)^n}
                {1 - \left( \frac{(1-cost\_decrease)}{(1+wacc)}
                \right)^u} \right)

    Parameters
    ----------
    capex : float
        Capital expenditure for first investment. Net Present Value (NPV) or
        Net Present Cost (NPC) of investment
    n : int
        Horizon of the analysis, or number of years the annuity wants to be
        obtained for (n>=1)
    wacc : float
        Weighted average cost of capital (0<wacc<1)
    u : int
        Lifetime of the investigated investment. Might be smaller than the
        analysis horizon, 'n', meaning it will have to be replaced.
        Takes value 'n' if not specified otherwise (u>=1)
    cost_decrease : float
        Annual rate of cost decrease (due to, e.g., price experience curve).
        This only influences the result for investments corresponding to
        replacements, whenever u<n.
        Takes value 0, if not specified otherwise (0<cost_decrease<1)
    Returns
    -------
    float
        annuity
    """
    if u is None:
        u = n

    return (
        capex
        * (wacc * (1 + wacc) ** n)
        / ((1 + wacc) ** n - 1)
        * (
            (1 - ((1 - cost_decrease) / (1 + wacc)) ** n)
            / (1 - ((1 - cost_decrease) / (1 + wacc)) ** u)
        )
    )


def discount_values(df, IR, RH_startyear, startyear):
    """discount values in a given DataFrame using IR as interest rate"""
    return df.div((1 + IR) ** (RH_startyear - startyear))
