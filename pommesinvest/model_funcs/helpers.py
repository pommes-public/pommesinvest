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
