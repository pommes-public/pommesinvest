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

import math
from datetime import datetime
from itertools import compress

import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset

# Consider idealistic years, ignoring leap years and weekdays
FREQUENCY_TO_TIMESTEPS = {
    "1H": {"timesteps": 8760, "multiplier": 1},
    "4H": {"timesteps": 2190, "multiplier": 4},
    "8H": {"timesteps": 1095, "multiplier": 8},
    "24H": {"timesteps": 365, "multiplier": 24},
    "36H": {"timesteps": 244, "multiplier": 36},
    "48H": {"timesteps": 182, "multiplier": 48},
}


def years_between(y1, y2):

    """Calculate the difference in years between two dates using
    the dateutil.relativedelta package

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
        diff_in_hours / FREQUENCY_TO_TIMESTEPS[freq]["multiplier"]
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

    Time series indices ignore time shifts and can be interpreted as UTC time.
    Since they aren't localized, this cannot be detected by pandas and the
    correct frequency cannot be inferred. As a hack, only the first couple of
    time steps are checked, for which no problems should occur.

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
    # Ensure a datetime index
    try:
        timeseries.index = pd.DatetimeIndex(timeseries.index)
    except ValueError:
        raise ValueError(
            "Time series has an invalid index. "
            "A pd.DatetimeIndex is required."
        )

    try:
        original_freq = pd.infer_freq(timeseries.index, warn=True)
    except ValueError:
        original_freq = "AS"

    # Hack for problems with recognizing abolishing the time shift
    if not original_freq:
        try:
            original_freq = pd.infer_freq(timeseries.index[:5], warn=True)
        except ValueError:
            raise ValueError("Cannot detect frequency of time series!")

    # Introduce common timestamp to be able to compare different frequencies
    common_dt = pd.to_datetime("2000-01-01")

    if common_dt + to_offset(freq) > common_dt + to_offset(original_freq):
        # do downsampling
        resampled_timeseries = timeseries.resample(freq).agg(aggregation_rule)

    else:
        # do upsampling
        resampled_timeseries = timeseries.resample(freq).interpolate(
            method=interpolation_rule
        )

    cut_leap_days(resampled_timeseries)

    return resampled_timeseries


def cut_leap_days(time_series):
    """Take a time series index with real dates and cut the leap days out

    Actual time stamps cannot be interpreted. Instead consider 8760 hours
    of a synthetical year

    Parameters
    ----------
    time_series : pd.Series or pd.DataFrame
        original time series with real life time index

    Returns
    -------
    time_series : pd.Series or pd.DataFrame
        Time series, simply cutted down to 8 760 hours per year
    """
    years = sorted(list(set(getattr(time_series.index, "year"))))
    for year in years:
        if is_leap_year(year):
            try:
                time_series.drop(
                    time_series.loc[
                        (time_series.index.year == year)
                        & (time_series.index.month == 12)
                        & (time_series.index.day == 31)
                    ].index,
                    inplace=True,
                )
            except KeyError:
                continue

    return time_series


def convert_annual_limit(annual_limit, start_time, end_time):
    """Convert an annual limit to a sub- or multi-annual one

    Parameters
    ----------
    annual_limit: float or pd.Series of dtype float
        An annual limit (e.g. for emissions, investment budgets)
        if start_time and end_time are within the same year,
        or a pd.Series of annual limits indexed by years if start_time and
        end_time are not within one year

    start_time: str
        The first date string; start_time of the optimization run

    end_time: str
        The second date string; end_time of the optimization run

    Returns
    -------
    new_limit: float
        A sub-annual / multi-annual limit for the optimization timeframe
    """
    dt_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    dt_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    start_year = dt_start.year
    end_year = dt_end.year

    new_limit = 0

    if start_year == end_year:
        day_diff = days_between(start_time, end_time)
        year_fraction = day_diff / float(365)
        if isinstance(annual_limit, float):
            new_limit = annual_limit * year_fraction
        else:
            new_limit = annual_limit[start_year] * year_fraction

    else:
        start_year_begin = str(start_year) + "-01-01 00:00:00"
        end_year_end = str(end_year) + "-12-31 23:59:59"
        # POMMES doesn't care about leap years
        if is_leap_year(end_year):
            end_year_end = str(end_year) + "-12-30 23:59:59"
        day_diff_start = days_between(start_year_begin, start_time)
        day_diff_end = days_between(end_time, end_year_end)

        start_year_fraction = (365 - day_diff_start) / float(365)
        end_year_fraction = (365 - day_diff_end) / float(365)
        full_years = end_year - start_year - 1

        # Add annual limits for full years within optimization time frame
        for i in range(full_years):
            new_limit += annual_limit.loc[start_year + i + 1]

        # Add limits for fractions of the start year and end year
        new_limit += (
            annual_limit.loc[start_year] * start_year_fraction
            + annual_limit.loc[end_year] * end_year_fraction
        )

    return new_limit


def days_between(d1, d2):
    """Calculate the difference in days between two days

    Parameters
    ----------
    d1 : str
        The first date string
    d2 : str
        The second date string

    Returns
    -------
    day_diff: int
        The difference between the two dates in days
    """
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    day_diff = abs((d2 - d1).days)

    return day_diff
