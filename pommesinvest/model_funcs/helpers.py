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

# Import datetime for conversion between date string and its element (year, month, ...)
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset

# Compress can be used to slice lists using another list of boolean values
# -> used by method MultipleLeapYears
from itertools import compress

# Imports for creating holidays in pandas
# See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html, accessed 03.09.2019
# Pandas on git hub: https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/holiday.py, accessed 03.09.2019
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    EasterMonday,
    GoodFriday,
)
from pandas.tseries.offsets import Day, Easter

### NOTE: SOME OF THE FUNCTIONS ARE NOT USED, BUT MAY AT SOME POINT BE USEFUL (OR NOT)

# JK: FUNCTION NOT TESTED YET!
def csv_to_excel(
    path_from, path_to="./inputs/", filename_to="power_market_input_data", *args
):

    """Reads in several csv-datasheets and puts them together to an Excel workbook.

    Parameters
    ----------
    path_from: :obj:`str`
        The path the csv files are stored at

    path_to: :obj:`str`
        The path where the resulting .xlsx file shall be  stored at

    filename_to: :obj:`str`
        The filename for the resulting .xlsx file

    *args: :obj:`str`
        A list of the name of csv-files that will be put together

    Returns
    -------
    workbook: :obj:`file`
        An Excel Workbook containing all data in separate Worksheets
    """

    # Create a list of sheets (i.e. pd.DataFrames)
    sheets = [pd.read_csv(path_from + el, sep=";", decimal=",") for el in args]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(path_to + filename_to, engine="xlsxwriter")

    for sheet in sheets:
        sheet.to_excel(writer)

    # Save the excel file and save it as variable workbook which is returned
    writer.save()
    workbook = writer.book

    return workbook


# 27.06.2019, FM: tested
def datetime_index_from_demand_data(path, file):

    """Separately read in demand data from a .csv file.
    The first column of the demand file must be the index column containing entries
    that can be interpreted as dates.

    Parameters
    ----------
    path: :obj:`str`
        The path the csv file is stored at with/ in the end in order to connect the path
    file: :obj:`str`
        The name of the demand data file

    Returns
    -------
    datetime: :obj:`datetime index`
    """

    demand_df = pd.read_csv(
        path + file, sep=";", decimal=",", parse_dates=True, index_col=0
    )
    # 27.06.2019, FM: old code: datetime= demand.df.index()
    datetime = demand_df.index

    return datetime


# 28.09.2019, JK: Function taken from Stack overflow issue, see: https://stackoverflow.com/questions/4436957/pythonic-difference-between-two-dates-in-years, accessed 28.08.2019
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
    year_diff = abs(relativedelta(y1, y2).years)

    return year_diff


# 09.12.2018, JK: Function taken from Stack overflow issue, see: https://stackoverflow.com/questions/8419564/difference-between-two-dates-in-python, accessed 09.12.2018
# 27.06.2019, FM: tested
def days_between(d1, d2):

    """Calculate the difference in days between two days using the datetime package

    Parameters:
    ----------
    d1: :obj:`str`
        The first date string
    d2: :obj:`str`
        The second date string

    Returns
    -------
    day_diff: :obj:`int`
        The difference between the two dates in days


    """

    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    day_diff = abs((d2 - d1).days)

    return day_diff


# 16.12.2018, JK: Function inspired from Stack overflow issue, see: https://stackoverflow.com/questions/24217641/how-to-get-the-difference-between-two-dates-in-hours-minutes-and-seconds, accessed 16.12.2018
# 05.07.2019, FM: tested
# TODO: YW: Search for pandas predefined function
def hours_between(h1, h2):
    """Calculate the difference in days between two days using the datetime package

    Parameters:
    ----------
    h1: :obj:`str`
        The first date string
    h2: :obj:`str`
        The second date string

    Returns
    -------
    hour_diff: :obj:`int`
        The difference between the two dates in hours


    """

    h1 = datetime.strptime(h1, "%Y-%m-%d %H:%M:%S")
    h2 = datetime.strptime(h2, "%Y-%m-%d %H:%M:%S")
    diff = abs((h2 - h1))
    days, seconds = diff.days, diff.seconds
    # 16.12.2018, JK: Double slash here is used for floor division (rounded down to nearest number)
    # Actually not really needed here, since only full hours are simulated, at least so far...
    hour_diff = days * 24 + math.floor(seconds / 3600)
    print("hour_diff: " + str(hour_diff))

    return hour_diff


def time_steps_between_timestamps(ts1, ts2, freq):
    """Calculate the difference in hours between two timesteps

    Parameters
    ----------
    ts1 : pd.Timestamp
        The first timestamp
    ts2 : pd.Timestamp
        The second timestamp
    freq: str
        The frequency information, e.g. '60min', '15min'

    Returns
    -------
    hour_diff: int
        The difference between the two dates in hours
    """
    time_steps_seconds = {"60min": (24, 3600), "15min": (96, 900)}

    diff = ts2 - ts1

    return diff.days * time_steps_seconds[freq][0] + math.floor(
        diff.seconds / time_steps_seconds[freq][1]
    )


# 09.12.2018, JK: Routine to check whether a given year is a leapyear or not.
# Could be useful for adapting the number of days resp. timesteps per year.
# 05.07.2019, FM: tested
def IsLeapYear(year):

    """Check whether given year is a leap year or not.

    Parameters:
    -----------
    year: :obj:`int`
        year which shall be checked

    Returns:
    --------
    LeapYear: :obj:`boolean`
        True if year is a leap year and False else

    """

    # Underlying rules:
    # every fourth year is a leap year unless it can be divided by 100.
    # every fourhundredth year in turn is a leap year.

    LeapYear = False

    if year % 4 == 0:
        LeapYear = True
    if year % 100 == 0:
        LeapYear = False
    if year % 400 == 0:
        LeapYear = True

    return LeapYear


# 05.07.2019,FM: tested
def MultipleLeapYears(years):

    """Checks a list of multiple years to find out which one of them is a leap year.
    In turn calls function IsLeapYear.

    Parameters:
    -----------
    years: :obj:`list`
        list of years which shall be checked

    Returns:
    --------
    LeapYears:
        list of LeapYears

    """
    LeapYearsBoolean = [IsLeapYear(el) for el in years]
    # 11.12.2018, JK: To improve functionality, a set could be used to remove duplicates which may occur in input list
    # Since this is more or less a throwaway function, there is no need to do that here.
    # 11.12.2018, JK: compress is used here to be able to filter list years using LeapYearsBoolean, i.e. a list with boolean filter values
    LeapYears = list(compress(years, LeapYearsBoolean))
    return LeapYears


# TODO: Resume with resampled timeseries (JK)
# -> There probably exist some routines from demand regio which may be used...
# 05.07.2019, FM:Not yet tested
def resample_timeseries(
    timeseries, freq, aggregation_rule="sum", interpolation_rule="linear"
):

    """Resample a timeseries to the frequency provided.
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

    # 16.05.2019, JK: determine frequency of timeseries given
    # For an alternative approach see: https://stackoverflow.com/questions/31517728/python-pandas-detecting-frequency-of-time-series
    try:
        original_freq = pd.infer_freq(timeseries.index, warn=True)

    # Error occurs when only two neigbouring timesteps are simulated which
    # might be a potential use case for the model
    # Function needs at least 3 values to obtain freq of a timeseries
    except ValueError:
        original_freq = "AS"

    # 16.05.2019, JK: Determine whether upsampling or downsampling is needed
    # 16.05.2019, JK: Workaround needed here
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


def convert_annual_limit(annual_limit, starttime, endtime):
    """Convert an annual limit to a sub- or multi-annual one

    Parameters
    ----------
    annual_limit: :obj:`float` or :obj:`pd.Series`of :class:`float`
        An annual limit (e.g. for emissions, investment budgets)
        if starttime and endtime are within the same year
        or a pd.Series of annual limits indexed by years if starttime and
        endtime are not within one year

    starttime: :obj:`str`
        The first date string; starttime of the optimization run

    endtime: :obj:`str`
        The second date string; endtime of the optimization run

    Returns
    -------
    new_limit: :obj:`float`
        A sub-annual / multi-annual limit for the optimization timeframe

    """
    dt_start = datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    dt_end = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S")
    start_year = dt_start.year
    end_year = dt_end.year

    new_limit = 0

    if start_year == end_year:
        day_diff = days_between(starttime, endtime)
        year_fraction = day_diff / float(365)
        if isinstance(annual_limit, float):
            new_limit = annual_limit * year_fraction
        else:
            new_limit = annual_limit[start_year] * year_fraction

    else:
        start_year_begin = str(start_year) + "-01-01 00:00:00"
        end_year_end = str(end_year) + "-12-31 23:59:59"
        day_diff_start = days_between(starttime, start_year_begin)
        day_diff_end = days_between(end_year_end, endtime)

        start_year_fraction = (365 - day_diff_start) / float(365)
        end_year_fraction = (365 - day_diff_end) / float(365)
        full_years = end_year - start_year - 1

        # Add annual limits for full years within optimization time frame
        for i in range(full_years):
            new_limit += annual_limit[start_year + i + 1]

        # Add limits for fractions of the start year and end year
        new_limit += (
            annual_limit[start_year] * start_year_fraction
            + annual_limit[end_year] * end_year_fraction
        )

    return new_limit


# Taken from oemof.tools.economics, but removed check for infeasible values
def calc_annuity(capex, n, wacc, u=None, cost_decrease=0):
    """Calculates the annuity of an initial investment 'capex', considering the
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


class GermanHolidayCalendar(AbstractHolidayCalendar):
    """Define rules for all national German holidays which are used
    for data preparation."""

    rules = [
        Holiday("New Years", month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday("Ascension Day", month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday("Whitmonday", month=1, day=1, offset=[Easter(), Day(50)]),
        Holiday("Labour Day", month=5, day=1),
        Holiday("German Union Day", month=10, day=3),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Boxing Day", month=12, day=26),
    ]
