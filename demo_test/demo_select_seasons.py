#!/usr/bin/env/python

import datetime as dt
import dateutil
import numpy as np
import xarray as xr
import pandas as pd
import utilities as utils

def construct_season_dt_ranges(da):
    all_dtimes = [pd.Timestamp(i) for i in da["time"].values]
    current_dt =  all_dtimes[0]
    
    season_dt_ranges = []
    while (utils.datetime2unix(current_dt) <= utils.datetime2unix(all_dtimes[-1])):
        if (current_dt.month < 3):
            start_dt_next_season = pd.Timestamp(current_dt.year, 3, 1)
        elif (current_dt.month < 6):
            start_dt_next_season = pd.Timestamp(current_dt.year, 6, 1)
        elif (current_dt.month < 9):
            start_dt_next_season = pd.Timestamp(current_dt.year, 9, 1)
        elif (current_dt.month < 12):
            start_dt_next_season = pd.Timestamp(current_dt.year, 12, 1)
        else: # December case: next season starts in March of NEXT year
            start_dt_next_season = pd.Timestamp(current_dt.year + 1, 3, 1)

        # Since slicing will be inclusive, the dt range should end on the final day of the current season
        season_dt_range = [current_dt, start_dt_next_season - pd.Timedelta(days = 1)]
        season_dt_ranges.append(season_dt_range)
        current_dt = start_dt_next_season

    return season_dt_ranges

x = np.arange(36)
first_dtime = pd.Timestamp(2017,1,15)
last_dtime = pd.Timestamp(2019,12,15)

valid_monthly_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.MONTHLY, dtstart = first_dtime, until = last_dtime))
valid_annual_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.YEARLY, dtstart = first_dtime, until = last_dtime))
xda = xr.DataArray(x, coords = [valid_monthly_dt_list], dims = "time")

season_dt_ranges = construct_season_dt_ranges(xda)

# Almost! The above gets Jan 2017, Feb 2017, Dec 2017. But for the first winter we want [Dec 2016], Jan 2017, Feb 2017
# The three remaining seasons will be easier. You can follow the logic above since the entire season falls in one year.
# For winter, we need to include logic that if the month is December, select from the PREVIOUS year, accounting
# for instances when that year is not in the dataset (e.g., our dataset starts in January)

#djf_month_list = [12,1,2]
#mam_month_list = [3,4,5]
#jja_month_list = [6,7,8]
#son_month_list = [9,10,11]

#is_winter_month = xda.time.dt.month.isin(djf_month_list)
#is_2017 = xda.time.dt.year.isin([2017])

#xda_winter = xda.sel(time = is_winter_month)
#is_winter_2017 = xda_winter.time.dt.year.isin([2017])
#xda_winter_2017 = xda_winter.sel(time = is_winter_2017)
#xda_winter_2017

# Easier method (possibly):
# Figure out current_dt, the datetime at which the dataset starts
# Based on this, set the beginning of the next season, dt_next_season. For example:
    # If data starts 2002-01-01, dt_next_season = 2002-03-01
    # If data starts 2002-08-17, dt_next_season = 2002-09-01
    # If data starts 2008-03-01, dt_next_season = 2008-06-01
# Select data between current_dt and (dt_next_season - 1day)
# Repeat, incrementing current_dt forward such that it now equals dt_next_season
