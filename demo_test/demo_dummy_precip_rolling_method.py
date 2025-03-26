#!/usr/bin/env python

import datetime as dt
import numpy as np
import xarray as xr

# PROOF OF CONCEPT for using the xarray roller object to efficiently calculate sums, averages, etc.

num_precip_rate_values = 12

# Original valid dtime list based on IMERG times: will represent beginning of each half-hourly time period
valid_dt_list = [dt.datetime(2019,5,1,0,0) + dt.timedelta(hours = 0.5 * i) for i in range(num_precip_rate_values)] 

# Make times period ending
half_hourly_precip_rates = np.arange(0, num_precip_rate_values, 1)
valid_dt_list_period_end = [i + dt.timedelta(hours = 0.5) for i in valid_dt_list]
dummy_precip = xr.DataArray(half_hourly_precip_rates, dims = "time", coords = dict(time = valid_dt_list_period_end))
#dummy_precip = xr.DataArray(np.arange(0, num_precip_rate_values, 1), dims = "time", coords = dict(time=imerg.valid_dt_list[0:num_precip_rate_values]))
print("*** Original period-ending valid datetimes and precip rates:")
for i in dummy_precip.time.values:
    print(i, ": ", dummy_precip.sel(time = i).values)

# Create the rolling object, then calculate hourly sums 
roller = dummy_precip.rolling(time = 2)
dummy_precip_hourly_sums = roller.sum()
# dummy_precip_hourly_amount.sel(time = "2019-05-01 01:00:00")

# Now index for only the desired values (sums valid at top of hour, over that previous full hour)
precip_hourly = 0.5 * dummy_precip_hourly_sums[1::2]
print("*** Hourly valid datetimes and precip amounts over the previous hour:")
for i in precip_hourly.time.values:
    print(i, ": ", precip_hourly.sel(time = i).values)

# Calculate three-hourly precip amounts
time_period_hours = 3
roller_3hr = precip_hourly.rolling(time = time_period_hours)
precip_3hourly = roller_3hr.sum()[(time_period_hours - 1)::time_period_hours]

# Calculate 6-hourly precip amounts
time_period_hours = 6
roller_6hr = precip_hourly.rolling(time = time_period_hours)
precip_6hourly = roller_6hr.sum()[(time_period_hours - 1)::time_period_hours]
