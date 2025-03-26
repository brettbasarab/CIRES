#!/usr/bin/env python

import argparse
import datetime as dt
import pandas as pd
import numpy as np
import os
import utilities as utils
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--temporal_res", dest = "temporal_res", type = int, default = 1)
parser.add_argument("-m", "--nc_write_method", dest = "nc_write_method", choices = ["label", "integer"])
args = parser.parse_args()

# Write to netCDF
temporal_res = args.temporal_res 
data_size = int(72/temporal_res)
first_valid_dt = pd.Timestamp(year = 2000, month = 1, day = 1, hour = temporal_res)
data = np.random.randint(0, high = 10, size = data_size) 
coords = pd.date_range(first_valid_dt, periods = data_size, freq = dt.timedelta(hours = temporal_res))
da = xr.DataArray(data, coords = [coords], dims = ["time"]) 
da.name = "accum_precip"
da.attrs["long_name"] = "Hourly accumulated precip mm"
da.attrs["short_name"] = "accum_precip"
da.to_netcdf("/home/bbasarab/netcdf/test/test.nc")

# Write to netCDF, broken out by day
# Assuming precip is accumulated over prior hour, we want
# all values valid on the current day and the value valid at 00z the next day.

# Explicit method: xarray by-label positional indexing
if (args.nc_write_method == "label"):
    valid_dt = pd.Timestamp(da["time"].values[0])
    next_valid_dt = valid_dt + dt.timedelta(hours = (24 - temporal_res)) 
    valid_day = utils.numpy_dt64_to_pandas_datetime_ymd(valid_dt)
    while (valid_day != da["time"].values[-1]):
        valid_dt_str = valid_dt.strftime("%Y-%m-%d %H:%M:%S")
        next_valid_dt_str = next_valid_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(valid_dt_str)
        print(next_valid_dt_str)

        #data_to_write = da.sel(time = slice(valid_day_str, next_valid_day_str))
        data_to_write = da.loc[valid_dt_str:next_valid_dt_str]
        ncf_name = f"test.{valid_dt:%Y%m%d}.nc"
        ncf_path = os.path.join("/home/bbasarab/netcdf/test", ncf_name)
        print(f"Writing {ncf_path}")
        data_to_write.to_netcdf(ncf_path)

        valid_dt += dt.timedelta(days = 1) 
        next_valid_dt = valid_dt + dt.timedelta(hours = (24 - temporal_res))
        valid_day += dt.timedelta(days = 1)

# Simpler method: xarray by-integer position indexing
if (args.nc_write_method == "integer"):
    start_day_index = 0
    step = int(24/temporal_res)
    end_day_index = step 
    while (end_day_index <= da["time"].shape[0]):
        print(start_day_index, end_day_index)
        data_to_write = da[start_day_index:end_day_index]
        print(data_to_write.values)

        valid_dt = pd.Timestamp(da["time"].values[start_day_index])
        ncf_name = f"test.{valid_dt:%Y%m%d}.nc"
        ncf_path = os.path.join("/home/bbasarab/netcdf/test", ncf_name)
        print(f"Writing {ncf_path}")
        data_to_write.to_netcdf(ncf_path)

        start_day_index += step 
        end_day_index += step 

