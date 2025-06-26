#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("valid_dt_str", help = "Stage IV valid time to convert to nc (YYYYmmddHH)")
parser.add_argument("-p", "--plot_only", dest = "plot_only", default = False, action = "store_true",
                    help = "Set to make plots only, don't convert Stage IV grib to netCDF (e.g., if Stage IV netCDFs already exist)")
parser.add_argument("-r", "--region", dest = "region", default = "CONUS",
                    help = "Region for plotting; default CONUS")
args = parser.parse_args()

valid_dt = dt.datetime.strptime(args.valid_dt_str, "%Y%m%d%H")
main_dir = "/data/bbasarab/netcdf/testing"

# Input Stage IV grib file
input_file = f"st4_conus.{args.valid_dt_str}.01h.grb2"
input_fpath = os.path.join(main_dir, input_file)

# Output Stage IV netCDF file
output_dir = main_dir
output_file = f"StageIV.{args.valid_dt_str}.nc"
output_fpath = os.path.join(output_dir, output_file)

if not(args.plot_only):
    if not(os.path.exists(input_fpath)):
        print(f"Error: Input file {input_fpath} does not exist")
        sys.exit(1)

    print(f"Reading input grib file {input_fpath}")
    ds = xr.open_dataset(input_fpath, engine = "cfgrib")

    print(f"Writing output netcdf {output_fpath}")
    ds.to_netcdf(output_fpath)

print("Plotting data")

# Read and plot StageIV data
# Data labelled with end of the hour
stageiv_precip = xr.open_dataset(output_fpath).tp
data_name = "StageIV"
short_name = f"{data_name}.{args.valid_dt_str}"
pputils.plot_cmap_single_panel(stageiv_precip, data_name, args.region, np.arange(0, 55, 5), short_name = args.valid_dt_str)

# Read and plot AORC data from web server
# Daily data, each time is the end of the accumulation hour, from 00z current day to 23z current day 
input_fpath_aorc_web = os.path.join(main_dir, f"prate.aorc.{valid_dt:%Y%m%d}.nc")
aorc_web_precip = xr.open_dataset(input_fpath_aorc_web).prate.sel(time = valid_dt)
data_name = "AORC.web"
short_name = f"{data_name}.{args.valid_dt_str}"
pputils.plot_cmap_single_panel(aorc_web_precip, data_name, args.region, np.arange(0, 55, 5), short_name = args.valid_dt_str) 

# Read and plot AORC data from PSL Linux servers
# Daily data, each time is the end of the accumulation hour, from 01z current day to 00z next day
# THEREFORE, need PREVIOUS day's file to get precip valid at 00z
if (valid_dt.hour == 0):
    valid_dt_for_aorc_psl = valid_dt - dt.timedelta(days = 1)
else:
    valid_dt_for_aorc_psl = valid_dt
    
input_fpath_aorc_psl = os.path.join(main_dir, f"AORC.{valid_dt_for_aorc_psl:%Y%m%d}.precip.nc")
aorc_psl_precip = xr.open_dataset(input_fpath_aorc_psl).precrate.sel(time = valid_dt)
data_name = "AORC.psl"
short_name = f"{args.valid_dt_str}"
pputils.plot_cmap_single_panel(aorc_psl_precip, data_name, args.region, np.arange(0, 55, 5), short_name = args.valid_dt_str) 

