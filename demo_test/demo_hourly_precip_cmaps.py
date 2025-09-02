#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import precip_data_processors as pdp
import precip_plotting_utilities as ppu
import precip_verification_processor as pvp
import sys
import utilities as utils
import xarray as xr

utils.suppress_warnings()

def create_precip_colorbar():
    color_list = ["white",
                  "lightyellow", "oldlace", "tan", "peachpuff",
                  "lightgreen", "green", "forestgreen",
                  "deepskyblue", "dodgerblue", "blue",
                  "orange", "red", "violet", "magenta", "purple"]
    bounds = [0,1,2,3,4,5,6]
   
    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, bounds, norm

parser = argparse.ArgumentParser()
parser.add_argument("start_dt_str",
                    help = "Start date/time of verification; format YYYYmmdd.HH and PERIOD-ENDING")
parser.add_argument("end_dt_str",
                    help = "End date/time of verification; format YYYYmmdd.HH and PERIOD-ENDING")
parser.add_argument("--region", dest = "region", default = "US-Central", 
                    help = "Region for which to perform verification; default US-Central")
parser.add_argument("--plot_levels", dest = "plot_levels", nargs = "+", type = int, default = None,
                    help = "Plot levels for contour maps; pass as space-separate list with <min> <max> <step>")
parser.add_argument("--data_names_list", dest = "data_names_list", default = ["AORC", "NestedReplay", "CONUS404"], nargs = "+",
                    help = "List of data names to verify; pass space-separated with the truth data name listed first")
args = parser.parse_args()

truth_data_name = "AORC"

# Determine plot levels
if (args.plot_levels is None):
    match args.region:
        case "US-Central":
            plot_levels = np.arange(0, 42, 2)
        case "US-Mountain":
            plot_levels = np.arange(0, 32, 2)
        case "US-WestCoast":
            plot_levels = np.arange(0, 32, 2)
        case _:
            plot_levels = np.arange(0, 32, 2)
else:
    plot_levels = np.arange(args.plot_levels[0], args.plot_levels[1], args.plot_levels[2])

start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d.%H")
end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d.%H")

# 3-hourly data and period beginning; adjust start and end times accordingly
# Get one additional earlier time so have sufficient data from which to temporally interpolate
era5_start_dt = start_dt - dt.timedelta(hours = 4)
era5_end_dt = end_dt - dt.timedelta(hours = 3)

# Construct DataArray dictionary
da_dict = {}
for data_name in args.data_names_list:
    match data_name:
        case "AORC":
            processor = pdp.AorcDataProcessor(args.start_dt_str, args.end_dt_str, region = args.region) 
            da = processor.get_precip_data(load = True)
        case "CONUS404": 
            processor = pdp.CONUS404DataProcessor(args.start_dt_str, args.end_dt_str, region = args.region) 
            da = processor.get_precip_data(load = True)
        case "NestedReplay":
            processor = pdp.NestedReplayDataProcessor(args.start_dt_str, args.end_dt_str, region = args.region) 
            da = processor.get_precip_data(load = True)
        case "ERA5":
            processor = pdp.ERA5DataProcessor(era5_start_dt.strftime("%Y%m%d.%H"), era5_end_dt.strftime("%Y%m%d.%H"), region = args.region)
            da = processor.get_precip_data(load = True)

    da_dict[data_name] = da

# Temporally interpolate ERA5 3-hourly data to hourly
# Put in loop in case we don't have ERA5 data: in this case this step will just be skipped 
for data_name, da in da_dict.items():
    if (data_name == "ERA5"):
        print("Temporally interpolating 3-hourly ERA5 data to hourly")
        da_dict[data_name] = da.interp(period_end_time = da_dict[truth_data_name].period_end_time)

        pdp.add_attributes_to_data_array(da,
                                         short_name = "01-hour precipitation",
                                         long_name = "Precipitation accumulated over the prior 1 hour(s)",
                                         interval_hours = 1)

# Instantiate PrecipVerificationProcessor class with USE_EXTERNAL_DA_DICT = True
verif = pvp.PrecipVerificationProcessor(args.start_dt_str, args.end_dt_str,
                                        LOAD_DATA = False,
                                        USE_EXTERNAL_DA_DICT = True,
                                        external_da_dict = da_dict,   
                                        data_names = args.data_names_list, 
                                        truth_data_name = truth_data_name, 
                                        region = args.region, 
                                        temporal_res = 1)

# Plot contour maps
#verif.plot_cmap_multi_panel(data_dict = verif.da_dict, plot_levels = plot_levels, extend = "max")

# Use functions from precip_plotting_utilities.py
ppu.plot_cmap_multi_panel(da_dict, truth_data_name, args.region,
                          plot_levels = plot_levels,
                          short_name = "01_hour_precipitation",
                          extend = "max")
