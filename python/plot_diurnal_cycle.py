#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_data_processors as pdp
import precip_plotting_utilities as ppu
import precip_verification_processor as pvp
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to process data")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to process data")
    parser.add_argument("--exclude_zeros", dest = "exclude_zeros", default = False, action = "store_true",
                        help = "Set to exclude zero values from precip calculations")
    parser.add_argument("--region", dest = "region", default = "US-Central",
                        help = "For plotting only: region to zoom plots to (default US-Central)")
    args = parser.parse_args()

    verif = pvp.PrecipVerificationProcessor(args.start_dt_str, 
                                            args.end_dt_str,
                                            data_names = ["AORC", "CONUS404", "NestedReplay"], 
                                            truth_data_name = "AORC", 
                                            data_grid = "AORC",
                                            region = args.region,
                                            temporal_res = 1)

    utc_hour_list = np.arange(0, 24)
    local_hour_list = [convert_utc_hour_to_local_hour(ihr, 5) for ihr in utc_hour_list]

    # Organize into nested dictionary of the form data_name[local_hour][precip_for_that_local_hour]
    diurnal_precip_dict = {}
    for data_name, da in verif.da_dict.items():
        for ihr in utc_hour_list:
            ihr_precip = da.sel(period_end_time = da.period_end_time.dt.hour.isin( [ihr] ))

            if args.exclude_zeros:
                ihr_mean = ihr_precip.where(ihr_precip > 0.0).mean().item()
            else:
                ihr_mean = ihr_precip.mean().item()
   
            if (ihr == 0): # First instance, initialize dictionary
                diurnal_precip_dict[data_name] = {local_hour_list[ihr]: ihr_mean} 
            else:
                diurnal_precip_dict[data_name][ local_hour_list[ihr] ] = ihr_mean
    
    # Plot
    if args.exclude_zeros:
        ylims = [0.0, 2.4]
        ystep = 0.2
    else:
        ylims = [0.0, 0.24]
        ystep = 0.02
    plot_diurnal_cycle(diurnal_precip_dict, args.start_dt_str, args.end_dt_str, args.region,
                       ylims = ylims, ystep = ystep, exclude_zeros = args.exclude_zeros)

    return diurnal_precip_dict 

def convert_utc_hour_to_local_hour(utc_hour, tz_offset):
    if (utc_hour < tz_offset):
        return (utc_hour - tz_offset) + 24 
    return utc_hour - tz_offset

def plot_diurnal_cycle(diurnal_precip_dict, start_dt_str, end_dt_str, region,
                   ylims = [0.0, 0.24], ystep = 0.02, exclude_zeros = False):
    # Create figure
    plt.figure(figsize = (15, 10))
    axis = plt.gca()
    plt.title(f"Diurnal cycle hourly mean precip; {start_dt_str}-{end_dt_str}; {region}", size = 15)

    # Plot
    for data_name, precip_dict in diurnal_precip_dict.items():
        sorted_hour_list = sorted(precip_dict.keys())
        precip_ts = [precip_dict[ihr] for ihr in sorted_hour_list]
        axis.plot(sorted_hour_list, precip_ts,
                  label = data_name,
                  color = ppu.time_series_color_dict[data_name],
                  linewidth = 3)
    plt.legend(loc = "best", prop = {"size": 15})

    # Set axes grid lines, limits, ticks
    plt.grid(True, linewidth = 0.5)
    plt.xlim(0, 23)
    plt.ylim(ylims[0], ylims[-1])
    axis.set_xticks(np.arange(0, 24))
    axis.set_yticks(np.arange(ylims[0], ylims[-1] + ystep, ystep))
    plt.xlabel("Hour of day (local time)", size = 15)
    plt.ylabel("Average hourly precip amount (mm)", size = 15)
    axis.tick_params(axis = "both", labelsize = 15)

    # Save figure
    data_names_str = ppu.get_data_names_str(diurnal_precip_dict) 
    exclude_zeros_str = ""
    if exclude_zeros:
        exclude_zeros_str = "exclude_zeros."
    fig_name = f"diurnal_cycle_precip_mean.{data_names_str}{exclude_zeros_str}{start_dt_str}-{end_dt_str}.{region}.png"
    fig_path = os.path.join(utils.plot_output_dir, fig_name)
    print(f"Saving {fig_path}")  
    plt.savefig(fig_path)

if __name__ == "__main__":
    diurnal_precip_dict = main()
