#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_data_processors as pdp
import precip_plotting_utilities as ppu
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

    # Read data
    proc_aorc = pdp.AorcDataProcessor(args.start_dt_str,
                                      args.end_dt_str,
                                      region = args.region) 

    proc_conus404 = pdp.CONUS404DataProcessor(args.start_dt_str,
                                              args.end_dt_str,
                                              region = args.region)

    proc_nr = pdp.NestedReplayDataProcessor(args.start_dt_str,
                                            args.end_dt_str,
                                            region = args.region)

    # Get hourly precip data
    aorc_precip_hourly = proc_aorc.get_precip_data(load = True) 
    conus404_precip_hourly = proc_conus404.get_precip_data(load = True) 
    nr_precip_hourly = proc_nr.get_precip_data(load = True)

    # Organize into nested dictionary of the form data_name[local_hour][precip_for_that_local_hour]
    aorc_diurnal = {}
    conus404_diurnal = {}
    nr_diurnal = {}
    utc_hour_list = np.arange(0, 24)
    local_hour_list = [convert_utc_hour_to_local_hour(ihr, 5) for ihr in utc_hour_list] 
    for ihr in utc_hour_list:
        aorc_precip = aorc_precip_hourly.sel(period_end_time = aorc_precip_hourly.period_end_time.dt.hour.isin( [ihr] ))
        conus404_precip = conus404_precip_hourly.sel(period_end_time = conus404_precip_hourly.period_end_time.dt.hour.isin( [ihr] ))
        nr_precip = nr_precip_hourly.sel(period_end_time = nr_precip_hourly.period_end_time.dt.hour.isin( [ihr] ))

        if args.exclude_zeros:
            aorc_diurnal[local_hour_list[ihr]] = aorc_precip.where(aorc_precip > 0.0).mean().item()
            conus404_diurnal[local_hour_list[ihr]] = conus404_precip.where(conus404_precip > 0.0).mean().item()
            nr_diurnal[local_hour_list[ihr]] = nr_precip.where(nr_precip > 0.0).mean().item()
        else:
            aorc_diurnal[local_hour_list[ihr]] = aorc_precip.mean().item()
            conus404_diurnal[local_hour_list[ihr]] = conus404_precip.mean().item()
            nr_diurnal[local_hour_list[ihr]] = nr_precip.mean().item()

    diurnal_precip_dict = {"AORC": aorc_diurnal, "CONUS404": conus404_diurnal, "NestedReplay": nr_diurnal}

    # Plot
    plot_diurnal_cycle(diurnal_precip_dict, args.start_dt_str, args.end_dt_str, args.region,
                       exclude_zeros = args.exclude_zeros)

    return diurnal_precip_dict 

def convert_utc_hour_to_local_hour(utc_hour, tz_offset):
    if (utc_hour < tz_offset):
        return (utc_hour - tz_offset) + 24 
    return utc_hour - tz_offset

def plot_diurnal_cycle(diurnal_precip_dict, start_dt_str, end_dt_str, region,
                       ylims = [0.0, 0.2], ystep = 0.02, exclude_zeros = False):
    # Create figure
    plt.figure(figsize = (12,10))
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
