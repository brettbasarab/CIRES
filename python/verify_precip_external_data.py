#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import os
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import warnings
import xarray as xr

def main():
    utils.suppress_warnings()
    program_description = ("Program to create a single instance of PrecipVerificationProcessor and use its functionality to\n"
                          "read external data and make corresponding plots.")
    parser = argparse.ArgumentParser(description = program_description)
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format YYYYmmdd.HH")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format YYYYmmdd.HH")
    parser.add_argument("external_data_list", nargs = "+", 
                        help = ("Space-separated list of the type of external data to read from nc files and pass to PrecipVerificationProcessor.\n"
                                "Pass strings of format <time_period_type>.<stat_type>.<agg_type>, e.g., 'monthly.mean.space_time'."))
    parser.add_argument("--region", dest = "region", default = "Global",
                        help = "Region to zoom plot to; default Global")
    parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--poster", dest = "poster", action = "store_true", default = False,
                        help = "Set if plots to be produced are for a poster; will make plot fonts bigger")
    args = parser.parse_args()

    data_names, truth_data_name = precip_verification_processor.map_region_to_data_names(args.region) 

    for external_data_str in args.external_data_list:
        time_period_type, stat_type, agg_type = get_time_period_stat_agg_info(external_data_str)
        if (time_period_type is None):
            continue

        external_da_dict = read_data_from_nc_files(data_names, args.temporal_res, args.region, args.start_dt_str, args.end_dt_str,
                                                   on_replay_grid = True, time_period_type = time_period_type, stat_type = stat_type, agg_type = agg_type) 

        verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                          args.end_dt_str,
                                                                          LOAD_DATA_FLAG = False, 
                                                                          external_da_dict = external_da_dict, 
                                                                          data_names = data_names, 
                                                                          truth_data_name = truth_data_name,
                                                                          region = args.region,
                                                                          temporal_res = args.temporal_res,
                                                                          poster = args.poster)

        match agg_type:
            case "time":
                single_set_of_levels, plot_levels = set_cmap_plot_levels(args.region, time_period_type, stat_type) 
                verif.plot_cmap_multi_panel(single_colorbar = True, single_set_of_levels = single_set_of_levels, input_levels = plot_levels, plot_errors = False)
                verif.plot_cmap_multi_panel(single_colorbar = False, single_set_of_levels = single_set_of_levels, input_levels = plot_levels, plot_errors = True)
            case "space_time":
                verif.plot_timeseries(time_period_type = time_period_type, stat_type = stat_type,
                                      ann_plot = False, which_ann_text = "all", pct_errors_ann_text = True, write_stats = True)
            case _:
                print(f"Error: Unsupported agg_type {agg_type}")
                continue

    return verif

def get_time_period_stat_agg_info(external_data_str):
    str_split = external_data_str.split(".")
    time_period_type = str_split[0]
    if (len(str_split) == 3):
        stat_type = str_split[1]
        agg_type = str_split[2]
    elif (len(str_split) == 4): # Accounts for there being another period in pctl stat types (e.g., '99.0th_pctl')
        stat_type = str_split[1] + "." + str_split[2]
        agg_type = str_split[3]
    else:
        print(f"Error: Invalid external data type {external_data_str}")
        return None, None, None 

    return time_period_type, stat_type, agg_type 

def read_data_from_nc_files(data_names, temporal_res, region, start_dt_str, end_dt_str,
                            on_replay_grid = True, time_period_type = "monthly", stat_type = "mean", agg_type = "space_time"):
    print("**** Reading external data from nc files")
    var_name = f"{temporal_res:02d}_hour_precipitation"

    start_dt = dt.datetime.strptime(start_dt_str, "%Y%m%d.%H")
    end_dt = dt.datetime.strptime(end_dt_str, "%Y%m%d.%H")

    # For now, the dt_range_str is hard coded to be YYYYmm-YYYYmm; this may need to be generalized
    if (end_dt.day == 1 and end_dt.hour == 0): # We won't actually have data over the last month (e.g., if last time period is 20080201.00)
        end_dt = end_dt - dt.timedelta(days = 1)
    dt_range_str = f"{start_dt:%Y%m}-{end_dt:%Y%m}"

    da_dict = {}
    for data_name in data_names:
        print(f"Reading {data_name} data")
        if (on_replay_grid and data_name != "Replay"):
            dataset_name = data_name + ".ReplayGrid"
        else:
            dataset_name = data_name + ".NativeGrid"
        
        nc_dir = os.path.join(utils.data_nc_dir, f"{dataset_name}.{var_name}.stats")
        nc_fname = f"{dataset_name}.{var_name}.{time_period_type}.{stat_type}.{agg_type}.{dt_range_str}.{region}.nc"
        fpath = os.path.join(nc_dir, nc_fname)
        if (not os.path.exists(fpath)):
            print(f"Error: Input nc file path {fpath} does not exist")
            sys.exit(1)
        print(f"Reading nc file {fpath}")
        da = xr.open_dataset(fpath)[f"precipitation_{temporal_res}_hour"] 

        da_dict[data_name] = da
    return da_dict

def set_cmap_plot_levels(region, time_period_type, stat_type):
    single_set_of_levels = False
    plot_levels = None 

    if (stat_type == "mean") and (time_period_type != "monthly"):
        single_set_of_levels = True
        plot_levels = pputils.regions_info_dict[region].cm_mean_precip_range

    return single_set_of_levels, plot_levels

if __name__ == "__main__":
    verif = main()
