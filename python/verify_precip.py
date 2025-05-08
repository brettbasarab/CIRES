#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import warnings
import xarray as xr

def main():
    utils.suppress_warnings()
    program_description = ("Program to create an instance of PrecipVerificationProcessor and use its functionality to create\n"
                          "many different types of plots, summarized in the flag descriptions below.")
    parser = argparse.ArgumentParser(description = program_description)
    # Fundamental positional arguments and flags to configure verification properly
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "Region to zoom plot to; default CONUS")
    parser.add_argument("--region_ext", dest = "region_ext", nargs = "+", type = int,
                        help = "Lon and lat boundaries for a non-standard region; order llon, ulon, llat, ulat; region name must also be provided as --region flag") 
    parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--data_names_list", dest = "data_names_list", default = None, nargs = "+",
                        help = "List of data names to verify; pass space-separated with the truth data name listed first; if None, all possible datasets for the given region are verified")
    parser.add_argument("--grid_cell_size", dest = "grid_cell_size", type = float, default = utils.replay_grid_cell_size, 
                        help = f"Assumed grid cell size for FSS and other calculations; default Replay grid cell size of {utils.replay_grid_cell_size} degrees")
    parser.add_argument("--time_period_types", dest = "time_period_types", default = ["full_period", "common_seasonal", "common_monthly"], nargs = "+", 
                        help = "Time period types over which to perform analysis; default ['full_period', 'common_seasonal', 'common_monthly']")
    parser.add_argument("--high_res", dest = "high_res", action = "store_true", default = False,
                        help = "Set to verify CONUS high-resolution datasets (AORC, CONUS404, NestedReplay, etc.)")
    # Statistics flags: set which statistic(s) to calculated
    parser.add_argument("--cmaps", dest = "cmaps", action = "store_true", default = False,
                        help = "Set to plot contour maps (cmaps) of mean and percentiles of gridded data aggregated over various time periods")
    parser.add_argument("--fss", dest = "fss", action = "store_true", default = False,
                        help = "Calculate fractions skill score (FSS) averaged over full evaluation time period") 
    parser.add_argument("--fss_pctl_threshold", dest = "fss_pctl_threshold", action = "store_true", default = False,
                        help = "Set to calculate FSS based on percentile (rather than amount) thresholds")
    parser.add_argument("--fss_ari_grids", dest = "fss_ari_grids", action = "store_true", default = False,
                        help = "Set to calculate FSS based on ARI grid (rather than amount) thresholds")
    parser.add_argument("--pdfs", dest = "pdfs", action = "store_true", default = False,
                        help = "Calculate full-period and common-seasonal PDFs (and eventually, CDFs) of precip") 
    parser.add_argument("--timeseries", dest = "timeseries", action = "store_true", default = False,
                        help = "Calculate time series of precip stats aggregated over individual months, seasons, years, etc. in the specified time period")
    # Write to netCDF flags: set to write calculated statistic(s) to netCDF
    parser.add_argument("--write_to_nc", dest = "write_to_nc", default = False, action = "store_true",
                        help = "Set to write calcualted stats to netCDF")
    # Plotting flags
    parser.add_argument("--plot", dest = "plot", action = "store_true", default = False,
                        help = "Set to make plots of calculated stats; otherwise only output stats to netCDF files")
    parser.add_argument("--poster", dest = "poster", action = "store_true", default = False,
                        help = "Set if plots to be produced are for a poster; will make plot fonts bigger")
    parser.add_argument("--scatter_plot", dest = "scatter_plot", action = "store_true", default = False,
                        help = "Create additional scatter plots of accumulated precip over entire specified time period (most useful for case studies)")
    args = parser.parse_args()

    # Set up verification by instantiating the PrecipVerificationProcessor class
    if (type(args.data_names_list) is list) and (len(args.data_names_list) >= 1):
        data_names, truth_data_name, data_grid = args.data_names_list, args.data_names_list[0]
    else:
        data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region, high_res = args.high_res) 

    verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                      args.end_dt_str,
                                                                      LOAD_DATA_FLAG = True, 
                                                                      external_da_dict = None, 
                                                                      data_names = data_names, 
                                                                      truth_data_name = truth_data_name,
                                                                      data_grid = data_grid,
                                                                      region = args.region,
                                                                      region_extent = args.region_ext,
                                                                      temporal_res = args.temporal_res,
                                                                      poster = args.poster)

    # Calculate data for and plot contour maps of specified statistics valid at each grid point (so
    # contour maps can be made of the resulting data) and across various aggregation time periods.
    if args.cmaps:
        for time_period_type in args.time_period_types: 
            print(f"**** Calculating {time_period_type} aggregated data for contour maps")
            # Mean
            agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "mean", agg_type = "time", write_to_nc = args.write_to_nc)
            if args.plot:
                precip_range = pputils.regions_info_dict[verif.region].cm_mean_precip_range
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = True, plot_errors = False, input_levels = precip_range)
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = True, plot_errors = True, input_levels = precip_range)

            # 95th percentile
            agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 95, write_to_nc = args.write_to_nc)
            if args.plot:
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)

            # 99th percentile
            agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 99, write_to_nc = args.write_to_nc)
            if args.plot:
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)

            # 99.9th percentile
            agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 99.9, write_to_nc = args.write_to_nc)
            if args.plot:
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)
    
    # Calculate and plot fractions skill score (FSS) by radius and threshold
    if args.fss:
        print("**** Calculating FSS")
        eval_radius_list = utils.default_eval_radius_list_grid_cells * args.grid_cell_size 
        if args.fss_ari_grids:
            # Calculate FSS by radius, using an ARI grid as threshold
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius_ari_threshold", grid_cell_size = args.grid_cell_size,
                                                     fixed_ari_threshold = 2, 
                                                     eval_radius_list = eval_radius_list,
                                                     write_to_nc = args.write_to_nc)
            # Calculate FSS by ARI grid (i.e., using varying ARI grids as thresholds) 
            fss_dict_by_ari = verif.calculate_fss(eval_type = "by_ari_grid", grid_cell_size = args.grid_cell_size,
                                                  fixed_radius = 2 * args.grid_cell_size,
                                                  eval_ari_list = utils.default_eval_ari_list_years,
                                                  write_to_nc = args.write_to_nc)

            # Plot FSS, averaged across evaluation period 
            if args.plot:
                for time_period_type in args.time_period_types:
                    print(f"**** Calculating and plotting {time_period_type} aggregated FSS")
                    verif.plot_aggregated_fss(eval_type = "by_radius_ari_threshold", xaxis_explicit_values = False,
                                              time_period_type = time_period_type)
                    verif.plot_aggregated_fss(eval_type = "by_ari_grid", xaxis_explicit_values = True,
                                              time_period_type = time_period_type)
        else:
            is_pctl_threshold = False
            fixed_threshold = 10 # mm
            eval_threshold_list = utils.default_eval_threshold_list_mm
            if args.fss_pctl_threshold:
                is_pctl_threshold = True
                fixed_threshold = 95.0 # percentile (75th percentile)
                eval_threshold_list = utils.default_eval_threshold_list_pctl

            # Calculate FSS by radius and threshold for each valid time
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius", grid_cell_size = args.grid_cell_size,
                                                     fixed_threshold = fixed_threshold, 
                                                     eval_radius_list = eval_radius_list,
                                                     is_pctl_threshold = is_pctl_threshold,
                                                     include_zeros = False,
                                                     write_to_nc = args.write_to_nc)
            fss_dict_by_thresh = verif.calculate_fss(eval_type = "by_threshold", grid_cell_size = args.grid_cell_size,
                                                     fixed_radius = 2 * args.grid_cell_size,
                                                     eval_threshold_list = eval_threshold_list,
                                                     is_pctl_threshold = is_pctl_threshold,
                                                     include_zeros = False,
                                                     write_to_nc = args.write_to_nc)

            # Plot FSS, averaged across evaluation period 
            if args.plot:
                for time_period_type in args.time_period_types:
                    print(f"**** Calculating and plotting {time_period_type} aggregated FSS")
                    verif.plot_aggregated_fss(eval_type = "by_radius", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = is_pctl_threshold)
                    verif.plot_aggregated_fss(eval_type = "by_threshold", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = is_pctl_threshold,
                                              include_frequency_bias = True)

    # Plot PDFs and CDFs
    if args.pdfs:
        for time_period_type in args.time_period_types:
            print(f"**** Calculating {time_period_type} PDFs and CDFs")
            pdf_dict = verif.calculate_pdf(time_period_type = time_period_type, write_to_nc = args.write_to_nc)
            if args.plot:
                verif.plot_pdf(data_dict = pdf_dict, time_period_type = time_period_type) 

    # Plot monthly and seasonal mean timeseries
    if args.timeseries:
        for time_period_type in ["monthly", "seasonal"]:
            print(f"**** Calculating {time_period_type} timeseries stats")
            agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "mean", agg_type = "space_time", write_to_nc = args.write_to_nc) 
            if args.plot:
                verif.plot_timeseries(data_dict = agg_dict, time_period_type = time_period_type, stat_type = "mean")

    # Plot scatter plot of accumulated precip over entire time period (specified by args.start_dt_str-args.end_dt_str)
    if args.scatter_plot:
        print("**** Creating scatter plots of accumulated precip summed over entire time period")
        verif.plot_scatter_plot(summed_over_time_period = True)

    return verif

if __name__ == "__main__":
    verif = main()
